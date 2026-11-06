#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Circles of both hands through motion's pose chunks, to test its tracking without a policy.

- Each hand_root circles in the vertical x-z plane through its start pose with its start orientation:
  p(θ) = p0 + r·(cos θ − 1, 0, sin θ). Both hands keep their y, so the gap between them stays as at the start.
- θ̇ ramps in and out over --ramp with a cosine; Ctrl-C ramps it out on the circle.
- Every --chunk-period s a chunk of --horizon s, knots every --knot s, absolute in `world`.

Before anything is sent, the circle is solved to joint angles by IK from the commanded start and checked: joint limits,
joint speed against motion's max_joint_velocity, and PlanChecker's table and pillar margins. Run from ready:

    ros2 run oap_planner circle --radius 0.03 [--dry-run]
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time

import numpy as np
import pinocchio as pin
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Transform
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

from oap_planner.geometry import (LIMITS, READY_POSE, SAFETY_MARGIN_M, SIDES, TABLE_MARGIN_M,
                                  PlanChecker, joint_names)
from oap_planner.plan import NAMES, arm_model

CHECK_DT = 0.005  # [s]
CLEARANCE_DT = 0.1  # [s]
READY_TOL = 0.1  # [rad] the start must be this close to ready
MAX_JOINT_VELOCITY = 1.256  # [rad/s] oap_control config/controllers.yaml
IK_TOL = 1e-4  # [m, rad]


class Phase:
    """θ(t): θ̇ ramps up over `ramp`, holds ω, ramps down to end at turns·2π; stop(t_s) ramps it down from t_s."""

    def __init__(self, period, turns, ramp):
        self.w, self.ramp = 2.0 * math.pi / period, ramp
        self.total = 2.0 * math.pi * turns
        self.cruise = self.total / self.w - ramp
        if self.cruise < 0.0:
            raise ValueError("--ramp is longer than the turns allow")
        self.end = self.cruise + 2.0 * ramp
        self.t_stop = None

    def _ramp(self, tau):  # ∫ (1 − cos(π s/ramp))/2 ds over [0, tau]
        return 0.5 * tau - self.ramp / (2.0 * math.pi) * math.sin(math.pi * tau / self.ramp)

    def _plan(self, t):
        t = min(max(t, 0.0), self.end)
        if t < self.ramp:
            return self.w * self._ramp(t), self.w * 0.5 * (1.0 - math.cos(math.pi * t / self.ramp))
        if t < self.ramp + self.cruise:
            return self.w * (0.5 * self.ramp + t - self.ramp), self.w
        tau = self.end - t
        return self.total - self.w * self._ramp(tau), self.w * 0.5 * (1.0 - math.cos(math.pi * tau / self.ramp))

    def stop(self, t):
        if self.t_stop is None:
            self.t_stop = min(max(t, 0.0), self.end)
            self.theta_s, self.rate_s = self._plan(self.t_stop)

    def finished(self, t):
        return t >= (self.end if self.t_stop is None else self.t_stop + self.ramp)

    def __call__(self, t):
        if self.t_stop is None or t <= self.t_stop:
            return self._plan(t)[0]
        tau = min(t - self.t_stop, self.ramp)  # θ̇ = θ̇_s·(1 + cos(π τ/ramp))/2
        return self.theta_s + self.rate_s * (0.5 * tau + self.ramp / (2.0 * math.pi) * math.sin(math.pi * tau / self.ramp))


class Circle:
    def __init__(self, model, idx, q0, radius, phase):
        self.phase, self.radius = phase, radius
        data = model.createData()
        pin.framesForwardKinematics(model, data, q0)
        self.fid = {s: model.getFrameId(f"{s}_hand_root") for s in SIDES}
        self.start = {s: pin.SE3(data.oMf[self.fid[s]]) for s in SIDES}
        self.ts = np.arange(0.0, phase.end + CHECK_DT, CHECK_DT)
        self.Q = np.empty((len(self.ts), model.nq))
        self.residual = 0.0
        q = q0.copy()
        for i, t in enumerate(self.ts):
            for s in SIDES:
                q, res = self._ik(model, data, q, s, idx[s], self.target(s, t))
                self.residual = max(self.residual, res)
            self.Q[i] = q
        self.V = np.gradient(self.Q, CHECK_DT, axis=0)

    def target(self, side, t):
        th = self.phase(t)
        p = self.start[side].translation + self.radius * np.array([math.cos(th) - 1.0, 0.0, math.sin(th)])
        return pin.SE3(self.start[side].rotation, p)

    def _ik(self, model, data, q, side, cols, goal, iters=100, damping=1e-3):
        q = q.copy()
        for _ in range(iters):
            pin.framesForwardKinematics(model, data, q)
            err = pin.log6(data.oMf[self.fid[side]].actInv(goal)).vector
            if np.linalg.norm(err) < 1e-10:  # far below IK_TOL, so the joint path is smooth to differentiate
                break
            J = pin.computeFrameJacobian(model, data, q, self.fid[side], pin.LOCAL)[:, cols]
            q[cols] += J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
        pin.framesForwardKinematics(model, data, q)
        return q, float(np.linalg.norm(pin.log6(data.oMf[self.fid[side]].actInv(goal)).vector))


def problems(circle, idx) -> list[str]:
    out = []
    if circle.residual > IK_TOL:
        out.append(f"the IK misses the circle by {circle.residual * 1e3:.1f} mm")
    for s in SIDES:
        for name, j, (lo, hi) in zip(joint_names(s), idx[s], LIMITS[s]):
            if circle.Q[:, j].min() < lo or circle.Q[:, j].max() > hi:
                out.append(f"{name} leaves its limits")
    peak = np.abs(circle.V).max()
    if peak > 0.9 * MAX_JOINT_VELOCITY:
        out.append(f"a joint peaks at {peak:.2f} rad/s, over 90% of {MAX_JOINT_VELOCITY}")
    checker = PlanChecker()
    Qc = circle.Q[::int(round(CLEARANCE_DT / CHECK_DT))]
    for s in SIDES:
        c = [checker.clearance(s, q[idx[s]]) for q in Qc]
        table, pillar = min(t for t, _ in c), min(p for _, p in c)
        print(f"[circle] {s}: table {table * 1e3:+.0f} mm, pillar {pillar * 1e3:+.0f} mm")
        if pillar < SAFETY_MARGIN_M:
            out.append(f"{s}: pillar clearance {pillar * 1e3:.0f} mm, under the margin")
        if table < TABLE_MARGIN_M:
            out.append(f"{s}: table clearance {table * 1e3:.0f} mm, under the margin")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=float, default=0.03, help="[m]")
    ap.add_argument("--period", type=float, default=4.0, help="[s] one turn at full speed")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--ramp", type=float, default=2.0, help="[s] θ̇ in and out")
    ap.add_argument("--knot", type=float, default=0.1, help="[s] between a chunk's knots")
    ap.add_argument("--horizon", type=float, default=1.0, help="[s] covered by one chunk")
    ap.add_argument("--chunk-period", type=float, default=0.5, help="[s] between chunks")
    ap.add_argument("--dry-run", action="store_true", help="check from the ready pose; no ROS")
    args = ap.parse_args()
    model, idx = arm_model()
    ready = np.zeros(14)
    for s in SIDES:
        ready[idx[s]] = READY_POSE[s]
    rclpy.init()
    node = Node("circle")
    commanded = {}
    node.create_subscription(JointState, "/arm/command", lambda m: commanded.update(zip(m.name, m.position)), 10)
    try:
        q0 = ready
        if not args.dry_run:
            end = time.monotonic() + 3.0
            while not all(n in commanded for n in NAMES) and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
            if not all(n in commanded for n in NAMES):
                print("[circle] no /arm/command within 3 s", file=sys.stderr)
                return 1
            q0 = np.zeros(14)
            for s in SIDES:
                q0[idx[s]] = [commanded[n] for n in joint_names(s)]
        if np.abs(q0 - ready).max() > READY_TOL:
            print(f"[circle] the arms are {np.abs(q0 - ready).max():.2f} rad from ready; move there first",
                  file=sys.stderr)
            return 1
        phase = Phase(args.period, args.turns, args.ramp)
        circle = Circle(model, idx, q0, args.radius, phase)
        bad = problems(circle, idx)
        if bad:
            print("[circle] refusing:\n  " + "\n  ".join(bad), file=sys.stderr)
            return 1
        v = phase.w * args.radius
        print(f"[circle] {phase.end:.0f} s, radius {args.radius * 1e3:.0f} mm, peak hand speed {v:.3f} m/s; "
              "the checks pass")
        if args.dry_run:
            return 0

        pubs = {s: node.create_publisher(MultiDOFJointTrajectory, f"/arm/{s}/cmd_pose_trajectory", 10) for s in SIDES}
        t0 = time.time() + 0.5
        signal.signal(signal.SIGINT, lambda *_: phase.stop(time.time() - t0))
        n_knots = max(1, int(round(args.horizon / args.knot)))
        next_send = t0
        while rclpy.ok():
            t = time.time()
            if t >= next_send:
                last = phase.finished(t - t0 + args.horizon)
                stamp = node.get_clock().now()
                st = stamp.nanoseconds * 1e-9
                for s, pub in pubs.items():
                    m = MultiDOFJointTrajectory()
                    m.header.stamp = stamp.to_msg()
                    m.header.frame_id = "world"
                    m.joint_names = [f"{s}_hand_root"]
                    for k in range(1, n_knots + 1):
                        T = circle.target(s, t - t0 + k * args.knot)
                        q = pin.Quaternion(T.rotation)
                        tr = Transform()
                        tr.translation.x, tr.translation.y, tr.translation.z = (float(x) for x in T.translation)
                        tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w = q.x, q.y, q.z, q.w
                        d = t + k * args.knot - st
                        m.points.append(MultiDOFJointTrajectoryPoint(
                            transforms=[tr], time_from_start=Duration(sec=int(d), nanosec=int(round((d % 1.0) * 1e9)))))
                    pub.publish(m)
                if last:
                    break
                next_send += args.chunk_period
            rclpy.spin_once(node, timeout_sec=0.005)
        print("[circle] done; the hands hold their last target")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
