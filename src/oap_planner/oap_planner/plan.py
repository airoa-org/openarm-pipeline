#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Move both arms to the ready or park pose along one optimized trajectory.

One CasADi/IPOPT problem over both arms' 14 joints, from the commanded pose to the target in a fixed time: joint and
velocity limits, and one ellipsoid per gripper kept clear of the table box and the pillar cylinder after the start.
The arms first leave relax mode and the fingers go half open; a move with no solution exits with REFUSED, so the GUI
asks for the arms to be moved by hand. Settings: config/trajopt.yaml.

    ros2 run oap_planner plan --target ready [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import casadi as ca
import numpy as np
import pinocchio as pin
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from pinocchio import casadi as cpin
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from oap_planner.geometry import (LIMITS, READY_POSE, REFUSED, SAFETY_MARGIN_M, SIDES, TABLE_MARGIN_M, TABLE_TOP,
                                  TABLE_X, TABLE_Y, FINGER_CLOSED, FINGER_OPEN, joint_names)

# lifted: hands about 5.5 cm above park, off its singularity, so the arms can relax and sink without hitting anything
TARGETS = {"ready": READY_POSE, "park": {s: [0.0] * 7 for s in SIDES},
           "lifted": {"left": [0.6, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0], "right": [-0.6, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]}}
READY_GRIPPER = 0.72  # [rad] user space: fingers nearly fully open (FINGER_OPEN 0.7854)
TRANSIT_GRIPPER = 0.36  # half of READY_GRIPPER: a full close squeezes the fingers on every move
RELEASE_FRACTION = 0.2  # of the move a gripper that starts inside the table box may stay in it: 2 s of the 10 s move
DISCOVERY_S = 3.0
SETTLE_TOL = 0.05  # [rad] the target counts as reached within this
SETTLE_TIMEOUT_S = 4.0  # extra wait after the path ends
NAMES = [n for s in SIDES for n in joint_names(s)]


def load_config():
    with open(os.path.join(get_package_share_directory("oap_planner"), "config", "trajopt.yaml")) as f:
        return yaml.safe_load(f)


def arm_model():
    """The URDF reduced to the 14 arm joints, in NAMES order."""
    with open(os.path.join(get_package_share_directory("oap_description"), "urdf", "openarm_bimanual.urdf")) as f:
        full = pin.buildModelFromXML(f.read())
    lock = [j for j in range(1, full.njoints) if full.names[j] not in NAMES]
    model = pin.buildReducedModel(full, lock, pin.neutral(full))
    assert model.nq == 14, f"expected 14 arm joints, got {model.nq}"
    idx = {s: [model.idx_qs[model.getJointId(n)] for n in joint_names(s)] for s in SIDES}
    return model, idx


def table_separation(c, R, semi, lo, hi, k):
    """Smooth max over the table box's faces of the ellipsoid's distance outside each face plane; >= 0 means one face
    plane separates the ellipsoid from the box."""
    seps = []
    for axis in range(3):
        for sign, p in ((-1.0, lo[axis]), (1.0, hi[axis])):
            n = np.zeros(3)
            n[axis] = sign
            seps.append(sign * (c[axis] - p) - ca.norm_2(ca.DM(semi) * (R.T @ ca.DM(n))))
    return ca.logsumexp(k * ca.vertcat(*seps)) / k


def pillar_separation(c, R, semi, axis, radius):
    """Distance of the ellipsoid outside the vertical cylinder, along the horizontal direction from the axis to the
    ellipsoid center."""
    d = c[:2] - ca.DM(axis)
    dist = ca.norm_2(d)
    n = ca.vertcat(d / dist, 0)
    return dist - radius - ca.norm_2(ca.DM(semi) * (R.T @ n))


def separations(model, cfg):
    """sep(q) -> (table, pillar) separation of each gripper's ellipsoid, per side, as a CasADi function."""
    k = float(cfg["smooth_k"])
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    qs = ca.SX.sym("q", model.nq)
    cpin.framesForwardKinematics(cmodel, cdata, qs)
    lo, hi = (TABLE_X[0], TABLE_Y[0], -1.0), (TABLE_X[1], TABLE_Y[1], TABLE_TOP)
    table, pillar = [], []
    for s in SIDES:
        M = cdata.oMf[model.getFrameId(cfg["hand_frame"].format(side=s))]
        c = M.translation + M.rotation @ ca.DM(cfg["hand_center"])
        table.append(table_separation(c, M.rotation, cfg["hand_semi_axes"], lo, hi, k))
        pillar.append(pillar_separation(c, M.rotation, cfg["hand_semi_axes"], cfg["pillar_axis"],
                                        float(cfg["pillar_radius"])))
    return ca.Function("sep", [qs], [ca.vertcat(*table), ca.vertcat(*pillar)])


def released(model, cfg, q_start):
    """Per side, the samples after the start whose table clearance is not enforced: the first RELEASE_FRACTION of them
    when the gripper starts inside the table box, because holding it out from sample 2 throws the arm up; else none."""
    rows = int(np.ceil(RELEASE_FRACTION * int(cfg["steps"])))
    table, _ = separations(model, cfg)(q_start)
    return {s: rows if float(table[a]) < TABLE_MARGIN_M else 0 for a, s in enumerate(SIDES)}


def solve(model, idx, cfg, q_start, q_goal, release):
    """Return (Q (N+1, 14), knot velocities (N+1, 14), dt, ipopt status) in the model's joint order."""
    n, dt = int(cfg["steps"]), float(cfg["duration"]) / int(cfg["steps"])
    bias = np.log(6) / float(cfg["smooth_k"])  # logsumexp over 6 faces overestimates the max by <= this
    sep = separations(model, cfg)

    qmin, qmax = np.zeros(14), np.zeros(14)
    for s in SIDES:
        for j, (lo_, hi_) in zip(idx[s], LIMITS[s]):
            qmin[j], qmax[j] = lo_, hi_
    vmax = float(cfg["velocity_scale"]) * np.asarray(model.velocityLimit)

    opti = ca.Opti()
    Q, V = opti.variable(14, n + 1), opti.variable(14, n)
    opti.subject_to(Q[:, 0] == q_start)
    opti.subject_to(Q[:, n] == q_goal)
    opti.subject_to(V[:, 0] == 0)
    opti.subject_to(V[:, n - 1] == 0)
    for i in range(n):
        opti.subject_to(Q[:, i + 1] == Q[:, i] + dt * V[:, i])
        opti.subject_to(opti.bounded(-vmax, V[:, i], vmax))
    for i in range(2, n + 1):  # samples 0 and 1 are the start (V[0] = 0)
        opti.subject_to(opti.bounded(qmin, Q[:, i], qmax))
        t, p = sep(Q[:, i])
        for a, s in enumerate(SIDES):
            if i > 1 + release[s]:
                opti.subject_to(t[a] >= TABLE_MARGIN_M + bias)
        opti.subject_to(p >= SAFETY_MARGIN_M)
    opti.minimize(ca.sumsqr(V[:, 1:] - V[:, :-1]) + 1e-3 * ca.sumsqr(V))

    ramp = np.linspace(q_start, q_goal, n + 1).T
    opti.set_initial(Q, ramp)
    opti.set_initial(V, np.diff(ramp, axis=1) / dt)
    opti.solver("ipopt", {"print_time": False, "expand": True},
                {"print_level": 0, "max_iter": 3000, "tol": 1e-6, "acceptable_iter": 0})
    try:
        sol = opti.solve()
    except RuntimeError:
        return None, None, dt, opti.stats()["return_status"]
    Qs, Vs = np.asarray(sol.value(Q)).T, np.asarray(sol.value(V)).T
    Vk = np.vstack([np.zeros(14), (Vs[:-1] + Vs[1:]) / 2, np.zeros(14)])  # central differences: the spline follows the path
    return Qs, Vk, dt, sol.stats()["return_status"]


class Arms(Node):
    def __init__(self):
        super().__init__("planner")
        self.measured, self.commanded, self.fingers = {}, {}, {}
        self.create_subscription(JointState, "/joint_states", lambda m: self.measured.update(zip(m.name, m.position)), 10)
        self.create_subscription(JointState, "/arm/command", lambda m: self.commanded.update(zip(m.name, m.position)), 10)
        self.create_subscription(JointState, "/gripper/joint_states",
                                 lambda m: self.fingers.update(zip(m.name, m.position)), 10)
        self.path_pub = self.create_publisher(JointTrajectory, "/arm/joint_trajectory", 10)
        self.grip_pub = self.create_publisher(JointTrajectory, "/gripper/joint_trajectory", 10)
        self.hold = self.create_client(Trigger, "/arm/hold")
        self.relax = {s: self.create_client(SetBool, f"/openarm_hw_relax_{s}/set_relax") for s in SIDES}

    def spin_until(self, done, timeout):
        end = time.monotonic() + timeout
        while rclpy.ok() and not done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
        return done()

    def call(self, client, request, timeout=2.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        fut = client.call_async(request)
        self.spin_until(fut.done, timeout)
        return fut.result() if fut.done() else None

    def finger(self):
        """Measured opening per side in the URDF finger range; None where nothing publishes it."""
        got = {s: self.fingers.get(f"hand_{s}_motor_joint") for s in SIDES}
        return {s: None if v is None else min(max(v, FINGER_CLOSED), FINGER_OPEN) for s, v in got.items()}

    def set_gripper(self, opening, wait_s=3.0):
        m = JointTrajectory(joint_names=[f"hand_{s}_motor_joint" for s in SIDES])
        m.points = [JointTrajectoryPoint(positions=[opening, opening])]
        self.grip_pub.publish(m)
        self.spin_until(lambda: all(v is not None and abs(v - opening) < 0.05 for v in self.finger().values()), wait_s)
        return self.finger()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--dry-run", action="store_true", help="solve from the other target's pose; send nothing")
    args = ap.parse_args()
    tag = f"[plan {args.target}]"
    cfg = load_config()
    model, idx = arm_model()

    rclpy.init()
    node = Arms()
    try:
        if args.dry_run:
            other = "park" if args.target != "park" else "ready"
            q_start = np.zeros(14)
            for s in SIDES:
                q_start[idx[s]] = TARGETS[other][s]
        else:
            if not node.spin_until(lambda: all(n in node.commanded and n in node.measured for n in NAMES), DISCOVERY_S):
                print(f"{tag} no /arm/command or /joint_states within {DISCOVERY_S} s", file=sys.stderr)
                return 1
            node.spin_until(lambda: None not in node.finger().values(), 1.0)
            # The command first takes the measured pose, so leaving relax mode does not snap the arms back to it.
            if node.call(node.hold, Trigger.Request()) is None:
                print(f"{tag} no /arm/hold service: is motion active?", file=sys.stderr)
                return 1
            for s in SIDES:
                res = node.call(node.relax[s], SetBool.Request(data=False))
                if res is None or not res.success:  # mock hardware has no relax mode
                    print(f"{tag} {s}: relax mode not switched off ({'no service' if res is None else res.message})")
            node.spin_until(lambda: False, 0.3)
            node.call(node.hold, Trigger.Request())  # the pose the arms settled at leaving relax
            node.spin_until(lambda: False, 0.3)  # the next /arm/command carries the measured pose it holds
            node.commanded.clear()
            node.spin_until(lambda: all(n in node.commanded for n in NAMES), 1.0)
            q_start = np.zeros(14)
            for s in SIDES:
                q_start[idx[s]] = [node.commanded[n] for n in joint_names(s)]
            if None not in node.finger().values():  # no gripper node on mock hardware
                node.set_gripper(TRANSIT_GRIPPER)

        q_goal = np.zeros(14)
        for s in SIDES:
            q_goal[idx[s]] = TARGETS[args.target][s]
        release = released(model, cfg, q_start)
        t0 = time.perf_counter()
        Q, V, dt, status = solve(model, idx, cfg, q_start, q_goal, release)
        print(f"{tag} IPOPT: {status} ({time.perf_counter() - t0:.2f} s)")
        if Q is None:
            print(f"{tag} no safe trajectory; the arms stay where they are", file=sys.stderr)
            return REFUSED
        if args.dry_run:
            return 0

        lo, hi = np.zeros(14), np.zeros(14)
        for s in SIDES:
            lo[idx[s]], hi[idx[s]] = np.array(LIMITS[s]).T
        Q = np.clip(Q, lo, hi)  # the solver ends a hair outside a limit the target sits on (park: j4 = 0 = its lower limit)
        order = [idx[s][j] for s in SIDES for j in range(7)]
        m = JointTrajectory(joint_names=NAMES)
        for i, (q, v) in enumerate(zip(Q, V)):
            t = i * dt
            pt = JointTrajectoryPoint(positions=[float(q[j]) for j in order], velocities=[float(v[j]) for j in order])
            pt.time_from_start.sec, pt.time_from_start.nanosec = int(t), int((t % 1.0) * 1e9)
            m.points.append(pt)
        node.path_pub.publish(m)
        goal = {n: TARGETS[args.target][s][j] for s in SIDES for j, n in enumerate(joint_names(s))}
        if not node.spin_until(lambda: max(abs(node.measured[n] - goal[n]) for n in NAMES) < SETTLE_TOL,
                               len(Q) * dt + SETTLE_TIMEOUT_S):
            res = max(abs(node.measured[n] - goal[n]) for n in NAMES)
            print(f"{tag} did not converge ({res:.3f} rad residual)", file=sys.stderr)
            return 1
        if args.target == "ready" and None not in node.finger().values():
            node.set_gripper(READY_GRIPPER)
        print(f"{tag} POSITION REACHED")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
