#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Policy inference on OpenArm: wrist cameras and the hands' state → the AIRoA-0 policy server (policy/, HTTP) → hand
pose rows for motion and finger commands for the gripper, one request at a time:

- The runner captures both wrist images and the state: the right gripper tip seen from the left one (xyz and the 6D
  rotation) and both finger openings. It posts them to /infer and holds while it waits.
- A reply holds rows at the server's action_hz: per arm, the target tip pose relative to that tip at the observation,
  then both finger openings. The runner puts them on the commanded hand at the observation, as hand_root poses, and
  plays the first `execute_rows`; when the last is due it captures again.
- Each row's finger openings go to the gripper one row before the row is due.
- /inference/pause stops requesting: the rows already sent play out and the arms hold. /inference/start resumes with a
  new history on the server; it is refused while the arm is tripped. A refused chunk, a stale camera or state stream,
  or a trip pauses too (a trip also stops the finger rows); /inference/state says which.
- When the server cannot be reached or a reply is overdue, the run ends. While running with rows still to play,
  /arm/hold first stops the arms with the fingers.

    ros2 run oap_runner runner --config <runner.yaml>
"""
from __future__ import annotations

import argparse
import collections
import io as iolib
import json
import math
import os
import queue
import secrets
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pinocchio as pin
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Transform
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import (JointTrajectory, JointTrajectoryPoint, MultiDOFJointTrajectory,
                                 MultiDOFJointTrajectoryPoint)

SIDES = ("left", "right")
FINGERS = ["hand_left_motor_joint", "hand_right_motor_joint"]
VIEWS = {"left": "left_wrist_0_rgb", "right": "right_wrist_0_rgb"}  # the server's image keys
REPLY_TIMEOUT_S = 10.0  # a reply takes well under 1 s; the server compiles before it serves
STALE_S = 3.0  # a camera or state stream without a new sample for this long pauses the run
HOLD_TIMEOUT_S = 1.0  # /arm/hold on the way out, when the server is gone
EXIT_TIMEOUT_S = 3.0  # leaving the ROS graph on exit; past it the process ends anyway
ENTRY_MARGIN_S = 0.02  # the first row is due at least this long after install starts: covers the time to publish


def se3(p) -> pin.SE3:
    """[x, y, z, qx, qy, qz, qw] → SE3, with the quaternion normalized."""
    p = np.asarray(p, float)
    q = pin.Quaternion(p[6], p[3], p[4], p[5])
    q.normalize()
    return pin.SE3(q.matrix(), p[:3].copy())


def pose7(T: pin.SE3) -> np.ndarray:
    q = pin.Quaternion(T.rotation)
    return np.array([*T.translation, q.x, q.y, q.z, q.w])


def pose9(T: pin.SE3) -> np.ndarray:
    """xyz and the 6D rotation: the rotation's first two columns."""
    return np.concatenate([T.translation, T.rotation[:, 0], T.rotation[:, 1]])


def from_pose9(v) -> pin.SE3:
    """pose9 back to SE3, the two columns made orthonormal (Gram-Schmidt)."""
    v = np.asarray(v, float)
    a = v[3:6] / np.linalg.norm(v[3:6])
    b = v[6:9] - (a @ v[6:9]) * a
    b /= np.linalg.norm(b)
    return pin.SE3(np.column_stack([a, b, np.cross(a, b)]), v[:3].copy())


def stamp_msg(node: Node):
    return node.get_clock().now().to_msg()


def duration(d: float) -> Duration:
    """d seconds, in whole nanoseconds; a time already past is now."""
    ns = max(0, int(round(d * 1e9)))
    return Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


class Camera:
    """The newest frame of one V4L2 camera, as RGB, with its monotonic capture time."""

    def __init__(self, side: str, dev: str, width: int, height: int, fps: float, publish=None):
        if not os.path.exists(dev):
            raise RuntimeError(f"{side} camera {dev} does not exist: set cameras.{side}.dev in the runner config "
                               "(v4l2-ctl --list-devices, ls /dev/v4l/by-path)")
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open the {side} camera {dev}")
        self.dev, self.publish = dev, publish
        self.lock = threading.Lock()
        self.frame, self.stamp = None, 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            t = time.monotonic()
            with self.lock:
                self.frame, self.stamp = bgr[:, :, ::-1].copy(), t
            if self.publish:
                self.publish(bgr)

    def latest(self):
        with self.lock:
            return self.frame, self.stamp




class Io(Node):
    def __init__(self):
        super().__init__("runner")
        self.lock = threading.Lock()
        self.hand = {s: None for s in SIDES}  # measured (monotonic stamp, pose7)
        self.hand_hist = {s: collections.deque(maxlen=1000) for s in SIDES}  # the last 4 s, to read the hand at a capture
        self.commands = {s: collections.deque(maxlen=2000) for s in SIDES}  # commanded hand (wall time, SE3), 8 s at 250 Hz
        self.ik = {s: None for s in SIDES}  # (target, FK(q_cmd)) of the same motion tick
        self.fingers: dict[str, float] = {}
        self.finger_stamp = 0.0
        self.tripped = False
        self.closing = False  # set before the node goes: the camera threads stop publishing
        for s in SIDES:
            self.create_subscription(PoseStamped, f"/arm/{s}/hand_pose", lambda m, s=s: self._hand(s, m), 10)
            self.create_subscription(PoseStamped, f"/arm/{s}/command_pose", lambda m, s=s: self._command(s, m), 10)
            self.create_subscription(PoseStamped, f"/arm/{s}/ik_pose", lambda m, s=s: self._ik(s, m), 10)
        self.create_subscription(JointState, "/gripper/joint_states", self._fingers, 10)
        self.create_subscription(Bool, "/arm/tripped", self._trip, rclpy.qos.QoSProfile(
            depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL))
        self.chunk_pub = {s: self.create_publisher(MultiDOFJointTrajectory, f"/arm/{s}/cmd_pose_trajectory", 10)
                          for s in SIDES}
        self.grip_pub = self.create_publisher(JointTrajectory, "/gripper/joint_trajectory", 10)
        self.image_pub = {s: self.create_publisher(CompressedImage, f"/{s}/{s}_arm_cam/image_raw/compressed", 10)
                          for s in SIDES}
        self.state_pub = self.create_publisher(String, "/inference/state", rclpy.qos.QoSProfile(
            depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL))
        self.requests: queue.SimpleQueue = queue.SimpleQueue()  # "start" or "pause", taken by the loop in order
        self.hold_client = self.create_client(Trigger, "/arm/hold")
        self.create_service(Trigger, "/inference/start", lambda _q, r: self._request("start", r))
        self.create_service(Trigger, "/inference/pause", lambda _q, r: self._request("pause", r))

    def _request(self, what, response):
        if what == "start" and self.tripped:
            response.success, response.message = False, "the arm is tripped: choose a motion (Ready, Park, Hold) first"
            return response
        self.requests.put(what)
        response.success, response.message = True, what
        return response

    def hold(self, timeout) -> bool:
        """/arm/hold: motion stops where the arms are. Called from the loop while the executor spins."""
        t0 = time.monotonic()
        if not self.hold_client.wait_for_service(timeout_sec=timeout):
            return False
        done = threading.Event()
        future = self.hold_client.call_async(Trigger.Request())
        future.add_done_callback(lambda _f: done.set())
        return done.wait(max(0.0, timeout - (time.monotonic() - t0))) and bool(future.result().success)

    def set_state(self, state):
        self.state_pub.publish(String(data=state))
        self.get_logger().info(f"state: {state}")

    def _hand(self, side, m):
        p, o = m.pose.position, m.pose.orientation
        t = time.monotonic()
        with self.lock:
            self.hand[side] = (t, np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w]))
            self.hand_hist[side].append(self.hand[side])

    def _command(self, side, m):
        p, o = m.pose.position, m.pose.orientation
        t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        with self.lock:
            self.commands[side].append((t, se3([p.x, p.y, p.z, o.x, o.y, o.z, o.w])))

    def _ik(self, s, m):
        p, o = m.pose.position, m.pose.orientation
        t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        with self.lock:
            target = next((T for ts, T in reversed(self.commands[s]) if ts == t), None)
            if target is not None:
                self.ik[s] = (target, se3([p.x, p.y, p.z, o.x, o.y, o.z, o.w]))

    def ik_residual(self, side) -> pin.SE3:
        """Where the IK got, seen from the target, at the newest tick with both: identity while it reaches it."""
        with self.lock:
            return pin.SE3.Identity() if self.ik[side] is None else self.ik[side][0].actInv(self.ik[side][1])

    def _fingers(self, m):
        with self.lock:
            self.fingers.update(zip(m.name, m.position))
            self.finger_stamp = time.monotonic()

    def _trip(self, m):
        self.tripped = m.data  # /arm/hold clears it

    def command_at(self, side, t) -> Optional[pin.SE3]:
        """The commanded hand at wall time t: the newest sample at or before it."""
        with self.lock:
            best = None
            for ts, T in self.commands[side]:
                if ts > t:
                    break
                best = T
            return best

    def hand_at(self, side, mono) -> np.ndarray:
        """The measured hand at monotonic time mono: the newest sample at or before it, else the oldest kept."""
        with self.lock:
            best = self.hand_hist[side][0][1]
            for ts, p in self.hand_hist[side]:
                if ts > mono:
                    break
                best = p
            return best

    def command_now(self, side) -> Optional[pin.SE3]:
        with self.lock:
            return self.commands[side][-1][1] if self.commands[side] else None

    def send_fingers(self, positions, period):
        m = JointTrajectory(joint_names=list(FINGERS))
        pt = JointTrajectoryPoint(positions=[float(v) for v in positions])
        pt.time_from_start = duration(period)
        m.points = [pt]
        self.grip_pub.publish(m)

    def send_chunk(self, side, times, poses):
        m = MultiDOFJointTrajectory()
        m.header.stamp = stamp_msg(self)
        m.header.frame_id = "world"
        m.joint_names = [f"{side}_hand_root"]
        stamp = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        # Knots already due would all clamp to the stamp, motion drops a chunk whose times do not increase, and its
        # spline keeps only knots after its own time anyway: they are left out. install has left out those due
        # before it sends, so this only catches a stall between the two sides.
        skip = sum(1 for t in times if int(round((t - stamp) * 1e9)) <= 0)
        for t, T in zip(times[skip:], poses[skip:]):
            p = pose7(T)
            tr = Transform()
            tr.translation.x, tr.translation.y, tr.translation.z = (float(v) for v in p[:3])
            tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w = (float(v) for v in p[3:])
            pt = MultiDOFJointTrajectoryPoint(transforms=[tr])
            pt.time_from_start = duration(t - stamp)  # due times are wall clock, like the stamp
            m.points.append(pt)
        if m.points:  # motion drops an empty chunk, with a warning
            self.chunk_pub[side].publish(m)


@dataclass
class Ticket:
    """What the runner knew when it captured a request's observation."""
    wall: float
    mono: float
    generation: int


class Runner:
    def __init__(self, cfg, io: Io):
        self.cfg, self.io = cfg, io
        self.url = f"http://{cfg['server_host']}:{cfg['server_port']}"
        self.execute_rows = int(cfg.get("execute_rows", 0))  # 0: every row of a reply
        self.tip = pin.SE3(np.eye(3), np.asarray(cfg["tcp_offset_m"], float))  # hand_root -> gripper tip
        self.cams = {}
        for s in SIDES:
            c = cfg["cameras"][s]
            pub = None
            if cfg.get("publish_images", True):
                def pub(bgr, s=s):
                    if io.closing:  # the node is going: publishing raises
                        return
                    ok, jpg = cv2.imencode(".jpg", bgr)
                    if not ok:
                        return
                    m = CompressedImage(format="jpeg", data=jpg.tobytes())
                    try:
                        m.header.stamp = stamp_msg(io)
                        io.image_pub[s].publish(m)
                    except RuntimeError:  # InvalidHandle: the node went during the encode
                        if not io.closing:
                            raise
            self.cams[s] = Camera(s, c["dev"], int(c["width"]), int(c["height"]), float(c["fps"]), pub)
        self.meta = self.get("/metadata")
        if self.meta.get("state_dim") != 11 or self.meta.get("action_dim") != 20:
            raise RuntimeError(f"the server is not an AIRoA-0 policy with 11 states and 20 actions: {self.meta}")
        self.hz = float(self.meta["action_hz"])
        self.session = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        self.generation = 0
        self.history_reset = True
        self.latency = float(cfg.get("camera_latency_s", 0.0))
        self.finger_rows: collections.deque = collections.deque()  # (due, fingers) still to send
        self.last_due: Optional[float] = None  # wall time the last row sent is due
        self.last_capture = {s: 0.0 for s in SIDES}
        self.last_state_capture = 0.0  # monotonic, as the hand and finger stamps
        self.replies: queue.Queue = queue.Queue()
        self.awaiting = False
        self.paused = False
        self.log = io.get_logger()
        self.started = time.monotonic()  # a stream gets STALE_S from here for its first sample

    # --- the server ---------------------------------------------------------------------------------------------------
    def get(self, path) -> dict:
        try:
            with urllib.request.urlopen(self.url + path, timeout=REPLY_TIMEOUT_S) as r:
                return json.load(r)
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise RuntimeError(f"no policy server at {self.url}: {e}") from None

    def post(self, path, body: bytes) -> dict:
        try:
            with urllib.request.urlopen(self.url + path, body, timeout=REPLY_TIMEOUT_S) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:  # the server refused the request: its reason is in the body
            try:
                why = json.load(e).get("error", e.reason)
            except ValueError:
                why = e.reason
            raise RuntimeError(f"the server refused the request: {why}") from None

    # --- requests -----------------------------------------------------------------------------------------------------
    def fresh(self) -> bool:
        """Every image and state is new since the last capture."""
        with self.io.lock:
            hands = all(self.io.hand[s] is not None and self.io.hand[s][0] > self.last_state_capture for s in SIDES)
            fingers = all(n in self.io.fingers for n in FINGERS) and self.io.finger_stamp > self.last_state_capture
        return hands and fingers and all(self.cams[s].latest()[1] > self.last_capture[s] for s in SIDES)

    def stale(self) -> Optional[str]:
        """The first stream without a new sample for STALE_S, or None."""
        old = time.monotonic() - STALE_S
        if self.started >= old:
            return None
        with self.io.lock:
            stamps = {f"the {s} hand pose": (self.io.hand[s] or (0.0,))[0] for s in SIDES}
            stamps["the finger state"] = self.io.finger_stamp
        stamps.update({f"the {s} camera": self.cams[s].latest()[1] for s in SIDES})
        for what, t in stamps.items():
            if t < old:
                return f"no new sample from {what} for {STALE_S:g} s"
        return None

    def observation(self, mono) -> dict:
        """The request for the observation at mono: both images, and the right tip seen from the left tip with both
        finger openings, measured at that instant."""
        tips = {s: se3(self.io.hand_at(s, mono)) * self.tip for s in SIDES}
        with self.io.lock:
            fingers = np.array([self.io.fingers[n] for n in FINGERS])
            self.last_state_capture = time.monotonic()
        obs = {"state": np.concatenate([pose9(tips["left"].actInv(tips["right"])), fingers]).astype(np.float32),
               "timestamp": np.float64(mono), "session": self.session, "reset": self.history_reset}
        for s in SIDES:
            frame, t = self.cams[s].latest()
            obs[VIEWS[s]] = frame
            self.last_capture[s] = t
        return obs

    def request(self) -> Ticket:
        # The frames left the sensor camera_latency_s before they were read: that is the observation's instant.
        now = time.monotonic()
        ticket = Ticket(time.time() - self.latency, now - self.latency, self.generation)
        buf = iolib.BytesIO()
        np.savez(buf, **self.observation(ticket.mono))
        self.history_reset = False
        self.awaiting = True
        self.requested_at = now

        def run():
            t0 = time.monotonic()
            try:
                self.replies.put((ticket, self.post("/infer", buf.getvalue()), time.monotonic() - t0))
            except Exception as e:  # noqa: BLE001 — handed to the loop
                self.replies.put((ticket, e, time.monotonic() - t0))
        threading.Thread(target=run, daemon=True).start()
        return ticket

    # --- chunks -------------------------------------------------------------------------------------------------------
    def install(self, ticket: Ticket, reply: dict, latency: float):
        """The rows on the commanded hand at the observation, from now: the arms held since then."""
        actions = np.asarray(reply.get("actions"), float)
        if actions.ndim != 2 or actions.shape[1] != 20 or not len(actions) or not np.isfinite(actions).all():
            raise RuntimeError(f"not a chunk of 20 values per row: actions {actions.shape}")
        hz = float(reply.get("action_hz", self.hz))
        n = len(actions) if self.execute_rows <= 0 else min(self.execute_rows, len(actions))
        now = time.time()
        due = now + ENTRY_MARGIN_S + (np.arange(n) + 1) / hz
        row_mm, row_deg = self.cfg.get("max_row_step_mm", math.inf), self.cfg.get("max_row_step_deg", math.inf)
        knots = {}
        for a, s in enumerate(SIDES):
            anchor, cur = self.io.command_at(s, ticket.wall), self.io.command_now(s)
            if anchor is None or cur is None:
                raise RuntimeError(f"no commanded {s} hand from motion")
            # Moved by what the IK falls short of (a joint limit), so the chunk starts where the arm can be.
            residual = self.io.ik_residual(s)
            tip, cur = anchor * residual * self.tip, cur * residual
            poses = [tip * from_pose9(actions[k, 9 * a:9 * a + 9]) * self.tip.inverse() for k in range(n)]
            self.check_first(s, cur, poses[0])
            for k in range(1, n):
                mm, deg = self.step(poses[k - 1], poses[k])
                if not (mm <= row_mm and deg <= row_deg):
                    raise RuntimeError(f"{s}: row {k} moves {mm:.0f} mm / {deg:.0f}° from the one before, over the "
                                       "row step limit")
            knots[s] = poses
        for s in SIDES:
            self.io.send_chunk(s, due, knots[s])
        self.hz = hz
        self.finger_rows.extend(zip(due, actions[:n, 18:20].copy()))
        self.last_due = due[-1]
        self.log.info(f"[policy] {n} rows at {hz:g} Hz, latency {latency:.3f} s, "
                      f"history slots {reply.get('history_slots_used', '-')}")

    def check_first(self, s, cur: pin.SE3, first: pin.SE3):
        mm, deg = self.step(cur, first)
        if not (mm <= self.cfg["max_step_mm"] and deg <= self.cfg["max_step_deg"]):  # NaN fails too
            raise RuntimeError(f"{s}: the first row moves {mm:.0f} mm / {deg:.0f}°, over the step limit")

    @staticmethod
    def step(a: pin.SE3, b: pin.SE3) -> tuple:
        """From a to b: mm, degrees."""
        d = a.actInv(b)
        return 1000.0 * np.linalg.norm(d.translation), math.degrees(np.linalg.norm(pin.log3(d.rotation)))

    def emit_fingers(self, now):
        now += 1.0 / self.hz  # a finger row goes out one row before its hand row is due
        while self.finger_rows and self.finger_rows[0][0] <= now:
            self.io.send_fingers(self.finger_rows.popleft()[1], 1.0 / self.hz)

    def wants_request(self) -> bool:
        """Once the last row is due: the arms hold there while the server answers."""
        if self.paused or self.awaiting:
            return False
        return self.last_due is None or time.time() - self.latency >= self.last_due

    def pause(self, why, stop_fingers=False):
        """Stop requesting: a reply in flight is dropped, the rows already sent play out and the arms hold. With
        stop_fingers (a trip: the arms are frozen), the finger rows stop too and the gripper holds the last command."""
        self.paused = True
        if stop_fingers:
            self.last_due = None
            self.finger_rows.clear()
        self.generation += 1
        self.io.set_state(f"paused: {why}")

    def resume(self):
        """A fresh start from the next observation, with a new history on the server."""
        self.paused = False
        self.generation += 1
        self.history_reset = True
        self.last_due = None
        self.finger_rows.clear()
        self.io.set_state("running")

    def hold(self):
        """Before the run ends (a server failure, Ctrl-C, SIGTERM, an error): while running with rows still to play, the
        arms stop where they are, with the fingers. Otherwise nothing: with no rows left the arms hold already, and while
        paused the operator may have started a planner move (Ready, Park), which /arm/hold would cancel. A tripped arm
        holds already, and /arm/hold would clear the trip. Once per run."""
        if getattr(self, "held", False):
            return
        self.held = True
        if self.paused or self.last_due is None or self.last_due <= time.time():
            return
        if not self.io.tripped and not self.io.hold(HOLD_TIMEOUT_S):
            self.log.error(f"/arm/hold failed or did not answer within {HOLD_TIMEOUT_S:g} s")

    def run(self) -> int:
        self.log.info(f"policy server {self.url}: {self.meta}")
        self.resume()
        while rclpy.ok():
            while not self.io.requests.empty():
                what = self.io.requests.get_nowait()
                if what == "pause" and not self.paused:
                    self.pause("requested")
                elif what == "start" and self.paused:
                    if self.io.tripped:
                        self.log.warn("the arm is tripped: choose a motion (Ready, Park, Hold) first")
                    else:
                        self.resume()
            if self.io.tripped and (not self.paused or self.last_due is not None):  # paused: the finger rows stop
                self.pause("the arm tripped: a joint fell behind its command", stop_fingers=True)
            why = None if self.paused else self.stale()
            if why:
                self.log.error(why)
                self.pause(why)
            self.emit_fingers(time.time())
            try:
                ticket, reply, latency = self.replies.get_nowait()
                self.awaiting = False
                if isinstance(reply, Exception) and not str(reply).startswith("the server refused"):
                    self.hold()  # the server is gone: the arms stop with the fingers, then exit
                    raise reply
                if ticket.generation == self.generation:
                    try:
                        if isinstance(reply, Exception):
                            raise reply
                        self.install(ticket, reply, latency)
                    # a refused request or chunk; the server is fine
                    except (RuntimeError, ValueError, TypeError, KeyError, IndexError) as e:
                        self.log.error(str(e))
                        self.pause(str(e))
            except queue.Empty:
                if self.awaiting and time.monotonic() - self.requested_at > REPLY_TIMEOUT_S:
                    self.hold()
                    raise RuntimeError(f"the server did not answer within {REPLY_TIMEOUT_S:.0f} s") from None
            if self.wants_request() and self.fresh():
                self.request()
            time.sleep(0.002)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, help="the policy server's, instead of the config's")
    args, ros_args = ap.parse_known_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.port:
        cfg["server_port"] = args.port
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)  # Ctrl-C is a normal stop, below
    for sig in (signal.SIGINT, signal.SIGTERM):  # also where the shell started it with SIGINT ignored
        signal.signal(sig, signal.default_int_handler)
    io = Io()
    ex = SingleThreadedExecutor()
    ex.add_node(io)

    def spin():
        try:
            ex.spin()
        except ExternalShutdownException:
            pass
    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    code, runner = 0, None
    try:
        runner = Runner(cfg, io)
        code = runner.run()
    except KeyboardInterrupt:
        io.get_logger().info("stopped: Ctrl-C or SIGTERM")
    except Exception as e:  # noqa: BLE001 — say why the run stopped
        io.get_logger().error(f"stopped: {e}")
        code = 1
    leave(io, ex, spinner, code, runner)


def leave(io: Io, ex: SingleThreadedExecutor, spinner: threading.Thread, code: int, runner: Optional[Runner] = None):
    """Stop the arms if rows still play (Runner.hold, unless the run ended by itself or held already), leave the ROS
    graph, then end the process. Without the context torn down, the DDS participant stays in the graph until its lease
    runs out (10 s), and a control launch stopped meanwhile hangs deleting its writers. The camera and request
    threads block in C calls, so the process ends with os._exit; a watchdog ends it the same way if leaving the graph
    hangs."""
    for sig in (signal.SIGINT, signal.SIGTERM):  # a second Ctrl-C must not cut the hold or the teardown short
        signal.signal(sig, signal.SIG_IGN)

    def stop():
        sys.stdout.flush()
        os._exit(code)
    watchdog = threading.Timer(HOLD_TIMEOUT_S + EXIT_TIMEOUT_S, stop)
    watchdog.daemon = True
    watchdog.start()
    if runner is not None and rclpy.ok():  # the executor still spins: /arm/hold can answer
        try:
            runner.hold()
        except Exception as e:  # noqa: BLE001 — leaving anyway
            print(f"runner: /arm/hold on exit failed: {e}", file=sys.stderr)
    io.closing = True
    try:
        ex.shutdown(timeout_sec=1.0)
        spinner.join(1.0)
        io.destroy_node()
        context = rclpy.get_default_context()
        rclpy.try_shutdown()
        context.destroy()
    except Exception as e:  # noqa: BLE001 — leaving anyway
        print(f"runner: leaving the ROS graph failed: {e}", file=sys.stderr)
    stop()


if __name__ == "__main__":
    sys.exit(main())
