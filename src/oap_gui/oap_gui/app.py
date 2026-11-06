#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""One window for a run: the control chain, the robot moves, the runner, the camera views, a bag and its overlay video.
Every button starts the same command the README lists, so the GUI adds no second path to the robot.

    ros2 run oap_gui gui        # in the container, from /ws
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.executors
from PyQt5 import QtCore, QtGui, QtWidgets
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from oap_planner.geometry import READY_POSE, REFUSED, joint_names

WS = os.environ.get("OAP_WS", "/ws")
CHECKPOINTS = os.environ.get("OAP_CHECKPOINTS", "/checkpoints")  # scripts/oap mounts the repo's checkpoints/ here
OUTPUT = os.path.join(WS, "output")  # scripts/record.sh writes the bags here
VIDEO = "overlay.mp4"  # in the bag's folder (oap_gui overlay)
# the arms' CAN interfaces; empty means the default. Start control rejects a bad or shared name
CAN = {"left": os.environ.get("OAP_LEFT_CAN") or "can1", "right": os.environ.get("OAP_RIGHT_CAN") or "can0"}
SETTINGS = os.path.join(WS, ".cache", "gui.json")
RUN = "/tmp/oap"  # the processes' pid, log, exit and meta files: they live as long as the container
PROC = "/proc"
# [s] after a launch's SIGINT: then SIGTERM to its group, then SIGKILL. Past launch's own escalation for
# ros2_control_node (robot.launch.py: SIGTERM after 20 s, SIGKILL 10 s later), so the GUI's comes only if launch hangs.
ESCALATE_S = (35.0, 5.0)
POLICY_STOP_S = 5.0  # [s] Stop control waits this long for the runner to exit before it stops control
PAUSE_WAIT_S = 2.0  # [s] a move waits this long for a running inference to report paused, else it does not start
NAMES = ("control", "server", "prewarm", "policy", "record", "move", "overlay")
TABS = ("GUI", "Control", "Moves", "Server", "Inference", "Recording", "Data")
TAB = {"control": "Control", "move": "Moves", "server": "Server", "prewarm": "Server", "policy": "Inference",
       "record": "Recording", "overlay": "Data"}
BAGS_EVERY_S = 2.0  # [s] the bag list is read again this often
SIDES = ("left", "right")
FINGERS = ["hand_left_motor_joint", "hand_right_motor_joint"]
OPEN, CLOSED = 0.72, 0.0  # [rad] the planner's READY_GRIPPER, and closed
READY_TOL = 0.1  # [rad] as the circle test's
COLOR = {"Stopped": "#c62828", "Starting": "#b8860b", "Paused": "#b8860b", "Started": "#2e7d32"}


def show(label, state, text=None):
    """One status style everywhere: Stopped red, Starting or Paused yellow, Started green."""
    label.setText(text or state)
    label.setStyleSheet(f"color: {COLOR[state]}; font-weight: bold")


def load_settings() -> dict:
    try:
        with open(SETTINGS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(**kv):
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    with open(SETTINGS, "w") as f:
        json.dump({**load_settings(), **kv}, f)


def checkpoints() -> list:
    """The checkpoints serve_policy.sh can start: released AIRoA-0 checkpoints, with a model_config.json."""
    if not os.path.isdir(CHECKPOINTS):
        return []
    found = os.listdir(CHECKPOINTS)
    return sorted(d for d in found if os.path.isfile(os.path.join(CHECKPOINTS, d, "model_config.json")))


def bags() -> list:
    """The bags in OUTPUT, newest first."""
    if not os.path.isdir(OUTPUT):
        return []
    found = [d for d in os.listdir(OUTPUT) if os.path.isfile(os.path.join(OUTPUT, d, "metadata.yaml"))]
    return sorted(found, key=lambda d: os.path.getmtime(os.path.join(OUTPUT, d)), reverse=True)


def control_cmd(hardware, can) -> str:
    # --noninteractive: launch forwards a SIGINT to its nodes only when noninteractive (else it expects the terminal's
    # Ctrl-C to reach them itself); Stop signals the launch alone (Proc.stop), so it must forward
    launch = (f"ros2 launch --noninteractive oap_control robot.launch.py hardware:={shlex.quote(hardware)}"
              f" left_can:={shlex.quote(can['left'])} right_can:={shlex.quote(can['right'])}")
    if hardware != "real":
        return launch
    up = f"./scripts/canup.sh {shlex.quote(can['right'])} && ./scripts/canup.sh {shlex.quote(can['left'])}"
    return "bash -c " + shlex.quote(f"{up} && exec {launch}")


def server_cmd(checkpoint, port, pa_text="") -> str:
    cmd = f"./scripts/serve_policy.sh {shlex.quote(os.path.join(CHECKPOINTS, checkpoint))} --port {int(port)}"
    return cmd + (f" --pa-text={shlex.quote(pa_text)}" if pa_text else "")  # "=": a text may start with -


def runner_cmd(config, port) -> str:
    return f"ros2 run oap_runner runner --config {shlex.quote(os.path.expanduser(config))} --port {int(port)}"


def record_cmd(tag) -> str:
    tag = "".join(c for c in tag.strip() if c.isalnum() or c in "-_")
    return "./scripts/record.sh" + (f" {shlex.quote(tag)}" if tag else "")


def overlay_cmd(bag) -> str:
    return f"ros2 run oap_gui overlay {shlex.quote(os.path.join(OUTPUT, bag))}"


def stat(pid):
    """/proc/<pid>/stat from field 3 (the state) on, or None when there is no such process."""
    try:
        with open(os.path.join(PROC, str(pid), "stat")) as f:
            return f.read().rpartition(")")[2].split()
    except OSError:
        return None


def start_time(pid):
    """Field 22 of /proc/<pid>/stat: when the process started, in clock ticks since boot. A reused pid differs."""
    fields = stat(pid)
    return fields[19] if fields and len(fields) > 19 else None


def group(pgid):
    """{pid: argv} of the live processes in group `pgid`. A group id is not reused while one of them lives."""
    members = {}
    for d in os.listdir(PROC):
        fields = stat(d) if d.isdigit() else None
        if not fields or len(fields) < 3 or fields[2] != str(pgid) or fields[0] == "Z":
            continue
        try:
            with open(os.path.join(PROC, d, "cmdline"), "rb") as f:
                members[int(d)] = f.read().rstrip(b"\0").split(b"\0")
        except OSError:
            continue
    return members


def is_launch(argv) -> bool:
    """`ros2 launch ...`, run directly or by an interpreter: skip argv[0] unless it is ros2 itself, then the
    interpreter's - options (python3 -s -u), then require basename ros2 followed by launch."""
    i = 0 if os.path.basename(argv[0]) == b"ros2" else 1
    while i < len(argv) and argv[i].startswith(b"-"):
        i += 1
    return i + 1 < len(argv) and os.path.basename(argv[i]) == b"ros2" and argv[i + 1] == b"launch"


def launch_pid(pgid):
    """The `ros2 launch` process in group `pgid`, or None: under the exit-code shell, and on real hardware under the
    canup shell that execs it."""
    return next((pid for pid, argv in sorted(group(pgid).items()) if argv and is_launch(argv)), None)


STALE_S = 0.5


class Ros(Node):
    """What the window shows, filled by subscriptions on a background executor."""

    def __init__(self):
        super().__init__("oap_gui")
        self.lock = threading.Lock()
        self.command_t = 0.0
        self.tripped = False
        self.fingers = {}
        self.images = {s: None for s in SIDES}
        self.create_subscription(JointState, "/arm/command", self._command, 10)
        self.joints = {}
        self.create_subscription(JointState, "/joint_states", self._joints, 10)
        self.create_subscription(Bool, "/arm/tripped", lambda m: setattr(self, "tripped", m.data),
                                 rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL))
        self.inference_state = ""
        self.create_subscription(String, "/inference/state", lambda m: setattr(self, "inference_state", m.data),
                                 rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(JointState, "/gripper/joint_states",
                                 lambda m: self.fingers.update(zip(m.name, m.position)), 10)
        for s in SIDES:
            self.create_subscription(CompressedImage, f"/{s}/{s}_arm_cam/image_raw/compressed",
                                     lambda m, s=s: self._image(s, m), 2)
        self.grip_pub = self.create_publisher(JointTrajectory, "/gripper/joint_trajectory", 10)
        self.hold = self.create_client(Trigger, "/arm/hold")
        self.inference = {w: self.create_client(Trigger, f"/inference/{w}") for w in ("start", "pause")}
        self.relax = {s: self.create_client(SetBool, f"/openarm_hw_relax_{s}/set_relax") for s in SIDES}
        self.manual = {s: self.create_client(SetBool, f"/openarm_hw_relax_{s}/set_manual") for s in SIDES}
        self.follow = self.create_client(SetBool, "/arm/follow")

    def _joints(self, m):
        with self.lock:
            self.joints.update(zip(m.name, m.position))

    def _command(self, _m):
        self.command_t = time.monotonic()

    def _image(self, side, m):
        with self.lock:
            self.images[side] = bytes(m.data)

    def distance_from_ready(self):
        """The largest joint distance from the ready pose [rad], or None before the joint states arrive."""
        with self.lock:
            q = dict(self.joints)
        names = [n for s in SIDES for n in joint_names(s)]
        if not all(n in q for n in names):
            return None
        ready = [v for s in SIDES for v in READY_POSE[s]]
        return max(abs(q[n] - r) for n, r in zip(names, ready))

    def alive(self) -> bool:
        return time.monotonic() - self.command_t < STALE_S

    def gripper(self, opening):
        m = JointTrajectory(joint_names=list(FINGERS))
        m.points = [JointTrajectoryPoint(positions=[opening, opening])]
        self.grip_pub.publish(m)


class Proc(QtCore.QObject):
    """One process, detached in its own session so it outlives the GUI: its pid, log, exit code and start settings live
    in RUN/<name>.*, so a reopened GUI finds it again. Stop sends SIGINT to its whole group, so ros2 bag closes cleanly
    and uv's server child stops with uv. A launch (`launch=True`) gets it alone instead: it forwards one SIGINT to its
    nodes (being --noninteractive), so ros2_control_node deactivates the hardware before it exits, where a second SIGINT
    straight from the group signal would kill it midway. Launch escalates per node itself; the group gets SIGTERM, then
    SIGKILL, only when it is still up after ESCALATE_S. A launch counts as running while any of its group lives."""

    finished = QtCore.pyqtSignal(str, int)

    def __init__(self, name, log, launch=False):
        super().__init__()
        self.name, self.log, self.launch = name, log, launch
        self.stopping = None  # (pid, start time) of the launch this GUI already sent its SIGINT
        self.path = {k: os.path.join(RUN, f"{name}.{k}") for k in ("pid", "start", "log", "exit", "meta")}
        self.child = None  # the Popen, when this GUI started it: polling it reaps it
        self.was_running = self.running()
        size = os.path.getsize(self.path["log"]) if os.path.exists(self.path["log"]) else 0
        self.offset = max(0, size - 4000) if self.was_running else size  # a running one shows its recent output

    def start(self, cmd: str, meta: str = "") -> bool:
        if self.running():
            self.log(f"[gui] {self.name} is already running (pid {self.pid()}): not starting it again")
            return False
        self.log(f"$ {cmd}")
        for k in ("exit", "pid", "start"):
            if os.path.exists(self.path[k]):
                os.remove(self.path[k])
        with open(self.path["meta"], "w") as f:
            f.write(meta)
        with open(self.path["log"], "w") as out:
            # its own session before exec, so a Stop right after the start already finds the group
            self.child = subprocess.Popen(["bash", "-c", f"{cmd}; echo $? > {shlex.quote(self.path['exit'])}"],
                                          cwd=WS, start_new_session=True, stdin=subprocess.DEVNULL, stdout=out,
                                          stderr=subprocess.STDOUT)
        with open(self.path["start"], "w") as f:  # "<pid> <start time>"; the pid file stays a bare pid
            f.write(f"{self.child.pid} {start_time(self.child.pid) or ''}".strip())
        with open(self.path["pid"], "w") as f:
            f.write(str(self.child.pid))
        self.offset, self.was_running = 0, True
        return True

    def pid(self):
        try:
            return int(open(self.path["pid"]).read())
        except (OSError, ValueError):
            return None

    def _start_time(self, pid):
        """The start time recorded for `pid`, or None (an older GUI started it: checked by pid only)."""
        try:
            fields = open(self.path["start"]).read().split()
        except OSError:
            return None
        return fields[1] if len(fields) > 1 and fields[0] == str(pid) else None

    def running(self) -> bool:
        if self.child is not None:
            self.child.poll()
        pid = self.pid()
        started = self._start_time(pid)
        fields = stat(pid) if pid is not None else None
        if fields and fields[0] != "Z":
            return started is None or (len(fields) > 19 and fields[19] == started)  # else the pid was reused
        # a launch's leader gone: its nodes may linger in the group, and they still hold the hardware
        return self.launch and pid is not None and bool(group(pid))

    def meta(self) -> str:
        try:
            return open(self.path["meta"]).read()
        except OSError:
            return ""

    def started(self) -> float:
        try:
            return os.path.getmtime(self.path["pid"])
        except OSError:
            return time.time()

    def stop(self, sig=signal.SIGINT):
        if not self.running():
            return
        pid = self.pid()
        if self.launch and sig == signal.SIGINT:
            key = (pid, start_time(pid))
            if self.stopping == key:  # a second SIGINT would reach the nodes twice
                self.log(f"[gui] {self.name} is stopping: its group gets SIGTERM {ESCALATE_S[0]:.0f} s after Stop"
                         " if it hangs")
                return
            self.stopping = key
            self.sigint_launch(key, retry=True)
            return
        self.send(pid, sig)

    def ours(self, key) -> bool:
        """Still the process `key` names: the leader with that start time, or its group once the leader is gone."""
        return self.running() and self.pid() == key[0] and start_time(key[0]) in (key[1], None)

    def sigint_launch(self, key, retry):
        """One SIGINT to the group's ros2 launch alone: it forwards one to each node and escalates per node itself.
        Without a launch (just started, or exited and nodes left), the group, loudly."""
        if not self.ours(key):
            return
        target = launch_pid(key[0])
        if target is None and retry:
            QtCore.QTimer.singleShot(200, lambda: self.sigint_launch(key, retry=False))
            return
        if target is not None and start_time(target) is not None:  # re-read: still alive right before the kill
            self.send(target, signal.SIGINT, os.kill)
        else:
            others = [os.path.basename(a[0]).decode(errors="replace") for a in group(key[0]).values()
                      if a and os.path.basename(a[0]) != b"bash"]
            self.log(f"[gui] WARNING {self.name}: no ros2 launch in its group, SIGINT to the whole group"
                     + (f" ({', '.join(others)})" if others else ""))
            self.send(key[0], signal.SIGINT)
        QtCore.QTimer.singleShot(int(ESCALATE_S[0] * 1000), lambda: self.escalate(key, signal.SIGTERM))

    def send(self, pid, sig, kill=None):
        try:
            (kill or os.killpg)(pid, sig)
        except (ProcessLookupError, PermissionError) as e:  # it exited meanwhile, or the pid is not ours
            self.log(f"[gui] {self.name}: {e}")

    def escalate(self, key, sig):
        """The launch did not end after its SIGINT: `sig` to the group, if it is still the same process."""
        if not self.ours(key):
            return
        self.log(f"[gui] {self.name} is still running: {signal.Signals(sig).name} to its group")
        self.send(key[0], sig)
        if sig == signal.SIGTERM:
            QtCore.QTimer.singleShot(int(ESCALATE_S[1] * 1000), lambda: self.escalate(key, signal.SIGKILL))

    def poll(self):
        """New output to the log; `finished` once it stops."""
        try:
            with open(self.path["log"], "rb") as f:
                f.seek(self.offset)
                data = f.read()
                self.offset += len(data)
            for line in data.decode(errors="replace").splitlines():
                self.log(line)
        except OSError:
            pass
        now = self.running()
        if self.was_running and not now:
            try:
                code = int(open(self.path["exit"]).read())
            except (OSError, ValueError):
                code = -1  # stopped by a signal before the shell could write it
            self.finished.emit(self.name, code)
        self.was_running = now


class Player(QtWidgets.QDialog):
    """A video in real time, from its first frame; closing it stops it."""

    def __init__(self, parent, path):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(os.path.dirname(path)))
        self.cap = cv2.VideoCapture(path)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lay = QtWidgets.QVBoxLayout(self)
        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.view, 1)
        row = QtWidgets.QHBoxLayout()
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, maximum=max(self.n - 1, 0))
        self.slider.sliderMoved.connect(self.seek)
        row.addWidget(self.pause_btn)
        row.addWidget(self.slider, 1)
        lay.addLayout(row)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(int(1000 / fps))
        self.resize(1300, 560)

    def step(self):
        ok, bgr = self.cap.read()
        if not ok:
            self.timer.stop()
            self.pause_btn.setText("Play")
            return
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.view.setPixmap(QtGui.QPixmap.fromImage(img).scaled(self.view.size(), QtCore.Qt.KeepAspectRatio))
        self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pause_btn.setText("Play")
            return
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.n:  # at the end: from the start
            self.seek(0)
        self.timer.start()
        self.pause_btn.setText("Pause")

    def seek(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        if not self.timer.isActive():
            self.step()

    def closeEvent(self, e):
        self.timer.stop()
        self.cap.release()
        e.accept()


class Window(QtWidgets.QMainWindow):
    def __init__(self, ros: Ros):
        super().__init__()
        self.ros = ros
        self.setWindowTitle("OpenArm pipeline")
        self.procs = {}
        self.action = 0  # bumped by every action: a pending prepare() whose token is stale does nothing more
        self.pending = None  # (token, what) of the prepare() still on its way to its `then`
        self.stopping_control = False  # from Stop control until control has exited: no new action meanwhile
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        robot = QtWidgets.QGroupBox("Robot")
        r = QtWidgets.QHBoxLayout(robot)
        self.hardware = QtWidgets.QComboBox()
        self.hardware.addItems(["mock", "real"])
        # the style's warning icon: the container has no color emoji font, so ⚠️ draws grey
        self.warning = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)
        self.control_btn = QtWidgets.QPushButton("Start control")
        # wide enough for either label: the text changes with the state
        self.control_btn.setMinimumWidth(self.control_btn.fontMetrics().horizontalAdvance("Stop control")
                                         + self.control_btn.iconSize().width() + 30)
        self.control_btn.clicked.connect(self.toggle_control)
        self.alive_lbl, self.trip_lbl = QtWidgets.QLabel(), QtWidgets.QLabel()
        for w in (QtWidgets.QLabel("hardware"), self.hardware, self.control_btn, self.alive_lbl, self.trip_lbl):
            r.addWidget(w)
        r.addStretch()
        outer.addWidget(robot)

        motion = QtWidgets.QGroupBox("Motion")
        m = QtWidgets.QHBoxLayout(motion)
        self.move_btns = []
        for label, fn in (("Ready", lambda: self.move("plan --target ready")),
                          ("Park", lambda: self.move("plan --target park"))):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            self.move_btns.append(b)
            m.addWidget(b)
        for label, fn in (("Hold", self.hold), ("Manual", self.manual)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            m.addWidget(b)
        m.addStretch()
        relax = QtWidgets.QPushButton(self.warning, "Relax")  # apart: the arms sink
        relax.clicked.connect(self.relax)
        m.addWidget(relax)
        test = QtWidgets.QPushButton("Test…")
        test.clicked.connect(lambda: (self.tests.show(), self.tests.raise_()))
        m.addWidget(test)
        outer.addWidget(motion)

        # moves one runs once or twice: a tracking check, and a pose where the arms can relax and sink safely
        self.tests = QtWidgets.QDialog(self)
        self.tests.setWindowTitle("Test motions")
        t = QtWidgets.QHBoxLayout(self.tests)
        self.radius = QtWidgets.QDoubleSpinBox(suffix=" mm", decimals=0, minimum=5, maximum=60, value=30)
        for label, fn in (("Lifted", lambda: self.move("plan --target lifted")),
                          ("Circle", lambda: self.move(f"circle --radius {self.radius.value() / 1000:.3f}"))):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            self.move_btns.append(b)
            t.addWidget(b)
        t.addWidget(self.radius)

        gripper = QtWidgets.QGroupBox("Gripper")
        g = QtWidgets.QHBoxLayout(gripper)
        for label, fn in (("Open", lambda: ros.gripper(OPEN)), ("Close", lambda: ros.gripper(CLOSED))):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            g.addWidget(b)
        g.addStretch()
        outer.addWidget(gripper)

        server = QtWidgets.QGroupBox("Policy server")
        sv = QtWidgets.QHBoxLayout()
        self.checkpoint = QtWidgets.QComboBox()
        self.checkpoint.addItems(checkpoints())
        self.checkpoint.setCurrentText(load_settings().get("checkpoint", ""))
        self.checkpoint.currentTextChanged.connect(lambda c: save_settings(checkpoint=c))
        self.pa_text = QtWidgets.QLineEdit(load_settings().get("pa_text", ""))
        self.pa_text.setPlaceholderText("the sub-task (PA); empty: the server's default")
        self.pa_text.editingFinished.connect(lambda: save_settings(pa_text=self.pa_text.text().strip()))
        self.port = QtWidgets.QSpinBox(minimum=1024, maximum=65535, value=8000)
        self.server_btn = QtWidgets.QPushButton("Start server")
        self.server_btn.clicked.connect(self.toggle_server)
        self.server_lbl = QtWidgets.QLabel()
        self.server_state = "Stopped"
        for w in (QtWidgets.QLabel("checkpoint"), self.checkpoint, QtWidgets.QLabel("port"), self.port, self.server_btn,
                  self.server_lbl):
            sv.addWidget(w)
        sv.addStretch()
        svl = QtWidgets.QVBoxLayout()
        svl.addLayout(sv)
        pa = QtWidgets.QHBoxLayout()
        pa.addWidget(QtWidgets.QLabel("PA"))
        pa.addWidget(self.pa_text, 1)
        svl.addLayout(pa)
        server.setLayout(svl)
        outer.addWidget(server)

        inference = QtWidgets.QGroupBox("Inference")
        i = QtWidgets.QGridLayout(inference)
        self.config = QtWidgets.QLineEdit("src/oap_runner/config/runner.yaml")
        self.runner_btn = QtWidgets.QPushButton("Start inference")
        self.runner_btn.clicked.connect(self.start_inference)
        self.pause_btn = QtWidgets.QPushButton("Pause inference")
        self.pause_btn.clicked.connect(self.pause_inference)
        self.state_lbl = QtWidgets.QLabel("")
        i.addWidget(QtWidgets.QLabel("config"), 0, 0)
        i.addWidget(self.config, 0, 1)
        i.addWidget(self.runner_btn, 0, 2)
        i.addWidget(self.pause_btn, 0, 3)
        i.addWidget(self.state_lbl, 1, 3)
        cams = QtWidgets.QHBoxLayout()
        self.cam = {}
        for s in SIDES:
            lbl = QtWidgets.QLabel(f"{s} camera")  # the runner publishes the frames while it runs
            lbl.setFixedSize(320, 240)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet("background: #222; color: #aaa")
            self.cam[s] = lbl
            cams.addWidget(lbl)
        i.addLayout(cams, 2, 0, 1, 3)
        outer.addWidget(inference)

        rec = QtWidgets.QGroupBox("Recording")
        b = QtWidgets.QHBoxLayout(rec)
        self.tag = QtWidgets.QLineEdit()
        self.tag.setPlaceholderText("tag, e.g. cups")
        self.rec_btn = QtWidgets.QPushButton("Record")
        self.rec_btn.clicked.connect(self.toggle_record)
        self.rec_lbl = QtWidgets.QLabel("not recording")
        for w in (QtWidgets.QLabel("bag"), self.tag, self.rec_btn, self.rec_lbl):
            b.addWidget(w)
        b.addStretch()
        outer.addWidget(rec)

        data = QtWidgets.QGroupBox("Data")
        dl = QtWidgets.QHBoxLayout(data)
        self.bag = QtWidgets.QComboBox()
        self.bag.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.video_btn = QtWidgets.QPushButton("Generate video")
        self.video_btn.clicked.connect(self.generate_video)
        self.play_btn = QtWidgets.QPushButton("Play video")
        self.play_btn.clicked.connect(self.play_video)
        self.video_lbl = QtWidgets.QLabel()
        for w in (QtWidgets.QLabel("bag"), self.bag, self.video_btn, self.play_btn, self.video_lbl):
            dl.addWidget(w)
        dl.addStretch()
        outer.addWidget(data)
        self.bags_read = 0.0

        self.log_tabs = QtWidgets.QTabWidget()
        self.log_views = {}
        for name in TABS:
            v = QtWidgets.QPlainTextEdit(readOnly=True)
            v.setMaximumBlockCount(5000)
            v.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            self.log_views[name] = v
            self.log_tabs.addTab(v, name)
        self.log_tabs.currentChanged.connect(lambda i: self.log_tabs.setTabText(i, TABS[i]))  # seen: drop the ●
        outer.addWidget(self.log_tabs, 1)

        self.rec_start = None
        os.makedirs(RUN, exist_ok=True)
        self.reattach()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.refresh)
        timer.start(100)
        self.refresh()

    # --- processes ----------------------------------------------------------------------------------------------------
    def log(self, line, tab="GUI"):
        self.log_views[tab].appendPlainText(line)
        i = TABS.index(tab)
        if i != self.log_tabs.currentIndex():
            self.log_tabs.setTabText(i, f"{tab} ●")

    def proc(self, name) -> Proc:
        if name not in self.procs:
            p = Proc(name, lambda line, tab=TAB[name]: self.log(line, tab), launch=name == "control")
            p.finished.connect(self.on_finished)
            self.procs[name] = p
        return self.procs[name]

    def running(self, name) -> bool:
        return name in self.procs and self.procs[name].running()

    def reattach(self):
        """Find the processes a previous GUI started: they kept running when it closed."""
        for name in NAMES:
            p = self.proc(name)
            if p.running():
                self.log(f"[gui] {name} is running (started {time.strftime('%H:%M:%S', time.localtime(p.started()))})")
        if self.running("control"):
            self.hardware.setCurrentText(self.procs["control"].meta() or "mock")
        if self.running("server"):
            meta = self.procs["server"].meta()
            checkpoint, port = meta.rsplit(" ", 1) if " " in meta else (meta, "")  # the name may hold a space
            self.checkpoint.setCurrentText(checkpoint)
            self.port.setValue(int(port) if port.isdigit() else 8000)
            prewarm = self.procs["prewarm"]
            code = open(prewarm.path["exit"]).read().strip() if os.path.exists(prewarm.path["exit"]) else ""
            self.server_state = "Starting" if prewarm.running() else "Started" if code == "0" else "Stopped"
        if self.running("record"):
            self.rec_start = time.monotonic() - (time.time() - self.procs["record"].started())

    def on_finished(self, name, code):
        self.log(f"exited with {code}", TAB[name])
        self.log(f"[gui] {name} exited with {code}")
        if name == "prewarm":
            ok = code == 0 and self.running("server")
            self.server_state = "Started" if ok else "Stopped"
            if not ok:
                self.log("[gui] the server did not come up: see its log above")
        elif name == "server":
            self.server_state = "Stopped"
            self.proc("prewarm").stop()
        if name == "record":
            self.rec_start = None
        if name == "move" and code == REFUSED:
            QtWidgets.QMessageBox.warning(self, "Move refused",
                                          "Could not plan a safe motion, Move the robot with Manual mode")

    def toggle_control(self):
        if self.running("control"):
            if self.stopping_control:
                self.log("[gui] control is stopping already")
                return
            if self.hardware.currentText() == "real" and QtWidgets.QMessageBox.question(
                    self, "Stop control", "The motors are disabled and the arms go limp. Are they parked?") \
                    != QtWidgets.QMessageBox.Yes:
                return
            self.cancel()
            self.stop_control()
        else:
            # checked whatever the hardware: the names go into the shell command line
            names_ok = all(re.fullmatch(r"[A-Za-z0-9_.-]{1,15}", i) for i in CAN.values())
            if CAN["left"] == CAN["right"] or not names_ok:
                QtWidgets.QMessageBox.warning(self, "Start control", f"OAP_LEFT_CAN / OAP_RIGHT_CAN: {CAN} must "
                                              "be two different network interface names")
                return
            self.stopping_control = False  # the last control exited: its Stop's wait must not stop this one
            self.proc("control").start(control_cmd(self.hardware.currentText(), CAN), meta=self.hardware.currentText())

    def stop_control(self):
        """The runner first, and control once it has exited (at most POLICY_STOP_S): a runner still on the graph can
        keep ros2_control_node waiting while it deletes its writers."""
        self.stopping_control = True
        self.proc("policy").stop()
        end = time.monotonic() + POLICY_STOP_S

        def then():
            if not self.stopping_control:  # control exited meanwhile: a new one is not this Stop's
                return
            if self.running("policy") and time.monotonic() < end:
                QtCore.QTimer.singleShot(100, then)
                return
            if self.running("policy"):
                self.log(f"[gui] the runner is still up after {POLICY_STOP_S:.0f} s: stopping control anyway")
            self.proc("control").stop()
        then()

    def toggle_server(self):
        """Start the policy server on the checkpoint, and wait until it serves (prewarm); or stop it."""
        if self.running("server"):
            if self.pending and self.pending[1] == "start inference":  # a pending move stays: it needs no server
                self.cancel()
            self.proc("policy").stop()
            self.proc("prewarm").stop()
            self.proc("server").stop()
            return
        if not self.checkpoint.currentText():
            self.log(f"[gui] no checkpoint with a model_config.json in {CHECKPOINTS}")
            return
        if self.running("prewarm"):  # the last server's, still stopping: its start would be refused
            self.proc("prewarm").stop()
            self.log("[gui] the last server's prewarm is still stopping: start the server again in a moment")
            return
        port = self.port.value()
        self.server_state = "Starting"
        self.log("[gui] starting the server: the first start builds its venv (minutes), then it compiles")
        save_settings(pa_text=self.pa_text.text().strip())
        if self.proc("server").start(server_cmd(self.checkpoint.currentText(), port, self.pa_text.text().strip()),
                                     meta=f"{self.checkpoint.currentText()} {port}"):
            self.proc("prewarm").start(f"ros2 run oap_runner prewarm --port {port}")

    def move(self, args):
        """A planner move; a running inference is paused first, or its chunks would play once the path ends."""
        start = lambda: self.proc("move").start(f"ros2 run oap_planner {args}")
        if self.stopping_control:
            self.cancel()
            self.log(f"[gui] {args}: control is stopping, not now")
            return
        if not self.running("policy") or self.ros.inference_state.startswith("paused"):
            self.prepare(args, start)
            return
        self.cancel()
        token = self.action
        self.pending = (token, args)
        self.log(f"[gui] {args}: pause inference first")
        self.pause_inference()
        end = time.monotonic() + PAUSE_WAIT_S

        def wait():
            if token != self.action:  # Hold, Relax or another action came first
                return
            if self.ros.inference_state.startswith("paused"):
                self.pending = None
                self.prepare(args, start)
            elif time.monotonic() < end:
                QtCore.QTimer.singleShot(50, wait)
            else:
                self.pending = None
                self.log(f"[gui] inference did not pause within {PAUSE_WAIT_S:.0f} s: {args} not started")
        QtCore.QTimer.singleShot(50, wait)

    def hold(self):
        """Pause inference and end a move, so nothing sends more commands; then the reset of prepare()."""
        if self.running("policy") and not self.ros.inference_state.startswith("paused"):
            self.pause_inference()
        if self.running("move"):
            self.proc("move").stop(signal.SIGTERM)  # SIGINT ramps a circle out, with more chunks
        self.prepare("hold")

    def cancel(self):
        """Drop a pending prepare(): its remaining steps and its `then` do not run."""
        self.action += 1
        if self.pending is not None:
            self.log("[gui] the pending action is cancelled")
        self.pending = None

    def prepare(self, what, then=None):
        """Before any action: clear a trip (which puts the command on the measured pose), leave relax mode, let the arms
        settle, and clear again at the pose they settled at. Only then `then`. A later action (Hold, Relax, another
        move, Stop control) cancels it; Stop server cancels only a pending inference start. While Stop control waits,
        only Hold (no `then`) goes through."""
        self.cancel()
        if self.stopping_control and then:
            self.log(f"[gui] {what}: control is stopping, not now")
            return
        token = self.action
        self.pending = (token, what)
        self.log(f"[gui] {what}: clear a trip and leave relax mode first")

        def steps():
            yield self.ros.hold.call_async(Trigger.Request())
            for s in SIDES:  # mock hardware has no relax service: skip it
                if self.ros.relax[s].service_is_ready():
                    yield self.ros.relax[s].call_async(SetBool.Request(data=False))
            yield 0.3
            yield self.ros.hold.call_async(Trigger.Request())
            self.pending = None
            if then:
                then()
        self.run_steps(steps(), token)

    def run_steps(self, gen, token):
        """Drive `gen`: it yields futures (wait until done, at most 2 s each) or delays in seconds. It stops once
        another action has bumped `self.action` past `token`."""
        if token != self.action:
            return
        try:
            item = next(gen)
        except StopIteration:
            return
        except Exception as e:  # in a timer callback it would only reach stderr and keep the buttons off
            self.log(f"[gui] the action failed: {e!r}")
            if self.pending and self.pending[0] == token:
                self.pending = None
            return
        start = time.monotonic()

        def wait():
            done = (time.monotonic() - start >= item) if isinstance(item, float) else \
                (item.done() or time.monotonic() - start > 2.0)
            if not done:
                QtCore.QTimer.singleShot(20, wait)
                return
            if not isinstance(item, float) and not item.done():
                self.log("[gui] a service did not answer within 2 s: stopping here")
                if self.pending and self.pending[0] == token:
                    self.pending = None
                return
            self.run_steps(gen, token)
        QtCore.QTimer.singleShot(20, wait)

    def relax(self):
        """Relax mode: kp 0, the arms sink, damped. Any action leaves it (prepare)."""
        self.cancel()  # a pending action would leave relax mode again
        for s in SIDES:
            self.ros.relax[s].call_async(SetBool.Request(data=True))
        self.log("[gui] relax on: any action (Hold, a move, Start inference) leaves it")

    def manual(self):
        """Manual: relax with gravity compensated, to move the arms by hand. Inference pauses and a move ends first,
        then /arm/hold, so motion commands no motion and the torque controller sends gravity alone (no friction: its
        sign is the commanded velocity); /arm/follow keeps the command on the arms as they move, so they do not trip.
        Any action leaves it (prepare: its /arm/hold ends the following)."""
        if self.running("policy") and not self.ros.inference_state.startswith("paused"):
            self.pause_inference()
        if self.running("move"):
            self.proc("move").stop(signal.SIGTERM)
        self.cancel()
        token = self.action

        def steps():
            yield self.ros.hold.call_async(Trigger.Request())
            for s in SIDES:  # mock hardware has no relax node
                if self.ros.manual[s].service_is_ready():
                    yield self.ros.manual[s].call_async(SetBool.Request(data=True))
            yield self.ros.follow.call_async(SetBool.Request(data=True))  # motion follows the arms, /arm/hold ends it
            self.log("[gui] manual on: move the arms by hand; any action (Hold, a move, Start inference) leaves it")
        self.run_steps(steps(), token)

    def start_inference(self):
        """The first time, start the runner, which warms up and runs; after a pause, resume it. Asks first when the
        arms are not at ready."""
        far = self.ros.distance_from_ready()
        if far is None or far > READY_TOL:
            text = "No joint states yet." if far is None else f"The arms are {far:.2f} rad from ready."
            if QtWidgets.QMessageBox.question(self, "Start inference", f"{text} Start anyway?") != \
                    QtWidgets.QMessageBox.Yes:
                return
        self.prepare("start inference", self._start_inference)

    def _start_inference(self):
        if self.running("policy"):
            self.ros.inference["start"].call_async(Trigger.Request())
            self.log("[gui] inference: start")
            return
        self.proc("policy").start(runner_cmd(self.config.text().strip(), self.port.value()))

    def pause_inference(self):
        self.ros.inference["pause"].call_async(Trigger.Request())
        self.log("[gui] inference: pause")

    def toggle_record(self):
        if self.running("record"):
            self.proc("record").stop()
        elif self.proc("record").start(record_cmd(self.tag.text())):
            self.rec_start = time.monotonic()

    def generate_video(self):
        bag = self.bag.currentText()
        if bag:
            self.proc("overlay").start(overlay_cmd(bag), meta=bag)

    def play_video(self):
        path = os.path.join(OUTPUT, self.bag.currentText(), VIDEO)
        if os.path.isfile(path):
            Player(self, path).show()

    def refresh_bags(self):
        """The bag list, read again while the GUI runs: a new bag is selected, else the selection stays."""
        names = bags()
        shown = [self.bag.itemText(k) for k in range(self.bag.count())]
        if names != shown:
            new = [n for n in names if n not in shown]
            current = new[0] if new else self.bag.currentText()
            self.bag.blockSignals(True)
            self.bag.clear()
            self.bag.addItems(names)
            self.bag.setCurrentText(current)
            self.bag.blockSignals(False)
        self.bags_read = time.monotonic()

    # --- status -------------------------------------------------------------------------------------------------------
    def refresh(self):
        for p in list(self.procs.values()):
            p.poll()
        control = self.running("control")
        self.stopping_control &= control
        self.control_btn.setText("Stop control" if control else "Start control")
        self.control_btn.setIcon(self.warning if control else QtGui.QIcon())
        self.hardware.setEnabled(not control)
        alive = self.ros.alive()
        show(self.alive_lbl, "Started" if alive else "Starting" if control else "Stopped")
        self.trip_lbl.setText("TRIPPED: choose a motion (Ready, Park, Hold) to resume" if self.ros.tripped else "")
        self.trip_lbl.setStyleSheet("color: #c62828; font-weight: bold")
        moving = self.running("move")
        policy = self.running("policy")
        state = self.ros.inference_state if policy else ""
        paused = state.startswith("paused")
        idle = alive and not moving and (not policy or paused) and self.pending is None \
            and not self.stopping_control  # Hold and Relax stay enabled
        for b in self.move_btns:
            b.setEnabled(idle)
        server = self.running("server")
        self.server_btn.setText("Stop server" if server else "Start server")
        self.checkpoint.setEnabled(not server)
        self.pa_text.setEnabled(not server)
        self.port.setEnabled(not server)
        show(self.server_lbl, self.server_state)
        self.runner_btn.setEnabled(idle and self.server_state == "Started")
        self.pause_btn.setEnabled(policy and not paused)
        if not policy:
            show(self.state_lbl, "Stopped")
        elif state == "running":
            show(self.state_lbl, "Started")
        elif paused:
            show(self.state_lbl, "Paused", "Paused" + state[len("paused"):].replace(": requested", ""))
        else:
            show(self.state_lbl, "Starting")
        recording = self.running("record")
        self.rec_btn.setText("Stop recording" if recording else "Record")
        if recording:
            show(self.rec_lbl, "Started", f"Recording {time.monotonic() - (self.rec_start or time.monotonic()):.0f} s")
        else:
            show(self.rec_lbl, "Stopped")
        if time.monotonic() - self.bags_read > BAGS_EVERY_S:
            self.refresh_bags()
        bag = self.bag.currentText()
        making = self.running("overlay")
        made = bool(bag) and os.path.isfile(os.path.join(OUTPUT, bag, VIDEO))
        # the newest bag while recording is still open
        open_bag = recording and self.bag.count() > 0 and bag == self.bag.itemText(0)
        self.video_btn.setEnabled(bool(bag) and not making and not open_bag)
        self.play_btn.setEnabled(made)
        if making:
            show(self.video_lbl, "Starting", f"Generating {self.procs['overlay'].meta()}")
        else:
            self.video_lbl.setText("" if made or not bag else "no video yet")
            self.video_lbl.setStyleSheet("")
        with self.ros.lock:
            images = dict(self.ros.images)
        for s, data in images.items():
            if data is None:
                continue
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            h, w, _ = rgb.shape
            img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            self.cam[s].setPixmap(QtGui.QPixmap.fromImage(img).scaled(320, 240, QtCore.Qt.KeepAspectRatio))

    def closeEvent(self, e):  # the processes keep running: a reopened GUI finds them, and only Stop stops them
        e.accept()


def main() -> int:
    """Ctrl-C or SIGTERM closes the window like its close button: the spin thread and rclpy stop here, in order, and
    the processes the GUI started keep running in their own sessions."""
    stopping = threading.Event()

    def on_signal(_sig, _frame):
        stopping.set()
        QtWidgets.QApplication.quit()

    # rclpy's own SIGINT handler would shut the context down under the spinning executor: tracebacks, and Qt runs on
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    ros = Ros()
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(ros)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    app = QtWidgets.QApplication(sys.argv)
    w = Window(ros)
    w.resize(900, 900)
    w.show()
    wake = QtCore.QTimer()  # Python runs a signal handler only between bytecodes: wake it while Qt waits
    wake.timeout.connect(lambda: None)
    wake.start(200)
    code = 0 if stopping.is_set() else app.exec_()
    w.close()
    ex.shutdown(timeout_sec=2.0)
    spin.join(timeout=2.0)
    ros.destroy_node()
    rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
