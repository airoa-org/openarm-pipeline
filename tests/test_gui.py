# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""The GUI's process bookkeeping and command lines without a robot: pid files checked against /proc (a fake one, and a
real child), quoting of the interpolated values, and Hold or Relax cancelling a pending move.

    QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_gui.py
"""
import os
import shlex
import signal
import subprocess
import sys
import threading
import time

import pytest

for _mod in ("rclpy", "cv2", "numpy", "yaml", "ament_index_python"):
    pytest.importorskip(_mod)
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ("oap_gui", "oap_planner"):  # appended, so an installed workspace wins
    if os.path.join(ROOT, "src", _pkg) not in sys.path:
        sys.path.append(os.path.join(ROOT, "src", _pkg))

from oap_gui import app  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def fake_stat(proc, pid, state="S", start="12345", pgrp=None, argv=None):
    """A /proc/<pid>/stat line, the comm holding a space and a ')' as real ones may; and its cmdline."""
    os.makedirs(proc / str(pid), exist_ok=True)
    rest = [state, "1", str(pgrp or pid)] + [str(i) for i in range(6, 22)] + [start, "0", "0"]  # fields 3..24
    (proc / str(pid) / "stat").write_text(f"{pid} (a b) c) " + " ".join(rest) + "\n")
    (proc / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv or ["bash"]) + b"\0")


@pytest.fixture
def run(tmp_path, monkeypatch):
    proc, run = tmp_path / "proc", tmp_path / "run"
    proc.mkdir()
    run.mkdir()
    monkeypatch.setattr(app, "PROC", str(proc))
    monkeypatch.setattr(app, "RUN", str(run))
    monkeypatch.setattr(app, "WS", str(tmp_path))
    return proc, run


def test_start_time_is_field_22(run):
    proc, _ = run
    fake_stat(proc, 42, start="777")
    assert app.start_time(42) == "777"
    assert app.start_time(43) is None


def test_running_checks_the_start_time(run):
    proc, rundir = run
    p = app.Proc("move", lambda line: None)
    assert not p.running()
    (rundir / "move.pid").write_text("42")
    (rundir / "move.start").write_text("42 777")
    assert not p.running()  # no such process
    fake_stat(proc, 42, start="777")
    assert p.running()
    fake_stat(proc, 42, start="999")
    assert not p.running()  # the pid was reused after a container restart
    fake_stat(proc, 42, state="Z", start="777")
    assert not p.running()


def test_running_accepts_an_old_pid_file(run):
    proc, rundir = run
    (rundir / "move.pid").write_text("42")  # an older GUI wrote the pid only, and no start file
    fake_stat(proc, 42, start="999")
    assert app.Proc("move", lambda line: None).running()
    (rundir / "move.start").write_text("41 777")  # left by an earlier start: not this pid's
    assert app.Proc("move", lambda line: None).running()
    (rundir / "move.pid").write_text("garbage")
    assert not app.Proc("move", lambda line: None).running()


def test_start_refuses_a_running_process(run, monkeypatch):
    proc, rundir = run
    fake_stat(proc, 42, start="777")
    (rundir / "move.pid").write_text("42")
    (rundir / "move.start").write_text("42 777")
    lines = []
    p = app.Proc("move", lines.append)
    monkeypatch.setattr(app.subprocess, "Popen", lambda *a, **k: pytest.fail("started twice"))
    assert p.start("ros2 run oap_planner plan --target ready") is False
    assert (rundir / "move.pid").read_text() == "42"
    assert any("already running" in s for s in lines)


@pytest.mark.parametrize("error", [ProcessLookupError, PermissionError])
def test_stop_survives_a_vanished_process(run, monkeypatch, error):
    proc, rundir = run
    fake_stat(proc, 42, start="777")
    (rundir / "move.pid").write_text("42")

    def killpg(pid, sig):
        raise error(pid)
    monkeypatch.setattr(app.os, "killpg", killpg)
    lines = []
    app.Proc("move", lines.append).stop()
    assert lines


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="needs Linux /proc")
def test_a_real_child(run, monkeypatch):
    monkeypatch.setattr(app, "PROC", "/proc")
    _, rundir = run
    p = app.Proc("record", lambda line: None)
    assert p.start("sleep 30")
    pid = (rundir / "record.pid").read_text()  # a bare pid, as an older GUI and `oap down` read it
    assert int(pid) == p.child.pid
    assert (rundir / "record.start").read_text() == f"{pid} {app.start_time(pid)}"
    assert p.running() and not p.start("sleep 30")
    p.stop(signal.SIGTERM)
    p.child.wait(5)
    assert not p.running()


def test_commands_quote_their_values(monkeypatch):
    monkeypatch.setattr(app, "CHECKPOINTS", "/check points")
    assert shlex.split(app.server_cmd("my task; rm", 8000)) == \
        ["./scripts/serve_policy.sh", "/check points/my task; rm", "--port", "8000"]
    assert shlex.split(app.server_cmd("cups", 8001, "pick the 'red' cup"))[2:] == \
        ["--port", "8001", "--pa-text=pick the 'red' cup"]
    assert shlex.split(app.server_cmd("cups", 8000, "-x"))[-1] == "--pa-text=-x"
    assert shlex.split(app.runner_cmd("my cfg.yaml", 8001)) == \
        ["ros2", "run", "oap_runner", "runner", "--config", "my cfg.yaml", "--port", "8001"]
    assert shlex.split(app.runner_cmd("~/c.yaml", 8000))[5:] == [os.path.expanduser("~/c.yaml"), "--port", "8000"]
    assert app.record_cmd("  cups ") == "./scripts/record.sh cups"
    assert app.record_cmd("a b;c'") == "./scripts/record.sh abc"
    assert app.record_cmd(" ") == "./scripts/record.sh"


def test_checkpoints_lists_only_released_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "CHECKPOINTS", str(tmp_path / "none"))
    assert app.checkpoints() == []
    monkeypatch.setattr(app, "CHECKPOINTS", str(tmp_path))
    for name in ("b ok", "a ok"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model_config.json").write_text("{}")
    (tmp_path / "no config").mkdir()
    (tmp_path / "dir config" / "model_config.json").mkdir(parents=True)
    assert app.checkpoints() == ["a ok", "b ok"]


def test_control_command():
    assert shlex.split(app.control_cmd("mock", {"left": "can 1", "right": "can0"})) == \
        ["ros2", "launch", "--noninteractive", "oap_control", "robot.launch.py", "hardware:=mock", "left_can:=can 1",
         "right_can:=can0"]
    outer = shlex.split(app.control_cmd("real", {"left": "can1", "right": "can0"}))
    assert outer[:2] == ["bash", "-c"] and len(outer) == 3
    assert outer[2] == ("./scripts/canup.sh can0 && ./scripts/canup.sh can1 && exec ros2 launch --noninteractive "
                        "oap_control robot.launch.py hardware:=real left_can:=can1 right_can:=can0")


class Done:
    def done(self):
        return True


class FakeRos:
    def __init__(self):
        self.lock = threading.Lock()
        self.tripped, self.inference_state, self.images = False, "", {s: None for s in app.SIDES}
        self.calls = []
        self.fail = False
        client = lambda name: type("C", (), {"call_async": lambda _, req: self._call(name, req),
                                             "service_is_ready": lambda _: True})()
        self.hold = client("hold")
        self.relax = {s: client(f"relax_{s}") for s in app.SIDES}
        self.inference = {w: client(w) for w in ("start", "pause")}

    def _call(self, name, req):
        if self.fail:
            raise RuntimeError("no context")
        self.calls.append((name, req))
        return Done()

    def alive(self):
        return True

    def gripper(self, opening):
        pass


@pytest.fixture(scope="module")
def qapp():
    """One QApplication for the module: a collected one aborts the next widget."""
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(run, tmp_path, monkeypatch, qapp):
    monkeypatch.setattr(app, "CHECKPOINTS", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(app, "SETTINGS", str(tmp_path / "gui.json"))
    started = []
    monkeypatch.setattr(app.Proc, "start", lambda self, cmd, meta="": started.append((self.name, cmd)) or True)
    monkeypatch.setattr(app.os, "killpg", lambda pid, sig: None)  # the fake /proc's pids are not ours
    w = app.Window(FakeRos())
    yield w, started, qapp
    w.close()


def settle(qapp, s=0.8):
    end = time.monotonic() + s
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)


def wait_for(qapp, cond, s=2.0):
    """Process events until `cond()` holds, at most `s`: a timer's slack on a loaded machine is not a failure."""
    end = time.monotonic() + s
    while not cond() and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    return cond()


def test_prepare_runs_its_action(window):
    w, started, qapp = window
    w.move("plan --target ready")
    assert w.pending is not None
    w.refresh()
    assert not any(b.isEnabled() for b in w.move_btns)  # no second click while it is pending
    settle(qapp)
    assert started == [("move", "ros2 run oap_planner plan --target ready")]
    assert w.pending is None
    w.refresh()
    assert all(b.isEnabled() for b in w.move_btns)  # enabled again once it went through


@pytest.mark.parametrize("action", ["hold", "relax"])
def test_hold_or_relax_cancels_a_pending_move(window, action):
    w, started, qapp = window
    w.move("plan --target ready")
    settle(qapp, 0.1)  # inside the 0.3 s settle
    getattr(w, action)()
    settle(qapp)
    assert started == []
    if action == "relax":  # nothing left relax mode after Relax
        assert w.ros.calls[-1][0].startswith("relax_") and w.ros.calls[-1][1].data is True


def test_a_new_move_replaces_a_pending_one(window):
    w, started, qapp = window
    w.move("plan --target ready")
    w.move("plan --target park")
    settle(qapp)
    assert started == [("move", "ros2 run oap_planner plan --target park")]


def running_inference(w, run, state="running"):
    proc, rundir = run
    fake_stat(proc, 60)
    (rundir / "policy.pid").write_text("60")
    w.ros.inference_state = state


def test_a_move_pauses_a_running_inference_first(window, run):
    w, started, qapp = window
    running_inference(w, run)
    w.move("plan --target park")
    settle(qapp, 0.3)
    assert [n for n, _ in w.ros.calls] == ["pause"] and started == []  # no prepare() yet: still running
    w.refresh()
    assert not any(b.isEnabled() for b in w.move_btns)
    w.ros.inference_state = "paused: requested"
    settle(qapp)
    assert started == [("move", "ros2 run oap_planner plan --target park")]
    assert w.ros.calls[1][0] == "hold" and w.pending is None


def test_a_move_does_not_start_when_inference_does_not_pause(window, run, monkeypatch):
    w, started, qapp = window
    monkeypatch.setattr(app, "PAUSE_WAIT_S", 0.3)
    running_inference(w, run)
    w.move("plan --target park")
    settle(qapp, 0.8)
    assert started == [] and w.pending is None
    assert [n for n, _ in w.ros.calls] == ["pause"]
    assert "did not pause" in w.log_views["GUI"].toPlainText()


def test_hold_cancels_a_move_waiting_for_the_pause(window, run):
    w, started, qapp = window
    running_inference(w, run)
    w.move("plan --target park")
    w.ros.inference_state = "paused: requested"  # Hold itself sees it paused and only resets
    w.hold()
    settle(qapp)
    assert started == []


def test_no_move_or_pause_while_control_stops(window, run):
    w, started, qapp = window
    running_inference(w, run)
    w.stopping_control = True
    w.move("plan --target park")
    settle(qapp, 0.3)
    assert started == [] and w.ros.calls == [] and w.pending is None


def test_a_move_with_inference_paused_needs_no_pause(window, run):
    w, started, qapp = window
    running_inference(w, run, "paused: requested")
    w.move("plan --target park")
    settle(qapp)
    assert "pause" not in [n for n, _ in w.ros.calls]
    assert started == [("move", "ros2 run oap_planner plan --target park")]


def test_a_failing_step_clears_the_pending_action(window):
    w, started, qapp = window
    w.ros.fail = True
    w.move("plan --target ready")
    settle(qapp, 0.2)
    assert started == [] and w.pending is None
    w.refresh()
    assert all(b.isEnabled() for b in w.move_btns)
    assert "the action failed" in w.log_views["GUI"].toPlainText()


def fake_server(w, run):
    proc, rundir = run
    fake_stat(proc, 42)
    (rundir / "server.pid").write_text("42")
    w.proc("server")


def test_stop_server_keeps_a_pending_move(window, run):
    w, started, qapp = window
    fake_server(w, run)
    w.move("plan --target park")
    w.toggle_server()  # stops the server
    settle(qapp)
    assert started == [("move", "ros2 run oap_planner plan --target park")]


def test_stop_server_cancels_a_pending_inference_start(window, run):
    w, started, qapp = window
    fake_server(w, run)
    w.prepare("start inference", w._start_inference)
    w.toggle_server()
    settle(qapp)
    assert started == [] and w.pending is None


def test_start_server_waits_for_a_leftover_prewarm(window, run, tmp_path):
    w, started, qapp = window
    (tmp_path / "checkpoints" / "b").mkdir(parents=True)
    w.checkpoint.addItem("b")
    w.checkpoint.setCurrentText("b")
    proc, rundir = run
    fake_stat(proc, 43)
    (rundir / "prewarm.pid").write_text("43")
    w.toggle_server()
    assert started == [] and w.server_state == "Stopped"


def test_reattach_reads_a_meta_without_a_port(run, tmp_path, monkeypatch, qapp):
    proc, rundir = run
    fake_stat(proc, 42)
    (rundir / "server.pid").write_text("42")
    (rundir / "server.meta").write_text("task")
    monkeypatch.setattr(app, "CHECKPOINTS", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(app, "SETTINGS", str(tmp_path / "gui.json"))
    w = app.Window(FakeRos())
    assert w.port.value() == 8000 and w.checkpoint.currentText() == ""  # "task" is not in the empty checkpoints
    w.close()


GUI = """
import sys
from oap_gui import app
app.RUN = app.WS = sys.argv[1]
app.CHECKPOINTS = app.SETTINGS = sys.argv[1] + "/none"
init = app.Window.__init__
def started(self, ros):
    init(self, ros)
    self.proc("record").start("sleep 60")
    print("ready", flush=True)
app.Window.__init__ = started
sys.exit(app.main())
"""


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="needs Linux /proc")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_closes_the_gui_and_keeps_its_processes(tmp_path, sig):
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": os.pathsep.join(sys.path)}
    gui = subprocess.Popen([sys.executable, "-c", GUI, str(tmp_path)], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    try:
        head = []  # Qt's notes come first
        while not head or head[-1] not in ("ready", ""):
            head.append(gui.stdout.readline().strip())
        assert head[-1] == "ready", head
        time.sleep(0.5)
        gui.send_signal(sig)
        out, _ = gui.communicate(timeout=10)
    finally:
        gui.kill()
    assert gui.returncode == 0, out
    assert "Traceback" not in out and "Exception" not in out, out
    pid = int((tmp_path / "record.pid").read_text())
    try:
        assert app.stat(pid) is not None and app.stat(pid)[0] != "Z"  # the recording outlives the window
    finally:
        os.killpg(pid, signal.SIGKILL)


def test_launch_pid_finds_the_launch_in_the_group(run):
    proc, _ = run
    fake_stat(proc, 10, argv=["bash", "-c", "ros2 launch x; echo $? > exit"])  # the exit-code shell, the leader
    fake_stat(proc, 11, pgrp=10, argv=["/usr/bin/python3", "/opt/ros/humble/bin/ros2", "launch", "oap_control"])
    fake_stat(proc, 12, pgrp=10, argv=["/ws/install/lib/controller_manager/ros2_control_node", "--ros-args"])
    fake_stat(proc, 20, argv=["/usr/bin/python3", "/opt/ros/humble/bin/ros2", "launch", "other"])  # another group
    fake_stat(proc, 30, pgrp=30, argv=["ros2", "run", "oap_planner", "plan"])
    assert app.launch_pid(10) == 11
    assert app.launch_pid(30) is None


@pytest.mark.parametrize("argv, launch", [
    (["/usr/bin/python3", "/opt/ros/humble/bin/ros2", "launch", "x"], True),
    (["python3", "-s", "-u", "/opt/ros/humble/bin/ros2", "launch", "x"], True),
    (["/opt/ros/humble/bin/ros2", "launch", "--noninteractive", "x"], True),
    (["/usr/bin/python3", "/opt/ros/humble/bin/ros2", "run", "x"], False),
    (["bash", "-c", "ros2 launch x; echo $? > exit"], False),
    (["/usr/bin/python3", "/opt/ros/humble/bin/ros2"], False),
    ([""], False),
])
def test_is_launch(argv, launch):
    assert app.is_launch([a.encode() for a in argv]) is launch


def test_a_launch_runs_while_its_group_lives(run):
    proc, rundir = run
    (rundir / "control.pid").write_text("10")
    fake_stat(proc, 12, pgrp=10, argv=["ros2_control_node"])  # the leader and the launch gone, a node left
    assert app.Proc("control", [].append, launch=True).running()
    assert not app.Proc("control", [].append, launch=False).running()


def control(proc, rundir, lines, launch=True):
    fake_stat(proc, 10, start="1")
    fake_stat(proc, 11, pgrp=10, argv=["python3", "/opt/ros/humble/bin/ros2", "launch", "oap_control"])
    (rundir / "control.pid").write_text("10")
    return app.Proc("control", lines.append, launch=launch)


def test_stop_control_signals_the_launch_once(run, monkeypatch, qapp):
    proc, rundir = run
    sent = []
    monkeypatch.setattr(app.os, "kill", lambda pid, sig: sent.append(("kill", pid, sig)))
    monkeypatch.setattr(app.os, "killpg", lambda pid, sig: sent.append(("killpg", pid, sig)))
    monkeypatch.setattr(app, "ESCALATE_S", (0.1, 0.1))
    lines = []
    p = control(proc, rundir, lines)
    p.stop()
    p.stop()  # a second click: no second SIGINT
    assert sent == [("kill", 11, signal.SIGINT)]
    assert wait_for(qapp, lambda: len(sent) == 3)  # it hangs: SIGTERM, then SIGKILL to the group
    assert sent[1:] == [("killpg", 10, signal.SIGTERM), ("killpg", 10, signal.SIGKILL)]


def test_escalation_spares_a_new_process(run, monkeypatch, qapp):
    proc, rundir = run
    sent = []
    monkeypatch.setattr(app.os, "kill", lambda pid, sig: sent.append(("kill", pid, sig)))
    monkeypatch.setattr(app.os, "killpg", lambda pid, sig: sent.append(("killpg", pid, sig)))
    monkeypatch.setattr(app, "ESCALATE_S", (0.1, 0.1))
    p = control(proc, rundir, [])
    p.stop()
    fake_stat(proc, 10, start="2")  # it exited and the pid came back as another process
    settle(qapp, 0.3)
    assert sent == [("kill", 11, signal.SIGINT)]


def test_other_processes_and_sigterm_signal_the_group(run, monkeypatch):
    proc, rundir = run
    sent = []
    monkeypatch.setattr(app.os, "kill", lambda pid, sig: sent.append(("kill", pid, sig)))
    monkeypatch.setattr(app.os, "killpg", lambda pid, sig: sent.append(("killpg", pid, sig)))
    control(proc, rundir, [], launch=False).stop()
    control(proc, rundir, [], launch=True).stop(signal.SIGTERM)
    assert sent == [("killpg", 10, signal.SIGINT), ("killpg", 10, signal.SIGTERM)]


def test_no_launch_found_signals_the_group_loudly_and_escalates(run, monkeypatch, qapp):
    proc, rundir = run
    sent = []
    monkeypatch.setattr(app.os, "kill", lambda pid, sig: sent.append(("kill", pid, sig)))
    monkeypatch.setattr(app.os, "killpg", lambda pid, sig: sent.append(("killpg", pid, sig)))
    monkeypatch.setattr(app, "ESCALATE_S", (0.3, 0.1))
    lines = []
    p = control(proc, rundir, lines)
    (proc / "11" / "cmdline").write_bytes(b"/opt/ros/lib/ros2_control_node\x00")  # no launch: a node left
    p.stop()
    assert sent == []  # it scans once more first
    assert wait_for(qapp, lambda: sent)
    assert sent == [("killpg", 10, signal.SIGINT)]
    assert any("WARNING" in s and "ros2_control_node" in s for s in lines)
    p.stop()  # stopping already: nothing more
    assert wait_for(qapp, lambda: len(sent) == 3)
    assert sent[1:] == [("killpg", 10, signal.SIGTERM), ("killpg", 10, signal.SIGKILL)]


def test_stop_control_stops_the_runner_first(window, run, monkeypatch):
    w, started, qapp = window
    proc, rundir = run
    stopped = []
    monkeypatch.setattr(app.Proc, "stop", lambda self, sig=signal.SIGINT: stopped.append(self.name))
    fake_stat(proc, 50)
    (rundir / "policy.pid").write_text("50")
    fake_stat(proc, 60)
    (rundir / "control.pid").write_text("60")
    w.ros.inference_state = "paused"
    w.stop_control()
    settle(qapp, 0.3)
    assert stopped == ["policy"]  # control waits while the runner is up
    w.refresh()
    assert not any(b.isEnabled() for b in w.move_btns) and not w.runner_btn.isEnabled()  # no move meanwhile
    w.move("plan --target ready")
    assert w.pending is None and started == []
    w.hold()  # Hold still clears a trip and holds the measured pose
    assert wait_for(qapp, lambda: [c[0] for c in w.ros.calls].count("hold") == 2) and started == []
    os.remove(proc / "50" / "stat")
    assert wait_for(qapp, lambda: stopped == ["policy", "control"])
    os.remove(proc / "60" / "stat")
    w.refresh()
    assert not w.stopping_control


def test_stop_control_skips_a_control_that_exited_meanwhile(window, run, monkeypatch):
    w, started, qapp = window
    proc, rundir = run
    stopped = []
    monkeypatch.setattr(app.Proc, "stop", lambda self, sig=signal.SIGINT: stopped.append(self.name))
    fake_stat(proc, 50)
    (rundir / "policy.pid").write_text("50")
    fake_stat(proc, 60)
    (rundir / "control.pid").write_text("60")
    w.stop_control()
    os.remove(proc / "60" / "stat")  # control exited by itself, and a new one is started
    w.refresh()
    w.toggle_control()
    assert ("control", app.control_cmd("mock", app.CAN)) in started
    fake_stat(proc, 60, start="2")
    os.remove(proc / "50" / "stat")
    settle(qapp, 0.3)
    assert stopped == ["policy"]  # the old Stop's wait leaves the new control alone


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="needs Linux /proc")
def test_real_mock_control_stops_cleanly(run, monkeypatch, tmp_path):
    """The real chain, as the GUI runs it: mock control from control_cmd, stopped by Proc.stop."""
    import re
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    try:
        get_package_share_directory("oap_control")
    except PackageNotFoundError:
        pytest.skip("oap_control is not built and sourced")
    monkeypatch.setattr(app, "PROC", "/proc")
    p = app.Proc("control", lambda line: None, launch=True)
    assert p.start(app.control_cmd("mock", app.CAN))
    try:
        end = time.monotonic() + 60
        while "Configured and activated" not in open(p.path["log"]).read() and time.monotonic() < end:
            assert p.running(), open(p.path["log"]).read()
            time.sleep(0.2)
        time.sleep(1.0)
        t = time.monotonic()
        p.stop()
        while p.running() and time.monotonic() - t < 40:
            time.sleep(0.05)
        took = time.monotonic() - t
    finally:
        if p.running():
            os.killpg(p.pid(), signal.SIGKILL)
    log = open(p.path["log"]).read()
    assert re.search(r"\[ros2_control_node-\d+\]: process has finished cleanly", log), log
    assert "escalating" not in log, log
    assert open(p.path["exit"]).read().strip() == "0", log
    assert took < 20, took


LAUNCH = """#!{python}
# stands in for ros2 launch: forwards SIGINT once to its node, which counts the SIGINTs it gets
import signal, subprocess, sys
NODE = '''
import signal, sys, time
got = []
signal.signal(signal.SIGINT, lambda s, f: got.append(time.monotonic()))
open(sys.argv[1] + "/ready", "w").close()
while not got:
    time.sleep(0.01)
time.sleep(1.0)  # a second SIGINT would land while it deactivates
open(sys.argv[1] + "/sigints", "w").write(str(len(got)))
'''
node = subprocess.Popen([sys.executable, "-c", NODE, "{out}"])
signal.signal(signal.SIGINT, lambda s, f: node.send_signal(signal.SIGINT))
sys.exit(node.wait())
"""


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="needs Linux /proc")
def test_a_launch_gets_one_sigint_and_its_node_one(run, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "PROC", "/proc")
    _, rundir = run
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "ros2").write_text(LAUNCH.format(python=sys.executable, out=tmp_path))
    (bindir / "ros2").chmod(0o755)
    p = app.Proc("control", lambda line: None, launch=True)
    assert p.start(f"{shlex.quote(str(bindir / 'ros2'))} launch fake")
    end = time.monotonic() + 10
    while not (tmp_path / "ready").exists() and time.monotonic() < end:
        time.sleep(0.05)
    p.stop()
    p.child.wait(10)
    assert (tmp_path / "sigints").read_text() == "1"  # the group SIGINT before this fix made it 2
    assert (rundir / "control.exit").read_text().strip() == "0"
