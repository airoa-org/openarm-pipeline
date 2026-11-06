# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""The runner without a server, cameras or ROS graph: the request's state, the reply's rows turned into hand_root
poses, their timing, the step limits, trips, pauses, stale streams, a lost server, start/pause requests, prewarm and
leaving. run() is driven a few loop turns at a time with rclpy.ok patched."""
import collections
import io as iolib
import json
import math
import os
import queue
import random
import threading
import time

import pytest

np = pytest.importorskip("numpy")
pin = pytest.importorskip("pinocchio")
for _mod in ("rclpy", "cv2", "yaml", "std_srvs"):
    pytest.importorskip(_mod)

from builtin_interfaces.msg import Time  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from oap_runner import runner as rn  # noqa: E402

HZ, N = 20.0, 8  # the released checkpoints' rows per second and rows per reply
TCP = np.array([0.088, 0.0, 0.0])
IDENTITY9 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # xyz, then the rotation's first two columns
BASE = pin.SE3(pin.exp3(np.array([0.0, 0.0, math.pi / 2])), np.array([0.3, 0.2, 0.4]))
META = {"state_dim": 11, "action_dim": 20, "action_hz": 20.0, "chunk_steps": 8}


class FakeTime:
    """The runner's clock: still until something sleeps, so wall-time margins do not depend on the machine."""

    def __init__(self):
        self.wall, self.mono = 1_700_000_000.0, 1000.0

    def time(self):
        return self.wall

    def monotonic(self):
        return self.mono

    def sleep(self, d):
        self.wall += d
        self.mono += d


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """Every test runs on FakeTime (rn.time is the runner's time module)."""
    c = FakeTime()
    monkeypatch.setattr(rn, "time", c)
    return c


class Log:
    def __init__(self):
        self.errors = []

    def info(self, *_):
        pass

    def warn(self, *_):
        pass

    def error(self, msg):
        self.errors.append(msg)


class FakeIo:
    def __init__(self):
        self.lock = threading.Lock()
        self.tripped = False
        self.requests = queue.SimpleQueue()
        self.chunks, self.finger_cmds, self.states, self.holds = {}, [], [], []
        self.fingers = {n: 0.3 for n in rn.FINGERS}  # measured, as Io keeps them
        self.measured = {s: pin.SE3.Identity() for s in rn.SIDES}
        self.residual = {s: pin.SE3.Identity() for s in rn.SIDES}
        self.hold_ok = True
        self.touch()

    def touch(self):
        t = rn.time.monotonic()
        self.hand = {s: (t, rn.pose7(self.measured[s])) for s in rn.SIDES}
        self.finger_stamp = t

    def hand_at(self, side, mono):
        return rn.pose7(self.measured[side])

    def command_at(self, side, t):
        return BASE

    def command_now(self, side):
        return BASE

    def ik_residual(self, side):
        return self.residual[side]

    def send_chunk(self, side, times, poses):
        self.chunks[side] = (np.asarray(times), list(poses))

    def send_fingers(self, positions, period):
        self.finger_cmds.append(np.asarray(positions))

    def set_state(self, state):
        self.states.append(state)

    def hold(self, timeout):
        self.holds.append(timeout)
        return self.hold_ok


class FakeCam:
    def __init__(self):
        self.frozen = None  # a capture time that no longer advances

    def latest(self):
        return np.zeros((4, 4, 3), np.uint8), self.frozen if self.frozen is not None else rn.time.monotonic()


def make_runner(io, execute_rows=0, **cfg):
    r = rn.Runner.__new__(rn.Runner)  # skip __init__: it opens cameras and asks the server
    r.cfg = {"max_step_mm": 50.0, "max_step_deg": 20.0, **cfg}
    r.io, r.log = io, Log()
    r.url, r.meta, r.session = "http://test", {}, "s"
    r.execute_rows, r.hz, r.latency = execute_rows, HZ, 0.0
    r.tip = pin.SE3(np.eye(3), TCP)
    r.cams = {s: FakeCam() for s in rn.SIDES}
    r.generation, r.history_reset = 0, True
    r.finger_rows, r.last_due = collections.deque(), None
    r.last_capture, r.last_state_capture = {s: 0.0 for s in rn.SIDES}, 0.0
    r.replies, r.awaiting, r.paused = queue.Queue(), False, False
    r.started = rn.time.monotonic() - rn.STALE_S - 1.0  # past the start-up grace
    r.sent = 0

    def request():  # no server: count it, and wait for a reply the test puts
        r.sent += 1
        r.awaiting, r.requested_at = True, rn.time.monotonic()
    r.request = request
    return r


def actions(n=N, rows=None, fingers=None):
    """[n, 20]: left tip pose9, right tip pose9, two finger openings."""
    a = np.zeros((n, 20))
    for k in range(n):
        a[k, 0:9] = a[k, 9:18] = IDENTITY9 if rows is None else rows[k]
    a[:, 18:20] = np.arange(2 * n).reshape(n, 2) / 100.0 if fingers is None else fingers
    return a


def reply(a, hz=HZ):
    return {"actions": np.asarray(a).tolist(), "action_hz": hz, "history_slots_used": 0}


def ticket(r, age=0.0, generation=None):
    return rn.Ticket(wall=rn.time.time() - age, mono=rn.time.monotonic() - age,
                     generation=r.generation if generation is None else generation)


def assert_se3_close(a, b, atol=1e-9):
    np.testing.assert_allclose(a.translation, b.translation, atol=atol)
    np.testing.assert_allclose(a.rotation, b.rotation, atol=atol)


# --- pose helpers -----------------------------------------------------------------------------------------------------
def test_pose9_is_xyz_then_the_first_two_rotation_columns():
    T = pin.SE3(pin.exp3(np.array([0.3, -0.2, 0.5])), np.array([0.1, -0.2, 0.3]))
    v = rn.pose9(T)
    np.testing.assert_allclose(v, [*T.translation, *T.rotation[:, 0], *T.rotation[:, 1]])
    assert_se3_close(rn.from_pose9(v), T)


def test_from_pose9_makes_the_columns_orthonormal():
    v = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.1, 1.0, 0.0])  # scaled, and the second column leans on the first
    R = rn.from_pose9(v).rotation
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(R[:, 0], [1.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("d, sec, nanosec", [
    (-0.003, 0, 0),  # was 0.997 s
    (-1.5, 0, 0),
    (0.0, 0, 0),
    (1.25, 1, 250_000_000),
    (0.9999999999, 1, 0),  # was nanosec = 1e9
])
def test_duration(d, sec, nanosec):
    m = rn.duration(d)
    assert (m.sec, m.nanosec) == (sec, nanosec)


class FakePub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


class FakeClock:
    def __init__(self, t):
        self.t = t

    def now(self):
        return self

    def to_msg(self):
        ns = int(round(self.t * 1e9))
        return Time(sec=ns // 10**9, nanosec=ns % 10**9)


def knots_sent(due_from_stamp):
    """The time_from_start and x of each point send_chunk publishes, for knots at x = 0, 1, 2, ..."""
    node = type("Node", (), {})()
    stamp = 1_700_000_000.5
    node.get_clock = lambda: FakeClock(stamp)
    node.chunk_pub = {"left": FakePub()}
    poses = [pin.SE3(np.eye(3), np.array([float(k), 0.0, 0.0])) for k in range(len(due_from_stamp))]
    rn.Io.send_chunk(node, "left", stamp + np.asarray(due_from_stamp), poses)
    if not node.chunk_pub["left"].msgs:
        return None
    pts = node.chunk_pub["left"].msgs[0].points
    return ([p.time_from_start.sec + 1e-9 * p.time_from_start.nanosec for p in pts],
            [p.transforms[0].translation.x for p in pts])


def test_knots_already_due_are_left_out():
    """Motion's spline keeps only knots after its own time, which is later than the stamp."""
    t, x = knots_sent([-0.25, -0.003, 0.05, 0.2])
    assert t == pytest.approx([0.05, 0.2]) and x == [2.0, 3.0]


def test_a_chunk_all_past_is_not_sent():
    assert knots_sent([-0.3, -0.2, 0.0]) is None


def test_a_chunk_all_ahead_sends_every_knot():
    t, x = knots_sent([0.1, 0.2, 0.3])
    assert t == pytest.approx([0.1, 0.2, 0.3]) and x == [0.0, 1.0, 2.0]


# --- the request ------------------------------------------------------------------------------------------------------
def test_the_state_is_the_right_tip_seen_from_the_left_tip_then_the_fingers():
    io = FakeIo()
    io.measured["left"] = pin.SE3(pin.exp3(np.array([0.0, 0.3, 0.0])), np.array([0.3, 0.2, 0.4]))
    io.measured["right"] = pin.SE3(pin.exp3(np.array([0.1, 0.0, -0.4])), np.array([0.35, -0.2, 0.45]))
    io.fingers = {rn.FINGERS[0]: 0.4, rn.FINGERS[1]: 0.6}
    r = make_runner(io)
    obs = r.observation(rn.time.monotonic())
    X = pin.SE3(np.eye(3), TCP)
    rel = (io.measured["left"] * X).actInv(io.measured["right"] * X)
    np.testing.assert_allclose(obs["state"], [*rn.pose9(rel), 0.4, 0.6], atol=1e-6)
    assert obs["state"].dtype == np.float32
    assert set(obs) >= {"left_wrist_0_rgb", "right_wrist_0_rgb", "timestamp", "session", "reset"}


def test_the_first_request_after_a_start_resets_the_history(monkeypatch):
    r = make_runner(FakeIo())
    sent = []
    monkeypatch.setattr(r, "post", lambda path, body: sent.append(dict(np.load(iolib.BytesIO(body)))) or {})
    for _ in range(2):
        rn.Runner.request(r)
        deadline = time.monotonic() + 5.0
        while r.replies.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        r.replies.get_nowait()
    assert [bool(s["reset"]) for s in sent] == [True, False]


# --- the reply's rows -------------------------------------------------------------------------------------------------
def test_identity_rows_hold_the_commanded_hand():
    io = FakeIo()
    r = make_runner(io)
    r.install(ticket(r), reply(actions()), 0.1)
    now = rn.time.time()
    for s in rn.SIDES:
        times, poses = io.chunks[s]
        np.testing.assert_allclose(times, now + rn.ENTRY_MARGIN_S + (np.arange(N) + 1) / HZ)
        for T in poses:
            assert_se3_close(T, BASE)
    assert r.last_due == pytest.approx(io.chunks["left"][0][-1])


def test_a_turn_of_the_tip_keeps_the_tip_in_place():
    """The policy's poses are of the tip: a pure rotation turns the hand about the tip, 0.088 m ahead of hand_root."""
    turn = pin.SE3(pin.exp3(np.array([0.0, 0.0, 0.4])), np.zeros(3))  # 23 deg: hand_root swings 35 mm
    io = FakeIo()
    r = make_runner(io, max_step_deg=30.0)
    r.install(ticket(r), reply(actions(rows=[rn.pose9(turn)] * N)), 0.1)
    X = pin.SE3(np.eye(3), TCP)
    for T in io.chunks["left"][1]:
        np.testing.assert_allclose((T * X).translation, (BASE * X).translation, atol=1e-12)
        np.testing.assert_allclose(T.rotation, (BASE * turn).rotation, atol=1e-12)


def test_a_tip_step_moves_hand_root_by_the_same_step():
    step = pin.SE3(np.eye(3), np.array([0.01, 0.0, 0.0]))
    io = FakeIo()
    r = make_runner(io)
    r.install(ticket(r), reply(actions(rows=[rn.pose9(step)] * N)), 0.1)
    # BASE is turned 90 deg about z: the hand's +x is the world's +y
    np.testing.assert_allclose(io.chunks["right"][1][0].translation, BASE.translation + [0.0, 0.01, 0.0], atol=1e-12)


def test_the_rows_start_where_the_ik_got():
    io = FakeIo()
    io.residual["left"] = pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.02]))
    r = make_runner(io)
    r.install(ticket(r), reply(actions()), 0.1)
    assert_se3_close(io.chunks["left"][1][0], BASE * io.residual["left"])
    assert_se3_close(io.chunks["right"][1][0], BASE)


def test_execute_rows_plays_the_first_rows_only():
    io = FakeIo()
    r = make_runner(io, execute_rows=3)
    r.install(ticket(r), reply(actions()), 0.1)
    assert len(io.chunks["left"][1]) == 3 and len(r.finger_rows) == 3


def test_the_reply_sets_the_row_rate():
    io = FakeIo()
    r = make_runner(io)
    r.install(ticket(r), reply(actions(), hz=10.0), 0.1)
    np.testing.assert_allclose(np.diff(io.chunks["left"][0]), 0.1, atol=1e-6)
    assert r.hz == 10.0


@pytest.mark.parametrize("bad", [
    lambda a: a.__setitem__((3, 0), np.nan),
    lambda a: a.__setitem__((0, 12), np.inf),
])
def test_non_finite_actions_are_refused(bad):
    io = FakeIo()
    a = actions()
    bad(a)
    r = make_runner(io)
    with pytest.raises(RuntimeError, match="20 values per row"):
        r.install(ticket(r), reply(a), 0.1)
    assert io.chunks == {}


def test_a_reply_without_twenty_values_per_row_is_refused():
    r = make_runner(FakeIo())
    with pytest.raises(RuntimeError, match="20 values per row"):
        r.install(ticket(r), {"actions": [[0.0] * 16] * N}, 0.1)


def test_a_first_row_over_the_step_limit_is_refused():
    far = pin.SE3(np.eye(3), np.array([0.1, 0.0, 0.0]))  # 100 mm
    io = FakeIo()
    r = make_runner(io, max_step_mm=50.0)
    with pytest.raises(RuntimeError, match="first row moves 100 mm"):
        r.install(ticket(r), reply(actions(rows=[rn.pose9(far)] * N)), 0.1)
    assert io.chunks == {} and not r.finger_rows


def jump_rows():
    """A 200 mm step between rows 3 and 4."""
    return [rn.pose9(pin.SE3(np.eye(3), np.array([0.001 + (0.2 if k >= 4 else 0.0), 0.0, 0.0]))) for k in range(N)]


def test_a_row_step_over_the_limit_is_refused():
    r = make_runner(FakeIo(), max_row_step_mm=100.0)
    with pytest.raises(RuntimeError, match="row 4 moves 200 mm"):
        r.install(ticket(r), reply(actions(rows=jump_rows())), 0.1)


def test_without_a_row_limit_in_the_config_rows_are_not_limited():
    r = make_runner(FakeIo())
    r.install(ticket(r), reply(actions(rows=jump_rows())), 0.1)


# --- fingers and the next request -------------------------------------------------------------------------------------
def test_a_finger_row_goes_out_one_row_before_its_hand_row():
    io = FakeIo()
    r = make_runner(io)
    a = actions()
    r.install(ticket(r), reply(a), 0.1)
    due = io.chunks["left"][0]
    r.emit_fingers(due[0] - 1.0 / HZ - 1e-6)
    assert io.finger_cmds == []
    r.emit_fingers(due[2] - 1.0 / HZ)
    assert len(io.finger_cmds) == 3
    np.testing.assert_array_equal(io.finger_cmds[-1], a[2, 18:20])


def test_the_next_request_waits_for_the_last_row():
    io = FakeIo()
    r = make_runner(io)
    assert r.wants_request()  # nothing played yet
    r.install(ticket(r), reply(actions()), 0.1)
    assert not r.wants_request()
    rn.time.sleep(r.last_due - rn.time.time())
    assert r.wants_request()
    r.awaiting = True
    assert not r.wants_request()
    r.awaiting, r.paused = False, True
    assert not r.wants_request()


# --- the loop ---------------------------------------------------------------------------------------------------------
def drive(monkeypatch, r, turns, hook=None):
    """run() for this many loop turns; hook(turn) before each. The fake streams stay fresh."""
    count = [0]

    def ok():
        if count[0] >= turns:
            return False
        r.io.touch()
        if hook:
            hook(count[0])
        count[0] += 1
        return True
    monkeypatch.setattr(rn.rclpy, "ok", ok)
    return r.run()


def reply_on_first_turn(r, resp):
    """A reply for the generation run() starts in (resume() bumps it)."""
    def hook(turn):
        if turn == 0:
            r.awaiting = True
            r.replies.put((ticket(r), resp, 0.1))
    return hook


def test_a_trip_stops_the_finger_rows(monkeypatch):
    io = FakeIo()
    r = make_runner(io)
    at_trip = []
    first = reply_on_first_turn(r, reply(actions()))

    def hook(turn):
        first(turn)
        if len(io.finger_cmds) >= 2 and not io.tripped:
            io.tripped = True
            at_trip.append(len(io.finger_cmds))
    drive(monkeypatch, r, 300, hook)  # 0.6 s: the rows end at 0.42 s
    assert at_trip and len(io.finger_cmds) == at_trip[0]
    assert io.states[-1].startswith("paused: the arm tripped")


def test_an_operator_pause_lets_the_finger_rows_play_out(monkeypatch):
    io = FakeIo()
    r = make_runner(io)
    first = reply_on_first_turn(r, reply(actions()))

    def hook(turn):
        first(turn)
        if turn == 1:
            io.requests.put("pause")
    drive(monkeypatch, r, 300, hook)
    assert io.states[-1] == "paused: requested"
    assert len(io.finger_cmds) == N


def test_a_refused_chunk_pauses_the_run(monkeypatch):
    io = FakeIo()
    r = make_runner(io)
    a = actions()
    a[2, 0] = np.nan
    assert drive(monkeypatch, r, 5, reply_on_first_turn(r, reply(a))) == 0
    assert io.states[-1].startswith("paused: not a chunk of 20 values per row")
    assert io.chunks == {} and io.finger_cmds == [] and io.holds == []


def test_a_refused_request_pauses_instead_of_ending(monkeypatch):
    io = FakeIo()
    r = make_runner(io)
    err = RuntimeError("the server refused the request: ValueError: state must have shape (11,)")
    assert drive(monkeypatch, r, 5, reply_on_first_turn(r, err)) == 0
    assert io.states[-1].startswith("paused: the server refused") and io.holds == []


@pytest.mark.parametrize("freeze, what", [
    (lambda r: setattr(r.cams["left"], "frozen", rn.time.monotonic() - rn.STALE_S - 1), "left camera"),
    (lambda r: r.io.__setattr__("finger_stamp", rn.time.monotonic() - rn.STALE_S - 1), "finger state"),
    (lambda r: r.io.hand.__setitem__("right", (rn.time.monotonic() - rn.STALE_S - 1, None)), "right hand pose"),
])
def test_a_stale_stream_pauses_the_run(monkeypatch, freeze, what):
    io = FakeIo()
    r = make_runner(io)
    drive(monkeypatch, r, 5, lambda turn: freeze(r) if turn >= 2 else None)
    assert io.states[-1].startswith("paused: no new sample from") and what in io.states[-1]
    assert r.paused


def test_streams_not_yet_up_get_a_grace_from_start():
    io = FakeIo()
    io.hand, io.finger_stamp = {s: None for s in rn.SIDES}, 0.0
    r = make_runner(io)
    for s in rn.SIDES:
        r.cams[s].frozen = 0.0
    r.started = rn.time.monotonic()
    assert r.stale() is None
    r.started = rn.time.monotonic() - rn.STALE_S - 0.1
    assert "no new sample from the left hand pose" in r.stale()


def in_flight(r, then):
    """A chunk on turn 0, then(turn) from turn 2: the chunk's rows still play."""
    first = reply_on_first_turn(r, reply(actions()))

    def hook(turn):
        first(turn)
        if turn >= 2:
            then(turn)
    return hook


def lost_on(r, error, at=2):
    def then(turn):
        if turn == at:
            r.awaiting = True
            r.replies.put((ticket(r), error, 0.1))
    return then


def test_a_lost_server_holds_the_arms_before_the_run_ends(monkeypatch):
    io = FakeIo()
    r = make_runner(io)
    with pytest.raises(ConnectionError):
        drive(monkeypatch, r, 5, in_flight(r, lost_on(r, ConnectionError("gone"))))
    assert io.holds == [rn.HOLD_TIMEOUT_S]


@pytest.mark.parametrize("state", ["no chunk", "rows done", "paused"])
def test_a_lost_server_leaves_arms_it_does_not_move_alone(monkeypatch, state):
    """/arm/hold would cancel a planner move (Ready, Park) started meanwhile."""
    io = FakeIo()
    r = make_runner(io)
    lost = lost_on(r, ConnectionError("gone"), at=3)
    if state == "no chunk":
        hook = lost
    else:
        def then(turn):
            if turn == 2:
                if state == "rows done":
                    rn.time.sleep(r.last_due - rn.time.time())
                else:
                    io.requests.put("pause")
            lost(turn)
        hook = in_flight(r, then)
    with pytest.raises(ConnectionError):
        drive(monkeypatch, r, 6, hook)
    assert io.holds == []


def test_a_lost_server_ends_the_run_even_if_hold_fails(monkeypatch):
    io = FakeIo()
    io.hold_ok = False
    r = make_runner(io)
    with pytest.raises(RuntimeError, match="boom"):
        drive(monkeypatch, r, 5, in_flight(r, lost_on(r, RuntimeError("no policy server at http://test: boom"))))
    assert len(io.holds) == 1 and any("/arm/hold" in e for e in r.log.errors)


def test_an_overdue_reply_holds_the_arms_before_the_run_ends(monkeypatch):
    io = FakeIo()
    r = make_runner(io)

    def then(turn):
        if turn == 2:
            r.awaiting, r.requested_at = True, rn.time.monotonic() - rn.REPLY_TIMEOUT_S - 1.0
    with pytest.raises(RuntimeError, match="did not answer"):
        drive(monkeypatch, r, 5, in_flight(r, then))
    assert io.holds == [rn.HOLD_TIMEOUT_S]


def test_a_tripped_arm_is_not_held_again():
    io = FakeIo()
    r = make_runner(io)
    r.install(ticket(r), reply(actions()), 0.1)
    io.tripped = True  # received, but the loop has not paused yet
    r.hold()
    assert io.holds == []


def test_a_reply_from_before_a_pause_is_dropped(monkeypatch):
    io = FakeIo()
    r = make_runner(io)

    def hook(turn):
        if turn == 0:
            r.awaiting = True
            r.replies.put((ticket(r, generation=-1), reply(actions()), 0.1))
    drive(monkeypatch, r, 3, hook)
    assert io.chunks == {}


def test_requests_between_two_loop_turns_are_all_taken(monkeypatch):
    io = FakeIo()
    r = make_runner(io)

    def hook(turn):
        if turn == 1:
            io.requests.put("pause")
            io.requests.put("start")
    drive(monkeypatch, r, 3, hook)
    assert io.states[-2:] == ["paused: requested", "running"]


# --- the server's metadata --------------------------------------------------------------------------------------------
class Cam:
    pubs = {}

    def __init__(self, side, *a):
        Cam.pubs[side] = a[-1]


def construct(monkeypatch, meta):
    monkeypatch.setattr(rn, "time", time)  # __init__ names the session with time.strftime
    monkeypatch.setattr(rn, "Camera", Cam)
    monkeypatch.setattr(rn.Runner, "get", lambda self, path: meta)
    io = FakeIo()
    io.closing, io.get_logger, io.get_clock = False, Log, lambda: FakeClock(1_700_000_000.5)
    io.image_pub = {s: FakePub() for s in rn.SIDES}
    cam = {"dev": "/dev/null", "width": 4, "height": 4, "fps": 30}
    return io, rn.Runner({"server_host": "h", "server_port": 1, "tcp_offset_m": [0.088, 0.0, 0.0],
                          "cameras": {s: cam for s in rn.SIDES}}, io)


def test_a_server_with_another_layout_is_refused(monkeypatch):
    with pytest.raises(RuntimeError, match="11 states and 20 actions"):
        construct(monkeypatch, {**META, "state_dim": 9})


def test_camera_images_stop_once_the_node_is_closing(monkeypatch):
    io, _ = construct(monkeypatch, META)
    pubs = Cam.pubs
    frame = np.zeros((4, 4, 3), np.uint8)
    pubs["left"](frame)
    assert len(io.image_pub["left"].msgs) == 1
    io.closing = True
    pubs["left"](frame)
    assert len(io.image_pub["left"].msgs) == 1

    class GonePub:  # the node goes while the frame encodes, after the closing check
        def publish(self, _m):
            raise RuntimeError("InvalidHandle")
    io.image_pub["left"], io.closing = GonePub(), False
    real_encode = rn.cv2.imencode

    def encode(*a):
        io.closing = True
        return real_encode(*a)
    monkeypatch.setattr(rn.cv2, "imencode", encode)
    pubs["left"](frame)  # quiet: no InvalidHandle traceback from the camera thread
    monkeypatch.setattr(rn.cv2, "imencode", real_encode)
    io.closing = False
    with pytest.raises(RuntimeError):  # not closing: a failure still shows
        pubs["left"](frame)


# --- start and pause --------------------------------------------------------------------------------------------------
class RequestIo:
    def __init__(self, tripped):
        self.tripped = tripped
        self.requests = queue.SimpleQueue()


def test_start_while_tripped_is_refused_in_the_reply():
    io = RequestIo(tripped=True)
    res = rn.Io._request(io, "start", Trigger.Response())
    assert not res.success and "tripped" in res.message
    assert io.requests.empty()
    res = rn.Io._request(io, "pause", Trigger.Response())
    assert res.success and io.requests.get_nowait() == "pause"


def test_start_when_not_tripped_is_queued():
    io = RequestIo(tripped=False)
    assert rn.Io._request(io, "start", Trigger.Response()).success
    assert io.requests.get_nowait() == "start"


def test_io_hold_calls_the_arm_hold_service(monkeypatch):
    """A real rclpy graph, on this host and a random domain: Io.hold answers True from a /arm/hold server, and False
    within its timeout without one. Real time: the waits are rclpy's."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    monkeypatch.setattr(rn, "time", time)
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    taken = {88, int(os.environ.get("ROS_DOMAIN_ID") or 0)}  # never the robot's: motion's real /arm/hold
    monkeypatch.setenv("ROS_DOMAIN_ID", str(random.choice([d for d in range(1, 102) if d not in taken])))
    rclpy.init()
    ex, nodes = SingleThreadedExecutor(), []
    try:
        io = rn.Io()
        nodes.append(io)
        ex.add_node(io)
        threading.Thread(target=ex.spin, daemon=True).start()
        t0 = time.monotonic()
        assert not io.hold(0.3)
        assert time.monotonic() - t0 < 1.0
        calls = []

        def hold(_q, r):
            calls.append(1)
            r.success = True
            return r
        motion = rclpy.create_node("fake_motion")
        nodes.append(motion)
        motion.create_service(Trigger, "/arm/hold", hold)
        ex.add_node(motion)
        assert io.hold(5.0) and calls == [1]
    finally:
        ex.shutdown()
        for n in nodes:
            n.destroy_node()
        rclpy.try_shutdown()


# --- prewarm ----------------------------------------------------------------------------------------------------------
class Reply:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return self.body


def run_prewarm(monkeypatch, answer_on, timeout=25.0):
    from oap_runner import prewarm
    clock = FakeTime()
    monkeypatch.setattr(prewarm, "time", clock)
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        if answer_on is None or len(calls) < answer_on:
            raise prewarm.urllib.error.URLError("refused")
        return Reply(json.dumps(META).encode())
    monkeypatch.setattr(prewarm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(prewarm.sys, "argv", ["prewarm", "--port", "8123", "--timeout", str(timeout)])
    return prewarm.main(), calls


def test_prewarm_reports_while_waiting_and_on_giving_up(monkeypatch, capsys):
    code, _ = run_prewarm(monkeypatch, None)
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert code == 1
    assert lines[0].startswith("[prewarm] waiting for the policy server at http://localhost:8123/metadata")
    assert [ln.split("(")[1] for ln in lines[1:]] == ["10 s)", "20 s)"]
    assert "gave up: http://localhost:8123/metadata did not answer within 25 s" in err


def test_prewarm_stops_waiting_once_the_server_answers(monkeypatch, capsys):
    code, calls = run_prewarm(monkeypatch, 3)
    assert code == 0 and len(calls) == 3
    out = capsys.readouterr().out
    assert "still waiting" not in out and "ready after" in out


# --- leave(): Ctrl-C and SIGTERM stop the arms before the runner leaves the graph -------------------------------------
class Exited(Exception):
    pass


@pytest.fixture
def leaving(monkeypatch):
    """leave() with the process, signals, watchdog and rclpy faked; the order of what it did."""
    order = []

    class Timer:
        def __init__(self, t, fn):
            order.append(("watchdog", t))
            self.daemon = False

        def start(self):
            pass

    class Ex:
        def shutdown(self, timeout_sec=None):
            order.append("executor")

    class Spinner:
        def join(self, t):
            pass

    class Context:
        def destroy(self):
            order.append("context")

    def exit_(code):
        raise Exited(code)
    monkeypatch.setattr(rn.os, "_exit", exit_)
    monkeypatch.setattr(rn.signal, "signal", lambda sig, h: order.append(("signal", sig, h)))
    monkeypatch.setattr(rn.threading, "Timer", Timer)
    monkeypatch.setattr(rn.rclpy, "ok", lambda: True)
    monkeypatch.setattr(rn.rclpy, "get_default_context", Context)
    monkeypatch.setattr(rn.rclpy, "try_shutdown", lambda: order.append("rclpy"))

    def leave(io, code, runner):
        io.destroy_node = lambda: order.append("node")
        io.hold = lambda t: (order.append(("hold", t)), True)[1]
        with pytest.raises(Exited) as e:
            rn.leave(io, Ex(), Spinner(), code, runner)
        return e.value.args[0], order
    return leave


def running_with_rows(io):
    r = make_runner(io)
    r.install(ticket(r), reply(actions()), 0.1)
    return r


def test_ctrl_c_while_rows_play_holds_the_arms_first(leaving):
    io = FakeIo()
    r = running_with_rows(io)
    code, order = leaving(io, 0, r)
    assert code == 0
    for sig in (rn.signal.SIGINT, rn.signal.SIGTERM):  # a second Ctrl-C does not cut the hold short
        assert ("signal", sig, rn.signal.SIG_IGN) in order
    steps = [o for o in order if not (isinstance(o, tuple) and o[0] == "signal")]
    assert steps == [("watchdog", rn.HOLD_TIMEOUT_S + rn.EXIT_TIMEOUT_S), ("hold", rn.HOLD_TIMEOUT_S), "executor",
                     "node", "rclpy", "context"]


@pytest.mark.parametrize("state", ["paused", "tripped", "no rows left", "held already", "no runner"])
def test_leave_does_not_hold_arms_it_does_not_move(leaving, clock, state):
    io = FakeIo()
    r = running_with_rows(io)
    if state == "paused":
        r.paused = True
    elif state == "tripped":
        io.tripped = True
    elif state == "no rows left":
        clock.sleep(r.last_due - clock.time())
    elif state == "held already":
        r.held = True
    code, order = leaving(io, 1, None if state == "no runner" else r)
    assert code == 1 and not any(isinstance(o, tuple) and o[0] == "hold" for o in order)
    assert order[-4:] == ["executor", "node", "rclpy", "context"]


def test_leave_after_the_run_ended_by_itself_does_not_hold(leaving, monkeypatch):
    io = FakeIo()
    r = running_with_rows(io)
    monkeypatch.setattr(rn.rclpy, "ok", lambda: False)  # the context is gone: nothing could answer
    _, order = leaving(io, 0, r)
    assert not any(isinstance(o, tuple) and o[0] == "hold" for o in order)
