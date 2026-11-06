# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""oap_planner without a robot: the target poses, PlanChecker's table and pillar clearance against the installed URDF
and the gripper ellipsoid, the trajectory optimizer's separation terms, and circle's phase profile. Needs a built and
sourced workspace (colcon build; source install/setup.bash)."""
import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
ament = pytest.importorskip("ament_index_python.packages")
try:
    ament.get_package_share_directory("oap_planner")
    ament.get_package_share_directory("oap_description")
except ament.PackageNotFoundError:
    pytest.skip("oap_planner/oap_description are not installed: colcon build and source install/setup.bash",
                allow_module_level=True)

from oap_planner import geometry as geo  # noqa: E402


def within(q, limits, tol=1e-9):
    return all(lo - tol <= v <= hi + tol for v, (lo, hi) in zip(q, limits))


@pytest.fixture(scope="module")
def checker():
    return geo.PlanChecker()


# --- poses and limits ---------------------------------------------------------------------------------------------
def test_ready_pose_is_within_limits():
    for s in geo.SIDES:
        assert len(geo.READY_POSE[s]) == 7 and within(geo.READY_POSE[s], geo.LIMITS[s])


# --- PlanChecker ----------------------------------------------------------------------------------------------------
def test_checker_loads_both_arms_with_the_gripper_ellipsoid(checker):
    for s in geo.SIDES:
        assert any(name.endswith("_link7") for name in checker.geo[s])
        assert f"{s}_hand_root" in checker.geo[s]


@pytest.mark.parametrize("side", geo.SIDES)
def test_ready_pose_is_clear(checker, side):
    table, pillar = checker.clearance(side, geo.READY_POSE[side])
    assert table >= geo.TABLE_MARGIN_M, f"table {table * 1000:.1f} mm"
    assert pillar >= geo.SAFETY_MARGIN_M, f"pillar {pillar * 1000:.1f} mm"


@pytest.mark.parametrize("side", geo.SIDES)
def test_table_box_raised_over_the_arm_collides(checker, side, monkeypatch):
    clear, _ = checker.clearance(side, geo.READY_POSE[side])
    monkeypatch.setattr(geo, "TABLE_TOP", 5.0)
    monkeypatch.setattr(geo, "TABLE_X", (-5.0, 5.0))
    monkeypatch.setattr(geo, "TABLE_Y", (-5.0, 5.0))
    inside, _ = checker.clearance(side, geo.READY_POSE[side])
    assert inside < 0.0 < clear


@pytest.mark.parametrize("side", geo.SIDES)
def test_hand_reaching_down_hits_the_table(checker, side):
    table, _ = checker.clearance(side, DOWN_POSE[side])
    assert table < 0.0, f"table {table * 1000:.1f} mm"


@pytest.mark.parametrize("side", geo.SIDES)
def test_hand_swung_inward_hits_the_pillar(checker, side):
    _, pillar = checker.clearance(side, PILLAR_POSE[side])
    assert pillar < 0.0, f"pillar {pillar * 1000:.1f} mm"


def test_check_over_a_motion_is_the_worst_sample(checker):
    side = "left"
    start, down = np.array(geo.READY_POSE[side]), np.array(DOWN_POSE[side])
    assert checker.check(side, start, [start]) == pytest.approx(checker.clearance(side, start))
    samples = [checker.clearance(side, start + u * (down - start)) for u in np.linspace(0, 1, geo.CHECK_STEPS)]
    expected = (min(t for t, _ in samples), min(p for _, p in samples))
    assert checker.check(side, start, [down]) == pytest.approx(expected)
    assert expected[0] < 0.0  # the motion ends in the table


# --- plan.py --------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plan():
    for mod in ("casadi", "pinocchio.casadi", "rclpy"):
        pytest.importorskip(mod)
    from oap_planner import plan
    return plan


def test_targets_are_within_limits(plan):
    assert {"ready", "park", "lifted"} <= set(plan.TARGETS)
    for name, target in plan.TARGETS.items():
        for s in geo.SIDES:
            assert len(target[s]) == 7 and within(target[s], geo.LIMITS[s]), (name, s)


def test_lifted_target_is_clear(plan, checker):
    for s in geo.SIDES:
        table, pillar = checker.clearance(s, plan.TARGETS["lifted"][s])
        assert table >= geo.TABLE_MARGIN_M and pillar >= geo.SAFETY_MARGIN_M, (s, table, pillar)


def test_arm_model_has_the_fourteen_arm_joints(plan):
    model, idx = plan.arm_model()
    assert model.nq == 14
    assert sorted(i for s in geo.SIDES for i in idx[s]) == list(range(14))


def test_table_separation_sign(plan):
    import casadi as ca
    lo, hi, semi, R = (-0.5, -0.5, -1.0), (0.5, 0.5, 0.0), [0.05, 0.05, 0.05], ca.DM.eye(3)
    sep = lambda c: float(plan.table_separation(ca.DM(c), R, semi, lo, hi, 200.0))
    assert sep([0.0, 0.0, 0.2]) == pytest.approx(0.15, abs=0.01)  # 20 cm above the top, 5 cm semi-axis
    assert sep([0.0, 0.0, -0.2]) < 0.0  # inside the box
    assert sep([0.0, 0.0, 0.02]) < 0.0  # resting on the top


def test_pillar_separation_sign(plan):
    import casadi as ca
    semi, R = [0.05, 0.05, 0.05], ca.DM.eye(3)
    sep = lambda c: float(plan.pillar_separation(ca.DM(c), R, semi, [0.0, 0.0], 0.1))
    assert sep([0.3, 0.0, 0.5]) == pytest.approx(0.15)
    assert sep([0.0, 0.12, 0.5]) < 0.0


# --- circle.py ------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def circle():
    for mod in ("casadi", "pinocchio.casadi", "rclpy"):  # circle imports plan
        pytest.importorskip(mod)
    from oap_planner import circle
    return circle


def test_phase_runs_the_turns_with_a_smooth_ramp(circle):
    ph = circle.Phase(period=4.0, turns=2, ramp=1.0)
    ts = np.linspace(0.0, ph.end, 2001)
    th = np.array([ph(t) for t in ts])
    assert th[0] == pytest.approx(0.0) and th[-1] == pytest.approx(4 * math.pi)
    assert np.all(np.diff(th) >= -1e-12)
    rate = np.diff(th) / np.diff(ts)
    assert rate.max() == pytest.approx(2 * math.pi / 4.0, rel=1e-3)
    assert rate[0] < 0.01 and rate[-1] < 0.01  # ramps in and out
    dt = ts[1] - ts[0]
    assert np.abs(np.diff(rate)).max() <= 1.01 * ph.w * math.pi / 2 * dt  # no jump: at most the ramp's peak θ̈
    assert not ph.finished(ph.end - 1e-3) and ph.finished(ph.end)


def test_phase_stop_ramps_out_within_one_ramp(circle):
    ph = circle.Phase(period=4.0, turns=3, ramp=1.0)
    ph.stop(3.0)
    th = np.array([ph(t) for t in np.linspace(3.0, 4.0, 1001)])
    assert np.all(np.diff(th) >= -1e-12)
    assert th[-1] - th[0] == pytest.approx(0.5 * ph.w * 1.0)  # mean speed w/2 over the ramp
    assert ph(10.0) == pytest.approx(th[-1])
    assert ph.finished(4.0) and not ph.finished(3.9)


def test_phase_refuses_a_ramp_longer_than_the_turns(circle):
    with pytest.raises(ValueError):
        circle.Phase(period=1.0, turns=0.5, ramp=1.0)


# Configurations within the joint limits that collide, calibrated against PlanChecker: from park, the elbow bent 0.8 rad
# puts the hand about 230 mm into the table; joint 2 at its inner limit with joints 3 and 5 turned puts the hand about
# 30 mm into the pillar while it stays above the table.
DOWN_POSE = {"left": [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0]}
PILLAR_POSE = {"left": [0.0, 0.17, 1.5, 0.5, -1.5, 0.0, 0.0], "right": [0.0, -0.17, -1.5, 0.5, 1.5, 0.0, 0.0]}
