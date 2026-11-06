# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the hardware-free logic. The full suite runs in the ROS 2 Humble image after `colcon build` and
`source install/setup.bash`; elsewhere the modules that need rclpy, pinocchio or an installed workspace are skipped.

    pip3 install pytest==8.3.5 pytest-timeout==2.3.1   # newer pytest breaks ROS Humble's launch_testing plugin
    python3 -m pytest --timeout=120 tests/
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Without a sourced workspace, import the pure modules (e.g. flash) from the source tree. Appended, so an
# installed workspace wins.
for pkg in ("oap_runner", "oap_planner"):
    path = os.path.join(ROOT, "src", pkg)
    if path not in sys.path:
        sys.path.append(path)


def pytest_configure(config):
    config.addinivalue_line("markers", "ci_skip_ok: a skip of this test is intended, even under OAP_TESTS_NO_SKIP=1")


def pytest_sessionfinish(session, exitstatus):
    """OAP_TESTS_NO_SKIP=1 (scripts/oap test): a skip means a dependency went missing, so it fails the run. A test that
    is meant to skip there carries @pytest.mark.ci_skip_ok. Module-level skips (importorskip) carry no marker, so they
    always count."""
    if os.environ.get("OAP_TESTS_NO_SKIP") != "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None or session.exitstatus != 0:
        return
    bad = [r for r in reporter.stats.get("skipped", []) if "ci_skip_ok" not in getattr(r, "keywords", {})]
    if bad:
        reporter.write_line(f"OAP_TESTS_NO_SKIP=1: {len(bad)} test(s) skipped without @pytest.mark.ci_skip_ok:",
                            red=True)
        for r in bad:
            reporter.write_line(f"  {r.nodeid}", red=True)
        session.exitstatus = 1
