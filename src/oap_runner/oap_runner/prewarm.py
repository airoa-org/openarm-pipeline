#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Wait until the policy server answers GET /metadata: it compiles its first call before it listens, so then it is
ready for the runner.

    ros2 run oap_runner prewarm [--host localhost] [--port 8000] [--timeout 900]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PROGRESS_S = 10.0  # a line while waiting, so it does not look hung


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=900.0, help="[s]: a first start builds a venv, then compiles")
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}/metadata"
    print(f"[prewarm] waiting for the policy server at {url} (up to {args.timeout:.0f} s)", flush=True)
    t0 = shown = time.monotonic()
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as r:
                meta = json.load(r)
            print(f"[prewarm] ready after {time.monotonic() - t0:.0f} s: {meta}", flush=True)
            return 0
        except (urllib.error.URLError, OSError, ValueError):
            now = time.monotonic()
            if now - t0 > args.timeout:
                print(f"[prewarm] gave up: {url} did not answer within {args.timeout:.0f} s; is the policy server "
                      "running, on this host and port?", file=sys.stderr, flush=True)
                return 1
            if now - shown >= PROGRESS_S:
                shown = now
                print(f"[prewarm] still waiting ({now - t0:.0f} s)", flush=True)
            time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
