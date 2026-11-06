# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Minimal HTTP policy server for released AIRoA-0 post-trained checkpoints.

    python -m airoa0_inference.serve_policy --checkpoint examples/orange_pyramid

POST /infer  .npz body -> JSON {"actions": [8][20], ...}; POST /reset; GET /metadata.
See README.md for the request/response format and the proprio-history rule.
"""

import argparse
import collections
import http.server
import io
import json
import logging
import time

import numpy as np

from openpi.shared.proprio_frequency import resampling_plan

from airoa0_inference.policy import VIEWS, Policy, stage_pa_texts

HISTORY_INTERVAL_S = 0.1
ACTION_HZ = 20.0


class History:
    """Per-session (timestamp, state) pairs from earlier requests."""

    def __init__(self, slots, state_dim, keep=64):
        self.slots, self.state_dim = slots, state_dim
        self.sessions = collections.defaultdict(lambda: collections.deque(maxlen=keep))

    def reset(self, session):
        self.sessions.pop(session, None)

    def window(self, session, now):
        """(history [slots, state_dim], mask [slots], age_s [slots]), newest first:
        per 0.1 s grid point the nearest earlier state at most 0.1 s older."""
        past = self.sessions[session]
        if past and now <= past[-1][0]:
            past.clear()  # clock went backwards: start a new history
        history = np.zeros((self.slots, self.state_dim), np.float32)
        mask = np.zeros(self.slots, bool)
        ages = np.zeros(self.slots, np.float32)
        plan = resampling_plan(
            [t for t, _ in past], now, interval=HISTORY_INTERVAL_S,
            slots=self.slots, count=self.slots, interpolate=False,
        )
        for slot, entry in enumerate(plan):
            if entry is not None:
                history[slot], mask[slot], ages[slot] = past[entry[0]][1], True, entry[3]
        return history, mask, ages

    def add(self, session, now, state):
        self.sessions[session].append((now, np.asarray(state, np.float32)))


def handle_infer(policy, history, body):
    request = np.load(io.BytesIO(body), allow_pickle=False)
    session = str(request["session"]) if "session" in request else "default"
    now = float(request["timestamp"]) if "timestamp" in request else time.monotonic()
    if "reset" in request and bool(request["reset"]):
        history.reset(session)
    state = np.asarray(request["state"], np.float32)
    images = {view: request[view] for view in VIEWS}
    seed = int(request["seed"]) if "seed" in request else 0
    start = time.perf_counter()
    window = history.window(session, now)
    actions = policy.infer(images, state, *window, seed=seed)
    latency = time.perf_counter() - start
    if not np.isfinite(actions).all():
        raise ValueError("non-finite actions")
    history.add(session, now, state)  # only after a successful inference
    return {
        "actions": actions.tolist(), "action_hz": ACTION_HZ,
        "history_slots_used": int(window[1].sum()), "latency_s": latency,
    }


def make_handler(policy, history, metadata):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def do_GET(self):
            if self.path == "/metadata":
                return self._reply(200, metadata)
            self._reply(404, {"error": "unknown path"})

        def do_POST(self):
            try:
                if self.path == "/reset":
                    history.reset(self._body().decode() or "default")
                    return self._reply(200, {"ok": True})
                if self.path != "/infer":
                    return self._reply(404, {"error": "unknown path"})
                self._reply(200, handle_infer(policy, history, self._body()))
            except Exception as error:  # report, keep serving
                logging.exception("request failed")
                self._reply(400, {"error": f"{type(error).__name__}: {error}"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="e.g. examples/orange_pyramid")
    parser.add_argument("--pa-text", default=None,
                        help="fixed PA (subtask) text; default: the checkpoint's own PA for --stage")
    parser.add_argument("--stage", type=int, default=0, help="the checkpoint's stage whose PA is used (default 0)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    policy = Policy.from_checkpoint(args.checkpoint)
    stage_texts = stage_pa_texts(policy.config)
    if args.pa_text is None and not 0 <= args.stage < len(stage_texts):
        parser.error(f"--stage must be in 0..{len(stage_texts) - 1} for this checkpoint")
    if args.pa_text is None:
        args.pa_text = stage_texts[args.stage]
    policy.set_pa(args.pa_text)
    history = History(policy.history_slots, policy.state_dim)
    metadata = {
        "task": policy.task, "pa_text": args.pa_text, "stage_pa_texts": stage_texts,
        "state_dim": policy.state_dim, "action_dim": policy.action_dim,
        "chunk_steps": policy.config.flash_chunk_size, "action_hz": ACTION_HZ,
        "history_slots": policy.history_slots, "history_interval_s": HISTORY_INTERVAL_S,
    }
    logging.info("Compiling (first call)...")
    blank = np.zeros((480, 640, 3), np.uint8)
    warmup = History(policy.history_slots, policy.state_dim).window("warmup", 0.0)
    policy.infer({view: blank for view in VIEWS}, np.zeros(policy.state_dim), *warmup)
    server = http.server.HTTPServer((args.host, args.port), make_handler(policy, history, metadata))
    logging.info("Serving %s on http://%s:%d", metadata, args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
