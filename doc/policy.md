# Policy server protocol

What `oap_runner` sends to and expects from the policy server (`policy/airoa0_inference/serve_policy.py`), as
`oap_runner/runner.py` and `prewarm.py` use it. Checkpoint layout: [README](../README.md#policy).

Transport: HTTP, `http://<server_host>:<server_port>` (`runner.yaml`, default `localhost:8000`). A reply other than 200
carries `{"error": ...}`: the runner pauses on it. A server that cannot be reached, or a reply later than
`REPLY_TIMEOUT_S`, ends the run.

## `GET /metadata`

| Key | Use |
|---|---|
| `state_dim`, `action_dim` | must be 11 and 20 |
| `action_hz` | rows per second of a reply |
| `task`, `pa_text`, `stage_pa_texts`, `chunk_steps`, `history_slots`, `history_interval_s` | logged |

## `POST /infer`

The body is a `numpy.savez` archive:

| Key | Type | What |
|---|---|---|
| `left_wrist_0_rgb`, `right_wrist_0_rgb` | uint8 480×640×3 | the latest wrist camera frames, RGB |
| `state` | float32 [11] | the right gripper tip in the left tip's frame: xyz [m], then the 6D rotation (the rotation matrix's first two columns); then the left and right finger openings [rad], measured |
| `timestamp` | float | the observation's monotonic time [s]: the read time minus `camera_latency_s` |
| `session` | str | one per runner process |
| `reset` | bool | true on the first request after a start or a resume: the server drops the session's state history |

A tip is `tcp_offset_m` (0.088 m) along `<side>_hand_root`'s x; the hand poses are the measured ones at the
observation (`/arm/<side>/hand_pose`).

The reply is JSON:

| Key | Type | What |
|---|---|---|
| `actions` | [n][20] | per row: the left tip's target pose relative to the left tip at the observation (xyz, 6D rotation), the same for the right, then the left and right finger openings [rad] |
| `action_hz` | float | rows per second |
| `history_slots_used`, `latency_s` | | logged |

The runner puts each row on the commanded hand at the observation, shifted by what the IK falls short of, and turns
the tip pose into a hand_root pose. It plays the first `execute_rows` rows (0: all), row k due (k + 1) / `action_hz`
after the reply arrives, sends each row's finger openings one row before the row is due, and captures again once the
last row is due. Finger openings run from 0 (closed) to about 0.785 (open).
