# AIRoA-0 reference inference

A minimal policy server for the released AIRoA-0 post-trained checkpoints, for
example `examples/orange_pyramid/` from the Hugging Face repository.

AIRoA-0 itself is post-training only and is used unmodified, as a library.

## Install

Install AIRoA-0 (tested at commit `ab9c7ec`) as described in its README, then
put this folder on the Python path:

```sh
git clone https://github.com/airoa-org/AIRoA-0 && cd AIRoA-0 && git checkout ab9c7ec
python -m pip install -e '.[train,test]'
export PYTHONPATH=/path/to/this/folder:$PYTHONPATH
```

The code imports from AIRoA-0 only: `openpi.models` (`flash`, `model`,
`pi0_config`, `pi0`), `openpi.transforms`, `openpi.shared` (`normalize`,
`proprio_frequency`) and `post_training` (`data`, `language`). It calls private
`Pi0` methods, so pin the AIRoA-0 commit. The PaliGemma tokenizer is
downloaded on first use, as in AIRoA-0.

## Run

```sh
python -m airoa0_inference.serve_policy --checkpoint path/to/examples/orange_pyramid --port 8000
```

The PA is the checkpoint's own PA for `--stage` (default 0), decoded from its
`model_config.json`; `--pa-text` sets any other text instead. The server logs
every stage's PA at start and lists them in `GET /metadata` (`stage_pa_texts`).

## Request and response

`POST /infer` takes an `.npz` body (`numpy.savez`) with these keys:

- `left_wrist_0_rgb`, `right_wrist_0_rgb`: uint8 RGB `[H, W, 3]`. The
  checkpoints were trained on 640x480 frames.
- `state`: the 11-dim state (AIRoA-0 README, "State and action layout").
- `timestamp` (optional): capture time in seconds. Default: server clock.
- `session` (optional): history buffer name. Default `"default"`.
- `reset` (optional): clear the session's history first.
- `seed` (optional): noise seed. Default 0.

The reply is JSON `{"actions": [8][20], "action_hz": 20, "history_slots_used": n, "latency_s": t}`.
A malformed request gets HTTP 400 with `{"error": ...}`. `POST /reset` (body:
session name) clears one history; `GET /metadata` returns the task, the PA in use
(`pa_text`), the checkpoint's PA per stage (`stage_pa_texts`) and shapes.

```python
import io, json, urllib.request, numpy as np
buf = io.BytesIO()
np.savez(buf, left_wrist_0_rgb=left, right_wrist_0_rgb=right, state=state, timestamp=t, session="ep1")
reply = json.load(urllib.request.urlopen("http://HOST:8000/infer", buf.getvalue()))
actions = np.asarray(reply["actions"])  # [8, 20], 20 Hz
```

## Notes

- **Actions are TCP targets.** Per arm: the target TCP pose relative to that
  arm's TCP pose at the observation (xyz in meters, 6D rotation), then the two
  gripper openings. Unnormalized, at 20 Hz.
- **First 8 actions only.** The model predicts 16 steps as two FLASH chunks of
  8. Each request encodes the observation once and refines the two-chunk buffer
  in two rounds with the trained clock (levels `[0.1, 1.0]`, substeps
  `[[0.1, 0.0], [1.0, 0.7, 0.1]]`). Only the fully denoised first chunk is
  returned.
- **Fixed PA, no stage.** The prompt is the checkpoint's task instruction plus
  one fixed PA, encoded as in training (`" <PA>\nAction: "`). The stage head is
  not used, so the PA does not advance by itself.
- **Proprio history.** The model takes 10 past states, newest first, on a 0.1 s
  grid. Per session the server keeps the states of earlier requests. For each
  grid point it takes the latest earlier state at most 0.1 s older, each state
  once, with its true age. Other slots are masked, as with history dropout in
  training. Clients slower than 10 Hz get fewer slots. Send `reset` at the start
  of each episode.

## Files

- `airoa0_inference/flash_sampler.py`: FLASH cold-start sampling on an AIRoA-0
  `Pi0` model.
- `airoa0_inference/policy.py`: checkpoint loading, input transforms and
  unnormalization.
- `airoa0_inference/serve_policy.py`: HTTP server and proprio-history buffer.
- `tests/`: tiny random model on CPU: `JAX_PLATFORMS=cpu python -m pytest tests`.
