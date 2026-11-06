#!/usr/bin/env bash
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# The AIRoA-0 policy server (policy/) on one released checkpoint, e.g. examples/<task>/ from Hugging Face.
# In the container: ./scripts/serve_policy.sh /checkpoints/<task> [--port 8000] [--stage N | --pa-text "<sub-task>"]
# The first start clones AIRoA-0 at AIRoA0_REF into .cache/AIRoA-0 (or uses AIRoA0_DIR) and builds its venv (minutes).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
[ $# -ge 1 ] || { echo "usage: $0 /checkpoints/<task> [--port 8000] [--stage N | --pa-text TEXT]" >&2; exit 2; }
[ -f "$1/model_config.json" ] || { echo "no released checkpoint at $1 (model_config.json, params/, norm_stats.json)" >&2; exit 1; }
checkpoint=$(cd "$1" && pwd); shift
AIRoA0_REF=${AIRoA0_REF:-ab9c7ec}
airoa0=${AIRoA0_DIR:-$root/.cache/AIRoA-0}
if [ ! -d "$airoa0/.git" ]; then
  echo "[server] cloning AIRoA-0 into $airoa0"
  git clone https://github.com/airoa-org/AIRoA-0 "$airoa0"
fi
git -C "$airoa0" checkout -q "$AIRoA0_REF"
export UV_CACHE_DIR="$root/.cache/uv" UV_PROJECT_ENVIRONMENT="$root/.cache/venvs/airoa0-$AIRoA0_REF"
echo "[server] AIRoA-0 $AIRoA0_REF: venv $UV_PROJECT_ENVIRONMENT"
(cd "$airoa0" && uv sync --frozen --python 3.11 --extra train)
export PYTHONPATH="$root/policy${PYTHONPATH:+:$PYTHONPATH}"
exec "$UV_PROJECT_ENVIRONMENT/bin/python" -m airoa0_inference.serve_policy --checkpoint "$checkpoint" "$@"
