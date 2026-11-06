# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Load a released AIRoA-0 checkpoint and map one observation to actions."""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.shared import normalize
from post_training.data import ResizeWithMask, rgb_uint8
from post_training.language import StagePrefix, stage_tokens, tokenizer_for

from airoa0_inference import flash_sampler

VIEWS = ("left_wrist_0_rgb", "right_wrist_0_rgb")


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def pa_token_ids(config, *, pa_text=None, stage=0):
    """The checkpoint's encoded PA row for `stage`, or `pa_text` encoded the
    same way (" <PA>\\nAction: ")."""
    rows = np.asarray(config.stage_pa_token_ids, np.int32)
    if pa_text is None:
        return rows[stage]
    width = rows.shape[1]
    return np.asarray(stage_tokens(tokenizer_for(width), [pa_text], width)[0], np.int32)


def stage_pa_texts(config, sp=None):
    """The checkpoint's PA texts, one per stage, decoded from `stage_pa_token_ids`
    (each row encodes " <PA>\\nAction: "). A text that does not re-encode to its
    row raises, so a listed text always reproduces the trained PA."""
    rows = np.asarray(config.stage_pa_token_ids, np.int32)
    sp = sp or tokenizer_for(rows.shape[1])._tokenizer
    texts = []
    for stage, row in enumerate(rows):
        ids = [int(t) for t in row if t != 0]
        text = sp.decode(ids).rsplit("\nAction:", 1)[0].strip()
        if sp.encode(" " + text + "\nAction: ") != ids:
            raise ValueError(f"stage {stage}: the checkpoint's PA row does not decode to a text")
        texts.append(text)
    return texts


class Policy:
    """Two wrist RGB frames + 11-dim state + proprio history -> the first FLASH
    chunk of unnormalized actions ([8, 20] at 20 Hz for the released checkpoints)."""

    def __init__(self, model, config, norm_stats, *, task, pa_tokens):
        self.model, self.config, self.task = model, config, task
        self.history_slots = len(config.temporal_proprio_history_lags)
        self.state_dim = len(norm_stats["state"].q01)
        self.action_dim = len(norm_stats["actions"].q01)
        self.pa_tokens = jnp.asarray(np.asarray(pa_tokens, np.int32)[None])
        self._inputs = [
            transforms.Normalize(norm_stats, use_quantiles=True),
            ResizeWithMask(),
            StagePrefix(tokenizer_for(config.max_token_len)),
            transforms.PadStatesAndActions(config.action_dim),
        ]
        self._unnormalize = transforms.Unnormalize(
            {"actions": norm_stats["actions"]}, use_quantiles=True
        )
        self.sample = flash_sampler.jit_sampler(model)

    def set_pa(self, pa_text):
        """Condition every later request on `pa_text`, encoded as the checkpoint's PA rows."""
        self.pa_tokens = jnp.asarray(pa_token_ids(self.config, pa_text=pa_text)[None])

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, *, pa_text=None, stage=0):
        """Load `params/` (Orbax, cast to bfloat16), `model_config.json` and
        `norm_stats.json`, as published under `examples/<task>/`."""
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        meta = json.loads((checkpoint_dir / "model_config.json").read_text())
        config = pi0_config.Pi0Config(**{k: _tuples(v) for k, v in meta["model_fields"].items()})
        params = model_lib.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        (graphdef, state) = nnx.split(nnx.eval_shape(config.create, jax.random.key(0)))
        state.replace_by_pure_dict(params)
        return cls(
            nnx.merge(graphdef, state),
            config,
            normalize.load(checkpoint_dir),
            task=meta["task_instruction"],
            pa_tokens=pa_token_ids(config, pa_text=pa_text, stage=stage),
        )

    def observation(self, images, state, state_history, state_history_mask, state_history_age):
        """The training input transforms, applied to one observation; batch of 1."""
        data = {
            "image": {view: rgb_uint8(images[view]) for view in VIEWS},
            "image_mask": {view: np.bool_(True) for view in VIEWS},
            "state": np.asarray(state, np.float32),
            "state_history": np.asarray(state_history, np.float32),
            "state_history_mask": np.asarray(state_history_mask, np.bool_),
            "state_history_age": np.asarray(state_history_age, np.float32),
            "prompt": self.task,
        }
        if data["state"].shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},)")
        if data["state_history"].shape != (self.history_slots, self.state_dim):
            raise ValueError("state_history must be [slots, state_dim], newest first")
        for transform in self._inputs:
            data = transform(data)
        batched = jax.tree.map(lambda v: np.asarray(v)[None], data)
        return jax.tree.map(jnp.asarray, model_lib.Observation.from_dict(batched))

    def infer(self, images, state, state_history, state_history_mask, state_history_age, *, seed=0):
        observation = self.observation(
            images, state, state_history, state_history_mask, state_history_age
        )
        normalized = self.sample(jax.random.key(seed), observation, self.pa_tokens)
        normalized = np.asarray(jax.device_get(normalized), np.float32)[0]
        actions = self._unnormalize({"actions": normalized})["actions"]
        return actions[:, : self.action_dim]
