# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""FLASH sampler, policy and server helpers on a tiny random AIRoA-0 model (CPU)."""

import dataclasses
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.shared.normalize import NormStats

from airoa0_inference import flash_sampler
from airoa0_inference.policy import VIEWS, Policy, stage_pa_texts
from airoa0_inference.serve_policy import History, handle_infer

# C2/N2 tiny analogue of the released C8/N2 clock.
LEVELS = (0.1, 1.0)
SUBSTEPS = ((0.1, 0.0), (1.0, 0.7, 0.1))


def tiny_config():
    return Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        siglip_variant="mu/14",
        dtype="float32",
        action_horizon=4,
        max_token_len=64,
        base_view_names=(),
        wrist_view_names=VIEWS,
        visual_budget_gate_hidden_dim=8,
        visual_budget_gate_tau_start=0.805,
        visual_budget_gate_tau_end=0.805,
        visual_budget_mpa_tau=0.805,
        visual_budget_s2_alpha_warmup_steps=500,
        use_temporal_proprio_memory=True,
        temporal_proprio_history_lags=tuple(range(1, 11)),
        temporal_hidden_dim=8,
        proprio_frequency_enabled=True,
        flash_enabled=True,
        flash_chunk_size=2,
        flash_buffer_size=2,
        flash_flow_times=LEVELS,
        flash_flow_substeps=SUBSTEPS,
        stage_num_classes=2,
        stage_pa_conditioning=True,
        stage_pa_token_ids=((5, 6, 7, 0), (8, 9, 0, 0)),
    )


def open_action_attention_gate(model):
    """Emulate a trained checkpoint: random AdaRMS attention gates start at zero."""
    flat = nnx.state(model, nnx.Param).flat_state()
    for path, value in flat.items():
        if tuple(path[-3:]) == ("pre_attention_norm_1", "Dense_0", "bias"):
            width = value.value.shape[-1] // 3
            flat[path] = value.replace(value.value.at[..., 2 * width :].set(0.5))
    nnx.update(model, nnx.State.from_flat_path(flat))


@pytest.fixture(scope="module")
def setup():
    config = tiny_config()
    model = config.create(jax.random.key(3))
    open_action_attention_gate(model)  # let the suffix see the prefix
    stats = {
        "state": NormStats(mean=np.zeros(11), std=np.ones(11), q01=-np.ones(11), q99=np.ones(11)),
        "actions": NormStats(
            mean=np.zeros(20), std=np.ones(20),
            q01=np.linspace(-2, 0, 20), q99=np.linspace(1, 3, 20),
        ),
    }
    return config, model, stats


@pytest.fixture(scope="module")
def policy(setup):
    config, model, stats = setup
    return Policy(model, config, stats, task="Stack cups.", pa_tokens=config.stage_pa_token_ids[0])


def request(rng, size=(48, 64)):
    images = {view: rng.integers(0, 256, (*size, 3), dtype=np.uint8) for view in VIEWS}
    return images, rng.uniform(-1, 1, 11)


def test_solver_phases_match_released_clock():
    phases = flash_sampler.solver_phases(SUBSTEPS, levels=LEVELS, buffer_size=2)
    assert phases == (((0.1, 1.0), (0.1, 0.7)), ((0.1, 0.7), (0.0, 0.1)))
    x = jnp.ones((1, 4, 3))
    out = flash_sampler.update_phase(
        x, jnp.ones_like(x), before=(0.1, 0.7), after=(0.0, 0.1), chunk_size=2
    )
    np.testing.assert_allclose(out[0, :, 0], [0.9, 0.9, 0.4, 0.4], rtol=1e-6)


def test_append_pa_matches_airoa0_stage_rows(setup):
    config, model, _ = setup
    ids = jnp.asarray([[3, 4, 5, 6, 7]])
    tokens = model.PaliGemma.llm(ids, method="embed")
    prefix_mask = jnp.ones(ids.shape, bool)
    (_, kv_cache) = model.PaliGemma.llm(
        [tokens, None],
        mask=make_attn_mask(prefix_mask, jnp.zeros(ids.shape, bool)),
        positions=jnp.cumsum(prefix_mask, axis=1) - 1,
    )
    for stage in range(2):
        expected = model._append_stage_pa(tokens, prefix_mask, kv_cache, jnp.asarray([stage]))
        ours = flash_sampler.append_pa(
            model, tokens, prefix_mask, kv_cache, jnp.asarray([config.stage_pa_token_ids[stage]])
        )
        jax.tree.map(np.testing.assert_array_equal, ours, expected)


def test_sample_actions_shape_and_determinism(setup):
    config, model, _ = setup
    obs = config.fake_obs()
    sample = flash_sampler.jit_sampler(model)
    (pa0, pa1) = (jnp.asarray([row]) for row in config.stage_pa_token_ids)
    first = sample(jax.random.key(0), obs, pa0)
    assert first.shape == (1, config.flash_chunk_size, config.action_dim)
    assert np.isfinite(np.asarray(first)).all()
    np.testing.assert_array_equal(first, sample(jax.random.key(0), obs, pa0))
    eager = flash_sampler.sample_actions(model, jax.random.key(0), obs, pa0)
    np.testing.assert_allclose(first, eager, rtol=1e-5, atol=1e-5)
    assert not np.array_equal(first, sample(jax.random.key(1), obs, pa0))
    assert not np.array_equal(first, sample(jax.random.key(0), obs, pa1))


def test_non_flash_checkpoints_are_rejected():
    config = dataclasses.replace(tiny_config(), flash_flow_substeps=None)
    model = config.create(jax.random.key(4))
    with pytest.raises(NotImplementedError):
        flash_sampler.sample_actions(
            model, jax.random.key(0), config.fake_obs(), jnp.ones((1, 4), jnp.int32)
        )


def test_policy_returns_first_chunk_unnormalized(setup, policy):
    config, _, stats = setup
    rng = np.random.default_rng(0)
    (images, state) = request(rng)
    history = rng.uniform(-1, 1, (10, 11)).astype(np.float32)
    mask = np.arange(10) < 3
    ages = np.where(mask, (np.arange(10) + 1) * 0.1, 0.0)
    args = (images, state, history, mask, ages)
    actions = policy.infer(*args, seed=7)
    assert actions.shape == (config.flash_chunk_size, 20)
    assert np.isfinite(actions).all()
    np.testing.assert_array_equal(actions, policy.infer(*args, seed=7))

    obs = policy.observation(*args)
    assert obs.tokenized_prompt_mask.sum() > 0 and obs.state.shape == (1, 32)
    normalized = np.asarray(policy.sample(jax.random.key(7), obs, policy.pa_tokens))[0, :, :20]
    (q01, q99) = (stats["actions"].q01, stats["actions"].q99)
    np.testing.assert_allclose(
        actions, (normalized + 1) / 2 * (q99 - q01 + 1e-6) + q01, rtol=1e-5, atol=1e-5
    )


def test_history_window_takes_latest_state_per_grid_point_once():
    history = History(slots=10, state_dim=11)
    for k, t in enumerate([0.0, 0.1, 0.2, 0.35]):
        history.add("s", t, np.full(11, k))
    (states, mask, ages) = history.window("s", 0.5)  # grid 0.4, 0.3, 0.2, ...
    assert mask.tolist() == [True, True, False, True, True] + [False] * 5
    np.testing.assert_allclose(ages[mask], [0.15, 0.3, 0.4, 0.5], atol=1e-6)
    np.testing.assert_array_equal(states[mask, 0], [3, 2, 1, 0])
    history.reset("s")
    assert not history.window("s", 1.0)[1].any()


def test_handle_infer_keeps_history_and_rejects_bad_requests(policy):
    history = History(policy.history_slots, policy.state_dim)
    rng = np.random.default_rng(1)
    used = []
    for k in range(3):
        (images, state) = request(rng)
        buffer = io.BytesIO()
        np.savez(buffer, **images, state=state, timestamp=0.1 * k, session="ep", reset=k == 0)
        reply = handle_infer(policy, history, buffer.getvalue())
        assert np.asarray(reply["actions"]).shape == (2, 20)
        used.append(reply["history_slots_used"])
    assert used == [0, 1, 2]
    buffer = io.BytesIO()
    np.savez(buffer, **images, state=np.zeros(5), timestamp=1.0, session="ep")
    with pytest.raises(ValueError):
        handle_infer(policy, history, buffer.getvalue())



class TableTokenizer:
    """A stand-in SentencePiece that knows two encoded PAs and nothing else."""

    table = {" Stack cups.\nAction: ": [5, 6, 7], " Rest\nAction: ": [8, 7]}

    def encode(self, text):
        return self.table.get(text, [99])

    def decode(self, ids):
        return next((text for text, row in self.table.items() if row == ids), "?")


def test_stage_pa_texts_round_trip_and_reject_rows_that_do_not():
    sp = TableTokenizer()
    config = dataclasses.replace(tiny_config(), stage_pa_token_ids=((5, 6, 7, 0), (8, 7, 0, 0)))
    assert stage_pa_texts(config, sp) == ["Stack cups.", "Rest"]
    broken = dataclasses.replace(config, stage_pa_token_ids=((9, 5, 0, 0), (8, 7, 0, 0)))
    with pytest.raises(ValueError, match="stage 0"):
        stage_pa_texts(broken, sp)
