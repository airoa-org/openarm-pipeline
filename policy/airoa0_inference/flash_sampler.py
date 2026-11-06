# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
# Modified by AI Robot Association (AIRoA) in 2026 for openarm-pipeline.
# Derived from openpi (Apache-2.0), revision 215abfb, as vendored in AIRoA-0 at ab9c7ec (`openpi.models.pi0`).
"""FLASH cold-start sampling for an unmodified AIRoA-0 `Pi0` model.

Only the public AIRoA-0 package is imported; the model's own methods build the
prefix exactly as in post-training. Tested against AIRoA-0 commit ab9c7ec.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from openpi.models import flash
from openpi.models import model as _model
from openpi.models.pi0 import _suffix_mpa_attention_bias, make_attn_mask


def solver_phases(paths, *, levels, buffer_size):
    """Right-align per-slot solver paths into shared decoder evaluations.

    Returns ((pre-step time per slot), (post-step time per slot)) for each
    evaluation; a slot with a shorter path keeps its time (zero step) until
    its own steps start.
    """
    paths = flash.flow_substeps(paths, levels=levels, buffer_size=buffer_size)
    count = max(len(p) - 1 for p in paths)
    padded = [(p[0],) * (count - len(p) + 1) + p for p in paths]
    return tuple(
        (tuple(p[i] for p in padded), tuple(p[i + 1] for p in padded))
        for i in range(count)
    )


def update_phase(x, velocity, *, before, after, chunk_size):
    """One Euler step of every slot from its `before` time to its `after` time."""
    times = jnp.repeat(jnp.asarray(before, jnp.float32), chunk_size)
    next_times = jnp.repeat(jnp.asarray(after, jnp.float32), chunk_size)
    return x - (times - next_times)[None, :, None] * velocity


def append_pa(model, prefix_tokens, prefix_mask, kv_cache, pa_token_ids):
    """`Pi0._append_stage_pa`, with the PA token row given instead of a stage id."""
    ids = jnp.asarray(pa_token_ids, dtype=jnp.int32)
    valid = ids != 0
    tokens = model.PaliGemma.llm(ids, method="embed")
    local = valid[:, :, None] & valid[:, None, :]
    cross = valid[:, :, None] & prefix_mask[:, None, :]
    mask = jnp.concatenate([cross, local], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(valid, axis=-1) - 1
    ((_, _), cache) = model.PaliGemma.llm(
        [tokens, None], mask=mask, positions=positions, kv_cache=kv_cache
    )
    return (
        jnp.concatenate([prefix_tokens, tokens], axis=1),
        jnp.concatenate([prefix_mask, valid], axis=1),
        cache,
    )


def encode_prefix(model, observation, pa_token_ids):
    """Prefix KV cache as in post-training, followed by one fixed PA.

    Visual-budget soft gate at its final schedule values, mask-prior attention
    bias at full strength, proprio-history tokens; the stage head is not used.
    Returns (obs, prefix_mask, kv_cache, prefix_prior).
    """
    obs = _model.preprocess_observation(
        None,
        observation,
        train=False,
        image_keys=model.image_view_names,
        image_baked_padding_by_view=model.image_baked_padding_by_view,
    )
    image_tokens = model._embed_image_tokens_by_name(obs, train=False)
    prompt_tokens = model._embed_prompt_tokens(obs)
    dtype = next(iter(image_tokens.values())).dtype
    beta = jnp.asarray(model.visual_budget_gate_prior_bias_end, dtype=dtype)
    mask_logits = model._compute_visual_budget_mask_logits(
        obs, image_tokens, prompt_tokens, beta=beta
    )
    (gated_image_tokens, _) = model._apply_visual_budget_soft_gate(
        obs,
        image_tokens,
        prompt_tokens,
        beta=beta,
        tau=jnp.asarray(model.visual_budget_gate_tau_end, dtype=dtype),
        gate_floor=jnp.asarray(model.visual_budget_gate_floor_end, dtype=dtype),
        mask_logits_by_name=mask_logits,
    )
    (image_priors, _) = model._compute_visual_budget_mpa_log_priors(
        obs, image_tokens, prompt_tokens, mask_logits_by_name=mask_logits
    )
    (proprio_tokens, proprio_mask) = model._build_proprio_memory_tokens(obs, prompt_tokens)
    (proprio_tokens, proprio_mask, _) = model._apply_temporal_proprio_prefix_gate(
        proprio_tokens, proprio_mask, dtype=dtype
    )
    (prefix_tokens, prefix_mask, prefix_ar_mask) = model._compose_prefix_tokens(
        obs,
        gated_image_tokens,
        prompt_tokens,
        proprio_memory_tokens=proprio_tokens,
        proprio_memory_mask=proprio_mask,
    )
    prefix_prior = model._compose_prefix_mpa_log_prior(
        obs, image_priors, prompt_tokens=prompt_tokens, proprio_memory_tokens=proprio_tokens
    )
    (_, kv_cache) = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=make_attn_mask(prefix_mask, prefix_ar_mask),
        positions=jnp.cumsum(prefix_mask, axis=1) - 1,
    )
    (_, prefix_mask, kv_cache) = append_pa(
        model, prefix_tokens, prefix_mask, kv_cache, pa_token_ids
    )
    prefix_prior = jnp.pad(prefix_prior, ((0, 0), (0, pa_token_ids.shape[1])))
    return obs, prefix_mask, kv_cache, prefix_prior


def sample_actions(model, rng, observation, pa_token_ids):
    """The first action chunk, still normalized: [b, flash_chunk_size, action_dim].

    Supports FLASH (nonuniform levels and substeps) PA-conditioned checkpoints.
    Every call is a cold start on one prefix encoding: the buffer is filled
    over `flash_buffer_size` rounds (first round: only the last chunk is valid;
    after each round it shifts left by one chunk and fresh noise enters the
    last slot), each round running the trained per-slot solver phases. Only
    the fully denoised first chunk is returned.
    """
    if not (
        model.flash_enabled
        and model.stage_pa_conditioning
        and model.flash_flow_times is not None
        and model.flash_flow_substeps is not None
    ):
        raise NotImplementedError(
            "Only FLASH (nonuniform levels and substeps) PA-conditioned checkpoints are supported"
        )
    (obs, prefix_mask, kv_cache, prefix_prior) = encode_prefix(model, observation, pa_token_ids)
    (batch, horizon) = (prefix_mask.shape[0], model.action_horizon)
    (chunk, slots) = (model.flash_chunk_size, model.flash_buffer_size)
    logit_bias = _suffix_mpa_attention_bias(prefix_prior, horizon)
    prefix_to_suffix = jnp.broadcast_to(
        prefix_mask[:, None, :], (batch, horizon, prefix_mask.shape[1])
    )
    phases = solver_phases(
        model.flash_flow_substeps, levels=model.flash_flow_times, buffer_size=slots
    )

    def refine(x, valid):
        (suffix_mask, local_positions) = flash.attention(valid, chunk_size=chunk, buffer_size=slots)
        mask = jnp.concatenate([prefix_to_suffix, suffix_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + local_positions
        for before, after in phases:
            time = jnp.broadcast_to(
                jnp.repeat(jnp.asarray(before, jnp.float32), chunk), (batch, horizon)
            )
            (tokens, _, _, adarms_cond) = model.embed_suffix(obs, x, time)
            tokens = jnp.where(valid[..., None], tokens, 0)
            adarms_cond = jnp.where(valid[..., None], adarms_cond, 0)
            ((_, suffix_out), _) = model.PaliGemma.llm(
                [None, tokens],
                mask=mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                attn_logit_bias=logit_bias,
            )
            velocity = model.action_out_proj(suffix_out[:, -horizon:])
            x = update_phase(x, velocity, before=before, after=after, chunk_size=chunk)
        return x

    rngs = jax.random.split(rng, slots)
    x = jax.random.normal(rngs[0], (batch, horizon, model.action_dim))
    for round_index in range(slots):
        if round_index:
            fresh = jax.random.normal(rngs[round_index], (batch, chunk, model.action_dim))
            x = jnp.concatenate([x[:, chunk:], fresh], axis=1)
        valid = jnp.arange(horizon) >= (slots - 1 - round_index) * chunk
        x = refine(x, jnp.broadcast_to(valid, (batch, horizon)))
    return x[:, :chunk]


def jit_sampler(model):
    """Compiled `sample_actions(rng, observation, pa_token_ids)` with the
    model's weights frozen (as `openpi.shared.nnx_utils.module_jit`)."""
    (graphdef, state) = nnx.split(model)
    compiled = jax.jit(
        lambda state, rng, observation, pa_token_ids: sample_actions(
            nnx.merge(graphdef, state), rng, observation, pa_token_ids
        )
    )
    return lambda rng, observation, pa_token_ids: compiled(state, rng, observation, pa_token_ids)
