"""Exact keys (and queries): for every key position t and key direction j, the sum over the queries s of the exactly
evaluated change of the attention output, scored against each query's attention-output credit:

    score[t, j] = sum_s  ch / (1 + ch) * C[s, t],   ch = A[s, t] * (exp(dl_j[s, t]) - 1),
    A[s, t] = exp(q_s . k_t * scale - lse_s),       dl_j[s, t] = q_s . dk_{t, j} * scale,
    C[s, t] = hub_s . v_t - hub_s . o_s,            over the causal window (0 <= s - t <= window).

kernels/exact_moments.py evaluates these sums on tensor cores from this file's closed-form coefficients
(`closed_coefficients`, `rotated_columns`); lse_s and o_s come from the clean attention. `reference` /
`query_reference` evaluate the same scores in plain jnp from explicit change tensors dk / dq. This file also holds
the values' exact credit (`value_credit`, a fused kernel).

A query move changes only its own output, through all of its weights: with ch = A[s, t] * (exp(dl_j[s, t]) - 1),
dl_j = dq_{s, j} . k_t * scale, the new output is o_s + sum_t ch C[s, t] / (1 + sum_t ch) scored against the hub credit
(sum_t A C = 0).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltr


def reference(q, k, v, hub, dk, lse, hub_out, window):
    """Plain jnp: q, k, v, hub [BH, T, D]; dk [BH, J, T, D]; lse, hub_out [BH, T] -> [BH, T, J] fp32."""
    t, d = q.shape[1], q.shape[2]
    scale = d ** -0.5
    f32 = lambda x: x.astype(jnp.float32)
    lag = jnp.arange(t)[:, None] - jnp.arange(t)[None, :]
    allowed = (lag >= 0) & (lag <= window)
    a = jnp.where(allowed, jnp.exp(jnp.einsum("bsd,btd->bst", f32(q), f32(k)) * scale - lse[..., None]), 0.0)
    c = jnp.einsum("bsd,btd->bst", f32(hub), f32(v)) - hub_out[..., None]
    dl = jnp.einsum("bsd,bjtd->bjst", f32(q), f32(dk)) * scale
    ch = a[:, None] * jnp.expm1(dl)
    return jnp.moveaxis(jnp.sum(ch / (1 + ch) * c[:, None], axis=2), 1, 2)


def query_reference(q, k, v, hub, dq, lse, hub_out, window):
    """Plain jnp: q, k, v, hub [BH, T, D]; dq [BH, J, T, D]; lse, hub_out [BH, T] -> [BH, T, J] fp32."""
    t, d = q.shape[1], q.shape[2]
    scale = d ** -0.5
    f32 = lambda x: x.astype(jnp.float32)
    lag = jnp.arange(t)[:, None] - jnp.arange(t)[None, :]
    allowed = (lag >= 0) & (lag <= window)
    a = jnp.where(allowed, jnp.exp(jnp.einsum("bsd,btd->bst", f32(q), f32(k)) * scale - lse[..., None]), 0.0)
    c = jnp.einsum("bsd,btd->bst", f32(hub), f32(v)) - hub_out[..., None]
    ch = a[:, None] * jnp.expm1(jnp.einsum("bjsd,btd->bjst", f32(dq), f32(k)) * scale)
    return jnp.moveaxis(jnp.sum(ch * c[:, None], -1) / (1 + jnp.sum(ch, -1)), 1, 2)


def value_reference(q, k, hub, lse, window):
    """Plain jnp: the values' exact credit sum_s A[s, t] hub_s, [BH, T, D] fp32."""
    t, d = q.shape[1], q.shape[2]
    lag = jnp.arange(t)[:, None] - jnp.arange(t)[None, :]
    allowed = (lag >= 0) & (lag <= window)
    f32 = lambda x: x.astype(jnp.float32)
    a = jnp.where(allowed, jnp.exp(jnp.einsum("bsd,btd->bst", f32(q), f32(k)) * d ** -0.5 - lse[..., None]), 0.0)
    return jnp.einsum("bst,bsd->btd", a, f32(hub))


def _value_kernel(q_ref, k_ref, hub_ref, lse_ref, out_ref, *, window, block_s):
    t_total, d = q_ref.shape
    block_t = k_ref.shape[0]
    scale = d ** -0.5
    t0 = pl.program_id(1) * block_t
    k = k_ref[...]
    key_pos = t0 + jnp.arange(block_t)
    first = t0 // block_s
    last = jnp.minimum((t0 + block_t - 1 + window) // block_s + 1, t_total // block_s)

    def tile(i, total):
        s0 = i * block_s
        q, hub = q_ref[pl.ds(s0, block_s), :], hub_ref[pl.ds(s0, block_s), :]
        lse = lse_ref[pl.ds(s0, block_s)]
        lag = (s0 + jnp.arange(block_s))[:, None] - key_pos[None, :]
        allowed = (lag >= 0) & (lag <= window)
        logits = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        a = jnp.where(allowed, jnp.exp(logits * scale - lse[:, None]), 0.0)
        return total + jax.lax.dot_general(a.astype(hub.dtype), hub, (((0,), (0,)), ((), ())),
                                           preferred_element_type=jnp.float32)

    out_ref[...] = jax.lax.fori_loop(first, last, tile, jnp.zeros((block_t, d), jnp.float32))


@functools.partial(jax.jit, static_argnames=("window", "block_t", "block_s", "num_warps", "num_stages"))
def value_credit(q, k, hub, lse, window, block_t=64, block_s=64, num_warps=4, num_stages=2):
    """The values' exact credit sum_s A[s, t] hub_s [BH, T, D] (fp32): q, k, hub [BH, T, D] (cast to bf16), lse
    [BH, T] fp32 from the clean attention. Moving position t's value moves query s's output by A[s, t] times the move."""
    bh, t, d = q.shape
    if t % block_t or t % block_s:
        raise ValueError(f"T {t} must divide by the blocks {block_t} / {block_s}")
    b16 = lambda x: x.astype(jnp.bfloat16)
    row = lambda: pl.BlockSpec((None, t, d), lambda i, j: (i, 0, 0))
    return pl.pallas_call(
        functools.partial(_value_kernel, window=window, block_s=block_s),
        grid=(bh, t // block_t),
        in_specs=[row(), pl.BlockSpec((None, block_t, d), lambda i, j: (i, j, 0)), row(),
                  pl.BlockSpec((None, t), lambda i, j: (i, 0))],
        out_specs=pl.BlockSpec((None, block_t, d), lambda i, j: (i, j, 0)),
        out_shape=jax.ShapeDtypeStruct((bh, t, d), jnp.float32),
        compiler_params=pltr.CompilerParams(num_warps=num_warps, num_stages=num_stages),
        name="exact_values",
    )(b16(q), b16(k), b16(hub), lse.astype(jnp.float32))


# Closed-form changes. A raw key (query) step s e_j moves the processed (rotated, RMS-normed) key by
# A k_hat + B R e_j: A = r' / r - 1 from the norm's change (|x|^2 + 2 s x_j + s^2), B = s r', and R e_j touches only
# dims i = j mod d/2 and i + d/2 with the rotation's (cos, -sin) or (sin, cos) at that position. So each direction's
# logit change is dl_j[s, t] = scale (A_tj S[s, t] + B_tj (u_tj q_s[i] + w_tj q_s[i + d/2])), S = q_s . k_t: the same
# exactly evaluated experiments as `reference` / `query_reference`, with no per-direction dot and no change tensors.


def closed_coefficients(raw, cos, sin, dims, sigma):
    """For moves of `raw` [BH, T, D] (f32) by sigma times its RMS along the dims `dims` [J]: coefficients
    [BH, 3, T, J] (A, B u, B w) and the steps [BH, T, 1]. cos, sin [T, D/2]: the rotation at each position."""
    d = raw.shape[-1]
    half = d // 2
    sq = jnp.sum(raw * raw, -1, keepdims=True)
    steps = sigma * jnp.maximum(jnp.sqrt(sq / d), 1e-6)
    change = (2 * steps * raw[..., dims] + steps * steps) / d / (sq / d + 1e-6)  # relative change of the mean square
    a = jnp.expm1(-0.5 * jnp.log1p(change))  # r' / r - 1
    b = steps * jax.lax.rsqrt(sq / d + 1e-6) * (1 + a)  # s r'
    i = dims % half
    first = (dims < half)[None, :]
    u = jnp.where(first, cos[:, i], sin[:, i])  # [T, J]
    w = jnp.where(first, -sin[:, i], cos[:, i])
    return jnp.stack([a, b * u, b * w], 1), steps


def rotated_columns(processed, dims):
    """The processed (rotated, normed) vectors' columns [BH, 2, T, J] (i, i + D/2) that a move along dims j touches."""
    half = processed.shape[-1] // 2
    i = dims % half
    return jnp.stack([processed[..., i], processed[..., i + half]], 1)
