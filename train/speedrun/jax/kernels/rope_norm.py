"""RoPE + per-head RMS norm of q/k in one pass, bf16 out (Pallas Triton).

XLA computes the rotation as an fp32 concatenate (written to HBM), then normalizes it in a second kernel. Here each
program loads a block of head rows once, takes the RMS statistic (RoPE rotates channel pairs, so it keeps the RMS),
rotates both halves and stores bf16: one read of the input, one bf16 write.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltr

EPS = 1e-6


def reference(x, cos, sin):
    """The trainer's process_qk: x [B, T, H, D]; cos, sin [1, T, 1, D/2] -> bf16 [B, T, H, D]."""
    x = x.astype(jnp.float32)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = jnp.concatenate([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], axis=-1)
    return (rotated * jax.lax.rsqrt(jnp.mean(rotated * rotated, axis=-1, keepdims=True) + EPS)).astype(jnp.bfloat16)


def _kernel(x_ref, cos_ref, sin_ref, out_ref, *, half):
    x1 = x_ref[:, pl.ds(0, half)].astype(jnp.float32)
    x2 = x_ref[:, pl.ds(half, half)].astype(jnp.float32)
    scale = jax.lax.rsqrt(((x1 * x1).sum(axis=1) + (x2 * x2).sum(axis=1)) / (2 * half) + EPS)[:, None]
    cos, sin = cos_ref[...], sin_ref[...]
    out_ref[:, pl.ds(0, half)] = ((x1 * cos + x2 * sin) * scale).astype(out_ref.dtype)
    out_ref[:, pl.ds(half, half)] = ((x2 * cos - x1 * sin) * scale).astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("block_rows", "num_warps"))
def rope_norm(x, cos, sin, block_rows=64, num_warps=8):
    """x [B, T, H, D] (bf16 or fp32); cos, sin [1, T, 1, D/2] fp32 -> bf16 [B, T, H, D]. Default tiling: the fastest
    on H100 for [128, 2048, 8, 64] (from a tiling sweep)."""
    b, t, h, d = x.shape
    half = d // 2
    period = t * h  # head rows per sequence; the rotation angle depends on the row's position t
    if period % block_rows:
        raise ValueError(f"T * H = {period} must divide by {block_rows}")
    # one (cos, sin) row per head row of a sequence: row r of x has position (r // H) % T
    cos_rows = jnp.repeat(cos.reshape(t, half), h, axis=0).astype(jnp.float32)
    sin_rows = jnp.repeat(sin.reshape(t, half), h, axis=0).astype(jnp.float32)
    blocks_per_sequence = period // block_rows
    angle_spec = pl.BlockSpec((block_rows, half), lambda i: (i % blocks_per_sequence, 0))
    out = pl.pallas_call(
        functools.partial(_kernel, half=half),
        grid=(b * period // block_rows,),
        in_specs=[pl.BlockSpec((block_rows, d), lambda i: (i, 0)), angle_spec, angle_spec],
        out_specs=pl.BlockSpec((block_rows, d), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((b * period, d), jnp.bfloat16),
        compiler_params=pltr.CompilerParams(num_warps=num_warps, num_stages=1),
        name="rope_norm",
    )(x.reshape(b * period, d), cos_rows, sin_rows)
    return out.reshape(b, t, h, d)
