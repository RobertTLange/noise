"""Residual add (+ jitter, + x0 mixing) and RMS norm in one pass over the fp32 residual stream (Pallas Triton).

XLA materializes `x + delta`, `lambda x + mu x0` and `norm(...)` as separate fp32 [rows, 512] passes, so a block reads
and writes the residual stream ~4 times between its matmuls. Here each program loads a block of rows once and writes
the new stream (fp32) and its RMS-normed bf16 copy: the next matmul's input.

delta is a bf16 matmul output; jitter, when given, is `sigma * noise` added to it with the trainer's bf16 rounding
(`_jitter`: sigma becomes bf16, then a bf16 product and a bf16 sum). Every operand may have fewer rows than the
output, dividing them: rerun copies of a batch then share one array (row r reads row r % rows of that operand), so the
cached clean state is never tiled in memory.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltr

EPS = 1e-6
BLOCK_ROWS = 16  # the fastest tiling on H100 for [262144, 512] (~2.8 TB/s)
NUM_WARPS = 4


def _round_bf16(x):
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def _norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=1, keepdims=True) + EPS)


def _kernel(*refs, sigma, has_delta, has_noise, has_x0):
    refs = list(refs)
    scales_ref = refs.pop(0) if has_x0 else None
    x = refs.pop(0)[...]
    if has_delta:
        delta = refs.pop(0)[...].astype(jnp.float32)
        if has_noise:
            delta = _round_bf16(delta + _round_bf16(sigma * refs.pop(0)[...].astype(jnp.float32)))
        x = x + delta
    if has_x0:
        x = scales_ref[0] * x + scales_ref[1] * refs.pop(0)[...]
    out_ref, normed_ref = refs
    out_ref[...] = x
    normed_ref[...] = _norm(x).astype(jnp.bfloat16)


def _bf16(sigma):
    """A Python float times a bf16 array is a bf16 product: sigma is rounded to bf16 first."""
    return float(np.asarray(sigma, jnp.bfloat16))


def _call(x, delta, noise, x0, scales, sigma):
    operands = [a for a in (x, delta, noise, x0) if a is not None]
    rows, width = max(a.shape[0] for a in operands), x.shape[1]
    for a in operands:
        if a.shape[1] != width or rows % a.shape[0] or a.shape[0] % BLOCK_ROWS:
            raise ValueError(f"operand {a.shape} must tile [{rows}, {width}] in blocks of {BLOCK_ROWS} rows")
    row_spec = lambda a: pl.BlockSpec((BLOCK_ROWS, width), lambda i, n=a.shape[0] // BLOCK_ROWS: (i % n, 0))
    in_specs = [row_spec(a) for a in operands]
    if x0 is not None:
        operands, in_specs = [scales] + operands, [pl.BlockSpec((2,), lambda i: (0,))] + in_specs
    out_spec = pl.BlockSpec((BLOCK_ROWS, width), lambda i: (i, 0))
    kernel = functools.partial(
        _kernel, sigma=_bf16(sigma), has_delta=delta is not None, has_noise=noise is not None, has_x0=x0 is not None
    )
    return pl.pallas_call(
        kernel,
        grid=(rows // BLOCK_ROWS,),
        in_specs=in_specs,
        out_specs=[out_spec, out_spec],
        out_shape=[jax.ShapeDtypeStruct((rows, width), jnp.float32), jax.ShapeDtypeStruct((rows, width), jnp.bfloat16)],
        compiler_params=pltr.CompilerParams(num_warps=NUM_WARPS, num_stages=1),
        name="residual_norm",
    )(*operands)


@functools.partial(jax.jit, static_argnames=("sigma",))
def add_norm(x, delta, noise=None, sigma=0.0):
    """x' = x + (delta + sigma noise) -> (x' fp32, norm(x') bf16). x fp32, delta / noise bf16 (noise optional), all
    [rows_i, D] with rows_i dividing the output rows."""
    return _call(x.astype(jnp.float32), delta, noise, None, None, sigma)


@functools.partial(jax.jit, static_argnames=("sigma",))
def add_mix_norm(x, delta, noise, x0, resid_lambda, x0_lambda, sigma=0.0):
    """mixed = resid_lambda (x + delta + sigma noise) + x0_lambda x0 -> (mixed fp32, norm(mixed) bf16). delta and noise
    may be None (no pending residual add); shapes as in add_norm."""
    scales = jnp.stack([resid_lambda, x0_lambda]).astype(jnp.float32)
    return _call(x.astype(jnp.float32), delta, noise, x0.astype(jnp.float32), scales, sigma)


def _rows(a, rows):
    return jnp.tile(a, (rows // a.shape[0], 1))


def reference_add_norm(x, delta, noise=None, sigma=0.0):
    """The trainer's jnp path: bf16 jitter add, fp32 residual add, RMS norm to bf16."""
    rows = max(a.shape[0] for a in (x, delta, noise) if a is not None)
    delta = _rows(delta, rows)
    if noise is not None:
        delta = delta + (sigma * _rows(noise, rows)).astype(delta.dtype)
    x = _rows(x, rows) + delta.astype(jnp.float32)
    return x, (x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + EPS)).astype(jnp.bfloat16)


def reference_add_mix_norm(x, delta, noise, x0, resid_lambda, x0_lambda, sigma=0.0):
    rows = max(a.shape[0] for a in (x, delta, noise, x0) if a is not None)
    x = _rows(x, rows)
    if delta is not None:
        x, _ = reference_add_norm(x, delta, noise, sigma)
    mixed = resid_lambda * x + x0_lambda * _rows(x0, rows)
    return mixed, (mixed * jax.lax.rsqrt(jnp.mean(mixed * mixed, axis=-1, keepdims=True) + EPS)).astype(jnp.bfloat16)
