"""Boost's readout probes: for every token n and probe direction p, the partition function of the soft-capped head at
the moved stream, without materializing the [tokens, directions, vocab] logits:

    total[n, p] = sum_v exp(c tanh(z / c)),  z = (base[n, v] + step[n] C[p, v]) * inv[n, p],  c = 15

(base = the clean stream's raw logits, C = each direction's logits, inv = the moved stream's inverse RMS). Each
program takes `block_rows` tokens and `block_dirs` directions and walks the vocabulary in `block_vocab` tiles: a tile
of base stays in registers while the program loops over its directions, so each direction's logit row is read once
per tile and token block. exp(softcap(z)) uses head_ce's hardware ex2 / rcp approximations (~1e-6 relative); the
softcap bounds z to [-15, 15], so no running max is needed.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltr

from kernels.head_ce import SOFTCAP, _exp_softcap


def reference(base, c, steps, inv):
    """Plain-jnp partition functions: base [N, V], c [P, V], steps [N, 1], inv [N, P] -> [N, P] fp32."""
    z = (base[:, None, :] + steps[:, :, None] * c[None]) * inv[..., None]
    return jnp.sum(jnp.exp(SOFTCAP * jnp.tanh(z / SOFTCAP)), axis=-1)


def _kernel(base_ref, c_ref, steps_ref, inv_ref, out_ref, *, block_dirs, block_vocab):
    rows, vocab = base_ref.shape
    steps = steps_ref[...]  # [rows]

    def tile(j, totals):
        base = base_ref[:, pl.ds(j * block_vocab, block_vocab)]  # [rows, block_vocab]

        def direction(p, totals):
            row = c_ref[pl.ds(p, 1), pl.ds(j * block_vocab, block_vocab)]  # [1, block_vocab]
            inv = inv_ref[:, pl.ds(p, 1)]  # [rows, 1]
            terms = _exp_softcap((base + steps[:, None] * row) * inv).sum(axis=1)  # [rows]
            column = jnp.arange(block_dirs)[None, :] == p
            return totals + jnp.where(column, terms[:, None], 0.0)

        return jax.lax.fori_loop(0, block_dirs, direction, totals)

    out_ref[...] = jax.lax.fori_loop(0, vocab // block_vocab, tile, jnp.zeros((rows, block_dirs), jnp.float32))


def _kernel3d(base_ref, c_ref, steps_ref, inv_ref, out_ref, *, block_vocab):
    """One [rows, dirs, block_vocab] block per vocabulary tile: no loop over the directions."""
    vocab = base_ref.shape[1]
    steps, inv = steps_ref[...][:, None, None], inv_ref[...][:, :, None]

    def tile(j, totals):
        base = base_ref[:, pl.ds(j * block_vocab, block_vocab)][:, None, :]
        rows_c = c_ref[:, pl.ds(j * block_vocab, block_vocab)][None, :, :]
        return totals + _exp_softcap((base + steps * rows_c) * inv).sum(axis=2)

    out_ref[...] = jax.lax.fori_loop(0, vocab // block_vocab, tile, jnp.zeros(out_ref.shape, jnp.float32))


@functools.partial(jax.jit, static_argnames=("block_rows", "block_dirs", "block_vocab", "num_warps", "num_stages",
                                             "flat"))
def partitions(base, c, steps, inv, block_rows=64, block_dirs=16, block_vocab=256, num_warps=8, num_stages=2,
               flat=False):
    """total[n, p] (module docstring): base [N, V] f32, c [P, V] f32, steps [N, 1], inv [N, P] -> [N, P] fp32. V must be
    a multiple of block_vocab (the caller passes the unpadded vocabulary)."""
    n, vocab = base.shape
    dirs = c.shape[0]
    if n % block_rows or dirs % block_dirs or vocab % block_vocab:
        raise ValueError(f"tokens {n} / directions {dirs} / vocab {vocab} must divide by the blocks")
    return pl.pallas_call(
        (functools.partial(_kernel3d, block_vocab=block_vocab) if flat
         else functools.partial(_kernel, block_dirs=block_dirs, block_vocab=block_vocab)),
        grid=(n // block_rows, dirs // block_dirs),
        in_specs=[
            pl.BlockSpec((block_rows, vocab), lambda i, j: (i, 0)),
            pl.BlockSpec((block_dirs, vocab), lambda i, j: (j, 0)),
            pl.BlockSpec((block_rows,), lambda i, j: (i,)),
            pl.BlockSpec((block_rows, block_dirs), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((block_rows, block_dirs), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((n, dirs), jnp.float32),
        compiler_params=pltr.CompilerParams(num_warps=num_warps, num_stages=num_stages),
        name="probe_cycle",
    )(base.astype(jnp.float32), c.astype(jnp.float32), steps.reshape(-1).astype(jnp.float32), inv.astype(jnp.float32))
