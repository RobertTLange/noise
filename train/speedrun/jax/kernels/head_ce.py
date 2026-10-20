"""Per-token cross-entropy of the soft-capped LM head, without materializing the [rows, vocab] logits.

loss = log sum_v exp(c tanh(z_v / c)) - c tanh(z_target / c), z = x @ W, c = 15. Each program takes `block_rows`
rows of x and walks the vocabulary in `block_vocab` tiles (K-looped bf16 dot, fp32 accumulate), adding
exp(softcap(z)) to a running sum; the logits never leave the SM. The softcap bounds every logit to [-15, 15], so the
sum needs no running max (at most 50k x e^15 ~ 2e11). The target logit is one row-wise dot outside the kernel.
exp(softcap(z)) uses only the hardware ex2 / rcp approximations, accurate to ~1e-6 relative.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltr

SOFTCAP = 15.0
LOG2E = 1.4426950408889634


def reference(x, w, targets, vocab_size):
    """Plain-jnp per-token loss with fp32 logits: x [N, D], w [D, V_padded], targets [N] -> [N] fp32."""
    logits = jnp.dot(x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
    logits = SOFTCAP * jnp.tanh(logits / SOFTCAP)
    logits = jnp.where(jnp.arange(logits.shape[-1]) < vocab_size, logits, -jnp.inf)
    target_logit = jnp.take_along_axis(logits, targets[:, None], axis=-1)[:, 0]
    return jax.nn.logsumexp(logits, axis=-1) - target_logit


def _approx(op, x):
    asm, like = f"{op}.approx.ftz.f32 $0, $1;", [jax.ShapeDtypeStruct.like(x)]
    [result] = pltr.elementwise_inline_asm(asm, args=[x], constraints="=f,f", pack=1, result_shape_dtypes=like)
    return result


def _exp_softcap(z):
    """exp(c tanh(z / c)) = 2^(c log2(e) (1 - 2 / (1 + 2^(2 z log2(e) / c)))): hardware ex2 / rcp, ~1e-6 relative
    (tanh.approx would err by up to ~1e-3 near saturation, biasing the loss of confident tokens)."""
    tanh = 1 - 2 * _approx("rcp", 1 + _approx("ex2", z * (2 * LOG2E / SOFTCAP)))
    return _approx("ex2", tanh * (SOFTCAP * LOG2E))


def _kernel(x_ref, w_ref, lse_ref, *, vocab_size, block_vocab, block_k):
    rows, d = x_ref.shape

    def logits_tile(j):
        def accumulate(k, acc):
            x = x_ref[:, pl.ds(k * block_k, block_k)]  # [block_rows, block_k] bf16
            w = w_ref[pl.ds(j * block_vocab, block_vocab), pl.ds(k * block_k, block_k)]  # [block_vocab, block_k]
            return acc + jax.lax.dot_general(x, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

        return jax.lax.fori_loop(0, d // block_k, accumulate, jnp.zeros((rows, block_vocab), jnp.float32))

    full_tiles = vocab_size // block_vocab
    add_tile = lambda j, total: total + _exp_softcap(logits_tile(j)).sum(axis=1)
    total = jax.lax.fori_loop(0, full_tiles, add_tile, jnp.zeros((rows,), jnp.float32))
    if vocab_size % block_vocab:  # the partial last tile; later (all-padding) tiles are skipped
        columns = full_tiles * block_vocab + jnp.arange(block_vocab)
        terms = _exp_softcap(logits_tile(full_tiles))
        total += jnp.where(columns[None, :] < vocab_size, terms, 0.0).sum(axis=1)
    lse_ref[...] = jnp.log(total)


@functools.partial(
    jax.jit, static_argnames=("vocab_size", "block_rows", "block_vocab", "block_k", "num_warps", "num_stages")
)
def head_ce(x, w, targets, vocab_size, block_rows=128, block_vocab=256, block_k=64, num_warps=8, num_stages=3):
    """Per-token loss of softcap(x @ w) over the first vocab_size columns: x [N, D], w [D, V_padded], targets [N].
    Defaults: the fastest tiling on H100 at 262k rows for vocab 4096 and 50257 (from a tiling sweep)."""
    n, d = x.shape
    if n % block_rows or d % block_k or w.shape[1] < pl.cdiv(vocab_size, block_vocab) * block_vocab:
        raise ValueError(f"rows {n} / width {d} must divide by {block_rows} / {block_k}; vocab tiles must fit w")
    x = x.astype(jnp.bfloat16)
    w_rows = w.astype(jnp.bfloat16).T  # [V_padded, D]: one contiguous row per vocabulary entry
    lse = pl.pallas_call(
        functools.partial(_kernel, vocab_size=vocab_size, block_vocab=block_vocab, block_k=block_k),
        grid=(n // block_rows,),
        in_specs=[pl.BlockSpec((block_rows, d), lambda i: (i, 0)), pl.BlockSpec(w_rows.shape, lambda i: (0, 0))],
        out_specs=pl.BlockSpec((block_rows,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        compiler_params=pltr.CompilerParams(num_warps=num_warps, num_stages=num_stages),
        name="head_ce",
    )(x, w_rows)
    target_logit = jnp.einsum("nd,nd->n", x, w_rows[targets], preferred_element_type=jnp.float32)
    return lse - SOFTCAP * jnp.tanh(target_logit / SOFTCAP)
