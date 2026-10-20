"""Track 2's trainer: identity-path credit with ANVIL II and Adam on the benchmark's model, on 8 GPUs in one process,
with no backward pass. The defaults are the record's recipe; train/speedrun/run.py launches it on the pinned data,
exports the final weights and scores them with utils/eval.py.

    python train.py data=<dir with train.npy, val.npy> [key=value ...]    # train (logs/<run_id>.txt, checkpoints)
    python train.py export <checkpoint.npz> <out.npz>                     # utils/model.py's state dict

Sections: the recipe; the model (forward only, with the clean pass's cache the estimator reads); the estimator
(identity-path credit: one measured readout cycle at the top, one local step per site); the optimizers (Adam, and
ANVIL II batched by matrix shape); training (the compiled step, warmup, timer, log); the export. The Pallas kernels
are in kernels/.

Every update is one XLA program under shard_map over a 'pop' axis of all GPUs (each runs the estimate for its share
of the tokens; the collectives are inside), compiled ahead of time and run once on copies of the state before the
timer starts. The timer covers the first update to the end of the last, excluding validation. The run log
(logs/<run_id>.txt) opens with the source of this file and every kernel, so a record stays reproducible from its log.
"""

import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from kernels import exact_keys, exact_moments, head_ce, relu2_matmul, residual_norm, rope_norm

SOURCES = (exact_keys, exact_moments, head_ce, relu2_matmul, residual_norm, rope_norm)  # embedded in the run log
F32, BF16 = jnp.float32, jnp.bfloat16
EVAL_CHUNK_ROWS = 4  # validation rows per forward chunk: 544 val rows = 8 GPUs x 17 chunks


# ---------------------------------------------------------------------------------------------------------------------
# Recipe
# The record's recipe as defaults; key=value arguments override them.


@dataclass
class Hyperparameters:
    # data
    data: str = ""  # a directory with train.npy and val.npy: uint16 rows [N, T + 1] (run.py writes them)
    train_rows: int = 4880  # train on the first rows only (the benchmark's train.pt); 0: all
    max_epochs: int = 1  # passes over the train rows (each row feeds at most this many updates)
    batch_size: int = 4  # rows per update
    seq_len: int = 2048
    seq_warmup: str = "1024:780"  # e.g. "512:200,1024:400": 512-token rows until step 200 (the same tokens per update)
    # run
    num_iterations: int = 975
    steps_per_call: int = 15  # updates per dispatch (one compiled scan): the step boundaries below divide by it
    val_every: int = 150
    target: float = 4.2
    save_every: int = 0  # latest checkpoint at logs/<run_id>.npz (0 disables)
    save_steps: str = ""  # steps whose checkpoints are kept at logs/<run_id>-<step>.npz, e.g. "300,975"
    seed: int = 0
    rng: str = "rbg"  # rbg | unsafe_rbg | threefry2x32
    # model
    vocab_size: int = 4096
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    # optimizer: ANVIL II on the block matrices, Adam elsewhere, warmup-stable-decay
    lr: float = 0.003
    beta1: float = 0.7
    beta2: float = 0.98
    adam_eps: float = 1e-8
    lr_wte: float = 0.9
    lr_value_embeds: float = 0.9
    lr_lambdas: float = 0.003
    anvil_lr: float = 0.04
    anvil_wd: float = 2.25  # ANVIL II's cautious weight decay (times its scheduled rate)
    warmup_steps: int = 0
    cooldown_steps: int = 390  # the last 40%


def parse_args(argv):
    hp = Hyperparameters()
    types = {f.name: type(getattr(hp, f.name)) for f in fields(hp)}
    for arg in argv:
        name, value = arg.split("=", 1)
        if name not in types:
            raise SystemExit(f"unknown hyperparameter {name!r}")
        setattr(hp, name, types[name](value))
    if hp.batch_size * hp.seq_len % 128 or any(t * hp.n_head % 64 or hp.seq_len % t for t, _ in seq_stages(hp)):
        raise SystemExit("kernels tile batch_size * seq_len by 128, row length * n_head by 64; lengths divide seq_len")
    marks = [hp.num_iterations, *(until for _, until in seq_stages(hp)[:-1]), *(int(x) for x in hp.save_steps.split(",")
                                                                                  if x)]
    marks += [hp.val_every] if hp.val_every < hp.num_iterations else []
    if any(mark % hp.steps_per_call for mark in marks):
        raise SystemExit(f"steps_per_call={hp.steps_per_call} must divide the run's step boundaries {marks}")
    return hp


def seq_stages(hp):
    """[(row length, first step past it)] of the sequence-length warmup, ending with seq_len for the rest of the run."""
    stages = [tuple(map(int, stage.split(":"))) for stage in hp.seq_warmup.split(",") if stage]
    return [*stages, (hp.seq_len, math.inf)]


def seq_length(step, hp):
    return next(length for length, until in seq_stages(hp) if step < until)


# ---------------------------------------------------------------------------------------------------------------------
# Model
# The benchmark's model (utils/model.py) in JAX: modded-nanoGPT as in the Dust paper (8L, d512, RoPE + QK-norm, relu²,
# SSSL windows, value embeddings, x0 mixing, untied head, logit softcap). Linear weights are [in, out]; bf16 matmuls,
# fp32 residual stream. The forward pass is forward-only (Pallas kernels without gradient rules); `clean_pass` caches
# every block's inputs and outputs for the estimator (below).


GATE_CHANNELS = 32
SOFTCAP = 15.0
HEAD_SLAB = 256


@dataclass(frozen=True)
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 4096
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    window_pattern: str = "SSSL"

    @property
    def head_dim(self):
        return self.n_embd // self.n_head

    @property
    def padded_vocab(self):
        """Whole head slabs; padding columns are masked out of every softmax."""
        return math.ceil(self.vocab_size / HEAD_SLAB) * HEAD_SLAB

    def has_ve(self, layer):
        return layer % 2 == (self.n_layer - 1) % 2

    def window(self, layer):
        """Keys visible to a query: positions s - window .. s."""
        if layer == self.n_layer - 1:
            return self.sequence_len
        kind = self.window_pattern[layer % len(self.window_pattern)]
        return self.sequence_len if kind == "L" else self.sequence_len // 2


def norm(x):
    x = x.astype(F32)
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def linear(x, w):
    return jnp.dot(x.astype(BF16), w.astype(BF16))


def rotary(cfg, length):
    dim = cfg.head_dim
    inv_freq = 1.0 / 10000 ** (jnp.arange(0, dim, 2, dtype=F32) / dim)
    freqs = jnp.outer(jnp.arange(length, dtype=F32), inv_freq)
    return jnp.cos(freqs)[None, :, None, :], jnp.sin(freqs)[None, :, None, :]


def process_qk(raw, cos, sin):
    """RoPE then QK-norm, returned in bf16 (one fused pass; fp32 math)."""
    return rope_norm.rope_norm(raw, cos, sin)


def attention(q, k, v, window):
    """Causal attention over [batch, token, head, dim]; query s sees keys s - window .. s."""
    local = None if window >= q.shape[1] else (window, 0)
    return jax.nn.dot_product_attention(
        q.astype(BF16), k.astype(BF16), v.astype(BF16), is_causal=True, local_window_size=local, implementation="cudnn"
    )


def attention_with_lse(q, k, v, window):
    """`attention` and its log-sum-exp [batch, head, token] (cuDNN's residual), for the exact-credit kernels."""
    from jax._src.cudnn.fused_attention_stablehlo import MaskType, dot_product_attention

    return dot_product_attention(q.astype(BF16), k.astype(BF16), v.astype(BF16), scale=q.shape[-1] ** -0.5,
                                 mask_type=MaskType.CAUSAL, sliding_window_length=window + 1, return_residual=True)


def init_params(cfg, key):
    d, vp = cfg.n_embd, cfg.padded_vocab
    s = math.sqrt(3.0) * d**-0.5
    keys = iter(jax.random.split(key, 8 * cfg.n_layer + 8))
    uniform = lambda shape, bound: jax.random.uniform(next(keys), shape, F32, -bound, bound)
    proj = lambda fan_in, fan_out: uniform((fan_in, fan_out), 1.5 * math.sqrt(3.0 / fan_in))
    blocks = []
    for layer in range(cfg.n_layer):
        block = {
            "attn.c_q": uniform((d, d), s),
            "attn.c_k": uniform((d, d), s),
            "attn.c_v": uniform((d, d), s),
            "attn.c_proj": proj(d, d),
            "mlp.c_fc": uniform((d, 4 * d), s),
            "mlp.c_proj": proj(4 * d, d),
        }
        if cfg.has_ve(layer):
            block["attn.ve_gate"] = jnp.zeros((GATE_CHANNELS, cfg.n_head), F32)
        blocks.append(block)
    return {
        "wte": jax.random.normal(next(keys), (vp, d), F32).astype(BF16),
        "lm_head": 0.001 * jax.random.normal(next(keys), (d, vp), F32),
        "blocks": blocks,
        "value_embeds": {str(i): uniform((vp, d), s).astype(BF16) for i in range(cfg.n_layer) if cfg.has_ve(i)},
        "resid_lambdas": jnp.ones((cfg.n_layer,), F32),
        "x0_lambdas": jnp.full((cfg.n_layer,), 0.1, F32),
    }


def _rows(x):
    return None if x is None else x.reshape(-1, x.shape[-1])


def _residual(kernel, t, x, pending, *args):
    """Run a residual_norm kernel on [B, T, D] operands; `pending` is a matmul output still to be added to x."""
    stream, normed = kernel(_rows(x), _rows(pending), None, *args, sigma=0.0)
    shape = (stream.shape[0] // t, t, stream.shape[-1])
    return stream.reshape(shape), normed.reshape(shape)


def attention_half(cfg, params, layer, x, pending, x0, idx, cache, rope):
    """Residual add and mix, then attention: (mixed stream, attention output before attn.c_proj)."""
    p = params["blocks"][layer]
    b, t = idx.shape
    h, hd = cfg.n_head, cfg.head_dim
    lambdas = (_rows(x0), params["resid_lambdas"][layer], params["x0_lambdas"][layer])
    mixed, n1 = _residual(residual_norm.add_mix_norm, t, x, pending, *lambdas)
    q, k, v = (linear(n1, p[f"attn.c_{name}"]) for name in "qkv")
    values = v.reshape(b, t, h, hd)
    if cfg.has_ve(layer):
        ve = params["value_embeds"][str(layer)][idx]
        z = linear(n1[..., :GATE_CHANNELS], p["attn.ve_gate"])
        values = values + (2 * jax.nn.sigmoid(z.astype(F32)))[..., None].astype(BF16) * ve.reshape(b, t, h, hd)
    cos, sin = rope
    qp = process_qk(q.reshape(b, t, h, hd), cos, sin)
    kp = process_qk(k.reshape(b, t, h, hd), cos, sin)
    pre = f"h{layer}."
    if cache is None or jax.default_backend() != "gpu":  # cuDNN's residual: GPU only
        o = attention(qp, kp, values, cfg.window(layer)).reshape(b, t, -1)
    else:
        o, lse = attention_with_lse(qp, kp, values, cfg.window(layer))
        cache.update({pre + "attn.out": o, pre + "attn.lse": lse})  # the exact-credit kernels' clean attention
        o = o.reshape(b, t, -1)
    if cache is not None:
        cache.update({pre + "in.attn": n1, pre + "attn.c_q": q, pre + "attn.c_k": k, pre + "attn.c_v": v})
        if cfg.has_ve(layer):
            cache[pre + "attn.ve_gate"] = z
            cache[f"ve{layer}"] = ve
    return mixed, o


def projection_and_mlp(params, layer, mixed, o, cache):
    """attn.c_proj and its residual add, then the MLP: (stream, MLP output still to be added to it)."""
    w = params["blocks"][layer]
    x, n2 = _residual(residual_norm.add_norm, o.shape[1], mixed, linear(o, w["attn.c_proj"]))
    if cache is None:  # c_fc's output is not needed: fused matmul + relu²
        act = relu2_matmul.relu2_matmul(n2.reshape(-1, n2.shape[-1]), w["mlp.c_fc"]).reshape(*n2.shape[:-1], -1)
    else:
        hidden = linear(n2, w["mlp.c_fc"])
        act = jnp.square(jax.nn.relu(hidden))
        cache.update({f"h{layer}.in.attn.c_proj": o, f"h{layer}.in.mlp.c_fc": n2, f"h{layer}.mlp.c_fc": hidden,
                      f"h{layer}.in.mlp.c_proj": act})
    return x, linear(act, w["mlp.c_proj"])


def softcap(x):
    return SOFTCAP * jnp.tanh(x / SOFTCAP)


def mask_padding(logits, vocab_size):
    return jnp.where(jnp.arange(logits.shape[-1]) < vocab_size, logits, -jnp.inf)


def head_losses(cfg, final, lm_head, targets):
    """Per-token losses [rows, T]: fused LM head + softcap + cross-entropy (fp32 logits, never materialized)."""
    rows, t, d = final.shape
    return head_ce.head_ce(final.reshape(-1, d), lm_head, targets.reshape(-1), cfg.vocab_size).reshape(rows, t)


def forward(cfg, params, idx, targets, cache=None):
    """Per-token losses [B, T] (fp32); fills `cache` (if given) with what the estimator reads."""
    rope = rotary(cfg, idx.shape[1])
    x = norm(params["wte"][idx])
    x0, pending = x, None
    if cache is not None:
        cache["x0"] = x0
    for layer in range(cfg.n_layer):
        mixed, o = attention_half(cfg, params, layer, x, pending, x0, idx, cache, rope)
        x, pending = projection_and_mlp(params, layer, mixed, o, cache)
        if cache is not None:  # the stream after the block
            cache[f"h{layer}.out"] = x.astype(F32) + pending.astype(F32)
    _, final = _residual(residual_norm.add_norm, idx.shape[1], x, pending)
    if cache is not None:  # the head's credit moves these logits directly
        cache["in.lm_head"], cache["lm_head"] = final, linear(final, params["lm_head"])
    return head_losses(cfg, final, params["lm_head"], targets)


@dataclass
class Clean:
    """The clean forward pass the estimator's experiments are measured against."""

    cfg: GPTConfig
    params: dict
    idx: jax.Array
    targets: jax.Array
    cache: dict
    loss: jax.Array  # per-token clean loss [B, T]


def clean_pass(cfg, params, idx, targets):
    cache = {}
    loss = forward(cfg, params, idx, targets, cache=cache)
    return Clean(cfg, params, idx, targets, cache, loss)


def site_layer(site):
    return int(site.split(".")[0][1:])


def site_input(clean, site):
    """The cached input [B, T, in] of a weight site (not the embeddings)."""
    layer, name = site_layer(site), site.split(".", 1)[1]
    if name in ("attn.c_q", "attn.c_k", "attn.c_v"):
        return clean.cache[f"h{layer}.in.attn"]
    if name == "attn.ve_gate":
        return clean.cache[f"h{layer}.in.attn"][..., :GATE_CHANNELS]
    return clean.cache[f"h{layer}.in.{name}"]


def set_grad(grads, site, value):
    if site == "wte":
        grads[site] = value
    elif site.startswith("ve"):
        grads["value_embeds"][site[2:]] = value
    else:
        grads["blocks"][site_layer(site)][site.split(".", 1)[1]] = value


def coordinate_change(raw, targets, vocab, sigma):
    """The LM head's credit on raw logits (..., padded vocab): for every token and logit j, the token's loss with raw
    logit j moved up by sigma minus moved down, over 2 sigma. Moving one logit changes the partition by one term, so
    each moved loss is the clean loss plus log1p(p_j expm1(z_j' - z_j)), minus the target logit's own move."""
    raw = raw.astype(F32)
    z = mask_padding(softcap(raw), vocab)
    up, down = softcap(raw + sigma), softcap(raw - sigma)
    weight = jax.nn.softmax(z, axis=-1)  # the clean forward's probabilities (0 at the padding)
    moved = lambda new: jnp.log1p(weight * jnp.expm1(new - jnp.where(weight > 0, z, new)))
    change = moved(up) - moved(down)
    at_target = jax.nn.one_hot(targets, raw.shape[-1], dtype=F32)
    change = change - at_target * (up - down)
    return change / (2 * sigma)


# ---------------------------------------------------------------------------------------------------------------------
# Estimator
# Identity-path credit (record #7, Noise PR #16), each GPU running it for its own rows of the batch.
#
# One readout cycle per update measures each token's credit at the top stream: the stream moves along every row of a
# sign-flipped order-512 Hadamard matrix and the readout (norm, head, softcap, cross-entropy) is evaluated on each move.
# Every block's output stream gets that credit through the residual path alone (scaled by the residual scalars above
# it), the embedding through the same path plus the x0 injections. Everything else is one local step from those
# credits: the writers' gradients directly, the MLP hidden credit by one relu² slope per unit, the attention-output
# targets as `credit @ W_projᵀ` (a linear map of measured credit doesn't count as a step), and from them the queries,
# keys and values by single-coordinate experiments evaluated exactly (kernels/exact_keys.py, exact_moments.py), the
# gates by one sigmoid slope. No credit is read back through a block's weights into the block below.
#
# The work split: each GPU runs the clean forward pass on its own rows (in the 2048-token phase, the row it shares
# with a neighbour) and the estimate for its own tokens; for the queries, keys and values each GPU takes one head over
# the whole batch, its inputs and targets brought by all-to-all.


EPS = float(np.finfo(np.float32).eps)  # the readout norm's epsilon (torch's rms_norm default)
SIGMA = 0.005  # the top cycle's step, relative to the stream's RMS at that token
FC_SIGMA = 0.05  # the MLP units' central-difference step, relative to the pre-activation RMS
KEY_SIGMA = 0.003  # the query / key coordinate step, relative to the raw query's (key's) RMS at that token
HEAD_SIGMA = 0.1  # the head's logit step
GATE_STEP = 0.1  # the gates' central-difference step on the gate logit
TOKEN_CHUNK = 1024  # tokens per readout-cycle chunk
MOMENT_DEGREE = 2  # the exact query / key experiments' polynomial in the logit change (kernels/exact_moments.py)
CHUNK = 16  # query / key directions per moment-kernel chunk


def hadamard(d):
    """The Sylvester Hadamard matrix of order d (a power of two), entries +-1."""
    h = np.ones((1, 1), np.float32)
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return jnp.asarray(h)


def flipped(key, d):
    """A full cycle of d orthogonal directions: the order-d Hadamard matrix with randomly flipped columns."""
    return hadamard(d) * jax.random.rademacher(key, (d,), F32)


def cycle_derivatives(base, c, y, steps, sq, dots, dir_sq, width):
    """Forward differences of each token's readout loss along every direction j of a cycle: (moved - clean) / step.
    base (n, V) the clean raw logits; c (P, V) row j the logits of direction j; steps (n, 1); sq (n, 1) |x|^2;
    dots (n, P) x . D_j; dir_sq (P,) |D_j|^2. The moved stream's RMS comes from these, so the head's norm is exact.

    Each moved stream's partition function from moments on tensor cores: moving along direction j changes token n's
    soft-capped head input z_v = inv0 base_v by e_jv = delta_j z_v + sigma_j c_jv (delta_j = inv_j / inv0 - 1,
    sigma_j = step inv_j), and f(z) = exp(15 tanh(z / 15)) has f(z + e) / f(z) = 1 + g1 e + g2 e^2 / 2 + g3 e^3 / 6
    with g1..g3 functions of z alone. So sum_v f(z_v + e_jv) splits into sums over v of f z^a g_k (per token) times
    c_jv^b (per direction): six [n, V] x [V, P] products (the first-order one in fp32) and three row sums. |e| <~ 0.02,
    so the dropped terms are ~1e-8 of the total; the moved difference comes out directly (log1p of the ratio)."""
    inv = jax.lax.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / width + EPS)
    inv0 = jax.lax.rsqrt(sq / width + EPS)
    delta, sigma = inv / inv0 - 1, steps * inv  # (n, P)
    z = base * inv0  # (n, V)
    t = jnp.tanh(z / SOFTCAP)
    f = jnp.exp(SOFTCAP * t)
    g1 = 1 - t * t  # derivatives of log f: g1, d1, d2
    d1 = -2 * t * g1 / SOFTCAP
    d2 = -2 * g1 * (1 - 3 * t * t) / SOFTCAP ** 2
    g2, g3 = g1 * g1 + d1, g1 * g1 * g1 + 3 * g1 * d1 + d2  # second and third derivatives of f, over f
    total = jnp.sum(f, -1, keepdims=True)  # the clean partition function
    exact = lambda w, cols: jnp.dot(w, cols.T, precision=jax.lax.Precision.HIGHEST)
    fast = lambda w, cols: jnp.dot(w.astype(BF16), cols.astype(BF16).T, preferred_element_type=F32)
    rows = lambda w: jnp.sum(w, -1, keepdims=True)
    fz, fzz = f * z, f * z * z
    ratio = (sigma * exact(f * g1, c) + delta * sigma * fast(fz * g2, c) + 0.5 * delta ** 2 * sigma * fast(fzz * g3, c)
             + 0.5 * sigma ** 2 * fast(f * g2, c * c) + 0.5 * delta * sigma ** 2 * fast(fz * g3, c * c)
             + sigma ** 3 / 6 * fast(f * g3, c * c * c)
             + delta * rows(fz * g1) + 0.5 * delta ** 2 * rows(fzz * g2) + delta ** 3 / 6 * rows(fzz * z * g3))
    clean_partition = jnp.log(total)
    base_y = jnp.take_along_axis(base, y[:, None], axis=1)
    dirs_y = c.T[y]  # (n, P)
    clean = clean_partition - softcap(base_y * inv0)
    moved = clean_partition + jnp.log1p(ratio / total) - softcap((base_y + steps * dirs_y) * inv)
    return (moved - clean) / steps


def top_credit(params, key, x, y, vocab):
    """Each token's readout credit (m, d) at the top stream x (m, d), from the whole cycle of directions."""
    width = x.shape[-1]
    head = params["lm_head"][:, :vocab].astype(F32)
    h = flipped(key, width)  # (P, d): row j is h_j
    c, dir_sq = h @ head, jnp.sum(h * h, axis=1)
    steps = SIGMA * jnp.maximum(jnp.sqrt(jnp.mean(x * x, -1, keepdims=True)), 1e-6)

    def chunk(args):
        xs, ys, st = args
        derivative = cycle_derivatives(xs @ head, c, ys, st, jnp.sum(xs * xs, -1, keepdims=True), xs @ h.T, dir_sq,
                                       width)
        return derivative @ h / width

    m = x.shape[0]
    size = min(TOKEN_CHUNK, m)
    split = lambda a: a.reshape(m // size, size, *a.shape[1:])
    return jax.lax.map(chunk, (split(x), split(y), split(steps))).reshape(m, width)


def hidden_credit(u, credit, w_out):
    """MLP hidden credit (m, hidden) for every unit: each unit's central difference through relu² (a step of FC_SIGMA
    times the pre-activation RMS), read against the block-output credit through its output row (`credit @ W_outᵀ`,
    a linear map of measured credit)."""
    step = FC_SIGMA * jnp.maximum(jnp.sqrt(jnp.mean(u * u, -1, keepdims=True)), 1e-6)
    relu2 = lambda x: jnp.square(jax.nn.relu(x))
    slope = (relu2(u + step) - relu2(u - step)) / (2 * step)
    read = jnp.dot(credit.astype(BF16), w_out.astype(BF16).T, preferred_element_type=F32)
    return slope * read


def rope32(x, cos, sin):
    """RoPE then QK-norm in float32 (process_qk without its bf16 output)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = jnp.concatenate([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], axis=-1)
    return rotated * jax.lax.rsqrt(jnp.mean(rotated * rotated, axis=-1, keepdims=True) + 1e-6)


def local_clean(cfg, params, idx, targets, shards):
    """The clean pass on this GPU's rows, and `mine`: an array of that pass [rows, t, ...] -> this GPU's tokens
    (m, ...), the batch's tokens split evenly in order. Rows divide over the GPUs, or GPUs over rows (each GPU then
    runs its whole row and keeps its part)."""
    b, t = idx.shape
    g = jax.lax.axis_index("pop")
    if b % shards == 0:
        rows = b // shards
        take = lambda a: jax.lax.dynamic_slice_in_dim(a, g * rows, rows, 0)
        mine = lambda a: a.reshape(rows * t, -1)
    elif shards % b == 0:
        per = shards // b
        part = t // per
        take = lambda a: jax.lax.dynamic_slice_in_dim(a, g // per, 1, 0)
        mine = lambda a: jax.lax.dynamic_slice_in_dim(a.reshape(t, -1), (g % per) * part, part, 0)
    else:
        raise SystemExit(f"the rows ({b}) and the GPUs ({shards}) must divide one another")
    return clean_pass(cfg, params, take(idx), take(targets)), mine


def gathered_heads(clean, targets, mine):
    """Every block's inputs for this GPU's head over the whole batch, from this GPU's rows of the clean pass and its
    tokens' attention-output targets [layers, m, d], in one bf16 all-to-all (the queries, keys, values, clean attention
    outputs and targets are bf16 already; the log-sum-exp travels as its fp32 bits). Per block: (q_raw, k_raw, values,
    out, hub [b, t, hd], lse [b, t])."""
    cfg, cache = clean.cfg, clean.cache
    h, hd, t = cfg.n_head, cfg.head_dim, clean.idx.shape[1]
    heads_of = lambda x: x.reshape(-1, h, hd).astype(BF16)  # this GPU's tokens (m, d) -> (m, h, hd)
    own = lambda x: heads_of(mine(x))
    parts = []
    for i in range(cfg.n_layer):
        pre = f"h{i}."
        values = cache[pre + "attn.c_v"].astype(F32)
        if cfg.has_ve(i):
            z = cache[pre + "attn.ve_gate"].astype(F32)
            ve = cache[f"ve{i}"].astype(F32).reshape(*z.shape, hd)
            values = values + (2 * jax.nn.sigmoid(z)[..., None] * ve).reshape(values.shape)
        lse = mine(jnp.swapaxes(cache[pre + "attn.lse"].astype(F32), 1, 2))  # (m, h)
        parts.append(jnp.concatenate([own(cache[pre + "attn.c_q"]), own(cache[pre + "attn.c_k"]), own(values),
                                      own(cache[pre + "attn.out"]), heads_of(targets[i]),
                                      jax.lax.bitcast_convert_type(lse, BF16)], -1))  # (m, h, 5 hd + 2)
    whole = jax.lax.all_to_all(jnp.stack(parts), "pop", 2, 1, tiled=True)  # [layers, n, 1, 5 hd + 2]
    whole = whole.reshape(cfg.n_layer, -1, t, whole.shape[-1])  # [layers, b, t, 5 hd + 2]
    kinds = lambda x: [x[..., j * hd:(j + 1) * hd].astype(F32) for j in range(5)]
    lse_of = lambda x: jax.lax.bitcast_convert_type(x[..., 5 * hd:], F32)
    return [(*kinds(whole[i]), lse_of(whole[i])) for i in range(cfg.n_layer)]


def head_credits(q_raw, k_raw, values, out, hub, lse, rope, window):
    """One head's exact query, key and value credits over the whole batch, every dimension: q_raw, k_raw, values,
    out (the clean attention output), hub [b, t, hd], lse [b, t] -> [3, b t, 1, hd].

    A key move changes each query's output in one term, o'_s - o_s = (a' - a)(v_t - o_s) / (1 - a + a'),
    a' = a exp(dl); a query move reweights its own output's keys. Each change is scored against the query's
    attention-output credit (hub_s . (o'_s - o_s)): one local step. The values' credit is A^T hub, exactly."""
    b, t, hd = q_raw.shape
    cos, sin = rope
    q1, k1 = (rope32(x[:, :, None], cos, sin)[:, :, 0] for x in (q_raw, k_raw))
    hub_out = jnp.sum(hub * out, -1)
    rotation = cos.reshape(t, hd // 2), sin.reshape(t, hd // 2)
    dims = jnp.arange(hd)
    chunks = hd // CHUNK

    def credit(raw, other, kernel):  # [b, t, hd]; the chunks of directions ride in the kernel's batch axis
        coef, steps = exact_keys.closed_coefficients(raw, *rotation, dims, KEY_SIGMA)  # [b, 3, t, hd]
        cols = exact_keys.rotated_columns(other, dims)  # [b, 2, t, hd]
        batch = lambda x: jnp.moveaxis(x.reshape(*x.shape[:-1], chunks, CHUNK), -2, 1).reshape(-1, *x.shape[1:-1],
                                                                                                CHUNK)
        score = kernel(q1, k1, values, hub, batch(coef), batch(cols), lse, hub_out, window, degree=MOMENT_DEGREE,
                       repeat=chunks)  # [b chunks, t, CHUNK]
        return jnp.moveaxis(score.reshape(b, chunks, t, CHUNK), 1, 2).reshape(b, t, hd) / steps

    q = credit(q_raw, k1, exact_moments.moment_query_scores)
    k = credit(k_raw, q1, exact_moments.moment_scores)
    v = exact_keys.value_credit(q1, k1, hub, lse, window)
    return jnp.stack([q, k, v]).reshape(3, b * t, 1, hd)


def batched_products(deferred):
    """[(sites, their weight gradients [k, in, out])] from the deferred (site, input (m, in), error (m, out)): one
    batched input.T @ error product per shape."""
    groups = {}
    for site, x, error in deferred:
        groups.setdefault((x.shape[1], error.shape[1]), []).append((site, x, error))
    return [([site for site, _, _ in group],
             jnp.einsum("kmi,kmo->kio", jnp.stack([x for _, x, _ in group]), jnp.stack([e for _, _, e in group])))
            for group in groups.values()]


def identity_grad(cfg, params, idx, targets, key, shards):
    """(The identity-path gradient estimate, the mean clean token loss), inside shard_map over the 'pop' axis. Each
    GPU: the clean pass on its rows, the top cycle (all directions) and the MLP steps on its tokens, one head's
    queries, keys and values over the batch (`gathered_heads`, `head_credits`; their credits back by token in one
    all-to-all), and its share of every weight gradient. The block matrices' gradients stay this GPU's partial sums
    (`anvil_blocks` reduce-scatters them); the rest are summed over the GPUs here."""
    if jax.lax.axis_size("pop") != cfg.n_head:
        raise SystemExit(f"the exact internals run one head per GPU: {cfg.n_head} GPUs")
    (b, t), vocab = idx.shape, cfg.vocab_size
    n, layers, h = b * t, cfg.n_layer, cfg.n_head
    clean, mine = local_clean(cfg, params, idx, targets, shards)
    cache = clean.cache
    rope = rotary(cfg, t)
    stream = lambda name: mine(cache[name].astype(F32))
    y, tokens = mine(clean.targets)[:, 0], mine(clean.idx)[:, 0]
    grads = jax.tree.map(lambda p: jnp.zeros(p.shape, F32), params)
    deferred = []  # (site, input, error) of the matrices: one batched product per shape at the end

    def put(site, error):
        if site == "wte" or site.startswith("ve"):
            set_grad(grads, site, jnp.zeros((cfg.padded_vocab, error.shape[-1]), F32).at[tokens].add(error))
        elif site.endswith("ve_gate"):
            set_grad(grads, site, mine(site_input(clean, site)).astype(F32).T @ error)
        else:
            deferred.append((site, mine(site_input(clean, site)).astype(F32), error))

    coordinates = coordinate_change(mine(cache["lm_head"]), y, vocab, HEAD_SIGMA)
    grads["lm_head"] = mine(cache["in.lm_head"]).astype(F32).T @ coordinates
    credit_top = top_credit(params, jax.random.split(key)[0], stream(f"h{layers - 1}.out"), y, vocab)
    lam_r, lam_0 = params["resid_lambdas"].astype(F32), params["x0_lambdas"].astype(F32)
    x0 = stream("x0")
    above = jnp.cumprod(lam_r[::-1])[::-1]  # above[i] = lam_r[i] * ... * lam_r[last]
    credits = [credit_top * (above[i + 1] if i + 1 < layers else 1.0) for i in range(layers)]  # block outputs'
    for i, credit in enumerate(credits):
        w, pre = params["blocks"][i], f"h{i}."
        put(pre + "mlp.c_proj", credit)
        put(pre + "mlp.c_fc", hidden_credit(stream(pre + "mlp.c_fc"), credit, w["mlp.c_proj"]))
        put(pre + "attn.c_proj", credit)
    targets_o = jnp.stack([(credit @ params["blocks"][i]["attn.c_proj"].astype(F32).T).astype(BF16)
                           for i, credit in enumerate(credits)])  # attention-output targets; bf16 as the kernels read
    heads = gathered_heads(clean, targets_o, mine)
    stacked = jnp.stack([head_credits(*heads[i], rope, cfg.window(i)) for i in range(layers)])  # [L, 3, n, 1, hd]
    by_token = jax.lax.all_to_all(stacked, "pop", 2, 3, tiled=True)  # [L, 3, m, h, hd]
    by_token = by_token.reshape(layers, 3, by_token.shape[2], h * cfg.head_dim)
    scalars = []
    for i, credit in enumerate(credits):
        pre = f"h{i}."
        for j, kind in enumerate("qkv"):
            put(pre + f"attn.c_{kind}", by_token[i, j])
        if cfg.has_ve(i):
            z = mine(cache[pre + "attn.ve_gate"].astype(F32))  # (m, h)
            ve = mine(cache[f"ve{i}"].astype(F32)).reshape(-1, h, cfg.head_dim)
            value_credit = by_token[i, 2].reshape(-1, h, cfg.head_dim)
            slope = (jax.nn.sigmoid(z + GATE_STEP) - jax.nn.sigmoid(z - GATE_STEP)) / GATE_STEP
            put(pre + "attn.ve_gate", slope * jnp.sum(value_credit * ve, -1))
            put(f"ve{i}", (value_credit * 2 * jax.nn.sigmoid(z)[..., None]).reshape(-1, h * cfg.head_dim))
        prev = x0 if i == 0 else stream(f"h{i - 1}.out")
        scalars.append(jnp.stack([jnp.sum(credit * prev), jnp.sum(credit * x0)]))
    put("wte", sum(lam_0[i] * credit for i, credit in enumerate(credits)) + lam_r[0] * credits[0])  # x0's credit
    for sites, value in batched_products(deferred):
        for site, g in zip(sites, value, strict=True):
            set_grad(grads, site, g)
    grads["resid_lambdas"], grads["x0_lambdas"] = jnp.stack(scalars, axis=1)
    summed = lambda path, g: g if anvil_leaf(path) else jax.lax.psum(g, "pop")
    grads = jax.tree_util.tree_map_with_path(lambda path, g, p: (summed(path, g) / n).astype(p.dtype), grads, params)
    return grads, jax.lax.pmean(mine(clean.loss).mean(), "pop")


# ---------------------------------------------------------------------------------------------------------------------
# Optimizers
# The optimizers, as record #7's: ANVIL II (modded-nanogpt PR #360, the Noise benchmark's) on the block matrices and
# torch.optim.Adam (no weight decay) on the rest, with per-group learning rates times a warmup-stable-decay schedule.
# ANVIL II runs as one batched update per matrix shape, each GPU stepping its share of the matrices.


ANVIL_KINDS = ("attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")  # not the gate
ANVIL_MAPS = (  # six quintic maps toward the polar factor
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)
POLAR_MARGIN = 1.1  # the gradient's Frobenius norm times this bounds its spectral norm before the maps (PR #16's)
ANVIL_RUN = (1194, 240, 50, 514)  # PR #360's total, warmup, cooldown and engage steps; scaled to the run length


def lr_scale(step, hp):
    """Warmup-stable-decay: linear from 1/warmup_steps to 1 over the first warmup_steps, then 1, then linear to 0 over
    the last cooldown_steps before num_iterations."""
    warmup = min(1.0, (step + 1) / hp.warmup_steps) if hp.warmup_steps else 1.0
    remaining = hp.num_iterations - step
    return warmup * (1.0 if remaining >= hp.cooldown_steps else remaining / hp.cooldown_steps)


def learning_rates(params, hp):
    def rate(path, _):
        group = str(getattr(path[0], "key", path[0]))
        if group == "wte":
            return hp.lr_wte
        if group == "value_embeds":
            return hp.lr_value_embeds
        return hp.lr_lambdas if group.endswith("lambdas") else hp.lr

    return jax.tree_util.tree_map_with_path(rate, params)


def init_optimizer(params):
    """Adam's fp32 first and second moments, and ANVIL II's state per block matrix."""
    zeros = lambda: jax.tree.map(lambda p: jnp.zeros(p.shape, F32), params)
    anvil = [{kind: anvil_state(w) for kind, w in block.items() if kind in ANVIL_KINDS} for block in params["blocks"]]
    return {"m": zeros(), "v": zeros(), "anvil": anvil}


def anvil_leaf(path):
    """Whether a parameter-tree path is a block matrix ANVIL II steps."""
    return str(getattr(path[0], "key", path[0])) == "blocks" and str(getattr(path[-1], "key", "")) in ANVIL_KINDS


def adam_update(params, grads, moments, rates, step_index, hp):
    """torch.optim.Adam without weight decay: bias-corrected moments, step_index counting from 0. The matrices ANVIL
    II steps keep their weights and (unused) moments."""
    t = step_index.astype(F32) + 1
    correction1, correction2 = 1 - hp.beta1**t, 1 - hp.beta2**t
    adam = lambda f: lambda path, old, *xs: old if anvil_leaf(path) else f(old, *xs)
    with_path = jax.tree_util.tree_map_with_path
    m = with_path(adam(lambda m, g: hp.beta1 * m + (1 - hp.beta1) * g.astype(F32)), moments["m"], grads)
    v = with_path(adam(lambda v, g: hp.beta2 * v + (1 - hp.beta2) * jnp.square(g.astype(F32))), moments["v"], grads)
    direction = lambda m, v: m / correction1 / (jnp.sqrt(v) / jnp.sqrt(correction2) + hp.adam_eps)
    directions = jax.tree.map(direction, m, v)
    params = with_path(adam(lambda p, d, lr: (p.astype(F32) - lr * d).astype(p.dtype)), params, directions, rates)
    return params, {"m": m, "v": v}


def anvil_lane_axis(w):
    """The axis ANVIL II averages its per-lane energy over: weights here are (in, out), torch's (out, in)."""
    return 0 if w.shape[1] >= w.shape[0] else 1


def anvil_state(w):
    zeros = jnp.zeros(w.shape, F32)
    return {"fast": zeros, "slow": zeros, "energy": jnp.zeros_like(zeros.mean(anvil_lane_axis(w), keepdims=True))}


def polar(g):
    """Approximate orthogonalization (the polar factor) with ANVIL_MAPS, in bf16 on the wide orientation."""
    tall = g.shape[0] > g.shape[1]
    x = g.T if tall else g
    x = (x / (POLAR_MARGIN * jnp.linalg.norm(x) + 1e-6)).astype(BF16)
    for a, b, c in ANVIL_MAPS:
        sq = x @ x.T
        x = a * x + (b * sq + c * (sq @ sq)) @ x
    return (x.T if tall else x).astype(F32)


def anvil_betas(step_index, steps):
    """(rail beta, fast beta, fast weight) at a step: the rail beta warms from 0.85 to 0.93 and cools back over the
    scaled cooldown; after the scaled engage step the fast rail runs at 0.85 and mixes 0.4385 with the slow rail."""
    total, warmup, cooldown, engage = (round(k * steps / ANVIL_RUN[0]) for k in ANVIL_RUN)
    t, lo, hi = step_index.astype(F32), 0.85, 0.93
    beta = jnp.where(t < warmup, lo + (hi - lo) * t / max(warmup, 1), hi)
    beta = jnp.where(t > total - cooldown, hi - (hi - lo) * (t - (total - cooldown)) / max(cooldown, 1), beta)
    engaged = t >= engage
    return beta, jnp.where(engaged, 0.85, beta), jnp.where(engaged, 0.4385, 1.0)


def anvil_rate(shape):
    """ANVIL II's rate factor sqrt(max(1, out / in)) of an (in, out) weight."""
    return max(1.0, shape[1] / shape[0]) ** 0.5


def anvil_step(w, g, state, lr, betas, wd, rate):
    """ANVIL II on one matrix: a fast and a slow gradient average, the polar factor of the gradient pulled toward
    their mix, per-lane energy normalization at the same norm, cautious weight decay, and the rate factor (a matrix
    and its transpose step alike, so the wide and the tall kinds of one shape batch together)."""
    beta, fast_beta, weight = betas
    g = g.astype(F32)
    fast = state["fast"] + (1 - fast_beta) * (g - state["fast"])
    slow = state["slow"] + (1 - 0.98) * (g - state["slow"])
    update = polar(g + beta * (weight * fast + (1 - weight) * slow - g))
    energy = state["energy"] + (1 - 0.9) * (jnp.square(update).mean(anvil_lane_axis(w), keepdims=True) - state["energy"])
    gained = update * jax.lax.rsqrt(jnp.maximum(energy, 1e-10))
    update = gained * (jnp.linalg.norm(update) / jnp.maximum(jnp.linalg.norm(gained), 1e-10))
    w32 = w.astype(F32)
    decay = ((slow * w32) >= 0) * w32 * (wd * lr)
    w32 = w32 - lr * rate * (update + decay)
    return w32.astype(w.dtype), {"fast": fast, "slow": slow, "energy": energy}


def stack_kind(trees, kind):
    """The blocks' trees of one weight kind stacked along a leading layer axis."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *[tree[kind] for tree in trees])


def anvil_groups(block):
    """The ANVIL kinds of a block grouped by their wide shape: [(kinds, tall flags)] (tall kinds step transposed)."""
    groups = {}
    for kind, w in block.items():
        groups.setdefault(tuple(sorted(w.shape)), []).append((kind, w.shape[0] > w.shape[1]))
    return [tuple(zip(*members, strict=True)) for members in groups.values()]


def orient(x, tall):
    return jnp.swapaxes(x, -1, -2) if tall else x


def group_stack(trees, kinds, talls):
    """The blocks' matrices (or state trees) of a group's kinds, wide side out, along one leading axis."""
    stacks = [jax.tree.map(lambda x, tall=tall: orient(x, tall), stack_kind(trees, kind))
              for kind, tall in zip(kinds, talls, strict=True)]
    return jax.tree.map(lambda *xs: jnp.concatenate(xs), *stacks)


def group_unstack(stack, kinds, talls, layers):
    """{kind: per-block list} back from a group stack."""
    return {kind: [jax.tree.map(lambda x, j=j * layers + i, tall=tall: orient(x[j], tall), stack) for i in range(layers)]
            for j, (kind, tall) in enumerate(zip(kinds, talls, strict=True))}


def owned_slice(count, shards):
    """(first, size) of this GPU's matrices in a group of `count`: count / shards when they divide, else all."""
    if shards == 1 or count % shards:
        return 0, count
    per = count // shards
    return jax.lax.axis_index("pop") * per, per


def anvil_blocks(blocks, grads, states, lr, step_index, hp, shards):
    """The block matrices' ANVIL II steps (from the weights before this step) and the new states: one batched update
    per wide shape. `grads` are each GPU's partial sums: one reduce-scatter per group gives each GPU the summed
    gradients of its share, it steps them, the new weights are gathered, and each GPU keeps only its share's states
    current (`gather_anvil_states` collects them)."""
    betas = anvil_betas(step_index, hp.num_iterations)
    layers = len(blocks)
    stepped, new_states = [{} for _ in blocks], [{} for _ in blocks]
    for kinds, talls in anvil_groups({kind: blocks[0][kind] for kind in states[0]}):
        rates = jnp.repeat(jnp.asarray([anvil_rate(blocks[0][k].shape) for k in kinds], F32), layers)
        first, per = owned_slice(len(kinds) * layers, shards)
        mine = lambda x, first=first, per=per: jax.lax.dynamic_slice_in_dim(x, first, per, 0)
        old = group_stack(states, kinds, talls)
        update = jax.vmap(lambda w, g, st, rate: anvil_step(w, g, st, lr, betas, hp.anvil_wd, rate))
        if per == len(kinds) * layers:
            raise SystemExit(f"ANVIL's groups must split across the GPUs ({kinds})")
        summed = jax.lax.psum_scatter(group_stack(grads, kinds, talls), "pop", scatter_dimension=0, tiled=True)
        w, st = update(mine(group_stack(blocks, kinds, talls)), summed, jax.tree.map(mine, old), mine(rates))
        if per < len(kinds) * layers:
            w = jax.lax.all_gather(w, "pop", axis=0, tiled=True)
            st = jax.tree.map(lambda full, part, first=first: jax.lax.dynamic_update_slice_in_dim(full, part, first, 0),
                              old, st)
        for kind, ws in group_unstack(w, kinds, talls, layers).items():
            for i, x in enumerate(ws):
                stepped[i][kind] = x
        for kind, sts in group_unstack(st, kinds, talls, layers).items():
            for i, x in enumerate(sts):
                new_states[i][kind] = x
    return stepped, new_states


def gather_anvil_states(states, shards):
    """Every matrix's ANVIL state from the GPU that steps it (inside shard_map; for checkpoints)."""
    layers = len(states)
    gathered = [{} for _ in states]
    for kinds, talls in anvil_groups({k: v["fast"] for k, v in states[0].items()}):
        first, per = owned_slice(len(kinds) * layers, shards)
        stack = group_stack(states, kinds, talls)
        if per < len(kinds) * layers:
            mine = lambda x, first=first, per=per: jax.lax.dynamic_slice_in_dim(x, first, per, 0)
            stack = jax.tree.map(lambda x, mine=mine: jax.lax.all_gather(mine(x), "pop", axis=0, tiled=True), stack)
        for kind, sts in group_unstack(stack, kinds, talls, layers).items():
            for i, x in enumerate(sts):
                gathered[i][kind] = x
    return gathered


# ---------------------------------------------------------------------------------------------------------------------
# Training
# Data, the compiled step, validation, checkpoints and the timed loop.


def train_batches(rows, batch_size, seed, max_epochs):
    """[B, T] int32 (inputs, targets) over shuffled epochs (epoch e permutes rows with rng (seed, e))."""
    for epoch in range(max_epochs):
        order = np.random.default_rng((seed, epoch)).permutation(len(rows))
        for index in range(0, len(rows) - batch_size + 1, batch_size):
            batch = np.asarray(rows[np.sort(order[index : index + batch_size])], np.int32)
            yield batch[:, :-1], batch[:, 1:]


def make_train_step(cfg, hp, mesh):
    """steps_per_call updates in one compiled call: idx, targets [K, rows, T], step_index, scale [K] -> (params,
    buffers, the K updates' losses). One dispatch per K updates keeps the host's per-call cost (hundreds of state
    arrays on 8 GPUs) off the update's critical path."""
    shards = mesh.size

    def step(params, buffers, idx, targets, step_index, scale):
        key = jax.random.fold_in(jax.random.key(hp.seed), step_index)
        grads, loss = identity_grad(cfg, params, idx, targets, key, shards)
        rates = jax.tree.map(lambda lr: scale * lr, learning_rates(params, hp))
        stepped, anvil = anvil_blocks(params["blocks"], grads["blocks"], buffers["anvil"], scale * hp.anvil_lr,
                                      step_index, hp, shards)
        params, state = adam_update(params, grads, buffers, rates, step_index, hp)
        params["blocks"] = [b | s for b, s in zip(params["blocks"], stepped, strict=True)]  # ANVIL's block matrices
        return params, state | {"anvil": anvil}, loss

    def steps(params, buffers, idx, targets, step_index, scale):
        def update(state, batch):
            params, buffers, loss = step(*state, *batch)
            return (params, buffers), loss

        (params, buffers), losses = jax.lax.scan(update, (params, buffers), (idx, targets, step_index, scale))
        return params, buffers, losses

    sharded = jax.shard_map(steps, mesh=mesh, in_specs=(P(),) * 6, out_specs=(P(),) * 3, check_vma=False)
    return jax.jit(sharded, donate_argnums=(0, 1))


def make_evaluate(cfg, mesh):
    """Summed token loss over validation rows [N, T+1], rows split over the 'pop' axis."""

    def evaluate(params, rows):
        def chunk_loss(chunk):
            return forward(cfg, params, chunk[:, :-1].astype(jnp.int32), chunk[:, 1:].astype(jnp.int32)).sum()

        total = jax.lax.map(chunk_loss, rows.reshape(-1, EVAL_CHUNK_ROWS, rows.shape[-1])).sum()
        return jax.lax.psum(total, "pop")

    sharded = jax.shard_map(evaluate, mesh=mesh, in_specs=(P(), P("pop")), out_specs=P(), check_vma=False)
    return jax.jit(sharded)


def save_checkpoint(path, params, buffers, meta):
    """Parameters p0, p1, ... and optimizer state m0, m1, ... in tree-leaf order (fp32), with `meta` as json."""
    arrays = {f"p{i}": np.asarray(x, np.float32) for i, x in enumerate(jax.tree.leaves(params))}
    arrays |= {f"m{i}": np.asarray(x, np.float32) for i, x in enumerate(jax.tree.leaves(buffers))}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, meta=json.dumps(meta), **arrays)
    tmp.replace(path)


def load_rows(hp, mesh):
    data = Path(hp.data)
    if not (data / "train.npy").exists():
        raise SystemExit(f"no train.npy in {str(data)!r}: pass data=<dir> (train/speedrun/run.py writes it)")
    train_rows = np.load(data / "train.npy", mmap_mode="r")
    if hp.train_rows > len(train_rows) or hp.train_rows < 0:
        raise SystemExit(f"train_rows={hp.train_rows}: the train split has {len(train_rows)} rows")
    train_rows = train_rows[: hp.train_rows or len(train_rows)]
    val_rows = np.load(data / "val.npy")
    if train_rows.shape[1] != hp.seq_len + 1 or val_rows.shape[1] != hp.seq_len + 1:
        raise SystemExit(f"train/val rows of {train_rows.shape[1]}/{val_rows.shape[1]} tokens must be seq_len + 1")
    if len(val_rows) % (mesh.size * EVAL_CHUNK_ROWS):
        raise SystemExit(f"val rows ({len(val_rows)}) must divide by {mesh.size * EVAL_CHUNK_ROWS}")
    return train_rows, val_rows


def main():
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")  # read when the backend starts
    jax.config.update("jax_use_shardy_partitioner", False)  # Shardy's unused-inputs pass segfaults on this step
    hp = parse_args(sys.argv[1:])
    jax.config.update("jax_default_prng_impl", hp.rng)  # before any key exists
    run_id = str(uuid.uuid4())
    root = Path(__file__).parent
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    logfile = logs / f"{run_id}.txt"

    def print0(text="", console=True):
        if console:
            print(text, flush=True)
        with open(logfile, "a") as f:
            print(text, file=f)

    print0(Path(__file__).read_text(), console=False)
    for module in SOURCES:
        print0(f"{'=' * 100}\n# {Path(module.__file__).relative_to(root)}", console=False)
        print0(Path(module.__file__).read_text(), console=False)
    print0("=" * 100, console=False)
    print0(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, console=False)
    print0(f"run {run_id} | jax {jax.__version__} | {jax.device_count()}x {jax.devices()[0].device_kind}")
    print0(json.dumps(asdict(hp)))

    mesh = Mesh(np.array(jax.devices()), ("pop",))
    replicated = NamedSharding(mesh, P())
    put = lambda x: jax.device_put(x, replicated)
    cfg = GPTConfig(hp.seq_len, hp.vocab_size, hp.n_layer, hp.n_head, hp.n_embd)
    params = jax.jit(lambda k: init_params(cfg, k), out_shardings=replicated)(jax.random.key(hp.seed))
    buffers = jax.jit(init_optimizer, out_shardings=replicated)(params)
    print0(f"parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")
    train_rows, val_rows = load_rows(hp, mesh)
    val_tokens = val_rows[:, :-1].size
    val_rows = jax.device_put(val_rows, NamedSharding(mesh, P("pop")))
    batches = train_batches(train_rows, hp.batch_size, hp.seed + 1, hp.max_epochs)

    gather_states = jax.jit(jax.shard_map(lambda st: gather_anvil_states(st, mesh.size), mesh=mesh,
                                          in_specs=P(), out_specs=P(), check_vma=False))
    step_fn = make_train_step(cfg, hp, mesh)
    calls = hp.steps_per_call
    dummy = lambda length: put(np.zeros((calls, hp.batch_size * hp.seq_len // length, length), np.int32))
    schedule = lambda first: (jnp.arange(first, first + calls, dtype=jnp.int32),
                              jnp.asarray([lr_scale(s, hp) for s in range(first, first + calls)], F32))
    compile_start = time.perf_counter()
    lengths = sorted({seq_length(s, hp) for s in range(hp.num_iterations)})
    train_steps = {length: step_fn.lower(params, buffers, dummy(length), dummy(length), *schedule(0)).compile()
                   for length in lengths}
    for length, compiled in train_steps.items():  # first runs (CUDA graphs, collectives' setup) on copies
        jax.block_until_ready(compiled(jax.tree.map(jnp.copy, params), jax.tree.map(jnp.copy, buffers),
                                       dummy(length), dummy(length), *schedule(0)))
    evaluate = make_evaluate(cfg, mesh).lower(params, val_rows).compile()
    val_loss_of = lambda p: float(evaluate(p, val_rows)) / val_tokens
    print0(f"compiled in {time.perf_counter() - compile_start:.0f}s")

    tokens_through = lambda n: hp.seq_len * hp.batch_size * n
    step, train_time, reached, final = 0, 0.0, None, None
    exhausted = False  # the training data ran out: validate once more, then stop
    kept_steps = {int(x) for x in hp.save_steps.split(",") if x}
    segment_start = None  # wall clock at the first update dispatched since the last sync point (timed stretch)
    pending = None  # (step, loss on device) of the last dispatched update, logged once the next one is in flight

    def log_update(done, losses, elapsed):
        """Log the K updates ending at `done` (their losses come back together)."""
        losses = np.asarray(losses, np.float64)
        for update, loss in enumerate(losses, start=done - len(losses) + 1):
            print0(f"step:{update}/{hp.num_iterations} train_loss:{loss:.4f} train_time:{elapsed * 1000:.0f}ms "
                   f"step_avg:{elapsed * 1000 / update:.2f}ms")
            if not math.isfinite(loss):
                raise SystemExit("non-finite train loss")

    def sync(train_time, segment_start, pending):
        """End a timed stretch: wait for its updates, add its wall time, log the last update. (train_time, None, None)"""
        if segment_start is not None:
            jax.block_until_ready(params)
            train_time += time.perf_counter() - segment_start
        if pending:
            log_update(*pending, train_time)
        return train_time, None, None

    while True:
        last = step >= hp.num_iterations or exhausted
        if last or step % hp.val_every == 0:
            train_time, segment_start, pending = sync(train_time, segment_start, pending)
            val_loss = val_loss_of(params)
            print0(f"step:{step}/{hp.num_iterations} val_loss:{val_loss:.4f} tokens:{tokens_through(step):,} "
                   f"train_time:{train_time * 1000:.0f}ms step_avg:{train_time * 1000 / max(step, 1):.2f}ms")
            if last:
                final = dict(val_loss=val_loss, passed=val_loss <= hp.target)
            if reached is None and val_loss <= hp.target:
                reached = dict(step=step, tokens=tokens_through(step), train_time=train_time, val_loss=val_loss)
                print0(f"reached target {hp.target}: {json.dumps(reached)}")
        if last:
            break
        try:
            chunk = [next(batches) for _ in range(calls)]
        except StopIteration:
            print0(f"end of training data: max_epochs={hp.max_epochs} epochs of {len(train_rows)} rows")
            exhausted = True
            continue
        if segment_start is None:
            segment_start = time.perf_counter()
        length = seq_length(step, hp)
        inputs = put(np.stack([x.reshape(-1, length) for x, _ in chunk]))  # [K, rows, length]: rows cut into pieces
        targets = put(np.stack([y.reshape(-1, length) for _, y in chunk]))
        params, buffers, loss = train_steps[length](params, buffers, inputs, targets, *schedule(step))
        step += calls
        if pending:  # the previous update's loss: waits for it only, while this one runs
            log_update(*pending, train_time + time.perf_counter() - segment_start)
        pending = (step, loss)
        if (hp.save_every and step % hp.save_every == 0) or step in kept_steps:
            train_time, segment_start, pending = sync(train_time, segment_start, pending)
            buffers = buffers | {"anvil": gather_states(buffers["anvil"])}
            meta = dict(step=step, train_time=train_time, reached=reached, hparams=asdict(hp))
            if hp.save_every and step % hp.save_every == 0:
                save_checkpoint(logs / f"{run_id}.npz", params, buffers, meta)
            if step in kept_steps:
                save_checkpoint(logs / f"{run_id}-{step}.npz", params, buffers, meta)
    print0(f"peak memory per GPU: {jax.devices()[0].memory_stats()['peak_bytes_in_use'] / 2**30:.1f} GiB")
    print0(f"result: {json.dumps(dict(reached=reached, final=final, steps=step, train_time=train_time))}")


# ---------------------------------------------------------------------------------------------------------------------
# Export
# A checkpoint of the JAX trainer as the benchmark's state dict (named float32 arrays in torch layout):
#
# The trainer's block weights are (in, out) and its head (d, vocab); torch's Linear stores (out, in), so they are
# transposed.


def noise_state(params):
    """{state-dict name: array} of `utils/model.py`'s GPT from the trainer's parameter tree."""
    state = {"transformer.wte.weight": params["wte"], "lm_head.weight": params["lm_head"].T,
             "resid_lambdas": params["resid_lambdas"], "x0_lambdas": params["x0_lambdas"]}
    for layer, block in enumerate(params["blocks"]):
        state |= {f"transformer.h.{layer}.{kind}.weight": w.T for kind, w in block.items()}
    state |= {f"value_embeds.{i}.weight": w for i, w in params["value_embeds"].items()}
    return {name: np.asarray(value, np.float32) for name, value in state.items()}


def export(path, out):
    hp = parse_args([])
    cfg = GPTConfig(hp.seq_len, hp.vocab_size, hp.n_layer, hp.n_head, hp.n_embd)
    template = init_params(cfg, jax.random.key(0))
    data = np.load(path)  # the checkpoint's parameters are its arrays p0, p1, ... in the tree's leaf order
    leaves = [np.asarray(data[f"p{i}"], np.float32) for i in range(len(jax.tree.leaves(template)))]
    params = jax.tree.unflatten(jax.tree.structure(template), leaves)
    step = json.loads(str(data["meta"]))["step"]
    np.savez(out, meta=json.dumps({"step": step, "source": str(path)}), **noise_state(params))
    print(f"wrote {out} (step {step})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        export(*sys.argv[2:])
    else:
        main()
