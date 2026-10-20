"""The exact query and key experiments of kernels/exact_keys.py as attention-weighted moments on tensor cores.

Each experiment moves one raw query (key) coordinate j at one position; its logit change against a key (query) is
dl = A L + U + V, with L the clean logit, U = Bu X and V = Bw Y, X and Y the other side's two rotated columns that j
touches (scaled by d^-1/2), and A, Bu, Bw per moved position and direction (`exact_keys.closed_coefficients`). The
attention's response is a polynomial in dl (exp(dl) - 1, and ch / (1 + ch) for a key, to `degree`), so each
direction's sum over the other positions splits into monomials (A L)^m U^p V^q: per monomial, a weight that depends
on the (query, key) pair only (a c L^m, a L^m) times features that depend on the other position and j only (X^p Y^q).
The sums over positions are then matrix products [moved, other] x [other, 2J] on tensor cores, and the coefficients
(A^m Bu^p Bw^q) apply once per moved position at the end. The trainer uses degree 2: with |dl| <~ 0.01 the cubic
terms are ~1e-6 of each result, below the products' bf16 rounding (the sums are fp32).

J directions per call (a power of two >= 8). Feature groups (2J columns each, the directions' X or Y side by side):
[X | Y], [X^2 | Y^2], [XY | XY], [X^3 | Y^3], [X^2 Y | X Y^2]; a duplicated monomial's coefficient is halved, and the
two halves are summed after the kernel. With `repeat`, the coefficients' and columns' row i reads row i // repeat of
the shared inputs, so chunks of directions ride in the batch axis without copies.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltr

F32, BF16 = jnp.float32, jnp.bfloat16
GROUPS = 5  # feature groups: [X|Y], [X²|Y²], [XY|XY], [X³|Y³], [X²Y|XY²]


def features(cols, d):
    """Feature groups [BH, 5, T, 2J] (bf16) from the other side's rotated columns [BH, 2, T, J] (f32), scaled by
    d^-1/2 (X, Y in the logit's units)."""
    x, y = cols[:, 0] * d ** -0.5, cols[:, 1] * d ** -0.5
    side = lambda a, b: jnp.concatenate([a, b], -1)
    groups = [side(x, y), side(x * x, y * y), side(x * y, x * y), side(x * x * x, y * y * y),
              side(x * x * y, x * y * y)]
    return jnp.stack(groups, 1).astype(BF16)


def coefficients(coef):
    """[BH, 3, T, 2J] f32: [A | A], P = [Bu | Bw], Q = [Bw | Bu] from closed_coefficients' coef [BH, 3, T, J]."""
    a, bu, bw = coef[:, 0], coef[:, 1], coef[:, 2]
    side = lambda x, y: jnp.concatenate([x, y], -1)
    return jnp.stack([side(a, a), side(bu, bw), side(bw, bu)], 1).astype(F32)


def _group_coefficients(p, q):
    """Per feature group, the monomial coefficients with 1/(p! q!) (duplicated XY halved): [P, P²/2, PQ/2, P³/6,
    P²Q/2]."""
    return [p, 0.5 * p * p, 0.5 * p * q, p * p * p / 6.0, 0.5 * p * p * q]


GROUP_DEGREE = (1, 2, 2, 3, 3)  # p + q of each feature group


def _query_terms(degree):
    """(power m of L, feature group) of each product in a query sum, total degree m + p + q <= degree."""
    return [(m, g) for m in range(degree) for g in range(GROUPS) if m + GROUP_DEGREE[g] <= degree]


def _key_terms(degree):
    """(degree n, power m of L, feature group) of each product in a key sum."""
    return [(n, m, g) for n in range(1, degree + 1) for m in range(n) for g in range(GROUPS)
            if m + GROUP_DEGREE[g] == n]


def _power(x, m):
    out = jnp.ones_like(x)
    for _ in range(m):
        out = out * x
    return out


def _check(directions, t, block_t, block_s):
    if directions < 8 or directions & (directions - 1) or t % block_t or t % block_s:
        raise ValueError(f"needs a power of two >= 8 directions ({directions}) and T {t} dividing by the blocks "
                         f"{block_t} / {block_s}")


_mm = lambda x, y: jax.lax.dot_general(x, y, (((1,), (0,)), ((), ())), preferred_element_type=F32)  # x @ y
_mt = lambda x, y: jax.lax.dot_general(x, y, (((1,), (1,)), ((), ())), preferred_element_type=F32)  # x @ yᵀ


def _query_kernel(q_ref, k_ref, v_ref, hub_ref, coef_ref, feat_ref, lse_ref, hub_out_ref, out_ref, *, window,
                  block_t, width, degree):
    d = k_ref.shape[1]
    block_s = q_ref.shape[0]
    scale = d ** -0.5
    s0 = pl.program_id(1) * block_s
    q, hub = q_ref[...], hub_ref[...]
    lse, hub_out = lse_ref[...], hub_out_ref[...]
    query_pos = s0 + jnp.arange(block_s)
    first = jnp.maximum(s0 - window, 0) // block_t
    last = (s0 + block_s - 1) // block_t + 1

    terms = _query_terms(degree)
    groups = range(max(g for _, g in terms) + 1)

    def tile(i, totals):  # totals: (numerator, denominator), each [block_s, 2J] products and [block_s] sums
        t0 = i * block_t
        k = k_ref[pl.ds(t0, block_t), :]
        v = v_ref[pl.ds(t0, block_t), :]
        lag = query_pos[:, None] - (t0 + jnp.arange(block_t))[None, :]
        allowed = (lag >= 0) & (lag <= window)
        logits = _mt(q, k) * scale
        a = jnp.where(allowed, jnp.exp(logits - lse[:, None]), 0.0)
        c = _mt(hub, v) - hub_out[:, None]
        f = [feat_ref[g, pl.ds(t0, block_t), :] for g in groups]
        new = []
        for weight, (products, sums) in zip((a * c, a), totals, strict=True):
            powers = [weight]  # weight L^m
            for _ in range(degree):
                powers.append(powers[-1] * logits)
            step = [_mm(powers[m].astype(BF16), f[g]) for m, g in terms]
            new_sums = [jnp.sum(powers[m], axis=1) for m in range(1, degree + 1)]
            new.append((tuple(x + y for x, y in zip(products, step, strict=True)),
                        tuple(x + y for x, y in zip(sums, new_sums, strict=True))))
        return tuple(new)

    zero = lambda shape: jnp.zeros(shape, F32)
    side = lambda: (tuple(zero((block_s, width)) for _ in terms), tuple(zero((block_s,)) for _ in range(degree)))
    totals = jax.lax.fori_loop(first, last, tile, (side(), side()))
    a16 = coef_ref[0]  # [block_s, 2J]
    g = _group_coefficients(coef_ref[1], coef_ref[2])
    power = lambda m: _power(a16, m) / math.factorial(m)
    weights = [power(m) * g[k] for m, k in terms]  # A^m Bu^p Bw^q / (m! p! q!)
    halves = [0.5 * power(m) for m in range(1, degree + 1)]  # the L^m sums' A^m / m!, halved
    for side_index, (products, sums) in enumerate(totals):
        z = sum(p * w for p, w in zip(products, weights, strict=True))
        z = z + sum(s[:, None] * h for s, h in zip(sums, halves, strict=True))
        out_ref[side_index] = z


@functools.partial(jax.jit, static_argnames=("window", "block_t", "block_s", "num_warps", "num_stages", "degree",
                                             "repeat"))
def moment_query_scores(q, k, v, hub, coef, kcols, lse, hub_out, window, block_t=64, block_s=64, num_warps=4,
                        num_stages=2, degree=3, repeat=1):
    """`exact_keys.query_reference` for J directions: q, k, v, hub [BH, T, D] (processed rows), coef
    [BH repeat, 3, T, J] (closed_coefficients of the raw queries), kcols [BH repeat, 2, T, J] (rotated_columns of the
    keys), lse, hub_out [BH, T] -> [BH repeat, T, J] f32. With `repeat`, row i of the coefficients and columns reads
    row i // repeat of the others (chunks of directions in the batch axis, no copies)."""
    bh, t, d = coef.shape[0], *q.shape[1:]
    directions = coef.shape[-1]
    width = 2 * directions
    _check(directions, t, block_t, block_s)
    b16 = lambda x: x.astype(BF16)
    row = lambda: pl.BlockSpec((None, t, d), lambda i, j: (i // repeat, 0, 0))
    block = lambda: pl.BlockSpec((None, block_s, d), lambda i, j: (i // repeat, j, 0))
    vector = lambda: pl.BlockSpec((None, block_s), lambda i, j: (i // repeat, j))
    z = pl.pallas_call(
        functools.partial(_query_kernel, window=window, block_t=block_t, width=width, degree=degree),
        grid=(bh, t // block_s),
        in_specs=[block(), row(), row(), block(),
                  pl.BlockSpec((None, 3, block_s, width), lambda i, j: (i, 0, j, 0)),
                  pl.BlockSpec((None, GROUPS, t, width), lambda i, j: (i, 0, 0, 0)), vector(), vector()],
        out_specs=pl.BlockSpec((None, 2, block_s, width), lambda i, j: (i, 0, j, 0)),
        out_shape=jax.ShapeDtypeStruct((bh, 2, t, width), F32),
        compiler_params=pltr.CompilerParams(num_warps=num_warps, num_stages=num_stages),
        name="moment_queries",
    )(b16(q), b16(k), b16(v), b16(hub), coefficients(coef), features(kcols, d), lse.astype(F32),
      hub_out.astype(F32))
    pairs = z[..., :directions] + z[..., directions:]  # [BH, 2, T, J]
    return pairs[:, 0] / (1.0 + pairs[:, 1])


def _key_kernel(q_ref, k_ref, v_ref, hub_ref, coef_ref, feat_ref, lse_ref, hub_out_ref, out_ref, *, window,
                block_s, width, degree):
    t_total, d = q_ref.shape
    block_t = k_ref.shape[0]
    scale = d ** -0.5
    t0 = pl.program_id(1) * block_t
    k, v = k_ref[...], v_ref[...]
    key_pos = t0 + jnp.arange(block_t)
    first = t0 // block_s
    last = jnp.minimum((t0 + block_t - 1 + window) // block_s + 1, t_total // block_s)

    terms = _key_terms(degree)
    groups = range(max(g for _, _, g in terms) + 1)

    def tile(i, totals):  # totals: [block_t, 2J] products and [block_t] sums
        products, sums = totals
        s0 = i * block_s
        q = q_ref[pl.ds(s0, block_s), :]
        hub = hub_ref[pl.ds(s0, block_s), :]
        lse = lse_ref[pl.ds(s0, block_s)]
        hub_out = hub_out_ref[pl.ds(s0, block_s)]
        lag = (s0 + jnp.arange(block_s))[None, :] - key_pos[:, None]
        allowed = (lag >= 0) & (lag <= window)
        logits = _mt(k, q) * scale  # [block_t, block_s]
        a = jnp.where(allowed, jnp.exp(logits - lse[None, :]), 0.0)
        c = _mt(v, hub) - hub_out[None, :]
        # ch / (1 + ch) = a dl + (a/2 - a²) dl² + (a/6 - a² + a³) dl³: the weight of degree n, times L^m
        by_degree = [a * c, (0.5 * a - a * a) * c, (a / 6.0 - a * a + a * a * a) * c][:degree]
        powers = [[w] for w in by_degree]
        for n, row in enumerate(powers, start=1):
            for _ in range(n):
                row.append(row[-1] * logits)
        f = [feat_ref[g, pl.ds(s0, block_s), :] for g in groups]
        step = [_mm(powers[n - 1][m].astype(BF16), f[g]) for n, m, g in terms]
        new_sums = [jnp.sum(powers[n - 1][n], axis=1) for n in range(1, degree + 1)]
        return (tuple(x + y for x, y in zip(products, step, strict=True)),
                tuple(x + y for x, y in zip(sums, new_sums, strict=True)))

    zero = lambda shape: jnp.zeros(shape, F32)
    init = (tuple(zero((block_t, width)) for _ in terms), tuple(zero((block_t,)) for _ in range(degree)))
    products, sums = jax.lax.fori_loop(first, last, tile, init)
    a16 = coef_ref[0]  # [block_t, 2J]
    g = _group_coefficients(coef_ref[1], coef_ref[2])
    ratio = lambda n, m: math.factorial(n) / math.factorial(m)
    weights = [ratio(n, m) * _power(a16, m) * g[k] for n, m, k in terms]  # n! / (m! p! q!) A^m Bu^p Bw^q
    halves = [0.5 * _power(a16, n) for n in range(1, degree + 1)]  # the L^n sums (m = n): A^n, halved
    z = sum(p * w for p, w in zip(products, weights, strict=True))
    out_ref[...] = z + sum(s[:, None] * h for s, h in zip(sums, halves, strict=True))


@functools.partial(jax.jit, static_argnames=("window", "block_t", "block_s", "num_warps", "num_stages", "degree",
                                             "repeat"))
def moment_scores(q, k, v, hub, coef, qcols, lse, hub_out, window, block_t=64, block_s=64, num_warps=4,
                  num_stages=2, degree=3, repeat=1):
    """`exact_keys.reference` for J directions: coef [BH repeat, 3, T, J] (closed_coefficients of the raw keys), qcols
    [BH repeat, 2, T, J] (rotated_columns of the queries) -> [BH repeat, T, J] f32; `repeat` as in
    moment_query_scores."""
    bh, t, d = coef.shape[0], *q.shape[1:]
    directions = coef.shape[-1]
    width = 2 * directions
    _check(directions, t, block_t, block_s)
    b16 = lambda x: x.astype(BF16)
    row = lambda: pl.BlockSpec((None, t, d), lambda i, j: (i // repeat, 0, 0))
    z = pl.pallas_call(
        functools.partial(_key_kernel, window=window, block_s=block_s, width=width, degree=degree),
        grid=(bh, t // block_t),
        in_specs=[row(), pl.BlockSpec((None, block_t, d), lambda i, j: (i // repeat, j, 0)),
                  pl.BlockSpec((None, block_t, d), lambda i, j: (i // repeat, j, 0)), row(),
                  pl.BlockSpec((None, 3, block_t, width), lambda i, j: (i, 0, j, 0)),
                  pl.BlockSpec((None, GROUPS, t, width), lambda i, j: (i, 0, 0, 0)),
                  pl.BlockSpec((None, t), lambda i, j: (i // repeat, 0)),
                  pl.BlockSpec((None, t), lambda i, j: (i // repeat, 0))],
        out_specs=pl.BlockSpec((None, block_t, width), lambda i, j: (i, j, 0)),
        out_shape=jax.ShapeDtypeStruct((bh, t, width), F32),
        compiler_params=pltr.CompilerParams(num_warps=num_warps, num_stages=num_stages),
        name="moment_keys",
    )(b16(q), b16(k), b16(v), b16(hub), coefficients(coef), features(qcols, d), lse.astype(F32),
      hub_out.astype(F32))
    return z[..., :directions] + z[..., directions:]
