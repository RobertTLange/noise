"""relu(x @ w)² with the activation in the matmul epilogue (Hopper, Pallas Mosaic GPU).

The MLP's first layer: XLA runs it as a cuBLAS matmul with a fused relu, then squares in a separate kernel (a full
extra read + write of the [rows, 4d] activation). Here the square happens on the fp32 accumulator before the bf16
store. Adapted from jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py (Apache-2.0, The JAX Authors):
warp-specialized TMA pipeline, two compute warpgroups issuing wgmma, persistent grid over output tiles.
"""

import dataclasses
import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas.ops.gpu.hopper_matmul_mgpu import MatmulDimension, get_num_sms


def reference(x, w):
    """Plain-jnp relu(x @ w)² as the trainer computed it (bf16 matmul output, square in bf16)."""
    return jnp.square(jax.nn.relu(jnp.dot(x.astype(jnp.bfloat16), w.astype(jnp.bfloat16))))


@dataclasses.dataclass(frozen=True)
class Tiling:
    """Defaults: the fastest tiling for [262144, 512] x [512, 2048] on H100 (from a tiling sweep)."""

    tile_m: int = 64
    tile_n: int = 256
    tile_k: int = 64
    max_concurrent_steps: int = 4
    epi_tile_m: int = 64
    epi_tile_n: int = 64
    grid_minor_dim: MatmulDimension = MatmulDimension.M
    grid_tile_width: int = 4
    wg_dimension: MatmulDimension = MatmulDimension.M


def _kernel(a_gmem, b_gmem, out_gmem, tiling: Tiling):
    dtype, out_dtype = a_gmem.dtype, out_gmem.dtype
    m, k = a_gmem.shape
    _, n = b_gmem.shape
    tile_m, tile_n, tile_k = tiling.tile_m, tiling.tile_n, tiling.tile_k
    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
    transforms = (plgpu.TilingTransform((8, swizzle // jnp.dtype(dtype).itemsize)), plgpu.SwizzleTransform(swizzle))
    cta_tile_m = tile_m * (1 + (tiling.wg_dimension == MatmulDimension.M))
    cta_tile_n = tile_n * (1 + (tiling.wg_dimension == MatmulDimension.N))
    if m % cta_tile_m or n % cta_tile_n or k % tile_k:
        raise ValueError(f"({m}, {n}, {k}) must divide by ({cta_tile_m}, {cta_tile_n}, {tile_k})")
    m_iters, n_iters, k_iters = m // cta_tile_m, n // cta_tile_n, k // tile_k
    epi_tile_m, epi_tile_n = tiling.epi_tile_m, tiling.epi_tile_n
    num_out_slots = min(2, (tile_m * tile_n) // (epi_tile_m * epi_tile_n))
    out_swizzle = plgpu.find_swizzle(epi_tile_n * jnp.dtype(out_dtype).itemsize * 8)
    out_transforms = (
        plgpu.TilingTransform((8, out_swizzle // jnp.dtype(out_dtype).itemsize)),
        plgpu.SwizzleTransform(out_swizzle),
    )

    def get_pipeline(pipeline_body, compute_context):
        return plgpu.emit_pipeline_warp_specialized(
            pipeline_body,
            grid=(k_iters,),
            memory_registers=40,
            in_specs=[
                plgpu.BlockSpec((cta_tile_m, tile_k), lambda k: (0, k), transforms=transforms, memory_space=plgpu.SMEM),
                plgpu.BlockSpec((tile_k, cta_tile_n), lambda k: (k, 0), transforms=transforms, memory_space=plgpu.SMEM),
            ],
            wg_axis="wg",
            num_compute_wgs=2,
            max_concurrent_steps=tiling.max_concurrent_steps,
            compute_context=compute_context,
        )

    ignore = lambda *_, **__: None

    @functools.partial(
        pl.run_scoped,
        pipeline_allocs=get_pipeline(ignore, ignore).get_allocations(a_gmem, b_gmem),
        out_smem=plgpu.SMEM((2, num_out_slots, epi_tile_m, epi_tile_n), out_dtype, transforms=out_transforms),
        collective_axes="wg",
    )
    def _pipeline_scope(pipeline_allocs, out_smem):
        wg_idx = lax.axis_index("wg")

        @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
        def _mn_loop(loop_info: plgpu.NDLoopInfo):
            (lin_idx,) = loop_info.index
            m_idx, n_idx = plgpu.planar_snake(
                lin_idx, (m_iters, n_iters), tiling.grid_minor_dim, tiling.grid_tile_width
            )
            cta_m_slice = pl.ds(m_idx * cta_tile_m, cta_tile_m)
            cta_n_slice = pl.ds(n_idx * cta_tile_n, cta_tile_n)
            if tiling.wg_dimension == MatmulDimension.M:
                wg_m_slice, wg_n_slice = pl.ds(wg_idx * tile_m, tile_m), slice(None)
            else:
                wg_m_slice, wg_n_slice = slice(None), pl.ds(wg_idx * tile_n, tile_n)

            def compute_context(eval_pipeline):
                @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC((tile_m, tile_n), jnp.float32))
                def _acc_scope(acc_ref):
                    eval_pipeline(acc_ref)
                    acc = acc_ref[...]
                    acc = jnp.square(jnp.maximum(acc, 0.0)).astype(out_dtype)  # the fused relu² epilogue
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    for epi_mi in range(tile_m // epi_tile_m):
                        for epi_ni in range(tile_n // epi_tile_n):
                            epi_m_slice = slice(epi_mi * epi_tile_m, (epi_mi + 1) * epi_tile_m)
                            epi_n_slice = slice(epi_ni * epi_tile_n, (epi_ni + 1) * epi_tile_n)
                            slot = (epi_mi * (tile_n // epi_tile_n) + epi_ni) % 2
                            plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                            out_smem[wg_idx, slot] = acc[epi_m_slice, epi_n_slice]
                            plgpu.commit_smem()
                            tile_out = out_gmem.at[cta_m_slice, cta_n_slice].at[wg_m_slice, wg_n_slice]
                            plgpu.copy_smem_to_gmem(out_smem.at[wg_idx, slot], tile_out.at[epi_m_slice, epi_n_slice])

            def mma_body(_, a_smem, b_smem, acc_ref):
                plgpu.wgmma(acc_ref, a_smem.at[wg_m_slice], b_smem.at[:, wg_n_slice])
                plgpu.wgmma_wait(0)
                return acc_ref

            get_pipeline(mma_body, compute_context)(
                a_gmem.at[cta_m_slice, :], b_gmem.at[:, cta_n_slice], allocations=pipeline_allocs
            )

    plgpu.wait_smem_to_gmem(0, wait_read_only=True)


@functools.partial(jax.jit, static_argnames=("tiling",))
def relu2_matmul(x, w, tiling=Tiling()):
    """relu(x @ w)² in bf16: x [M, K], w [K, N] (cast to bf16), fp32 accumulate and activation."""
    x, w = x.astype(jnp.bfloat16), w.astype(jnp.bfloat16)
    kernel = plgpu.kernel(
        functools.partial(_kernel, tiling=tiling),
        out_type=jax.ShapeDtypeStruct((x.shape[0], w.shape[1]), jnp.bfloat16),
        grid=(get_num_sms(),),
        grid_names=("cluster_grid",),
        cluster=(1,),
        cluster_names=("cluster",),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return kernel(x, w)
