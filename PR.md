# Draft PR: track 2 (30M) record, 3.5 in ~23 s

Draft text for a PR from `RobertTLange:track2-fused-cut` to `qlabs-eng/noise` `main` (not opened).

**Title:** Track 2 (30M): 3.4989 in 22.7 s — #51's identity phase + Cut with two single roots, fused Gluon chain kernels and overlapped bf16 collectives (seed 0, one 8xH100 node)

---

## Result

**3.49886 in 22.71 s** with seed 0 (the log in `train/speedrun/log/`; PR_LOG_NOTE). The current record (#44) is
3.4988 in 499.6 s (8.3 min), so this is ×22 faster. It builds on the open PR #51 (3.4943 in 123.2 s), which is
×5.4 slower than this. Identity phase 1.87 s, end merge 0.14 s.

- **Command:** `uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py`, the default command with no
  flags.
- **Hardware variance:** the same code runs in 22.7–23.2 s across our four 8xH100 nodes and over the day (±0.2 s).
  Final val spreads ±0.0015 from run to run. 13 of 15 runs at this identity count passed.

![From PR #51 to this PR](https://raw.githubusercontent.com/RobertTLange/noise/track2-fused-cut-figures/pr-records.gif)

## What changed

This builds on PR #51: a 300-update identity phase, then Cut with paired MLP roots. Each Cut update is one CUDA
graph. Here:

1. **A different Cut estimator: two single roots.**
   - #51 replays roots in pairs ((0,1), (2,4), (6,7)). The partner moves add noise: block 0's credit is ~100%
     estimator noise (the identity-path control explains ~0%), and paired roots measured worse than single ones.
   - This PR replays **two single roots, blocks 4 and 6**. Each moves the mid stream along a full 512-direction
     signed Hadamard cycle and replays it through the routed MLPs to the readout. The root at block 4 also replays
     attention at block 5 (16 strongest keys per moved query).
   - The deeper blocks get their credit through the residual path (the identity-path control), as in #51.
   - MLP routes: the strongest 64 per token from block 2 on (128 before), computed only for the blocks the replays
     visit.
   - The query credit and the readout probe come from the degree-2 moment model (no probes).
   - Going from #51's roots to these two cut the Cut update by ~35% at the same final val (56.5 → 36.0 s across
     records #81–#88).
2. **Fused chain kernels** (Triton 3.6 Gluon on sm_90: TMA, wgmma). Each 64-move tile's 64 KB delta tile stays in
   shared memory instead of making a 1 GB HBM round trip per block:
   - `visit_chain3.py`: the block-4 replay's blocks 5 (with the attention replay's response), 6 and 7 (readout) in
     one kernel, 2.73 ms vs 5.04 ms for the Triton visits.
   - `visit_chain.py`: the block-6 root and block 7's readout in one kernel.
   - `visit_gluon_root.py`: the block-4 root, which also writes its 64-wide routed change.
   - `query_factored_gluon.py`: block 5's query change, δ·W_qᵀ = scale·(change·A_t + step·F_m) with
     A_t = Wo[routes_t]·W_qᵀ per token and F = frame·W_qᵀ. That is a K = 64 GEMM per token instead of K = 512
     against W_q: 0.71 ms vs 1.75 ms (cuBLAS + replay).
   - `moment_kernels.py`, `window_topk.py`, `project_values.py`: the key / query-moment / value-credit kernels.
     They write straight into the layout the gradient reassembly reads. The strong-key top-16 is bitwise equal to
     `torch.topk` (including tie order), and the routed-value projection is blocked.
3. **Leaner glue.** These are compiled or batched:
   - the capture forward (one q/k/v GEMM, weights cast to bf16 once per estimate);
   - the descending value chain and the weight credits (4 batched GEMMs instead of 48);
   - the readout probe's moments.
   The flat gradient buffer is packed inside the estimate graph.
4. **The update.**
   - ANVIL's matrices are stacked by shape. Each rank updates 1/8 of every batch (reduce-scatter → polar → all-gather),
     and each rank's two same-shape polars run as one.
   - The gradient collectives are bf16, issued async and waited where first read, so they run under the optimizer
     math.
   - AdamW is fused, and the next update's gate rows are drawn under the running estimate.
5. **The identity phase** (#51's, unchanged in method), 355 updates (300):
   - deduplicated moment kernels, a fused readout-cycle pass, and one-pass all-to-all packing and head prep;
   - `identity_grad` reordered so its two all-to-alls overlap independent work;
   - a second graph without the shaping steps after update 195, where they are exact no-ops.
   - The identity phase takes 1.87 s for 355 updates (#51: 2.06 s for 300).
6. **Schedule.**
   - Adam 0.0028, ANVIL 0.036 (identity ANVIL 0.04, decay 0.09).
   - Warmdown over the last 50% to 0.5 of the peak.
   - The end merge averages from 30% of the rows, one curvature setting, 2 decays × 2 coefficients, 4 keys, no
     sweeps (0.14 s).

The time per Cut update fell from 72 ms (#51's recipe on our nodes) to 14.1 ms, and the end merge from 3.5 s to 0.14 s.

## How we got here

We got here over 67 internal records, each a single seed-0 fixed-length run on this track's rules. The time panel
above shows them all; the milestones:

| # | Train time | Change |
| --- | --- | --- |
| 45 | 116.1 s | PR #51 with 32 query draws (our baseline run of #51) |
| 55 | 79.9 s | 64 MLP routes on the late blocks |
| 65 | 70.3 s | Gluon chain visits |
| 78 | 59.8 s | fused query replay (q-projection and replay in one kernel) |
| 81 | 56.5 s | faster moment kernels, batched value credits |
| 86 | 38.9 s | single roots at blocks 2 and 6 (no partners) |
| 88 | 36.0 s | single roots at blocks 4 and 6, attention replayed at block 5 |
| 93 | 31.9 s | async bf16 gradient collectives |
| 97 | 29.4 s | the replay response fused into the chain tail kernel |
| 99 | 26.7 s | fused glue (credit layouts, batched weight credits, compiled probe) |
| 102 | 25.6 s | factored query replay |
| 105 | 24.4 s | flat gradients in the graph, exact window top-k |
| 108 | 23.6 s | faster identity phase |
| 111 | 22.7 s | each rank's ANVIL polars batched |

**What didn't work:**
- Partial frames, token subsets and a learned low-rank frame lost too much: the deviation the replays measure isn't
  low-rank.
- An attention replay at block 4 behind a root at ≤ 3 also lost.
- Replaying no late root collapsed val.
- Applying each update one step late (to overlap the update with the next estimate) cost +0.03 val.
- Side-stream overlap of the optimizer math lost time on 8 GPUs (contention with the in-flight collectives).

## Exactness

Every systems change was gated at a fixed checkpoint and RNG. The gradient estimate matched the previous code at
per-tensor cosine 1.000000; most changes are bitwise, the others differ by fp32 summation order. The factored query
replay skips two bf16 roundings of δ: its weights move by ≤ 1 bf16 ulp, as much as the reference moves under one
fewer rounding of δ.

## Rules

- **No backprop.**
  - Every learning signal is a forward evaluation of the loss and takes at most one step through a nonlinear part.
  - The chain replays are #51's (finite moves along full Hadamard cycles, rerun to the loss).
  - Credit between blocks is carried linearly (the residual λ).
  - The moment models approximate forward evaluations, not derivatives through the network.
  - The factored query replay is an algebraic identity of the same forward move.
- **Fixed model and data.** The final weights load strictly into `utils/model.py`. Each training sequence feeds at
  most one update across both phases.
- **Timing.**
  - One `torchrun train/speedrun/run.py` process (helper kernel files next to it) and one timed region, from the
    first identity update to the end of the merge.
  - All compilation and CUDA-graph capture (both identity graphs, the Cut estimate graph, the merge forward) happens
    on random tokens before the timer.

## Reproducing

```bash
uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py
```

torch 2.10.0 (cu128), Triton 3.6 (Gluon kernels: Hopper / sm_90 only). The switches are `Settings` fields and
module constants with the record's values as defaults.
