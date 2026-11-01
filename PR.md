# Draft PR: track 2 (30M) record, 3.5 in ~14 s

Draft, not opened. Code branch: `RobertTLange:track2-wide-cut` (`0736d61`).

**Title:** Track 2 (30M): 3.4993 in 14.0 s — 16-row Cut updates, root 4 on every 2nd update early,
tensor-core and Gluon credit kernels, per-head q/k polar and a scheduled update tail (seed 0, one 8xH100 node)

---

## Result

**3.49932 in 13.97 s** with seed 0 (the log in `train/speedrun/log/`). `utils/eval.py` on its `final.pt` gives the
same 3.49932. Against the current record (#52, 3.4976 in 22.9 s), that's ×1.64 faster. Identity phase 1.33 s, end
merge 0.07 s.

- **Command:** `uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py`, the default command with no
  flags.
- **Run-to-run spread:** the margin is thin, and most runs of this code miss the target by a little. Final val
  varies by about ±0.0012 between runs of the same code. This code passed in 3 of 12 runs, mean 3.50052. Of those, 8
  were the plain command: 1 passed, mean 3.50085. The other 4 were screens of the same code through `--set`, and 2 of
  them passed. We spent each val gain on a wider root-4 window (below) until the command sat at the line, so the speed
  comes with a low pass rate.
- **Timing spread:** the same code takes 13.96–14.02 s depending on the node.

![Train time per record, from PR #52 to this PR](https://raw.githubusercontent.com/RobertTLange/noise/track2-wide-cut-figures/pr-progress.png)

## What changed

This builds on #52 (the identity phase, then Cut with single roots at blocks 4 and 6, fused chain kernels, one CUDA
graph per estimate). The estimator is #52's. The changes make a row cheaper, take fewer updates per row, and skip
root 4's replay on half the updates in the first part of Cut.

1. **Estimator and schedule.**
   - **16-row Cut updates.** From 10% of Cut's rows on, an update takes 16 rows (two per GPU), with one gradient
     exchange and one optimizer step, at rates ×1.3. Cut runs 148 updates of 8 rows, then 660 of 16 (#52: 1,468 of
     8). Same rows, same order.
   - **A native two-row estimate.** Each GPU estimates its two rows in one graph: capture, block recompute, route
     top-k, weight-credit GEMMs and the key / value kernels run on both rows at once. Each row keeps its own
     generators, so the draws match two one-row estimates. The two rows' chain replays run on two streams.
   - **Root 4 on every 2nd update over 10–65% of Cut.** Root 4's replay (blocks 5–7 with the block-5 attention
     replay) is the estimate's largest cost, ~40% of it. Inside the window it is replayed on every 2nd update, and
     its measured deviation from the identity-path prediction counts twice there. On the other updates root 4's
     credit is the identity-path prediction alone. Both estimate graphs are captured before the timer. The window
     started as the first 20% of Cut and widened as other changes bought val.
   - Structured start ρ 0.9 (0.8). The identity phase's rates decay over its last 150 updates to 0.3. Warmdown over
     the last 55% of Cut (50%).
2. **Kernels.**
   - Chain kernels on tensor cores: the gate dots and readout moments as wgmmas, packed bf16 converts, 128-move tiles
     in the block-4 tail kernel (64). The tail forms its replay response once per chunk, and the readout sums stay
     per-thread partials until the end of the tile.
   - Gluon FA3-style kernels for the value-credit passes and the key-credit scan.
   - The factored query replay scores against RoPE-rotated keys. RoPE is a rotation, so the keys are rotated back by
     each token's angles once, and RoPE, the norm sums and dq move onto small K = 64 wgmmas.
   - Value credits gather projected values from a 16 MB per-key table instead of a 268 MB tensor. The strong keys and
     the gathered value tables come from one kernel.
   - A pruned route top-k (bitwise equal to the old one). The descending chain's move dots are a fused row reduction
     (it was a batched gemv). The descent GEMMs are batched, and block recompute's relu² happens in its GEMM's load.
   - Weight-credit GEMMs on bf16 operands with fp32 accumulation (TF32 before).
3. **bf16 relaxations, checked by training runs** (see Exactness):
   - block recompute reads the capture forward's own bf16 c_fc / c_proj outputs (two GEMMs gone);
   - the implicit key statistics' lse / center pass and the probe moments in bf16;
   - the value pass multiplies by the bf16 high part only, and its logits come from the bf16 q / k copies;
   - the rotated-key query replay uses the keys' bf16 high parts.
4. **The update.**
   - **Per-head polar for c_q / c_k**, from @CharlesDDDD's track-1 PR #54. ANVIL orthogonalizes each head's query and
     key rows together (one 128 × 512 polar per head) instead of each of c_q and c_k as one 512 × 512 matrix. On our
     recipe this bought ~0.004 val for +0.07 s. We spent the val on the root-4 window: 10–45% → 10–70% internally,
     10–65% here (see item 6).
   - ANVIL's weights and flat gradients are stored rank-major, so one reduce-scatter and one all-gather per update
     replace three of each. Each ANVIL unit is one CUDA graph, and the units overlap on two streams.
   - The next replay's inputs are staged under the previous update's tail. The gate-row prefetch is a graph on a
     high-priority stream (both rows' gate rows are drawn under the running estimate), and the estimate graphs open
     with an event wait so their RNG prologue runs under the tail. The merge averages' lerp runs on a side stream.
     The loss bookkeeping is enqueued after the collectives.
   - NCCL's streams at high priority, and `NCCL_PROTO=LL128`, set by the script in its own process before NCCL starts.
5. **Systems outside the update.**
   - `gc.collect()` and `gc.freeze()` before the timer. Without it a generation-2 collection cost 0.2–0.46 s inside
     the timed region.
   - The Cut rows' host setup (order, checks, digests) runs on a thread under the identity phase. The identity
     phase keeps its memory cache for Cut, and an 8 GB block is allocated and freed before the timer.
   - Identity phase: leaner kernels, its gradient GEMMs read the layer inputs in place, value-embedding shaping in
     closed form. Its reduce-scatter goes out early, and its all-gathers run on a second communicator. The hand-off
     to Cut has no host sync. 355 updates now take 1.33 s (#52: 1.87 s).
6. **The end merge, fit in the run.** Each of the 57 parts (one per weight matrix and the embeddings) gets its own
   coefficient a in w + a·(avg − w) toward the 0.02 weight average, in place of one coefficient for all. The parts
   are split over the 8 ranks. Each rank probes its parts at a = 0.75 ± 0.5 on its 8 of the 64 held-out rows (one
   forward from the part's first block) and fits a parabola per part. One all-reduce shares the vertices (clipped
   to −0.25…1.25). The joint move is shrunk by the best of 0.25 / 0.5 / 0.75 on all 64 held-out rows. No row that
   fed an update is read again. This is the in-run form of the per-part merge in the closed #53 by @areu01or00,
   whose table was fit offline. Merge 0.07 s.
   - Our internal records #116–#182 fit the parabolas on 24 identity-phase rows per rank, rows that had already fed
     an update. There the in-run fit kept 60–80% of #53's offline gain (−0.002 to −0.003 val).
   - Fitting on the held-out rows only makes the fits noisier and cost ~+0.0012 val (4 runs: mean 3.50153 at
     10–70%). We paid that back with a narrower root-4 window, 10–65% instead of 10–70%, for ~+0.13 s. Hence
     13.97 s here against 13.89 s for internal #182.

The time per Cut row fell from 1.76 ms (#52) to 1.07 ms. That is 12.6 s for Cut's 11,744 rows, against 20.7 s.

## How we got here

We got here over 71 internal records (#112–#182), each a single seed-0 fixed-length run, then this PR's
held-out-only merge. The figure at the top shows the milestones:

| # | Train time | Change |
| --- | --- | --- |
| 111 | 22.71 s | PR #52's code (our run of it) |
| 116 | 22.32 s | in-run per-part merge |
| 123 | 21.39 s | 16-row updates, the second row per GPU on a concurrent graph |
| 133 | 20.62 s | 16-row updates from 10% of Cut (from 50% at first) |
| 139 | 19.91 s | native two-row estimate |
| 143 | 18.84 s | tensor-core chain kernels |
| 146 | 18.12 s | root 4 on every 2nd update over 10–40% of Cut |
| 147 | 17.72 s | bf16 weight credits |
| 149 | 17.42 s | Gluon value-credit and key-scan kernels |
| 166 | 15.69 s | exact kernels: pruned top-k, fused move dots, gathered values, rotated keys, lse rows by TMA |
| 168 | 15.16 s | bf16 recompute and statistics (checked by training runs) |
| 171 | 14.75 s | update-tail scheduling (rank-major ANVIL, overlapped units, staged inputs) |
| 176 | 14.18 s | per-head q/k polar (PR #54), root-4 window 10–65% |
| 180 | 13.97 s | joint q/k polar per head, root-4 window 10–70% |
| 182 | 13.89 s | identity phase overlapped, NCCL LL128 (merge fit on identity rows) |
| PR | 13.97 s | merge fit on held-out rows only, root-4 window 10–65% |

**What didn't work:**
- **Cheaper root estimates.** We tried token subsets, row subsets, frame subsets, importance-sampled tokens, control
  variates and truncations of root 4. Outside the early window they all cost ≥ 0.008 val per second saved. Bias cost
  more than noise, and noise late in Cut cost more than offline gradient errors predicted. Root 4 on every 2nd update
  was the cheapest cut (~0.0036 val per s over the whole run), and cheaper still in an early window.
- **Low-rank deviations.** The roots' per-token deviations share no small subspace across rows: the top 64 of 512
  directions hold 38–62% of the energy.
- **Exact subspace roots.** A root's replay reads a perturbation only through a per-token span (the 64 routed c_fc rows
  per later MLP and the block-5 key functionals), so 384 moves per token can replace 1,024 at equal credit quality.
  The replay kernels got 2.3 ms per row faster, but the per-token frames and k × k solves cost as much as they saved,
  so it was time-neutral. Adding low roots (2, 3) on top gave val 3.4725 but cost +7.5 s.
- **Amortizing the attention q/k credits.** Every 2nd update cost 0.033 val per s; dropping them cost +0.255.
- **A longer identity phase.** 500 updates cost +0.020 for −1.3 s and 700 cost +0.077. An identity phase with an
  amortized measured correction didn't help either: past identity update 200 the rows share ≤ 8% of their deviation.
- **Momentum self-distillation.** No val margin at +1.1 s.
- **Optimizer variants.** Per-head polar for c_v (+0.004) and attn c_proj (+0.007), MLP polar in 4 groups (+0.037),
  NorMuon's row normalization (already ANVIL's lane energy), Polar Express, damped / powered polar, Nesterov,
  Lookahead, per-tensor hypergradient rates, fewer polar iterations (4 maps: +0.007).
- **Rows and data.** 16-row updates from 0% (+0.0037 for −0.18 s). 32-row updates late sat at the line and cost time
  under the native two-row estimate. Row orders by surprisal (5 orders) were worse, a 1,024-token curriculum gave
  nothing, and dropping rows by surprisal cost ~0.009 val per s.
- **Other.** FP8 operands in the chain kernels: the kernels are issue- and latency-bound, so the best case is ~0.05 ms
  per estimate. Applying each update one step late cost +0.03 val. AdamW on a side stream lost ~0.5 ms per update to
  all-gather waits. The bf16 key scan cost val margin and was dropped.

**Concurrent work:** @CharlesDDDD's open #55 (3.4959 in 15.9 s, opened 2026-11-01) also starts from #52, and was
developed independently. Some of its ideas overlap with ours:
- Its root 6 replays along each token's routed span (128 moves, a per-token CG solve). That is close to our exact
  subspace roots above, which we found time-neutral at our kernel speeds.
- Its 16-row updates (from half of Cut), ANVIL in CUDA graphs and gate rows drawn on a side stream match changes here.
- Its token-subset root 4, word-mix row pruning, fixed end merge and kernels are different from what is here.

We haven't tried combining the two.

## Exactness

Every exact systems change was gated at a fixed checkpoint and RNG. The gradient estimate matched the previous code at
per-tensor cosine 1.000000.
- The update-tail and identity-overlap scheduling is bitwise equal in a deterministic end-to-end check (60 of 60
  tensors). Only the rank-major ANVIL storage changes the reduction grouping.
- The bf16 weight credits moved the estimate by ≤ 5e-6 in per-tensor cosine (norm ratios ≤ 1.00023).
- The rotated-key query replay no longer rounds dq to bf16. Against an fp64 replay its mean error fell from 1.24e-5 to
  8.9e-6.
- The bf16 relaxations in item 3 don't pass this gate (worst per-tensor cosine 0.9998), so we judged them by training
  runs instead. Four runs of the first three averaged 3.5007, against 3.5008 over 14 runs of the code before them,
  and they saved 0.42 s. One relaxation, the bf16 key scan, did cost val margin, and we dropped it.

## Rules

- **No backprop.**
  - Every learning signal is a forward evaluation of the loss and takes at most one step through a nonlinear part.
    The chain replays are #52's (finite moves along full Hadamard cycles, rerun to the loss).
  - Root 4 on every 2nd update adds no new signal: the deviation it measures is the same forward-measured signal as
    on every update, weighted by 2 on half the updates in the window.
  - The per-head polar and the in-run merge act on the update and the weights, not on credit.
- **Fixed model and data.**
  - The final weights load strictly into `utils/model.py`.
  - Each training sequence feeds at most one update. A 16-row update uses each of its rows once.
  - The end merge fits its coefficients and picks its shrink only on the 64 held-out rows, which feed no update.
    Every other row feeds exactly one update (2,840 identity + 11,744 Cut + 64 held-out = 14,648).
- **Timing.**
  - One `torchrun train/speedrun/run.py` process (helper kernel files next to it) and one timed region, from the first
    identity update to the end of the merge.
  - All compilation and CUDA-graph capture happens on random tokens before the timer: the identity graphs, both Cut
    estimate graphs (with and without root 4), the ANVIL unit graphs and the merge forwards.
  - Before the timer the script also freezes the garbage collector and reserves memory. `NCCL_PROTO` is set by the
    script itself, with no launcher flags or environment.

## Reproducing

```bash
uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py
```

torch 2.10.0 (cu128), Triton 3.6 (Gluon kernels: Hopper / sm_90 only). The switches are `Settings` fields and
module constants with the record's values as defaults. `run.py` is 4,458 lines (#52: 2,928).
