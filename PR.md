# Draft PR: track 3 (30M) record, 3.2115

Draft text for a PR from `RobertTLange:track3-ngram-record` to `qlabs-eng/noise` `main` (not opened).

**Title:** Track 3 (30M): 3.2115, a 12 × 1024 GPT with hashed n-gram value embeddings, trained by Cut-lite with input-key probes (seed 0, one 8xH100 node)

---

## Result

**3.21149** with seed 0 (the log in `train/arch/log/`), against the current track-3 record's 3.4095: −0.198.
Training took 10,604.9 s (2 h 57 min) on one 8xH100 node, end merge included (30.0 s).

- **Command:** `uv run torchrun --standalone --nproc_per_node=8 train/arch/run.py`, the default command with no
  flags.
- **For scale:** backprop on the fixed 8 × 512 model reaches 3.419 with Adam and 3.340 with ANVIL II (same data, one
  epoch). This entry trains a larger model (613.9M parameters, 402.7M of them in lookup tables), so the comparison
  isn't like for like. It does show that a forward-only learner can make use of a model this size within one epoch.

![Track-3 records](https://raw.githubusercontent.com/RobertTLange/noise/track3-ngram-record-figures/pr-progress.png)

## What changed

**The model** (`train/arch/model.py`; `GPTConfig()`'s defaults are the trained architecture):

| | Current record | This PR |
| --- | --- | --- |
| Blocks × width | 8 × 512, 8 heads | **12 × 1024, 16 heads of 64** |
| Value embeddings | every other block | **every block** |
| Hashed n-gram tables | none | **bigram, trigram and skip-bigram** tables (2^20 rows × 128 each), summed and projected into every block's value embedding |
| Parameters | 37.7M | 613.9M (tables 402.7M, blocks 151M, value embeddings 50.3M) |

For token t with value-embedding table E_l and projection P_l in block l:
`ve_l = E_l[x_t] + P_l (T2[h(x_{t−1}, x_t)] + T3[h(x_{t−2}, x_{t−1}, x_t)] + S[h(x_{t−2}, x_t)])`. `h` is a
multiplicative hash of the token ids, causal by construction: a row reads only the current and earlier tokens. The
projections start at zero.

**The training** (`train/arch/run.py`):

1. **Cut over every update, at 8 rows per update, without Boost.** This is the track-2 record's Cut-lite with its
   exact speedups: bf16 chain moves, readout moments, the fused query kernel, the norm fold, 8-key merge sweeps and
   ×750 / ×15 Adam multipliers on the embedding tables and residual scalars.
2. **Input-key probes along a full 1,024-direction cycle**, with no future-token probes.
3. **A warmdown over the last 60% of the updates** (was 40%).
4. **Credit for the tables.** The value path is linear, so each block's value error is exact once Cut has measured
   it. The value-embedding tables, the n-gram projections (an outer product) and the n-gram rows (a row update
   through the shared projections) all get exact credits from it. Each table adds 134M parameters and no reruns:
   the time per update is unchanged.
5. **The end merge**, scored on the last 64 training sequences, which feed no update.

![Validation curves](https://raw.githubusercontent.com/RobertTLange/noise/track3-ngram-record-figures/pr-curves.png)

## How we got here

We developed this over nine internal records, each a single seed-0 run on this track's rules. The figure above shows
each one's validation curve.

| # | Val | Change | Δ |
| --- | --- | --- | --- |
| 0 | 3.4095 | the current track-3 record (fixed 8 × 512, Boost → Cut with all probes, end merge) | |
| 1 | 3.4067 | 12 blocks, trained by Cut-lite at 8 rows per update (no future-token or input-key probes) | −0.003 |
| 2 | 3.3650 | width 1024, 16 heads | −0.042 |
| 3 | 3.3333 | value embeddings on every block | −0.032 |
| 4 | 3.2984 | 12 blocks at width 1024 | −0.035 |
| 5 | 3.2705 | input-key probes (512 directions), 60% warmdown | −0.028 |
| 6 | 3.2642 | the key probes along the full 1,024-direction cycle | −0.006 |
| 7 | 3.2269 | a hashed bigram table projected into every block's value embedding | −0.037 |
| 8 | 3.2168 | a trigram table, summed with it | −0.010 |
| 9 | **3.2115** | a skip-bigram table (the token two back, the token) | −0.005 |

- **Size first.** Under Cut a bigger model learns more per row at a fixed 30M tokens. Width 1024 beats 12 blocks,
  which beats the fixed model, and depth and width add about independently. Cut's chain credits are exact along
  each measured direction, so a larger model costs time, not estimate quality. Width 2048 and 16 blocks gained
  little per update (+0.004 at 16 × 1024).
- **Linear paths with exact credit.** The largest gains after size came from parameters that enter the network
  linearly and are credited exactly from a measured error, at no rerun cost:
  - value embeddings on every block;
  - the n-gram tables, which hold context that a value embedding (a function of x_t alone) can't.
- **Measured noise.** Two runs of one configuration at 8 × 1024 ended 0.0015 apart. #9's margin over #8 is about
  three times that, from one seed. It was level with #8 through 1,250 updates and ahead from there.

![Architecture screens](https://raw.githubusercontent.com/RobertTLange/noise/track3-ngram-record-figures/pr-screens.png)

**What didn't work.** The figure gives Δ against the matched baseline at each model size:

- **Held at 8 × 512, lost at 12 × 1024 (stopped early):**
  - memory slots (persistent key/values per attention). At 8 × 512 the gain turned out to be an attention sink: a
    single slot frozen at init gave most of it.
  - a fourth table, (t, t−2, t−3), on top of the skip-bigram.
- **Lost at 8 × 512:**
  - hash-routed MLP experts;
  - a depth hypernetwork (block matrices mixed from a shared bank);
  - per-block readouts (deep supervision);
  - a bilinear MLP (its estimate is as good as relu²'s; the units are weaker);
  - decayed linear attention;
  - residual-stream n-gram writes;
  - a 4-gram table.
- **Lost at every size tried:** an n-gram logit head (the tables read out directly into the logits). Its estimate
  matches autograd on a trained checkpoint, so it loses on training dynamics, not on the estimate.
- **Small screens didn't predict the record size.** The skip-bigram table won at 8 × 512, lost 0.027 at 8 × 1024 and
  won at 12 × 1024 with the full key cycle, so the record attempt ran next to the screen rather than after it.
- **Recipe changes that lost:** future-token probes on top of the key probes, 4 rows per update, rates ×0.7 / ×1.4,
  and the tables at ×0.3 / ×0.5 / ×3 rates, width 64 / 256, 2^19 / 2^21 rows.

## Rules

- **No backpropagation.** No autograd and no backward pass; the gradient buffers only receive the estimate for the
  optimizers. Every credit is measured by running the model forward against the loss and takes at most one step
  through a nonlinear part:
  - The chain credits, head coordinate differences and input-key probes are the previous records' (finite
    differences, rerun to the loss).
  - The readout moments approximate a forward evaluation, as in the merged track-2 record.
  - The new tables' credits pass only through linear maps (the value path, the gate's clean value, the projections).
- **Model.** `train/arch/model.py` keeps `utils/eval.py`'s interface: `GPT(GPTConfig())`, `init_weights()` and the
  summed loss from `forward`. The final weights load strictly (checked on `final.pt`).
- **Data.** 14,584 training sequences feed one update each. The last 64 of the 14,648 feed none and only score the
  end merge.
- **Score.** `val_loss` is `utils/eval.py`'s `evaluate()` on the final weights with this `model.py`. The validations
  every 125 updates are logged only; nothing is selected on them.
- **Seed and warmup.** Seed 0. All compilation runs on random tokens before the timer, then the model, optimizer and
  estimator are rebuilt from the seed.

## Reproducing

```bash
uv run torchrun --standalone --nproc_per_node=8 train/arch/run.py
uv run utils/eval.py logs/arch/final.pt --model train/arch/model.py
```

torch 2.10.0 (cu128), Triton from it. Peak memory is 48.9 GB in training and 58.8 GB in the end merge.
