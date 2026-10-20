# Track 2 record #8: the JAX trainer

`train/speedrun/run.py` is the record. It keeps the benchmark's harness (pinned data, the timing rule,
`utils/eval.py` on the exported weights) and runs the JAX trainer in this folder as one process on all 8 GPUs.

```text
run.py ─► train.py ─────────────────────────────► train.py export ─► utils/eval.py
          (recipe, data, warmup, timer, log;       (checkpoint →       (score)
           one update = the estimate + Adam /      utils/model.py
           ANVIL II's step)                        state dict)
```

`train.py` (990 lines) holds everything but the Pallas kernels, in this order:

| Section | Lines | What it holds |
| --- | ---: | --- |
| Recipe | 42–104 | The record's hyperparameters as defaults (`Hyperparameters`), the row-length schedule |
| Model | 106–356 | The benchmark's model, forward only, with the clean pass's cache the estimator reads; the head's credit (`coordinate_change`) |
| Estimator | 358–573 | **Identity-path credit** (`identity_grad`): the top readout cycle (`top_credit`), the MLP hidden credit (`hidden_credit`), the queries, keys, values and gates (`attention_credit`, `exact_blocks`) |
| Optimizers | 575–769 | Adam, and ANVIL II batched by matrix shape with each GPU stepping its share |
| Training | 771–957 | Data, the compiled step (`make_train_step`), untimed warmup, timer, logging, checkpoints (`main`) |
| Export | 959–990 | The final checkpoint as `utils/model.py`'s state dict (strict load): `python train.py export <ckpt> <out>` |

| Kernel | Lines | Role |
| --- | ---: | --- |
| `kernels/exact_keys.py` | 278 | Exact single-coordinate query / key experiments with closed-form logit changes, and the values' credit, fused |
| `kernels/probe_cycle.py` | 85 | Partition functions of the readout under every moved stream (the top credit) |
| `kernels/residual_norm.py`, `rope_norm.py`, `relu2_matmul.py`, `head_ce.py` | 403 | The forward pass |

**Reading order:** `identity_grad` → `top_credit` → `hidden_credit` → `attention_credit` → `exact_blocks` →
`kernels/exact_keys.py` (its docstring gives the score each experiment computes) → `make_train_step` → `main`
(warmup and timer).

## Credit paths and the no-backprop rule

The rule: every error signal is measured against the loss by running the model forward, and may then take at most
one step through a nonlinear part. Steps through linear maps don't count. In `identity_grad`, every signal starts at
one of two measurements and ends at a weight gradient after at most one nonlinear step:

| Parameters | Credit | Linear maps | Nonlinear step |
| --- | --- | --- | --- |
| `lm_head` | each logit coordinate moved ±σ, the loss's change evaluated exactly (`coordinate_change`) | `inputᵀ ·` | none |
| top stream (`h7.out`) | **measured**: one sign-flipped Hadamard cycle (512 directions) moves each token's stream, and the readout (norm, head, softcap, CE) is evaluated forward on each move (`top_credit`, `kernels/probe_cycle.py`) | | none |
| every block's output stream, `x0` | the top credit, read down the residual path | × residual scalars | none (the blocks above are skipped, not stepped through) |
| `mlp.c_proj`, `attn.c_proj`, residual scalars | the stream credit | `inputᵀ ·` | none |
| `mlp.c_fc` | `credit @ W_outᵀ` per hidden unit | `W_outᵀ`, `inputᵀ ·` | relu²: one central-difference slope per unit |
| `attn.c_v` | each value moved by one coordinate, its attention-output change scored against `hub = credit @ W_projᵀ` (exactly `Aᵀ · hub`) | `W_projᵀ`, `Aᵀ` (A from the forward pass) | none (the output is linear in the values) |
| `attn.ve_gate`, `value_embeds` | the value credit along the value embedding | | sigmoid gate: one central-difference slope (`value_embeds`: none) |
| `attn.c_q`, `attn.c_k` | each raw coordinate moved (RoPE, per-head RMS norm, softmax, all evaluated exactly in `kernels/exact_keys.py`), the attention-output change scored against `hub` | `W_projᵀ` | the attention: one forward experiment per coordinate |
| `wte` | the stream credit at `x0` | gathered per token | none (not stepped through the embedding norm) |

Nothing is chained. No credit reaches a block through another block's weights or nonlinearities: the only
cross-block route is the residual stream's identity path. Each local step's result ends at its weight gradient and
is never read back through `W_fcᵀ` or `W_{q,k,v}ᵀ`. The trainer has no code path that does. The query / key step is
the same kind of experiment as the exact keys of records #6 and #7: it moves one coordinate and evaluates the
attention forward, so a non-differentiable attention would still work.

## Timing and warmup

`main` compiles every step shape (1024- and 2048-token rows) ahead of time. It runs each once on dummy tokens
and on *copies* of the parameters and optimizer state (CUDA graphs and collectives get set up), and only then starts
the timer on the untouched state. The timer covers the first update to the end of the last. Training never
validates (`val_every=1000000`). The `step:0 val_loss` line in the trainer log is computed before the timer starts.
Compile and warmup took 23 s on the record run (8× H100 80GB HBM3, JAX 0.11.2, CUDA 13). `run.py` took about
1 min 40 s end to end: data cache, compile and warmup, 15.35 s of training, export, `utils/eval.py`.

## Reading `log/trainer.txt`

- It opens with the source of every module the run imported (the trainer embeds it so a log reproduces its run),
  then `nvidia-smi`, the hyperparameters, and one `step:N/975 train_loss:… train_time:…ms` line per timed update
  (`run.py` copies them into `metrics.jsonl`).
- The trainer's own final val (4.17966) matches `utils/eval.py` on the exported weights (4.17968).

## Reproducibility

Runs are not bitwise reproducible; we haven't pinned down which GPU reduction is nondeterministic. Two seed-0 runs
match the training loss to 4 decimals for the first updates, then drift at the 1e-4 level. Adam's normalization
magnifies the tiny gradient differences in near-zero coordinates, so single weights differ visibly between runs
(`lm_head` up to 0.3 of its max after 4 updates), while the val loss after 20 updates agrees to 1e-4.

**The trim.** This trainer is our research trainer cut down to this entry's path (it also carried the Dust, Boost and
Relay estimators, control variates and other optimizers). The two agree within that run-to-run noise. Over 20
updates at seed 0, every parameter group differs between them by as much as between two runs of either. Training
loss is equal to 4 decimals through update 6. Val at update 20 is 6.1070 and 6.1069 for each trainer's two runs.

**Seeds.** Seeds 42–46 ran the untrimmed trainer: 15.98 ± 0.07 s, val 4.163 ± 0.016, all five below 4.2 (one-sided
t-test, p = 0.003). The record run (trimmed, seed 0) gives 4.1797 in 15.35 s. The untrimmed trainer's seed-0 run took
15.93 s on another node.
