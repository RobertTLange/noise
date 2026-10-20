"""Track 2: reach validation loss TARGET on 8xH100 in the shortest time, without backpropagation (JAX).

Identity-path credit, as the previous record, in JAX (`jax/train.py`, see `jax/README.md`): one Hadamard cycle
measures each token's credit at the top stream (its partition functions from moments on tensor cores), every earlier
stream gets it through the residual path, and every site takes one local step from it: the writers directly, the MLP
hidden layer for every unit (relu² slope times `credit @ W_outᵀ`), the queries and keys by single-coordinate
experiments evaluated exactly (each logit change in closed form, the sums as attention-weighted moments on tensor
cores), the values, gates and value embeddings in closed form. Each GPU runs the clean pass and the estimate for its
own rows and one head's queries, keys and values over the batch; ANVIL II steps the block matrices batched by shape
and split across the GPUs.

This script keeps the benchmark's harness: it reads the pinned `data/train.pt` and `data/val.pt` (utils/prepare_data),
trains with the JAX trainer in a subprocess (one process, 8 GPUs; the timer runs from the first update to the end of
the last, after an untimed warmup of every compiled step on dummy tokens and a fresh state), exports the final
weights to `utils/model.py`'s state dict and scores them with `utils/eval.py`.

    uv pip install -r train/speedrun/jax/requirements.txt   # JAX next to the benchmark's torch, after uv sync
    uv run python train/speedrun/run.py                     # one process, all 8 GPUs
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.eval import evaluate
from utils.prepare_data import TRAIN_ROWS, VAL_ROWS, load_rows

SEED = 0
TARGET = 4.2
JAX = Path(__file__).resolve().parent / 'jax'
STEPS = 975
STEP_LINE = re.compile(r'^step:(\d+)/\d+ train_loss:([\d.]+) train_time:(\d+)ms')


def write_rows(data_dir, cache):
    """The pinned splits (hash-checked by load_rows) as the trainer's uint16 arrays."""
    cache.mkdir(parents=True, exist_ok=True)
    for name, rows, out in (('train.pt', TRAIN_ROWS, 'train.npy'), ('val.pt', VAL_ROWS, 'val.npy')):
        np.save(cache / out, load_rows(data_dir / name, rows).numpy().astype(np.uint16))


def shown(path):
    """A path as written to the logs: relative to the repository when it is inside it."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/speedrun'))
    args = parser.parse_args()
    args.output, args.data_dir = args.output.resolve(), args.data_dir.resolve()  # the trainer runs in jax/
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / 'rows'
    write_rows(args.data_dir, cache)

    # the recipe is train.py's defaults; validation only at the end (outside the timer)
    overrides = [f'seed={SEED}', f'num_iterations={STEPS}', f'data={cache}', f'save_steps={STEPS}',
                 'val_every=1000000']
    (args.output / 'config.json').write_text(json.dumps(dict(
        script=shown(__file__), trainer='train/speedrun/jax/train.py', world=8, batch=4, steps=STEPS,
        seed=SEED, target=TARGET, overrides=[f'data={shown(cache)}' if o.startswith('data=') else o for o in overrides],
    ), indent=2) + '\n')

    env = os.environ | {'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.85'}
    run_id, train_seconds = None, None
    with (args.output / 'metrics.jsonl').open('w') as log, subprocess.Popen(
            [sys.executable, '-u', str(JAX / 'train.py'), *overrides], cwd=JAX, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            if line.startswith('run '):
                run_id = line.split()[1]
            match = STEP_LINE.match(line)
            if match:
                record = dict(step=int(match[1]), train_loss=float(match[2]),
                              train_seconds=round(int(match[3]) / 1000, 2))
                log.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
            elif line.startswith('result: '):
                train_seconds = json.loads(line[len('result: '):])['train_time']
            elif 'Error' in line or 'Traceback' in line:
                print(line, end='', flush=True)
    if proc.returncode or run_id is None or train_seconds is None:
        raise SystemExit(f'training failed (exit {proc.returncode})')

    shutil.copy(JAX / 'logs' / f'{run_id}.txt', args.output / 'trainer.txt')  # the trainer's log, with its source
    weights = args.output / 'final.npz'
    checkpoint = JAX / 'logs' / f'{run_id}-{STEPS}.npz'
    subprocess.run([sys.executable, str(JAX / 'train.py'), 'export', str(checkpoint), str(weights)],
                   cwd=JAX, env=env | {'JAX_PLATFORMS': 'cpu'}, check=True)
    arrays = np.load(weights)
    state = {name: torch.from_numpy(arrays[name]) for name in arrays.files if name != 'meta'}
    torch.save(state, args.output / 'final.pt')
    val_loss = evaluate(state, args.data_dir)
    result = dict(val_loss=val_loss, train_seconds=round(train_seconds, 2), steps=STEPS,
                  tokens=STEPS * 4 * 2048, reached_target=val_loss <= TARGET, trainer_log='trainer.txt')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
