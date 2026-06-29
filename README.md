# wonntext

A denoising masked-language model based on the Winfree Oscillatory Neural
Network (WONN) Sudoku configuration from [Jiawen-Dai/WONN](https://github.com/Jiawen-Dai/WONN),
and a controlled study comparing it against modern transformer baselines on
arithmetic reasoning at matched compute.

The original WONN solves 9×9 Sudoku with a single Winfree layer, `T` recurrent
steps, and attention coupling on a 2-D grid. WONNText keeps that oscillator core
but makes four changes:

1. **Input** — token ids are embedded and the embedding directly becomes the
   oscillator's natural frequency `Ω_init`. Phases `Θ_init` are random `N(0, σ²)`.
2. **Coupling** — the 2-D grid becomes a 1-D token sequence with bidirectional
   attention and RoPE.
3. **Output head** — the final phase is mapped through `sin θ / cos θ` features to
   vocabulary logits, tied to the input embedding.
4. **Objective** — the Sudoku constraint loss becomes masked cross-entropy.

## The 3×3 study

We compare three architectures at three FLOP budgets (≈2M / 5M / 10M WONN
params), all sharing a modern, FLOP-efficient stack (RoPE, RMSNorm, bias-free
linears, Muon optimiser; the transformers add SwiGLU):

| | matches WONN on | note |
|---|---|---|
| **WONNText** | — | weight-tied oscillator (anchor) |
| **Universal Transformer** | params **and** FLOPs | one tied block × `T` steps |
| **Classical Transformer** | FLOPs only | untied → ~`T`× the params |

The task is online-generated `+`/`−` arithmetic (2–4 operands, 1–3 digit) over a
space far larger than any model's memorisation capacity, so **overfitting is
structurally impossible**. Evaluation is on two disjoint held-out sets:
in-distribution and 5-operand **extrapolation**.

## Project layout

```text
src/wonntext/
  layers.py       RMSNorm, RoPE, bias-free attention, SwiGLU block
  winfree.py      WONN oscillator layer + sequence attention coupling
  model.py        WONNText model and tied readout
  baseline_transformer.py   Classical + Universal transformers
  muon.py         Muon optimiser (+ AdamW for embeddings/norms)
  math_data.py    online +/- sampler + held-out set builder
  train_math.py   online (--steps) trainer with dual eval
  data.py         MLM collator + synthetic dataset (tests)
  utils.py        seeding helpers
scripts/
  prepare_math.py       build the held-out test sets
  make_experiments.py   compute the FLOP-matched 3×3 configs -> job spec
  ray_runner.py         dispatch jobs across GPUs via Ray
experiments/
  math_3x3.json   generated Ray job spec (9 cells × seeds)
```

## Quick start (local smoke)

```bash
uv run python scripts/prepare_math.py --out_dir data/math
uv run python -m wonntext.train_math --data_dir data/math --model wonn \
  --ch 64 --T 4 --heads 4 --steps 200 --device cpu
```

## Running the study on the GPU box (sophon, 8×H100)

Compute runs on a sudo-less Ray cluster on `sophon` (no SkyPilot/k8s). The repo
is the source of truth here; `sophon` only ever pulls.

```bash
# 1. here: commit + push, then on sophon pull (agent-forwarded SSH for git)
ssh -A -o ControlPath=none sophon 'cd /nvme/theo/wonntext && git pull'

# 2. on sophon: build held-out sets + regenerate the job spec
ssh sophon 'source /nvme/theo/env.sh && cd /nvme/theo/wonntext \
  && /nvme/theo/venv/bin/python scripts/prepare_math.py --out_dir data/math \
  && PYTHONPATH=src /nvme/theo/venv/bin/python scripts/make_experiments.py --seeds 0 1 2'

# 3. dispatch across the 8 GPUs (1 GPU/job, queued); watch the Ray dashboard
ssh sophon 'source /nvme/theo/env.sh && cd /nvme/theo/wonntext \
  && RAY_ADDRESS=auto /nvme/theo/venv/bin/python scripts/ray_runner.py experiments/math_3x3.json'
```

Per-job logs land in `/nvme/theo/logs/`, checkpoints + `metrics.json` in
`/nvme/theo/wonntext/runs/math/<exp>/`. Everything on sophon lives under
`/nvme/theo` (home is small); HuggingFace is reached via `hf-mirror.com`.

WONNText must train in **fp32** (`--amp false`): bf16 floors the oscillator's
phase precision and silently caps accuracy on exact-output tasks.

## Tests

```bash
just check   # lint + typecheck + pytest
```

## Vendor provenance

Original WONN code from `Jiawen-Dai/WONN` is vendored in `vendor/WONN/` for
reference. The implementation in `src/wonntext/` derives from its Sudoku
configuration. Earlier WikiText / two-digit-addition results are recorded in
`paper/main.typ`.

## Requirements

Python 3.10+, `torch >= 2.0`, `numpy`, `tqdm`. Managed with `uv` and `just`.
