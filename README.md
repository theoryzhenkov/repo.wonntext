# wonntext

A denoising masked-language model based on the Winfree Oscillatory Neural
Network (WONN) Sudoku configuration from [Jiawen-Dai/WONN](https://github.com/Jiawen-Dai/WONN).

The original WONN solves 9×9 Sudoku with discrete digit symbols, a single
Winfree layer (`L=1`), `T=16` recurrent steps, group size `1`, and attention
coupling on a 2-D grid. This project keeps that core but makes exactly four
changes:

1. **Input** — token ids are embedded and the embedding vector directly becomes
   the initial oscillator frequency `Ω_init = f_init(emb)`. Phases `Θ_init` are
   random `N(0, σ²)`. The mask token is an ordinary vocabulary entry.
2. **Coupling** — the 2-D grid is replaced by a 1-D token sequence with full
   bidirectional attention (no causal mask), using 1-D RoPE. Sequence length is
   configurable in the 128–256 range.
3. **Output head** — the final phase at each position is converted to
   `sin θ / cos θ` features, projected back to embedding dimension, and a linear
   layer maps to vocabulary logits. The output projection is tied to the input
   embedding.
4. **Objective** — the Sudoku constraint loss is replaced by masked cross-entropy
   (masked language modelling).

## Project layout

```text
src/wonntext/
  __init__.py     package metadata
  winfree.py      SequenceAttention, ThetaEmbedding1D, WinfreeTextLayer
  model.py        WONNText model and tied readout
  data.py         character corpus loader and MLM collator
  train.py        single-GPU training loop with masked CE
  utils.py        seeding helpers
```

## Quick start

```bash
just train --data_path data/sample.txt --seq_len 128 --epochs 5 --batchsize 4
```

For a smoke run with a synthetic dataset (no data file required):

```bash
just train --vocab_size 64 --seq_len 128 --epochs 2 --batchsize 4
```

## WikiText-2

Prepare a BPE-tokenized WikiText-2 dataset:

```bash
uv run --group data python scripts/prepare_wikitext.py --out_dir data/wikitext --vocab_size 10000
```

Train locally on CPU (slow) or GPU:

```bash
uv run python -m wonntext.train \
  --data_dir data/wikitext \
  --seq_len 256 --batchsize 32 --ch 256 --heads 8 --T 16 --epochs 50
```

## Run on RunPod with SkyPilot

**Important:** keep your RunPod API key out of git. Copy `.env.example` to `.env`, paste the key, and source it.

### Launch

In **bash/zsh**:

```bash
set -a && source .env && set +a
uv sync --group cloud
uv run --group cloud sky check runpod

# WONNText
uv run --group cloud sky launch -c wonntext sky/runpod_wikitext.yaml

# Transformer baseline (same data, ~2.8M params)
uv run --group cloud sky launch -c wonntext-baseline sky/runpod_baseline.yaml

# Ablations
uv run --group cloud sky launch -c wonntext-causal sky/runpod_ablation_causal.yaml
uv run --group cloud sky launch -c wonntext-omega sky/runpod_ablation_omega.yaml
```

In **fish** (use `set -x`, not `source .env`):

```fish
set -x RUNPOD_API_KEY rpa_...
uv sync --group cloud
uv run --group cloud sky check runpod

# WONNText
uv run --group cloud sky launch -c wonntext sky/runpod_wikitext.yaml

# Transformer baseline (same data, ~2.8M params)
uv run --group cloud sky launch -c wonntext-baseline sky/runpod_baseline.yaml

# Ablations
uv run --group cloud sky launch -c wonntext-causal sky/runpod_ablation_causal.yaml
uv run --group cloud sky launch -c wonntext-omega sky/runpod_ablation_omega.yaml
```

Inside the Nix devShell, `.venv/bin` is added to `PATH` after `uv sync --group cloud`, so you can also use `sky` directly.

Launches use the pre-baked runtime image (see below). The first time, make sure the image is already pushed to GHCR and that you have updated the `image_id` placeholder in the YAML files.

### Build the pre-baked image on GitHub Actions

The workflow in `.github/workflows/build-image.yml` builds and pushes the image
to Docker Hub. Secrets needed:

- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` — used to pull the public
  `runpod/pytorch` base image during the build (avoids Docker Hub rate limits) and
  to push the final image to Docker Hub.

The resulting image is public by default, so RunPod can pull it without extra
credentials. Update the `image_id` placeholders in `sky/*.yaml` to:

```yaml
image_id: docker:YOUR_DOCKERHUB_USERNAME/wonntext-runtime:latest
```

#### Authenticating Docker Hub pulls on RunPod (optional)

If you hit Docker Hub rate limits, authenticate pulls in one of two ways:

1. **RunPod console (easiest):** add your Docker Hub credentials in
   **RunPod settings → Container Registry Auth**. RunPod then uses them for every
   pod it starts, including SkyPilot clusters.

2. **Per-launch via SkyPilot CLI:** pass `--env` flags with your Docker Hub
   credentials when launching:

   ```bash
   uv run --group cloud sky launch -c wonntext-omega sky/runpod_ablation_omega.yaml \
     --env SKYPILOT_DOCKER_SERVER=docker.io \
     --env SKYPILOT_DOCKER_USERNAME="$DOCKERHUB_USERNAME" \
     --env SKYPILOT_DOCKER_PASSWORD="$DOCKERHUB_TOKEN"
   ```

## Two-digit addition fine-tuning

A small reasoning task is used to compare the pretrained WONNText and
Transformer checkpoints after fine-tuning. Each sample is a tokenized equation
such as `12+34=46` with the answer digits masked; the model must predict only the
answer tokens.

```bash
uv run --group data python scripts/prepare_arithmetic.py \
    --out_dir data/arithmetic \
    --tokenizer_path assets/wikitext/tokenizer.json
```

Run on RunPod:

```bash
# WONNText
uv run --group cloud sky launch -c arithmetic-wonn sky/runpod_arithmetic_wonn.yaml

# Transformer baseline
uv run --group cloud sky launch -c arithmetic-transformer sky/runpod_arithmetic_transformer.yaml
```

### Results on held-out test set

| Model | Token accuracy | Whole-answer accuracy |
|---|---|---|
| WONNText bidirectional | 99.71% | 99.38% |
| Transformer baseline | 77.29% | 43.95% |

The parameter-matched Transformer plateaus at roughly 44% whole-answer accuracy,
while WONNText reaches near-perfect addition.

## Tests

```bash
just test
```

## Vendor provenance

Original WONN code from `Jiawen-Dai/WONN` is vendored in `vendor/WONN/` for
reference. The new implementation in `src/wonntext/` is derived from the Sudoku
configuration in that repository.

## Requirements

- Python 3.10+
- `torch >= 2.0`
- `numpy`, `tqdm`

Managed with `uv` and `just` per the repository template. A local `uv.toml`
overrides the global `no-build = true` setting so that the current package can
be installed in editable mode.
