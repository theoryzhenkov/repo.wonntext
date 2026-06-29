"""Benchmark Winfree forward modes on the 2.8M model.

Measures, per mode:
  * forward FLOPs (torch FlopCounterMode, single batch)
  * number of full self-attention passes (analytic)
  * training wall-clock + final loss on the two-digit arithmetic task

Run on CPU (local VM has no GPU); relative numbers carry over to GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from wonntext.model import WONNText
from wonntext.utils import seed_everything

MODES = [
    ("recurrent", {}),
    ("predictor_corrector", {}),
    ("parallel_scan", {}),
    ("parallel_scan_refined", {}),
    ("lazy_coupling", {"lazy_k": 2}),
    ("lazy_coupling", {"lazy_k": 4}),
]


def load_arithmetic(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    meta = json.loads((data_dir / "metadata.json").read_text())
    ids = torch.load(data_dir / "train_ids.pt", weights_only=True).long()
    ans = torch.load(data_dir / "train_answer_mask.pt", weights_only=True).bool()
    return ids, ans, meta


def build_model(vocab_size: int, mask_token_id: int, mode: str, lazy_k: int) -> WONNText:
    seed_everything(0)
    return WONNText(
        vocab_size=vocab_size,
        ch=256,
        max_seq_len=256,
        L=1,
        T=8,
        heads=8,
        mask_token_id=mask_token_id,
        winfree_mode=mode,
        lazy_k=lazy_k,
    )


def count_flops(model: WONNText, ids: torch.Tensor, mask_id: int) -> int:
    model.eval()
    x = ids[:8].clone()
    x[:, -2:] = mask_id  # mask a couple positions like the real task
    flop_counter = FlopCounterMode(display=False)
    with flop_counter, torch.no_grad():
        model(x)
    return flop_counter.get_total_flops()


def train_eval(
    model: WONNText,
    ids: torch.Tensor,
    ans: torch.Tensor,
    mask_id: int,
    epochs: int,
    n: int,
    bs: int,
) -> tuple[float, float]:
    ids, ans = ids[:n], ans[:n]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    t0 = time.perf_counter()
    last_loss = float("nan")
    for _ in range(epochs):
        perm = torch.randperm(len(ids))
        for i in range(0, len(ids), bs):
            idx = perm[i : i + bs]
            inp, a = ids[idx].clone(), ans[idx]
            labels = torch.full_like(inp, -100)
            labels[a] = inp[a]
            inp[a] = mask_id
            out = model(inp, labels=labels)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
    return last_loss, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/arithmetic")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    ids, ans, meta = load_arithmetic(Path(args.data_dir))
    vocab_size = int(meta["vocab_size"])
    mask_id = int(meta["mask_token_id"])

    # Reference attention-pass count: recurrent does T*L per forward.
    T, L = 8, 1

    rows = []
    base_flops = None
    for mode, kw in MODES:
        lazy_k = kw.get("lazy_k", 2)
        label = mode if mode != "lazy_coupling" else f"lazy_coupling(k={lazy_k})"

        m = build_model(vocab_size, mask_id, mode, lazy_k)
        flops = count_flops(m, ids, mask_id)
        if mode == "recurrent":
            base_flops = flops

        # analytic full-attention passes
        if mode == "recurrent":
            passes = T * L
        elif mode == "predictor_corrector":
            passes = (2 + 3) * L
        elif mode == "parallel_scan":
            passes = T * L  # same FLOPs, batched (1 serial call)
        elif mode == "parallel_scan_refined":
            passes = 2 * T * L
        else:  # lazy_coupling
            passes = -(-T // lazy_k) * L  # ceil(T/k) * L

        m = build_model(vocab_size, mask_id, mode, lazy_k)
        loss, secs = train_eval(m, ids, ans, mask_id, args.epochs, args.n, args.bs)
        rows.append((label, loss, secs, flops, passes))

    print(f"\n{'mode':<26} {'loss':>6} {'time(s)':>8} {'GFLOPs':>9} "
          f"{'FLOPx':>6} {'attn passes':>12}")
    print("-" * 76)
    for label, loss, secs, flops, passes in rows:
        ratio = base_flops / flops if flops else float("nan")
        print(f"{label:<26} {loss:>6.3f} {secs:>8.1f} {flops/1e9:>9.2f} "
              f"{ratio:>5.2f}x {passes:>12}")


if __name__ == "__main__":
    main()
