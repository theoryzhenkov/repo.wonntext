"""Compute the FLOP-matched 3x3 configs and emit a Ray job spec.

WONNText anchors each budget (2M/5M/10M params). The Universal Transformer
(n_steps=T) and Classical Transformer (n_layers searched) are sized so their
measured forward FLOPs match WONN's. Params/FLOPs are verified by instantiating
the real modern models and measuring with torch FlopCounterMode (all three use
explicit Linear + SDPA, so the counter is consistent across architectures).

Writes experiments/math_3x3.json (one train_math command per cell x seed).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from wonntext.baseline_transformer import TransformerLM, UniversalTransformerLM
from wonntext.math_data import VOCAB_SIZE
from wonntext.model import WONNText

N = 32  # seq_len for the FLOP measurement
HEADS = 8


def _params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _flops(m: torch.nn.Module) -> int:
    m.eval()
    x = torch.randint(0, VOCAB_SIZE - 1, (1, N))
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        m(x)
    return fc.get_total_flops()


def _round16(x: float) -> int:
    return max(16, round(x / 16) * 16)


def wonn(ch: int, T: int) -> WONNText:
    return WONNText(vocab_size=VOCAB_SIZE, ch=ch, max_seq_len=N, L=1, T=T, heads=HEADS)


def make_universal(d: int, T: int) -> UniversalTransformerLM:
    return UniversalTransformerLM(vocab_size=VOCAB_SIZE, dim=d, heads=HEADS, n_steps=T)


def make_classical(d: int, nl: int) -> TransformerLM:
    return TransformerLM(vocab_size=VOCAB_SIZE, dim=d, heads=HEADS, n_layers=nl)


def search(predicate_lt: Callable[[int], bool], lo: int, hi: int, step: int = 16) -> int:
    """Smallest multiple-of-`step` value in [lo,hi] where predicate flips True->False."""
    best = lo
    for v in range(lo, hi + 1, step):
        if predicate_lt(v):
            best = v
        else:
            break
    return best


def nearest(flops_of: Callable[[int], int], target: int, lo: int, hi: int, step: int = 16) -> int:
    """Value (multiple of `step`) whose flops_of(v) is closest to target."""
    best, best_err = lo, float("inf")
    for v in range(lo, hi + 1, step):
        err = abs(flops_of(v) - target)
        if err < best_err:
            best, best_err = v, err
        elif flops_of(v) > target * 1.5:
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--arch", default="wonn,universal,classical",
                    help="comma-separated archs to include (wonn,universal,classical)")
    ap.add_argument("--out", default="experiments/math_3x3.json")
    args = ap.parse_args()
    T = args.T
    archs = {a.strip() for a in args.arch.split(",")}

    rows = []
    for budget in (2e6, 5e6, 10e6):
        # WONN: smallest ch (mult 16) reaching the param budget.
        ch = search(lambda c, b=budget: _params(wonn(c, T)) < b, 16, 4000)
        wp, wf = _params(wonn(ch, T)), _flops(wonn(ch, T))

        # UT: dim (mult 16) whose FLOPs are closest to WONN at n_steps=T.
        d = nearest(lambda dd: _flops(make_universal(dd, T)), wf, 16, 3000)
        up, uf = _params(make_universal(d, T)), _flops(make_universal(d, T))

        # Classical: same width, n_layers whose FLOPs are closest to WONN.
        nl = nearest(lambda n, dd=d: _flops(make_classical(dd, n)), wf, 1, 64, step=1)
        cp, cf = _params(make_classical(d, nl)), _flops(make_classical(d, nl))

        rows.append(dict(budget=budget, ch=ch, d=d, nl=nl,
                         wp=wp, wf=wf, up=up, uf=uf, cp=cp, cf=cf))

    print(f"T={T}  seq_len={N}  heads={HEADS}\n")
    hdr = (f"{'budget':>6} | {'WONN ch':>9} {'par':>6} {'GF':>6} | "
           f"{'UT d':>6} {'par':>6} {'GF':>6} | {'Cls d/nl':>9} {'par':>6} {'GF':>6}")
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{int(r['budget']/1e6):>4}M  | "
              f"ch={r['ch']:<5} {r['wp']/1e6:>4.2f}M {r['wf']/1e9:>5.2f} | "
              f"d={r['d']:<4} {r['up']/1e6:>4.2f}M {r['uf']/1e9:>5.2f} | "
              f"d={r['d']}/nl={r['nl']:<2} {r['cp']/1e6:>4.1f}M {r['cf']/1e9:>5.2f}")

    # Emit job spec.
    base = (
        "PYTHONPATH=src python -m wonntext.train_math --data_dir data/math "
        f"--heads {HEADS} --steps {args.steps} --optimizer muon --lr 0.02 "
        "--batch_size 512 --eval_every 1000 --amp false --save_dir runs/math"
    )
    jobs = []
    for r in rows:
        tag = f"{int(r['budget']/1e6)}M"
        for s in args.seeds:
            if "wonn" in archs:
                jobs.append({"name": f"wonn_{tag}_s{s}",
                             "cmd": f"{base} --model wonn --ch {r['ch']} --L 1 --T {T} "
                                    f"--seed {s} --exp_name wonn_{tag}_s{s}"})
            if "universal" in archs:
                jobs.append({"name": f"universal_{tag}_s{s}",
                             "cmd": f"{base} --model universal --dim {r['d']} --depth {T} "
                                    f"--seed {s} --exp_name universal_{tag}_s{s}"})
            if "classical" in archs:
                jobs.append({"name": f"classical_{tag}_s{s}",
                             "cmd": f"{base} --model classical --dim {r['d']} --depth {r['nl']} "
                                    f"--seed {s} --exp_name classical_{tag}_s{s}"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=2))
    print(f"\nwrote {out}  ({len(jobs)} jobs)")


if __name__ == "__main__":
    main()
