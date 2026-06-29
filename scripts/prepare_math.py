"""Build the fixed held-out sets for the online +/- arithmetic study.

Training samples fresh equations online (see wonntext.math_data.OnlineMathDataset),
so there is no train file. This script only writes the two reproducible held-out
sets and the exclude list that the online sampler avoids:

  <out_dir>/test_ids.pt, test_answer_mask.pt              - in-distribution (2-4 ops)
  <out_dir>/extrapolation_ids.pt, extrapolation_answer_mask.pt - 5 operands
  <out_dir>/heldout_questions.json                        - questions to exclude in training
  <out_dir>/metadata.json                                 - vocab, seq_len, sampler config
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wonntext.math_data import (
    DEFAULT_SEQ_LEN,
    MASK_ID,
    PAD_ID,
    STOI,
    VOCAB_SIZE,
    build_fixed_set,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build held-out sets for online +/- math.")
    ap.add_argument("--out_dir", default="data/math")
    ap.add_argument("--seed", type=int, default=137)
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    # Training distribution (the online sampler uses the same ranges).
    ap.add_argument("--min_operands", type=int, default=2)
    ap.add_argument("--max_operands", type=int, default=4)
    ap.add_argument("--min_digits", type=int, default=1)
    ap.add_argument("--max_digits", type=int, default=3)
    ap.add_argument("--n_test", type=int, default=10_000)
    ap.add_argument("--n_extrapolation", type=int, default=10_000)
    ap.add_argument("--extrapolation_operands", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # In-distribution held-out set.
    test_ids, test_mask, test_q = build_fixed_set(
        args.n_test, args.min_operands, args.max_operands,
        args.min_digits, args.max_digits, args.seq_len, args.seed,
    )
    torch.save(test_ids, out / "test_ids.pt")
    torch.save(test_mask, out / "test_answer_mask.pt")

    # Extrapolation set: more operands than trained, disjoint from the test set.
    ex_ids, ex_mask, ex_q = build_fixed_set(
        args.n_extrapolation, args.extrapolation_operands, args.extrapolation_operands,
        args.min_digits, args.max_digits, args.seq_len, args.seed + 1, exclude=test_q,
    )
    torch.save(ex_ids, out / "extrapolation_ids.pt")
    torch.save(ex_mask, out / "extrapolation_answer_mask.pt")

    heldout = sorted(test_q | ex_q)
    (out / "heldout_questions.json").write_text(json.dumps(heldout))

    metadata = {
        "vocab_size": VOCAB_SIZE,
        "pad_token_id": PAD_ID,
        "mask_token_id": MASK_ID,
        "seq_len": args.seq_len,
        "char_vocab": STOI,
        "ops": ["+", "-"],
        "train_distribution": {
            "operands": [args.min_operands, args.max_operands],
            "digits": [args.min_digits, args.max_digits],
        },
        "extrapolation": {"operands": args.extrapolation_operands},
        "n_test": len(test_q),
        "n_extrapolation": len(ex_q),
        "n_heldout": len(heldout),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {out}  vocab={VOCAB_SIZE} seq_len={args.seq_len}")
    print(f"  test (in-dist, {args.min_operands}-{args.max_operands} ops): {len(test_q)}")
    print(f"  extrapolation ({args.extrapolation_operands} ops): {len(ex_q)}")
    print(f"  heldout excluded from training: {len(heldout)}")


if __name__ == "__main__":
    main()
