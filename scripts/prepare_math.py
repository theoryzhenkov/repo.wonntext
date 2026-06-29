"""Hand-rolled +/- arithmetic dataset (character-level) for the WONNText study.

Equations use only addition and subtraction over two-digit operands (10-99),
with either two operands (``a±b``) or three (``a±b±c``), evaluated left to right
(``+``/``-`` share precedence). Negative results are allowed.

Character-level tokenisation keeps the vocabulary tiny (~15 symbols) so a 2-10M
model spends its parameters on the layers rather than an embedding table. The
answer (right of ``=``) is the masked span the model must predict.

Outputs (mirrors the masked-answer format consumed by train_arithmetic.py):
  <out_dir>/metadata.json            - vocab, special ids, seq_len, char_vocab
  <out_dir>/{train,valid,test}_ids.pt
  <out_dir>/{train,valid,test}_answer_mask.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import torch

# Fixed character vocabulary. PAD/MASK first so ids are stable.
PAD, MASK = "<pad>", "<mask>"
CHARS = list("0123456789+-=")
STOI = {PAD: 0, MASK: 1, **{c: i + 2 for i, c in enumerate(CHARS)}}
ITOS = {i: c for c, i in STOI.items()}
PAD_ID, MASK_ID = STOI[PAD], STOI[MASK]
VOCAB_SIZE = len(STOI)


def evaluate(operands: list[int], ops: list[str]) -> int:
    """Left-to-right evaluation of an add/sub chain."""
    acc = operands[0]
    for op, x in zip(ops, operands[1:], strict=True):
        acc = acc + x if op == "+" else acc - x
    return acc


def format_eq(operands: list[int], ops: list[str]) -> tuple[str, str]:
    """Return (question_with_equals, answer_str)."""
    q = str(operands[0])
    for op, x in zip(ops, operands[1:], strict=True):
        q += f"{op}{x}"
    return q + "=", str(evaluate(operands, ops))


def gen_two(rng: random.Random) -> tuple[list[int], list[str]]:
    return [rng.randint(10, 99), rng.randint(10, 99)], [rng.choice("+-")]


def gen_three(rng: random.Random) -> tuple[list[int], list[str]]:
    return (
        [rng.randint(10, 99) for _ in range(3)],
        [rng.choice("+-"), rng.choice("+-")],
    )


def build_pool(n_two: int, n_three: int, seed: int) -> list[tuple[str, str]]:
    """Build a deduplicated pool of (question, answer) strings."""
    rng = random.Random(seed)
    seen: set[str] = set()
    pool: list[tuple[str, str]] = []

    # Two-operand space is small (90*90*2 = 16 200); enumerate then sample.
    two_space = [
        ([a, b], [op])
        for a, b in itertools.product(range(10, 100), range(10, 100))
        for op in "+-"
    ]
    rng.shuffle(two_space)
    for operands, ops in two_space[: min(n_two, len(two_space))]:
        q, a = format_eq(operands, ops)
        if q not in seen:
            seen.add(q)
            pool.append((q, a))

    # Three-operand space is ~2.9M; sample unique by rejection.
    attempts = 0
    target = len(pool) + n_three
    while len(pool) < target and attempts < n_three * 20:
        attempts += 1
        operands, ops = gen_three(rng)
        q, a = format_eq(operands, ops)
        if q not in seen:
            seen.add(q)
            pool.append((q, a))

    rng.shuffle(pool)
    return pool


def encode(q: str, a: str, seq_len: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Encode 'q' + 'a' into padded ids + an answer-position mask."""
    s = q + a
    if len(s) > seq_len:
        return None
    ids = [STOI[c] for c in s] + [PAD_ID] * (seq_len - len(s))
    answer_mask = [False] * seq_len
    for i in range(len(q), len(s)):  # positions of the answer chars
        answer_mask[i] = True
    return torch.tensor(ids, dtype=torch.long), torch.tensor(answer_mask, dtype=torch.bool)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand-rolled +/- arithmetic dataset.")
    ap.add_argument("--out_dir", default="data/math")
    ap.add_argument("--seed", type=int, default=137)
    ap.add_argument("--n_two", type=int, default=16_200, help="two-operand equations (max 16200)")
    ap.add_argument("--n_three", type=int, default=180_000, help="three-operand equations")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--test_frac", type=float, default=0.05)
    args = ap.parse_args()

    pool = build_pool(args.n_two, args.n_three, args.seed)

    # Fixed seq_len: longest possible is "99+99+99=" (9) + "-188" (4) = 13.
    seq_len = max(len(q + a) for q, a in pool)

    n = len(pool)
    n_test = int(args.test_frac * n)
    n_val = int(args.val_frac * n)
    splits = {
        "train": pool[n_test + n_val :],
        "valid": pool[n_test : n_test + n_val],
        "test": pool[:n_test],
    }
    # Zero overlap is guaranteed: splits partition a deduplicated pool.

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, rows in splits.items():
        ids_list, mask_list = [], []
        for q, a in rows:
            enc = encode(q, a, seq_len)
            if enc is None:
                continue
            ids_list.append(enc[0])
            mask_list.append(enc[1])
        ids = torch.stack(ids_list)
        masks = torch.stack(mask_list)
        torch.save(ids, out / f"{name}_ids.pt")
        torch.save(masks, out / f"{name}_answer_mask.pt")
        sizes[name] = len(ids_list)

    metadata = {
        "vocab_size": VOCAB_SIZE,
        "pad_token_id": PAD_ID,
        "mask_token_id": MASK_ID,
        "seq_len": seq_len,
        "char_vocab": STOI,
        "ops": ["+", "-"],
        "operand_digits": 2,
        "operands": [2, 3],
        "negative_results": True,
        "sizes": sizes,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {args.out_dir}  vocab={VOCAB_SIZE} seq_len={seq_len}")
    for name, k in sizes.items():
        print(f"  {name}: {k}")
    print("Examples:")
    for q, a in splits["test"][:6]:
        print(f"  {q}{a}")


if __name__ == "__main__":
    main()
