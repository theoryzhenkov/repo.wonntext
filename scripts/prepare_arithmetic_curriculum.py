"""Generate two-stage arithmetic curriculum using the WikiText BPE tokenizer.

Stage 1: two-digit addition (a+b=c), all 90*90=8,100 equations.
Stage 2: three-number expressions (a op1 b op2 c=d) with +, -, *, / using
standard precedence. Operands are uniformly sampled from 1-99 (mixed 1-2 digits).
All intermediate and final results are non-negative integers; division must be
exact (no remainders).

Output layout:
  <out_dir>/stage1/metadata.json
  <out_dir>/stage1/{train,valid,test}_ids.pt
  <out_dir>/stage1/{train,valid,test}_answer_mask.pt
  <out_dir>/stage2/metadata.json
  <out_dir>/stage2/{train,valid,test}_ids.pt
  <out_dir>/stage2/{train,valid,test}_answer_mask.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

OPS = ("+", "-", "*", "/")
HIGH_PRECEDENCE = {"*", "/"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare arithmetic curriculum data.")
    parser.add_argument("--out_dir", type=str, default="data/arithmetic_curriculum")
    parser.add_argument("--tokenizer_path", type=str, default="assets/wikitext/tokenizer.json")
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--stage2_train_samples", type=int, default=50_000)
    parser.add_argument("--stage2_val_samples", type=int, default=5_000)
    parser.add_argument("--stage2_test_samples", type=int, default=5_000)
    parser.add_argument("--operand_min", type=int, default=1)
    parser.add_argument("--operand_max", type=int, default=99)
    parser.add_argument("--max_result", type=int, default=10_000)
    return parser.parse_args()


def _check_value(x: int, max_result: int) -> bool:
    return 0 <= x <= max_result


def _apply(op: str, x: int, y: int) -> int:
    if op == "+":
        return x + y
    if op == "-":
        return x - y
    if op == "*":
        return x * y
    # Division: require exact, non-negative dividend, positive divisor.
    if y == 0 or x < 0 or x % y != 0:
        raise ValueError("invalid division")
    return x // y


def evaluate_three_number(a: int, op1: str, b: int, op2: str, c: int, max_result: int) -> int:
    """Evaluate a op1 b op2 c with standard precedence.

    Raises ValueError if any intermediate is negative, any integer-division
    constraint is violated, or any value exceeds max_result.
    """
    if op1 in HIGH_PRECEDENCE and op2 in HIGH_PRECEDENCE:
        # Left-to-right for same high precedence.
        intermediate = _apply(op1, a, b)
        if not _check_value(intermediate, max_result):
            raise ValueError("intermediate out of range")
        result = _apply(op2, intermediate, c)
    elif op1 in HIGH_PRECEDENCE:
        intermediate = _apply(op1, a, b)
        if not _check_value(intermediate, max_result):
            raise ValueError("intermediate out of range")
        result = _apply(op2, intermediate, c)
    elif op2 in HIGH_PRECEDENCE:
        rhs = _apply(op2, b, c)
        if not _check_value(rhs, max_result):
            raise ValueError("intermediate out of range")
        result = _apply(op1, a, rhs)
    else:
        # Both low precedence: left-to-right.
        intermediate = _apply(op1, a, b)
        if not _check_value(intermediate, max_result):
            raise ValueError("intermediate out of range")
        result = _apply(op2, intermediate, c)

    if not _check_value(result, max_result):
        raise ValueError("result out of range")
    return result


def generate_stage1_equations(seed: int) -> list[tuple[str, int]]:
    """All two-digit addition equations, shuffled."""
    random.seed(seed)
    equations: list[tuple[str, int]] = []
    for a in range(10, 100):
        for b in range(10, 100):
            equations.append((f"{a}+{b}={a + b}", a + b))
    random.shuffle(equations)
    return equations


def generate_stage2_equations(
    n: int,
    seed: int,
    operand_min: int,
    operand_max: int,
    max_result: int,
) -> list[tuple[str, int]]:
    """Sample n valid three-number equations with standard precedence."""
    rng = random.Random(seed)
    equations: list[tuple[str, int]] = []
    attempts = 0
    max_attempts = n * 50
    while len(equations) < n and attempts < max_attempts:
        attempts += 1
        a = rng.randint(operand_min, operand_max)
        op1 = rng.choice(OPS)
        b = rng.randint(operand_min, operand_max)
        op2 = rng.choice(OPS)
        c = rng.randint(operand_min, operand_max)
        try:
            result = evaluate_three_number(a, op1, b, op2, c, max_result)
        except (ValueError, ZeroDivisionError):
            continue
        equations.append((f"{a}{op1}{b}{op2}{c}={result}", result))
    return equations


def split_equations(
    equations: list[tuple[str, int]], val_frac: float, test_frac: float
) -> dict[str, list[tuple[str, int]]]:
    n_total = len(equations)
    n_test = max(1, int(test_frac * n_total))
    n_val = max(1, int(val_frac * n_total))
    n_train = n_total - n_val - n_test
    return {
        "train": equations[:n_train],
        "valid": equations[n_train : n_train + n_val],
        "test": equations[n_train + n_val :],
    }


def encode_split(
    equations: list[tuple[str, int]],
    tokenizer: Tokenizer,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids_list: list[torch.Tensor] = []
    masks_list: list[torch.Tensor] = []
    eq_token_id = tokenizer.encode("=").ids[0]

    max_len = 0
    encoded: list[tuple[list[int], int]] = []
    for eq, _ in equations:
        ids = tokenizer.encode(eq).ids
        encoded.append((ids, len(ids)))
        max_len = max(max_len, len(ids))

    for ids, orig_len in encoded:
        padded = torch.full((max_len,), pad_token_id, dtype=torch.long)
        padded[:orig_len] = torch.tensor(ids, dtype=torch.long)

        eq_positions = (padded == eq_token_id).nonzero(as_tuple=True)[0]
        if len(eq_positions) == 0:
            raise ValueError(f"No '=' token found in {ids}")
        answer_start = int(eq_positions.item()) + 1

        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[answer_start:orig_len] = True

        ids_list.append(padded)
        masks_list.append(mask)

    return torch.stack(ids_list), torch.stack(masks_list)


def save_stage(
    out_dir: Path,
    stage_name: str,
    equations: list[tuple[str, int]],
    tokenizer: Tokenizer,
    pad_token_id: int,
    mask_token_id: int,
    vocab_size: int,
    val_frac: float,
    test_frac: float,
    extra_metadata: dict | None = None,
) -> dict[str, int]:
    stage_dir = out_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    splits = split_equations(equations, val_frac, test_frac)
    sizes = {name: len(split) for name, split in splits.items()}

    max_len = 0
    for name, split in splits.items():
        if not split:
            raise ValueError(f"{stage_name}/{name} split is empty")
        ids, answer_mask = encode_split(split, tokenizer, pad_token_id)
        max_len = max(max_len, ids.shape[1])
        torch.save(ids, stage_dir / f"{name}_ids.pt")
        torch.save(answer_mask, stage_dir / f"{name}_answer_mask.pt")

    metadata = {
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "mask_token_id": mask_token_id,
        "seq_len": max_len,
        "train_size": sizes["train"],
        "valid_size": sizes["valid"],
        "test_size": sizes["test"],
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with (stage_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return sizes


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    metadata_path = Path(args.tokenizer_path).with_name("metadata.json")
    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    pad_token_id = int(metadata["pad_token_id"])
    mask_token_id = int(metadata["mask_token_id"])
    vocab_size = int(metadata["vocab_size"])

    # Stage 1: two-digit addition.
    stage1_equations = generate_stage1_equations(args.seed)
    stage1_sizes = save_stage(
        out_dir,
        "stage1",
        stage1_equations,
        tokenizer,
        pad_token_id,
        mask_token_id,
        vocab_size,
        args.val_frac,
        args.test_frac,
        extra_metadata={"ops": ["+"], "operand_range": [10, 99]},
    )

    # Stage 2: three-number expressions.
    n_stage2 = args.stage2_train_samples + args.stage2_val_samples + args.stage2_test_samples
    stage2_equations = generate_stage2_equations(
        n=n_stage2,
        seed=args.seed + 1,
        operand_min=args.operand_min,
        operand_max=args.operand_max,
        max_result=args.max_result,
    )
    stage2_sizes = save_stage(
        out_dir,
        "stage2",
        stage2_equations,
        tokenizer,
        pad_token_id,
        mask_token_id,
        vocab_size,
        args.val_frac,
        args.test_frac,
        extra_metadata={
            "ops": list(OPS),
            "operand_range": [args.operand_min, args.operand_max],
            "max_result": args.max_result,
            "precedence": "standard",
        },
    )

    print(f"Saved curriculum to {out_dir}")
    print(
        f"  stage1: {stage1_sizes['train']} train, "
        f"{stage1_sizes['valid']} val, {stage1_sizes['test']} test"
    )
    print(
        f"  stage2: {stage2_sizes['train']} train, "
        f"{stage2_sizes['valid']} val, {stage2_sizes['test']} test"
    )


if __name__ == "__main__":
    main()
