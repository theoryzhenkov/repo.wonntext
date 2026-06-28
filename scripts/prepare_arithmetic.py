"""Generate a two-digit addition dataset encoded with the WikiText BPE tokenizer.

Output layout (mirrors WikiText structure):
  data/arithmetic/metadata.json
  data/arithmetic/train_ids.pt          # (N, L) token ids
  data/arithmetic/train_answer_mask.pt  # (N, L) bool mask over answer tokens
  data/arithmetic/valid_ids.pt
  data/arithmetic/valid_answer_mask.pt
  data/arithmetic/test_ids.pt
  data/arithmetic/test_answer_mask.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare two-digit arithmetic data.")
    parser.add_argument("--out_dir", type=str, default="data/arithmetic")
    parser.add_argument("--tokenizer_path", type=str, default="assets/wikitext/tokenizer.json")
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    metadata_path = Path(args.tokenizer_path).with_name("metadata.json")
    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    pad_token_id = int(metadata["pad_token_id"])
    mask_token_id = int(metadata["mask_token_id"])
    vocab_size = int(metadata["vocab_size"])

    low = 10 ** (args.digits - 1)
    high = 10**args.digits - 1

    equations: list[tuple[str, int]] = []
    for a in range(low, high + 1):
        for b in range(low, high + 1):
            # Format: no spaces so the tokenizer keeps digits as separate symbols.
            equations.append((f"{a}+{b}={a + b}", a + b))

    random.shuffle(equations)

    n_total = len(equations)
    n_train = int(args.train_frac * n_total)
    n_val = int(args.val_frac * n_total)
    n_test = n_total - n_train - n_val

    splits = {
        "train": equations[:n_train],
        "valid": equations[n_train : n_train + n_val],
        "test": equations[n_train + n_val :],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_len = 0
    encoded_splits: dict[str, list[torch.Tensor]] = {}
    for name, split in splits.items():
        encoded = []
        for eq, _ in split:
            ids = tokenizer.encode(eq).ids
            encoded.append(torch.tensor(ids, dtype=torch.long))
            max_len = max(max_len, len(ids))
        encoded_splits[name] = encoded

    arithmetic_metadata = {
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "mask_token_id": mask_token_id,
        "seq_len": max_len,
        "digits": args.digits,
        "train_size": n_train,
        "valid_size": n_val,
        "test_size": n_test,
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(arithmetic_metadata, f, indent=2)

    for name, encoded in encoded_splits.items():
        ids_list = []
        masks_list = []
        for ids in encoded:
            padded = torch.full((max_len,), pad_token_id, dtype=torch.long)
            padded[: len(ids)] = ids

            # Find the equal sign and mark everything after it as the answer.
            eq_pos = (ids == tokenizer.encode("=").ids[0]).nonzero(as_tuple=True)[0]
            if len(eq_pos) == 0:
                raise ValueError(f"No '=' token found in {ids.tolist()}")
            answer_start = int(eq_pos.item()) + 1

            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[answer_start: len(ids)] = True

            ids_list.append(padded)
            masks_list.append(mask)

        torch.save(torch.stack(ids_list), out_dir / f"{name}_ids.pt")
        torch.save(torch.stack(masks_list), out_dir / f"{name}_answer_mask.pt")

    print(f"Saved {out_dir}")
    print(f"  seq_len: {max_len}")
    print(f"  train: {n_train}, valid: {n_val}, test: {n_test}")


if __name__ == "__main__":
    main()
