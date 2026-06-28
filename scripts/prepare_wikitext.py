"""Download WikiText-2, train a small BPE tokenizer, and save encoded splits.

The output is a directory with:
  metadata.json        - vocab_size and special-token ids
  tokenizer.json       - trained tokenizer
  train_ids.pt         - torch.LongTensor of token ids
  valid_ids.pt
  test_ids.pt

Example:
    uv run --group data python scripts/prepare_wikitext.py --out_dir data/wikitext
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

SpecialTokens = dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WikiText-2 for WONNText.")
    parser.add_argument("--out_dir", type=str, default="data/wikitext")
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument(
        "--dataset", type=str, default="Salesforce/wikitext", help="HF dataset name."
    )
    parser.add_argument("--config", type=str, default="wikitext-2-raw-v1")
    return parser.parse_args()


def train_tokenizer(text: str, vocab_size: int) -> Any:
    # Import inside function so the main package does not depend on tokenizers.
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<mask>", "<unk>"],
        show_progress=True,
    )

    tokenizer.train_from_iterator([text], trainer=trainer)
    return tokenizer


def encode_split(tokenizer: object, examples: list[str]) -> torch.Tensor:
    encoded = tokenizer.encode_batch(examples)
    ids = [tok_id for enc in encoded for tok_id in enc.ids]
    return torch.tensor(ids, dtype=torch.long)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading WikiText-2...")
    dataset = load_dataset(args.dataset, args.config)

    train_text = "\n\n".join(dataset["train"]["text"])

    print("Training BPE tokenizer...")
    tokenizer = train_tokenizer(train_text, vocab_size=args.vocab_size)

    # Reserve special-token ids deterministically.
    special_tokens = {
        "pad_token_id": tokenizer.token_to_id("<pad>"),
        "mask_token_id": tokenizer.token_to_id("<mask>"),
        "unk_token_id": tokenizer.token_to_id("<unk>"),
        "vocab_size": tokenizer.get_vocab_size(),
    }

    print("Encoding splits...")
    train_ids = encode_split(tokenizer, dataset["train"]["text"])
    valid_ids = encode_split(tokenizer, dataset["validation"]["text"])
    test_ids = encode_split(tokenizer, dataset["test"]["text"])

    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(special_tokens, f, indent=2)

    torch.save(train_ids, out_dir / "train_ids.pt")
    torch.save(valid_ids, out_dir / "valid_ids.pt")
    torch.save(test_ids, out_dir / "test_ids.pt")

    print(f"Saved {out_dir}")
    print(f"  vocab_size: {special_tokens['vocab_size']}")
    print(f"  train tokens: {len(train_ids):,}")
    print(f"  valid tokens: {len(valid_ids):,}")
    print(f"  test tokens:  {len(test_ids):,}")


if __name__ == "__main__":
    main()
