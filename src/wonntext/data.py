"""Text datasets and MLM masking.

Provides a small character-corpus loader that works without internet access, a
tokenized-corpus loader for pre-encoded datasets (e.g., WikiText-2), plus a
generic masked-language-model collator. The mask token is just another entry
in the vocabulary, exactly as the input embedding handles it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

Sample = dict[str, torch.Tensor]


def _whitespace_tokenizer(text: str) -> list[str]:
    """Split text on whitespace for a tiny word-level baseline."""
    return text.split()


def _build_char_vocab(text: str) -> dict[str, int]:
    chars = sorted(set(text))
    return {ch: i for i, ch in enumerate(chars)}


class CharCorpusDataset(Dataset):
    """Character-level corpus chopped into fixed-length sequences.

    The mask token is automatically added as the last vocabulary entry.
    """

    def __init__(
        self,
        text_or_path: str,
        seq_len: int = 128,
        stride: int | None = None,
        mask_token: str = "<MASK>",
        pad_token: str | None = "<PAD>",
    ) -> None:
        super().__init__()

        self.seq_len = int(seq_len)
        self.stride = int(stride or seq_len)
        self.mask_token = mask_token
        self.pad_token = pad_token

        path = Path(text_or_path)
        text = path.read_text(encoding="utf-8") if path.is_file() else text_or_path

        # Build a deterministic character vocabulary.
        self.vocab = _build_char_vocab(text)

        # Reserve special tokens.
        special_tokens: list[str] = []
        if pad_token is not None and pad_token not in self.vocab:
            special_tokens.append(pad_token)
        if mask_token not in self.vocab:
            special_tokens.append(mask_token)
        for token in special_tokens:
            self.vocab[token] = len(self.vocab)

        self.pad_token_id = self.vocab.get(pad_token, 0) if pad_token else 0
        self.mask_token_id = self.vocab[mask_token]

        self.tokens = torch.tensor(
            [self.vocab[ch] for ch in text],
            dtype=torch.long,
        )

        self.indices = list(range(0, max(1, len(self.tokens) - self.seq_len + 1), self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        start = self.indices[index]
        chunk = self.tokens[start : start + self.seq_len]
        # Pad the last chunk if needed so every sample has the same length.
        if len(chunk) < self.seq_len:
            padding = torch.full((self.seq_len - len(chunk),), self.pad_token_id, dtype=torch.long)
            chunk = torch.cat([chunk, padding])
        return {
            "input_ids": chunk,
            "attention_mask": (chunk != self.pad_token_id).long(),
        }


class RandomTokenDataset(Dataset):
    """Synthetic token dataset for smoke tests."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 128,
        num_samples: int = 100,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Sample:
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


def make_mlm_collate_fn(
    mask_token_id: int,
    mask_probability: float = 0.15,
    pad_token_id: int = 0,
    ignore_index: int = -100,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Create a collator that masks random positions for MLM.

    For each position selected for masking, the input is replaced with the
    mask token id and the label is set to the original token id. All other
    positions are ignored via ``ignore_index``.
    """

    def _mask_sample(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.full_like(input_ids, ignore_index)
        valid_positions = (attention_mask != 0).nonzero(as_tuple=True)[0]
        if valid_positions.numel() == 0:
            return input_ids, labels

        n_mask = max(1, int(mask_probability * valid_positions.numel()))
        selected = valid_positions[torch.randperm(valid_positions.numel())[:n_mask]]

        masked_input = input_ids.clone()
        labels[selected] = input_ids[selected]
        masked_input[selected] = mask_token_id

        return masked_input, labels

    def collate(batch: Sequence[Sample]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)

        masked_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        for ids, mask in zip(input_ids, attention_mask, strict=True):
            masked, lbls = _mask_sample(ids, mask)
            masked_ids_list.append(masked)
            labels_list.append(lbls)

        return {
            "input_ids": torch.stack(masked_ids_list, dim=0),
            "attention_mask": attention_mask,
            "labels": torch.stack(labels_list, dim=0),
            "pad_token_id": torch.tensor(pad_token_id, dtype=torch.long),
        }

    return collate


class TokenizedCorpusDataset(Dataset):
    """Pre-tokenized corpus stored as `.pt` tensors and a `metadata.json`.

    Expected directory layout (created by ``scripts/prepare_wikitext.py``):
      data_dir/metadata.json   - vocab_size, pad_token_id, mask_token_id, ...
      data_dir/train_ids.pt    - torch.LongTensor of token ids
      data_dir/valid_ids.pt
      data_dir/test_ids.pt
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        seq_len: int = 256,
        stride: int | None = None,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.seq_len = int(seq_len)
        self.stride = int(stride or seq_len)

        metadata_path = self.data_dir / "metadata.json"
        ids_path = self.data_dir / f"{split}_ids.pt"

        with metadata_path.open(encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.pad_token_id = int(self.metadata["pad_token_id"])
        self.mask_token_id = int(self.metadata["mask_token_id"])

        self.tokens = torch.load(ids_path, map_location="cpu", weights_only=True).long()
        if self.tokens.dim() != 1:
            raise ValueError(
                f"Expected 1-D token tensor, got shape {tuple(self.tokens.shape)}"
            )

        self.indices = list(
            range(0, max(1, len(self.tokens) - self.seq_len + 1), self.stride)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        start = self.indices[index]
        chunk = self.tokens[start : start + self.seq_len]
        if len(chunk) < self.seq_len:
            padding = torch.full(
                (self.seq_len - len(chunk),), self.pad_token_id, dtype=torch.long
            )
            chunk = torch.cat([chunk, padding])
        return {
            "input_ids": chunk,
            "attention_mask": (chunk != self.pad_token_id).long(),
        }
