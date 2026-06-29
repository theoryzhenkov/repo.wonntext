"""Generic MLM masking utilities and a synthetic dataset for smoke tests.

The +/- arithmetic study uses :mod:`wonntext.math_data` (online sampler); this
module retains only the pieces used by the test suite. The mask token is just
another vocabulary entry, exactly as the input embedding handles it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.utils.data import Dataset

Sample = dict[str, torch.Tensor]


class RandomTokenDataset(Dataset):
    """Synthetic token dataset for smoke tests."""

    def __init__(self, vocab_size: int, seq_len: int = 128, num_samples: int = 100) -> None:
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
    """Create a collator that masks random positions for MLM."""

    def _mask_sample(
        input_ids: torch.Tensor, attention_mask: torch.Tensor
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
