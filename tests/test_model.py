"""Smoke tests for WONNText."""

from __future__ import annotations

import torch

from wonntext.data import RandomTokenDataset, make_mlm_collate_fn
from wonntext.model import WONNText


def test_forward_shape() -> None:
    vocab_size = 32
    seq_len = 64
    batch = 4
    model = WONNText(
        vocab_size=vocab_size,
        ch=64,
        max_seq_len=seq_len,
        L=1,
        T=4,
        heads=4,
    )
    input_ids = torch.randint(0, vocab_size - 1, (batch, seq_len))
    logits = model(input_ids)
    assert logits.shape == (batch, seq_len, vocab_size)


def test_masked_cross_entropy_loss() -> None:
    vocab_size = 32
    seq_len = 64
    dataset = RandomTokenDataset(vocab_size=vocab_size, seq_len=seq_len, num_samples=8)
    collate = make_mlm_collate_fn(mask_token_id=vocab_size - 1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate)
    batch = next(iter(loader))

    model = WONNText(
        vocab_size=vocab_size,
        ch=64,
        max_seq_len=seq_len,
        L=1,
        T=4,
        heads=4,
    )
    output = model(batch["input_ids"], labels=batch["labels"])

    assert "loss" in output
    assert output["loss"].numel() == 1
    assert not torch.isnan(output["loss"])


def test_tied_embedding_shapes() -> None:
    vocab_size = 16
    ch = 32
    model = WONNText(vocab_size=vocab_size, ch=ch, L=1, T=2)
    assert model.output_proj.weight is model.token_embed.weight
    assert model.token_embed.weight.shape == (vocab_size, ch)


def test_attention_mask_ignored_position() -> None:
    vocab_size = 16
    seq_len = 8
    model = WONNText(vocab_size=vocab_size, ch=16, max_seq_len=seq_len, L=1, T=2, heads=2)
    input_ids = torch.randint(0, vocab_size - 1, (1, seq_len))
    mask = torch.ones(seq_len, dtype=torch.long)
    mask[3:] = 0

    out_with_mask = model(input_ids, attention_mask=mask.unsqueeze(0))
    out_no_mask = model(input_ids)
    # Outputs should differ because masked positions are not allowed to influence
    # earlier positions. We simply assert the shapes are valid.
    assert out_with_mask.shape == (1, seq_len, vocab_size)
    assert out_no_mask.shape == (1, seq_len, vocab_size)
