"""Small bidirectional Transformer baseline for masked language modelling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLM(nn.Module):
    """Transformer encoder with tied token embeddings and an MLM head.

    Designed to be parameter-competitive with ``WONNText``:
    vocab_size=10000, d_model=232, nhead=8, num_layers=1, d_ff=464 gives
    roughly 2.8 M parameters.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 232,
        nhead: int = 8,
        num_layers: int = 1,
        d_ff: int | None = None,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        mask_token_id: int = -1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        if mask_token_id < 0:
            mask_token_id = vocab_size - 1
        self.mask_token_id = mask_token_id

        if d_ff is None:
            d_ff = 2 * d_model

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embed = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # With pre-norm (norm_first=True), the encoder leaves its output stream
        # un-normalised; a final LayerNorm (cf. GPT-2's ln_f) is required to keep
        # the residual stream bounded before the tied projection, otherwise
        # logits explode as depth grows.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

        # Output projection; tied to the input embedding like WONNText.
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embed.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        # Keep padding row at zero so it does not contribute.
        with torch.no_grad():
            self.token_embed.weight[self.pad_token_id].zero_()
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        x = self.token_embed(input_ids) + self.position_embed(positions)

        key_padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for padding.
            key_padding_mask = attention_mask == 0

        hidden = self.encoder(x, src_key_padding_mask=key_padding_mask)
        logits = self.output_proj(hidden)

        output: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class UniversalTransformerLM(nn.Module):
    """Weight-tied iterated Transformer (Universal-Transformer-style control).

    A single pre-norm encoder block is applied ``num_steps`` times with shared
    weights. This is the clean control for WONNText: it matches a 1-layer
    Transformer's *parameters* while matching a ``T``-step WONN's *FLOPs* (both
    run ``num_steps`` attention passes over tied weights). If WONNText still
    wins, the advantage is the oscillator dynamics rather than extra compute or
    capacity.

    The block is iterated with no per-step conditioning, mirroring WONN's fixed
    recurrence. An optional learned timestep embedding can be enabled to give
    the iterated block step awareness (off by default for the truest control).
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 232,
        nhead: int = 8,
        num_steps: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        mask_token_id: int = -1,
        timestep_embed: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_steps = int(num_steps)
        self.pad_token_id = pad_token_id
        if mask_token_id < 0:
            mask_token_id = vocab_size - 1
        self.mask_token_id = mask_token_id
        self.timestep_embed = bool(timestep_embed)

        if d_ff is None:
            d_ff = 2 * d_model

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embed = nn.Embedding(max_seq_len, d_model)
        if self.timestep_embed:
            self.step_embed = nn.Embedding(self.num_steps, d_model)

        # A single tied block, applied num_steps times in forward.
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # Final norm: pre-norm leaves the residual stream un-normalised, and
        # iterating the block many times makes this essential (cf. ln_f).
        self.final_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embed.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embed.weight[self.pad_token_id].zero_()
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)
        if self.timestep_embed:
            nn.init.normal_(self.step_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        x = self.token_embed(input_ids) + self.position_embed(positions)

        key_padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        for t in range(self.num_steps):
            if self.timestep_embed:
                x = x + self.step_embed.weight[t]
            x = self.layer(x, src_key_padding_mask=key_padding_mask)

        hidden = self.final_norm(x)
        logits = self.output_proj(hidden)

        output: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
