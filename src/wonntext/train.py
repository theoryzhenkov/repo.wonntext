"""Minimal training loop for WONNText masked-language modelling.

This intentionally drops the Sudoku-specific constraint loss and replaces it
with masked cross-entropy. Single-GPU only; DDP is left for future work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import tqdm

from wonntext.baseline_transformer import TransformerLM
from wonntext.data import (
    CharCorpusDataset,
    RandomTokenDataset,
    TokenizedCorpusDataset,
    make_mlm_collate_fn,
)
from wonntext.model import WONNText
from wonntext.utils import seed_everything, str2bool

Metrics = dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WONNText with masked cross-entropy.")

    parser.add_argument("--exp_name", type=str, default="wonntext_mlm")
    parser.add_argument("--save_dir", type=str, default="runs/wonntext")

    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--deterministic", type=str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--eval_batchsize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=0)
    parser.add_argument(
        "--model",
        type=str,
        choices=["wonn", "transformer"],
        default="wonn",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--omega_as_token_embed",
        type=str2bool,
        default=True,
        help="WONN: use token embedding as omega init (default True).",
    )
    parser.add_argument(
        "--causal",
        type=str2bool,
        default=False,
        help="WONN: use causal attention instead of bidirectional.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=232,
        help="Transformer embedding dimension (baseline only).",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Transformer attention heads (baseline only).",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Transformer encoder layers (baseline only).",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=None,
        help="Transformer FFN dimension (default 2*d_model; baseline only).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to a plain-text file. Mutually exclusive with --data_dir.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Directory containing metadata.json and {train,valid,test}_ids.pt. "
            "Use scripts/prepare_wikitext.py to create it."
        ),
    )
    parser.add_argument("--mask_prob", type=float, default=0.15)

    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--ch", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--theta_init_sigma", type=float, default=0.1)
    parser.add_argument("--rope", type=str2bool, default=True)

    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="If > 0, stop each epoch after this many training steps.")
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Metrics:
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output["loss"]

        mask = labels != -100
        preds = output["logits"].argmax(dim=-1)

        total_loss += float(loss.item()) * int(mask.sum().item())
        total_tokens += int(mask.sum().item())
        total_correct += int((preds == labels).masked_fill(~mask, False).sum().item())

    if total_tokens == 0:
        return {"loss": float("inf"), "ppl": float("inf"), "acc": 0.0}

    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "ppl": float(torch.exp(torch.tensor(avg_loss)).item()),
        "acc": total_correct / total_tokens,
    }


def main() -> None:
    args = parse_args()

    seed_everything(args.seed, deterministic=args.deterministic)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is not None:
        metadata_path = Path(args.data_dir) / "metadata.json"
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
        vocab_size = int(metadata["vocab_size"])
        pad_token_id = int(metadata["pad_token_id"])
        mask_token_id = int(metadata["mask_token_id"])
        train_set = TokenizedCorpusDataset(
            data_dir=args.data_dir,
            split="train",
            seq_len=args.seq_len,
        )
        eval_set = TokenizedCorpusDataset(
            data_dir=args.data_dir,
            split="valid",
            seq_len=args.seq_len,
        )
    elif args.data_path is not None:
        full_dataset = CharCorpusDataset(
            text_or_path=args.data_path,
            seq_len=args.seq_len,
        )
        vocab_size = len(full_dataset.vocab)
        n_train = int(0.9 * len(full_dataset))
        train_set, eval_set = torch.utils.data.random_split(
            full_dataset,
            [n_train, len(full_dataset) - n_train],
            generator=torch.Generator().manual_seed(args.seed),
        )
        pad_token_id = full_dataset.pad_token_id
        mask_token_id = full_dataset.mask_token_id
    else:
        vocab_size = args.vocab_size or 128
        train_set = RandomTokenDataset(
            vocab_size=vocab_size, seq_len=args.seq_len, num_samples=1000
        )
        eval_set = RandomTokenDataset(
            vocab_size=vocab_size, seq_len=args.seq_len, num_samples=200
        )
        pad_token_id = 0
        mask_token_id = vocab_size - 1

    collate_fn = make_mlm_collate_fn(
        mask_token_id=mask_token_id,
        mask_probability=args.mask_prob,
        pad_token_id=pad_token_id,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batchsize,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=args.eval_batchsize,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if args.model == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            max_seq_len=args.seq_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            dropout=args.gamma,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
        ).to(device)
    else:
        model = WONNText(
            vocab_size=vocab_size,
            ch=args.ch,
            max_seq_len=args.seq_len,
            L=args.L,
            T=args.T,
            heads=args.heads,
            gamma=args.gamma,
            rope=args.rope,
            causal=args.causal,
            theta_init_sigma=args.theta_init_sigma,
            omega_as_token_embed=args.omega_as_token_embed,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")

        for step, batch in enumerate(pbar):
            if args.max_steps > 0 and step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output["loss"]
            loss.backward()

            if args.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            metrics = evaluate(model, eval_loader, device)
            print(
                f"[eval] loss={metrics['loss']:.4f} "
                f"ppl={metrics['ppl']:.2f} acc={metrics['acc']:.4f}"
            )

    checkpoint_path = save_dir / "final.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
