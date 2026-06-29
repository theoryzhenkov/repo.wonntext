"""Fine-tune (or train from scratch) WONNText / Transformer on two-digit arithmetic.

The task is masked answer prediction. The input is a tokenized equation such as
"23+45=68" with the answer digits replaced by the mask token; the model must
predict only the answer digits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import tqdm

from wonntext.baseline_transformer import TransformerLM
from wonntext.model import WONNText
from wonntext.utils import seed_everything, str2bool

Metrics = dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/fine-tune arithmetic models.")

    parser.add_argument("--exp_name", type=str, default="arithmetic")
    parser.add_argument("--save_dir", type=str, default="runs/arithmetic")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional pretrained checkpoint to load before training.")

    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--deterministic", type=str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--eval_batchsize", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_min", type=float, default=0.0)
    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--grad_checkpoint", type=str2bool, default=False)
    parser.add_argument(
        "--winfree_mode",
        type=str,
        choices=["recurrent", "predictor_corrector", "parallel_scan", "parallel_scan_refined"],
        default="recurrent",
        help="Winfree layer forward mode (WONN only).",
    )
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument(
        "--model",
        type=str,
        choices=["wonn", "transformer"],
        default="wonn",
    )
    parser.add_argument("--omega_as_token_embed", type=str2bool, default=True)
    parser.add_argument("--causal", type=str2bool, default=False)

    parser.add_argument("--d_model", type=int, default=232)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=None)

    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--ch", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--theta_init_sigma", type=float, default=0.1)
    parser.add_argument("--rope", type=str2bool, default=True)

    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


class ArithmeticDataset(torch.utils.data.Dataset):
    """Pre-tokenized arithmetic equations with answer masks."""

    def __init__(self, data_dir: str | Path, split: str) -> None:
        data_dir = Path(data_dir)
        self.input_ids = torch.load(
            data_dir / f"{split}_ids.pt", map_location="cpu", weights_only=True
        ).long()
        self.answer_mask = torch.load(
            data_dir / f"{split}_answer_mask.pt",
            map_location="cpu",
            weights_only=True,
        ).bool()

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = self.input_ids[index]
        return {
            "input_ids": ids,
            "attention_mask": (ids != 0).long(),
            "answer_mask": self.answer_mask[index],
        }


def make_batch(
    batch: list[dict[str, torch.Tensor]],
    mask_token_id: int,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answer_mask = torch.stack([b["answer_mask"] for b in batch])

    labels = torch.full_like(input_ids, ignore_index)
    labels[answer_mask] = input_ids[answer_mask]

    masked_input = input_ids.clone()
    masked_input[answer_mask] = mask_token_id

    return {
        "input_ids": masked_input,
        "attention_mask": attention_mask,
        "labels": labels,
        "answer_mask": answer_mask,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = False,
) -> Metrics:
    model.eval()

    total_loss = 0.0
    total_answer_tokens = 0
    total_correct_tokens = 0
    total_full_correct = 0
    total_examples = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        answer_mask = batch["answer_mask"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output["loss"]

        logits = output["logits"]
        preds = logits.argmax(dim=-1)

        total_loss += float(loss.item()) * int(answer_mask.sum().item())
        total_answer_tokens += int(answer_mask.sum().item())
        total_correct_tokens += int(
            ((preds == labels) & answer_mask).sum().item()
        )

        # Whole-answer accuracy: every answer token must match for the example.
        answer_correct = ((preds == labels) | ~answer_mask).all(dim=-1)
        total_full_correct += int(answer_correct.sum().item())
        total_examples += input_ids.shape[0]

    return {
        "loss": total_loss / max(total_answer_tokens, 1),
        "ppl": float(torch.exp(torch.tensor(total_loss / max(total_answer_tokens, 1))).item()),
        "token_acc": total_correct_tokens / max(total_answer_tokens, 1),
        "answer_acc": total_full_correct / max(total_examples, 1),
    }


def _make_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    lr: float,
    lr_min: float,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(0, min(warmup_steps, total_steps))
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_step = step - warmup_steps
        progress = min(1.0, decay_step / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (lr_min + (lr - lr_min) * cosine) / lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.data_dir) / "metadata.json"
    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    vocab_size = int(metadata["vocab_size"])
    pad_token_id = int(metadata["pad_token_id"])
    mask_token_id = int(metadata["mask_token_id"])
    seq_len = int(metadata["seq_len"])

    train_set = ArithmeticDataset(args.data_dir, "train")
    val_set = ArithmeticDataset(args.data_dir, "valid")
    test_set = ArithmeticDataset(args.data_dir, "test")

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return make_batch(batch, mask_token_id=mask_token_id)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batchsize,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.eval_batchsize,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.eval_batchsize,
        shuffle=False,
        collate_fn=collate_fn,
    )

    if args.model == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            max_seq_len=max(seq_len, 256),
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
            max_seq_len=max(seq_len, 256),
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
            grad_checkpoint=args.grad_checkpoint,
            winfree_mode=args.winfree_mode,
        ).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"Checkpoint missing keys: {missing}")
        if unexpected:
            print(f"Checkpoint unexpected keys: {unexpected}")
        print(f"Loaded checkpoint from {args.checkpoint}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = _make_optimizer(args, model)
    scheduler = _make_scheduler(
        optimizer=optimizer,
        lr=args.lr,
        lr_min=args.lr_min,
        warmup_steps=args.warmup_steps,
        total_steps=args.epochs * len(train_loader),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_val_acc = 0.0
    best_state: dict | None = None

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                output = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = output["loss"]

            scaler.scale(loss).backward()

            if args.clip_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad_norm
                )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            val_metrics = evaluate(model, val_loader, device, amp=args.amp)
            print(
                f"[eval] loss={val_metrics['loss']:.4f} "
                f"ppl={val_metrics['ppl']:.2f} "
                f"token_acc={val_metrics['token_acc']:.4f} "
                f"answer_acc={val_metrics['answer_acc']:.4f}"
            )
            if val_metrics["answer_acc"] > best_val_acc:
                best_val_acc = val_metrics["answer_acc"]
                best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, amp=args.amp)
    print(
        f"[test] loss={test_metrics['loss']:.4f} "
        f"ppl={test_metrics['ppl']:.2f} "
        f"token_acc={test_metrics['token_acc']:.4f} "
        f"answer_acc={test_metrics['answer_acc']:.4f}"
    )

    final_path = save_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved checkpoint to {final_path}")

    metrics_path = save_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
