"""Online training on the +/- arithmetic stream for the 3x3 architecture study.

Trains WONNText / classical Transformer / Universal Transformer on fresh batches
from :class:`OnlineMathDataset` (no epochs, no overfitting), evaluating masked-
answer accuracy on the in-distribution and extrapolation held-out sets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wonntext.baseline_transformer import TransformerLM, UniversalTransformerLM
from wonntext.hier_model import HierarchicalWONNText
from wonntext.math_data import OnlineMathDataset
from wonntext.model import WONNText
from wonntext.muon import build_optimizer
from wonntext.utils import seed_everything, str2bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online +/- arithmetic training.")
    p.add_argument("--data_dir", default="data/math")
    p.add_argument(
        "--model", choices=["wonn", "wonn_hier", "classical", "universal"], required=True
    )
    p.add_argument("--exp_name", default="math")
    p.add_argument("--save_dir", default="runs/math")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")

    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_batch", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--adamw_lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--lr_min_frac", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--amp", type=str2bool, default=False)

    # WONN
    p.add_argument("--ch", type=int, default=256)
    p.add_argument("--L", type=int, default=1)
    p.add_argument("--T", type=int, default=8)
    # Hierarchical WONN (fast/slow): n_cycles*tau = total fast steps; `segments`
    # deep-supervision refinement passes; readout in {fast, slow, sum}.
    p.add_argument("--n_cycles", type=int, default=2)
    p.add_argument("--tau", type=int, default=4)
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--slow_scale", type=float, default=0.25)
    p.add_argument("--readout", choices=["fast", "slow", "sum"], default="slow")
    p.add_argument("--gamma", type=float, default=0.1,
                   help="Winfree coupling strength (vanilla gamma; hier gamma_f=gamma_s)")
    # transformers
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8,
                   help="n_layers (classical) / n_steps (universal)")
    # shared
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--qk_norm", type=str2bool, default=False)
    return p.parse_args()


def build_model(args: argparse.Namespace, meta: dict) -> torch.nn.Module:
    v, mask_id = meta["vocab_size"], meta["mask_token_id"]
    if args.model == "wonn":
        return WONNText(
            vocab_size=v, ch=args.ch, max_seq_len=meta["seq_len"], L=args.L,
            T=args.T, heads=args.heads, gamma=args.gamma, mask_token_id=mask_id,
        )
    if args.model == "wonn_hier":
        return HierarchicalWONNText(
            vocab_size=v, ch=args.ch, max_seq_len=meta["seq_len"], heads=args.heads,
            n_cycles=args.n_cycles, tau=args.tau, segments=args.segments,
            slow_scale=args.slow_scale, readout=args.readout,
            gamma_f=args.gamma, gamma_s=args.gamma, mask_token_id=mask_id,
        )
    cls = TransformerLM if args.model == "classical" else UniversalTransformerLM
    depth_kw = "n_layers" if args.model == "classical" else "n_steps"
    return cls(
        vocab_size=v, dim=args.dim, heads=args.heads, mask_token_id=mask_id,
        qk_norm=args.qk_norm, **{depth_kw: args.depth},
    )


def mask_batch(
    ids: torch.Tensor, amask: torch.Tensor, mask_id: int, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked = ids.clone()
    masked[amask] = mask_id
    labels = torch.full_like(ids, -100)
    labels[amask] = ids[amask]
    attn = (ids != pad_id).long()
    return masked, labels, attn


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, ids: torch.Tensor, amask: torch.Tensor,
    mask_id: int, pad_id: int, device: torch.device, bs: int,
) -> dict[str, float]:
    model.eval()
    tok_ok = tok_n = ans_ok = ans_n = 0
    for i in range(0, len(ids), bs):
        x, am = ids[i : i + bs].to(device), amask[i : i + bs].to(device)
        masked, labels, attn = mask_batch(x, am, mask_id, pad_id)
        out = model(masked, attention_mask=attn.to(device))
        logits = out["logits"] if isinstance(out, dict) else out
        pred = logits.argmax(-1)
        hit = (pred == labels) & am
        tok_ok += int(hit.sum())
        tok_n += int(am.sum())
        row_ok = ((pred == labels) | ~am).all(dim=1)
        ans_ok += int(row_ok.sum())
        ans_n += len(x)
    model.train()
    return {"token_acc": tok_ok / max(tok_n, 1), "answer_acc": ans_ok / max(ans_n, 1)}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    if args.model in ("wonn", "wonn_hier") and args.amp:
        raise SystemExit("WONN must run fp32 (--amp false); bf16 floors oscillator precision.")

    data = Path(args.data_dir)
    meta = json.loads((data / "metadata.json").read_text())
    pad_id, mask_id = meta["pad_token_id"], meta["mask_token_id"]
    heldout = set(json.loads((data / "heldout_questions.json").read_text()))
    test = (torch.load(data / "test_ids.pt"), torch.load(data / "test_answer_mask.pt"))
    extra = (
        torch.load(data / "extrapolation_ids.pt"),
        torch.load(data / "extrapolation_answer_mask.pt"),
    )

    td = meta["train_distribution"]
    lo, hi = td["operands"]
    dlo, dhi = td["digits"]
    stream = OnlineMathDataset(
        lo, hi, dlo, dhi, seq_len=meta["seq_len"],
        ops_pool=td.get("ops", "+-*/"), max_result=td.get("max_result", 1_000_000),
        exclude=heldout, seed=args.seed,
    )
    loader = iter(DataLoader(stream, batch_size=args.batch_size, num_workers=args.num_workers))

    model = build_model(args, meta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={n_params:,} device={device}", flush=True)

    if args.optimizer == "muon":
        opt = build_optimizer(model, lr=args.lr, adamw_lr=args.adamw_lr,
                              weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.adamw_lr,
                               betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_scale(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr_min_frac + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * prog))

    base_lrs = [g["lr"] for g in opt.param_groups]

    out = Path(args.save_dir) / args.exp_name
    out.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out / "tb"))
    except Exception as exc:
        writer = None
        print(f"tensorboard disabled: {exc}", flush=True)

    model.train()
    for step in range(args.steps):
        for g, base in zip(opt.param_groups, base_lrs, strict=True):
            g["lr"] = base * lr_scale(step)
        batch = next(loader)
        masked, labels, attn = mask_batch(batch["input_ids"], batch["answer_mask"], mask_id, pad_id)
        masked, labels, attn = masked.to(device), labels.to(device), attn.to(device)
        # bf16 autocast needs no GradScaler (that is for fp16).
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            loss = model(masked, attention_mask=attn, labels=labels)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            t = evaluate(model, *test, mask_id, pad_id, device, args.eval_batch)
            e = evaluate(model, *extra, mask_id, pad_id, device, args.eval_batch)
            print(
                f"step {step + 1}/{args.steps} loss={float(loss):.4f} "
                f"| indist tok={t['token_acc']:.4f} ans={t['answer_acc']:.4f} "
                f"| extrap tok={e['token_acc']:.4f} ans={e['answer_acc']:.4f}",
                flush=True,
            )
            if writer is not None:
                s = step + 1
                writer.add_scalar("train/loss", float(loss), s)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], s)
                writer.add_scalar("indist/token_acc", t["token_acc"], s)
                writer.add_scalar("indist/answer_acc", t["answer_acc"], s)
                writer.add_scalar("extrap/token_acc", e["token_acc"], s)
                writer.add_scalar("extrap/answer_acc", e["answer_acc"], s)
                writer.flush()

    torch.save(model.state_dict(), out / "final.pt")
    metrics = {
        "model": args.model, "params": n_params, "steps": args.steps,
        "in_distribution": evaluate(model, *test, mask_id, pad_id, device, args.eval_batch),
        "extrapolation": evaluate(model, *extra, mask_id, pad_id, device, args.eval_batch),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if writer is not None:
        final_indist = metrics["in_distribution"]["answer_acc"]
        final_extrap = metrics["extrapolation"]["answer_acc"]
        writer.add_scalar("final/indist_answer_acc", final_indist, args.steps)
        writer.add_scalar("final/extrap_answer_acc", final_extrap, args.steps)
        writer.close()
    print(f"saved {out}/final.pt\n{json.dumps(metrics, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
