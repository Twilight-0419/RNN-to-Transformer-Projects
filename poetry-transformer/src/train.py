from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.data import get_batch, load_token_file
from src.optimizer import (
    AdamW,
    Cross_Entropy_Loss,
    Gridient_Clipping,
    Lr_Cos_Schedule,
)
from src.tokenizer import WordTokenizer
from src.transformer_model import TransformerLM


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the poetry Transformer language model.")
    parser.add_argument("--train-data", type=Path, default=ROOT / "data" / "train.bin")
    parser.add_argument("--val-data", type=Path, default=ROOT / "data" / "val.bin")
    parser.add_argument("--vocab", type=Path, default=ROOT / "data" / "vocab.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--use-rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--cosine-cycle-iters", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many validation checks without improvement; 0 disables it.",
    )
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-name", default="poetry_transformer.pt")
    return parser.parse_args()


def make_model_config(args: argparse.Namespace, vocab_size: int) -> dict:
    return {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "num_layers": args.num_layers,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "use_rope": args.use_rope,
        "theta": args.theta,
    }


def build_model(config: dict, device: torch.device, dtype: torch.dtype) -> TransformerLM:
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        device=device,
        dtype=dtype,
        use_rope=config["use_rope"],
        theta=config["theta"],
    )
    return model.to(device=device, dtype=dtype)


def save_checkpoint(
    path: Path,
    model: TransformerLM,
    optimizer: AdamW,
    iteration: int,
    model_config: dict,
    tokenizer: WordTokenizer,
    training_config: dict,
    val_loss: float | None = None,
    best_val_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "model_config": model_config,
        "vocab": tokenizer.to_dict(),
        "training_config": training_config,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for inputs, targets in batches:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        total += Cross_Entropy_Loss(targets, logits).item()
    model.train()
    return total / len(batches)


def build_eval_batches(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    batches: int,
    seed: int = 12345,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    num_starts = len(data) - context_length
    if num_starts <= 0:
        raise ValueError("validation data must be longer than context_length")
    result = []
    for _ in range(batches):
        starts = rng.integers(0, num_starts, size=batch_size)
        indices = starts[:, None] + np.arange(context_length)[None, :]
        inputs = torch.as_tensor(data[indices].astype(np.int64), dtype=torch.long)
        targets = torch.as_tensor(data[indices + 1].astype(np.int64), dtype=torch.long)
        result.append((inputs, targets))
    return result


def main() -> None:
    args = parse_args()
    if args.warmup_iters >= args.max_iters:
        raise ValueError("--warmup-iters must be smaller than --max-iters")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    tokenizer = WordTokenizer.load(args.vocab)
    train_data = load_token_file(args.train_data)
    val_data = load_token_file(args.val_data)
    if len(train_data) <= args.context_length:
        raise ValueError("training data is shorter than --context-length")
    eval_batches = build_eval_batches(
        val_data,
        args.batch_size,
        args.context_length,
        args.val_batches,
        seed=args.seed,
    )

    model_config = make_model_config(args, tokenizer.vocab_size)
    model = build_model(model_config, device, dtype)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    start_iter = 0
    best_val_loss = float("inf")
    validation_checks_without_improvement = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("model_config") != model_config:
            raise ValueError(
                "Resume checkpoint model configuration differs from the current arguments."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = int(checkpoint["iteration"])
        best_val_loss = float(checkpoint.get("best_val_loss") or float("inf"))
        print(f"Resumed from {args.resume} at iteration {start_iter}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    cosine_cycle_iters = args.cosine_cycle_iters or args.max_iters
    cosine_cycle_iters = max(cosine_cycle_iters, args.warmup_iters + 1)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} dtype={args.dtype} vocab={tokenizer.vocab_size:,} "
        f"params={num_params:,}",
        flush=True,
    )
    print(
        f"train_tokens={len(train_data):,} val_tokens={len(val_data):,} "
        f"iterations={start_iter}->{args.max_iters}",
        flush=True,
    )

    model.train()
    running_loss = 0.0
    running_steps = 0
    tokens_since_log = 0
    last_log_time = time.time()
    last_iteration = start_iter
    stop_requested = False

    for iteration in range(start_iter, args.max_iters):
        last_iteration = iteration + 1
        lr = Lr_Cos_Schedule(
            iteration,
            args.lr,
            args.min_lr,
            args.warmup_iters,
            cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            device,
        )
        optimizer.zero_grad()
        logits = model(inputs)
        loss = Cross_Entropy_Loss(targets, logits)
        loss.backward()
        if args.grad_clip > 0:
            Gridient_Clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss.item()
        running_steps += 1
        tokens_since_log += args.batch_size * args.context_length

        if (iteration + 1) % args.log_interval == 0:
            now = time.time()
            elapsed = max(now - last_log_time, 1e-9)
            throughput = tokens_since_log / elapsed
            print(
                f"iter {iteration + 1}/{args.max_iters} | lr {lr:.2e} | "
                f"loss {running_loss / running_steps:.4f} | "
                f"{throughput:.0f} tok/s | {elapsed:.1f}s",
                flush=True,
            )
            running_loss = 0.0
            running_steps = 0
            tokens_since_log = 0
            last_log_time = now

        if (iteration + 1) % args.val_interval == 0:
            val_loss = evaluate(
                model,
                eval_batches,
                device,
            )
            print(
                f"iter {iteration + 1}/{args.max_iters} | "
                f"val loss {val_loss:.4f} | val ppl {math.exp(val_loss):.2f}",
                flush=True,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                validation_checks_without_improvement = 0
                best_path = args.checkpoint_dir / "checkpoint_best.pt"
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    iteration + 1,
                    model_config,
                    tokenizer,
                    training_config,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                )
                print(f"saved best checkpoint: {best_path}", flush=True)
            else:
                validation_checks_without_improvement += 1
                if (
                    args.early_stop_patience > 0
                    and validation_checks_without_improvement
                    >= args.early_stop_patience
                ):
                    print(
                        "early stopping: validation loss has not improved for "
                        f"{validation_checks_without_improvement} checks",
                        flush=True,
                    )
                    stop_requested = True

        if (iteration + 1) % args.save_interval == 0:
            latest_path = args.checkpoint_dir / "checkpoint_latest.pt"
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                iteration + 1,
                model_config,
                tokenizer,
                training_config,
                val_loss=val_loss if (iteration + 1) % args.val_interval == 0 else None,
                best_val_loss=best_val_loss,
            )
            print(f"saved checkpoint: {latest_path}", flush=True)

        if stop_requested:
            break

    final_path = args.checkpoint_dir / args.final_name
    save_checkpoint(
        final_path,
        model,
        optimizer,
        last_iteration,
        model_config,
        tokenizer,
        training_config,
        best_val_loss=best_val_loss,
    )
    print(f"saved final checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    main()
