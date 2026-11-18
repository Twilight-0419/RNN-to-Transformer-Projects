from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import PackedTranslationDataset, collate_translation_batch
from src.nmt_model import EncoderDecoderTransformer
from src.optimizer import AdamW, Cross_Entropy_Loss, Gridient_Clipping, Lr_Cos_Schedule
from src.text import PAD_TOKEN, load_chinese_vocabulary


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Chinese-English Transformer translator.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--chinese-vocab", type=Path, default=ROOT / "models" / "rnn_vocab.pt")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=80)
    parser.add_argument("--use-rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=3)

    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--max-val-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-name", default="translator_transformer.pt")
    return parser.parse_args()


def load_metadata(data_dir: Path) -> dict:
    return json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))


def model_config_from_args(args: argparse.Namespace, metadata: dict) -> dict:
    return {
        "source_vocab_size": metadata["chinese_vocab_size"],
        "target_vocab_size": metadata["english_vocab_size"],
        "d_model": args.d_model,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "max_seq_len": args.max_seq_len,
        "use_rope": args.use_rope,
        "theta": args.theta,
    }


def build_model(
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> EncoderDecoderTransformer:
    return EncoderDecoderTransformer(
        **config,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)


def save_checkpoint(
    path: Path,
    model: EncoderDecoderTransformer,
    optimizer: AdamW,
    model_config: dict,
    metadata: dict,
    chinese_word2index: dict,
    iteration: int,
    epoch: int,
    val_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "epoch": epoch,
        "model_config": model_config,
        "data_metadata": metadata,
        "chinese_word2index": chinese_word2index,
        "val_loss": val_loss,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = flat_labels != PAD_TOKEN
    return Cross_Entropy_Loss(flat_labels[valid], flat_logits[valid])


@torch.no_grad()
def evaluate(
    model: EncoderDecoderTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for batch in loader:
        if max_batches > 0 and batches >= max_batches:
            break
        source_ids = batch["source_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        target_labels = batch["target_labels"].to(device)
        source_mask = batch["source_mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        logits = model(source_ids, decoder_input_ids, source_mask, target_mask)
        total += masked_cross_entropy(logits, target_labels).item()
        batches += 1
    model.train()
    if batches == 0:
        raise ValueError("validation loader produced no batches")
    return total / batches


def main() -> None:
    args = parse_args()
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    metadata = load_metadata(args.data_dir)
    chinese_word2index, chinese_vocab_size = load_chinese_vocabulary(args.chinese_vocab)
    if chinese_vocab_size != metadata["chinese_vocab_size"]:
        raise ValueError("Chinese vocabulary does not match processed data metadata")
    config = model_config_from_args(args, metadata)
    model = build_model(config, device, dtype)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_dataset = PackedTranslationDataset(
        args.data_dir / "train_source.npy",
        args.data_dir / "train_target.npy",
    )
    val_dataset = PackedTranslationDataset(
        args.data_dir / "val_source.npy",
        args.data_dir / "val_target.npy",
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("processed train or validation split is empty")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_translation_batch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_translation_batch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    start_epoch = 0
    last_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    checks_without_improvement = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["model_config"] != config:
            raise ValueError("resume checkpoint model configuration does not match")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        last_epoch = start_epoch
        global_step = int(checkpoint.get("iteration", 0))
        best_val_loss = float(checkpoint.get("val_loss") or float("inf"))
        print(f"Resumed from {args.resume} at epoch={start_epoch}, step={global_step}")

    batches_per_epoch = len(train_loader)
    if args.max_batches_per_epoch > 0:
        batches_per_epoch = min(batches_per_epoch, args.max_batches_per_epoch)
    total_steps = max(
        args.warmup_steps + 1,
        args.epochs * batches_per_epoch,
    )
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} dtype={args.dtype} params={num_params:,} "
        f"train_pairs={len(train_dataset):,} val_pairs={len(val_dataset):,}",
        flush=True,
    )
    print(
        f"source_vocab={config['source_vocab_size']:,} "
        f"bpe_target_vocab={config['target_vocab_size']:,}",
        flush=True,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    running_loss = 0.0
    running_steps = 0
    last_log_time = time.time()
    stop_training = False
    model.train()

    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        for batch_index, batch in enumerate(train_loader):
            if args.max_batches_per_epoch > 0 and batch_index >= args.max_batches_per_epoch:
                break
            lr = Lr_Cos_Schedule(
                global_step,
                args.lr,
                args.min_lr,
                args.warmup_steps,
                total_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            source_ids = batch["source_ids"].to(device, non_blocking=True)
            decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
            target_labels = batch["target_labels"].to(device, non_blocking=True)
            source_mask = batch["source_mask"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(source_ids, decoder_input_ids, source_mask, target_mask)
            loss = masked_cross_entropy(logits, target_labels)
            loss.backward()
            if args.grad_clip > 0:
                Gridient_Clipping(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            running_steps += 1

            if global_step % args.log_interval == 0:
                now = time.time()
                elapsed = max(now - last_log_time, 1e-9)
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {global_step}/{total_steps} | "
                    f"lr {lr:.2e} | loss {running_loss / running_steps:.4f} | "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
                running_loss = 0.0
                running_steps = 0
                last_log_time = now

            if global_step % args.val_interval == 0:
                val_loss = evaluate(model, val_loader, device, args.max_val_batches)
                print(
                    f"step {global_step} | val loss {val_loss:.4f} | "
                    f"val ppl {math.exp(min(val_loss, 20.0)):.2f}",
                    flush=True,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checks_without_improvement = 0
                    best_path = args.checkpoint_dir / "translator_best.pt"
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        config,
                        metadata,
                        chinese_word2index,
                        global_step,
                        epoch,
                        val_loss,
                    )
                    print(f"saved best checkpoint: {best_path}", flush=True)
                else:
                    checks_without_improvement += 1
                    if (
                        args.early_stop_patience > 0
                        and checks_without_improvement >= args.early_stop_patience
                    ):
                        print("early stopping: validation loss did not improve", flush=True)
                        stop_training = True

            if global_step % 1000 == 0:
                latest_path = args.checkpoint_dir / "translator_latest.pt"
                save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    config,
                    metadata,
                    chinese_word2index,
                    global_step,
                    epoch,
                )
                print(f"saved checkpoint: {latest_path}", flush=True)

            if stop_training or (args.max_steps > 0 and global_step >= args.max_steps):
                stop_training = True
                break
        if stop_training:
            break

    final_path = args.checkpoint_dir / args.final_name
    save_checkpoint(
        final_path,
        model,
        optimizer,
        config,
        metadata,
        chinese_word2index,
        global_step,
        last_epoch,
        best_val_loss if math.isfinite(best_val_loss) else None,
    )
    print(f"saved final checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    main()
