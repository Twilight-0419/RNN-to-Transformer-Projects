from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.tokenizer import (
    build_tokenizer_from_corpus,
    build_tokenizer_from_rnn,
    encode_corpus_with_jieba,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact word IDs from the RNN poetry preprocessing."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "tang_poetry_corpus.txt",
    )
    parser.add_argument(
        "--preprocessing",
        type=Path,
        default=ROOT / "models" / "preprocessing.pt",
    )
    parser.add_argument("--vocab-size", type=int, default=12000)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--force-retokenize",
        action="store_true",
        help="Ignore preprocessing.pt and rebuild the frequency vocabulary from the corpus.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.5 <= args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in [0.5, 1.0)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.bin"
    val_path = args.output_dir / "val.bin"
    vocab_path = args.output_dir / "vocab.json"
    info_path = args.output_dir / "dataset_info.json"
    if not args.overwrite:
        existing = [path for path in (train_path, val_path, vocab_path) if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(f"{names} already exist; add --overwrite to replace them.")

    if args.preprocessing.exists() and not args.force_retokenize:
        print(f"Loading RNN preprocessing: {args.preprocessing}")
        tokenizer, encoded = build_tokenizer_from_rnn(
            args.preprocessing,
            args.vocab_size,
        )
        source = "preprocessing.pt"
    else:
        print(f"Rebuilding jieba vocabulary from: {args.corpus}")
        tokenizer = build_tokenizer_from_corpus(args.corpus, args.vocab_size)
        encoded = encode_corpus_with_jieba(args.corpus, tokenizer)
        source = "corpus_retokenization"

    split_at = int(len(encoded) * args.train_ratio)
    if split_at <= 1 or len(encoded) - split_at <= 1:
        raise ValueError("corpus is too small to create train and validation splits")
    train_tokens = encoded[:split_at]
    val_tokens = encoded[split_at:]

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    tokenizer.save(vocab_path)

    info = {
        "tokenizer_source": source,
        "vocab_size": tokenizer.vocab_size,
        "total_tokens": int(len(encoded)),
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "train_ratio": args.train_ratio,
        "dtype": "uint16",
    }
    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Vocabulary: {tokenizer.vocab_size:,} tokens")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {vocab_path}")


if __name__ == "__main__":
    main()
