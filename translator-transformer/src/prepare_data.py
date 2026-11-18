from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path

import numpy as np

from src.text import (
    ENGLISH_SPECIAL_TOKENS,
    SOS_TOKEN,
    encode_chinese,
    encode_english,
    load_chinese_vocabulary,
    load_english_bpe,
    special_id,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack the RNN translation corpus for Transformer training.")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "zh-en.txt")
    parser.add_argument("--chinese-vocab", type=Path, default=ROOT / "models" / "rnn_vocab.pt")
    parser.add_argument("--bpe-vocab", type=Path, default=ROOT / "models" / "bpe_vocab.json")
    parser.add_argument("--bpe-merges", type=Path, default=ROOT / "models" / "bpe_merges.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--max-source-tokens", type=int, default=32)
    parser.add_argument("--max-target-tokens", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 processes the full corpus.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def bucket_for_index(index: int) -> str:
    bucket = ((index + 1) * 2654435761) & 0xFFFFFFFF
    bucket %= 100
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.glob("*.npy"))
    if existing and not args.overwrite:
        raise FileExistsError("processed arrays already exist; add --overwrite to replace them")

    chinese_word2index, chinese_vocab_size = load_chinese_vocabulary(args.chinese_vocab)
    english_tokenizer = load_english_bpe(args.bpe_vocab, args.bpe_merges)
    english_pad = special_id(english_tokenizer, "<pad>")
    english_sos = SOS_TOKEN
    english_eos = special_id(english_tokenizer, "<eos>")

    splits = {
        name: {
            "source": array("H"),
            "target": array("H"),
            "source_offsets": array("Q", [0]),
            "target_offsets": array("Q", [0]),
            "pairs": 0,
        }
        for name in ("train", "val", "test")
    }

    processed = 0
    skipped = 0
    with args.corpus.open("r", encoding="utf-8") as corpus:
        for index, line in enumerate(corpus):
            if args.max_pairs > 0 and processed >= args.max_pairs:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                skipped += 1
                continue
            chinese, english = parts
            source_ids = encode_chinese(
                chinese,
                chinese_word2index,
                max_length=args.max_source_tokens,
            )
            target_ids = encode_english(
                english,
                english_tokenizer,
                max_length=args.max_target_tokens,
            )
            split = splits[bucket_for_index(index)]
            split["source"].extend(source_ids)
            split["target"].extend(target_ids)
            split["source_offsets"].append(len(split["source"]))
            split["target_offsets"].append(len(split["target"]))
            split["pairs"] += 1
            processed += 1
            if processed % 50000 == 0:
                print(f"processed {processed:,} pairs")

    metadata = {
        "chinese_vocab_size": chinese_vocab_size,
        "english_vocab_size": len(english_tokenizer.vocab),
        "english_special_tokens": ENGLISH_SPECIAL_TOKENS,
        "english_pad_id": english_pad,
        "english_sos_id": english_sos,
        "english_eos_id": english_eos,
        "max_source_tokens": args.max_source_tokens,
        "max_target_tokens": args.max_target_tokens,
        "processed_pairs": processed,
        "skipped_lines": skipped,
        "splits": {},
    }

    for name, split in splits.items():
        source_path = args.output_dir / f"{name}_source.npy"
        target_path = args.output_dir / f"{name}_target.npy"
        source_offsets_path = args.output_dir / f"{name}_source_offsets.npy"
        target_offsets_path = args.output_dir / f"{name}_target_offsets.npy"
        np.save(source_path, np.fromiter(split["source"], dtype=np.uint16))
        np.save(target_path, np.fromiter(split["target"], dtype=np.uint16))
        np.save(source_offsets_path, np.fromiter(split["source_offsets"], dtype=np.int64))
        np.save(target_offsets_path, np.fromiter(split["target_offsets"], dtype=np.int64))
        metadata["splits"][name] = {
            "pairs": split["pairs"],
            "source_tokens": len(split["source"]),
            "target_tokens": len(split["target"]),
            "source_path": source_path.name,
            "target_path": target_path.name,
            "source_offsets_path": source_offsets_path.name,
            "target_offsets_path": target_offsets_path.name,
        }
        print(
            f"{name}: {split['pairs']:,} pairs, "
            f"{len(split['source']):,} source tokens, "
            f"{len(split['target']):,} target tokens"
        )

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved processed data to {args.output_dir}")


if __name__ == "__main__":
    main()
