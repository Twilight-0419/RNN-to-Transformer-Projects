from __future__ import annotations

import argparse
from pathlib import Path

from src.translator import DEFAULT_CHECKPOINT, TransformerTranslator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chinese-English Transformer translator.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    translator = TransformerTranslator(args.checkpoint, device=args.device)

    if args.text is not None:
        print(translator.translate(
            args.text,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        ))
        return

    while True:
        sentence = input("请输入中文句子: ").strip()
        if not sentence:
            print("输入不能为空。")
            continue
        translation = translator.translate(
            sentence,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        print("翻译结果:", translation)
        if input("是否继续翻译？(y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    main()
