from __future__ import annotations

import argparse
from pathlib import Path

from src.poetry_generator import DEFAULT_CHECKPOINT, TransformerPoetryGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tang poetry with a Transformer.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--start", default=None, help="Optional prompt; skips the prompt.")
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Minimum number of new word tokens before waiting for a full stop.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--repetition-window", type=int, default=20)
    parser.add_argument("--line-width", type=int, default=12)
    parser.add_argument(
        "--max-extra-tokens",
        type=int,
        default=32,
        help="Maximum extra tokens allowed while waiting for the Chinese full stop.",
    )
    return parser.parse_args()


def prompt_for_value(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty.")


def main() -> None:
    args = parse_args()
    generator = TransformerPoetryGenerator(args.checkpoint, device=args.device)

    start_text = args.start
    if start_text is None:
        start_text = prompt_for_value("请输入起始字或句子：")

    length = args.length
    if length is None:
        while True:
            value = input("请输入最少生成词数：").strip()
            try:
                length = int(value)
            except ValueError:
                print("请输入一个整数。")
                continue
            if length > 0:
                break
            print("请输入一个正数。")

    poem = generator.generate(
        start_text=start_text,
        poem_length=length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        line_width=args.line_width,
        max_extra_tokens=args.max_extra_tokens,
    )
    print("\n生成结果：")
    print(poem)


if __name__ == "__main__":
    main()
