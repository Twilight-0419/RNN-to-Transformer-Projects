from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import jieba
import torch

from src.bpe_tokenizer import Tokenizer


PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
ENGLISH_SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>"]


def normalize_english(text: str) -> str:
    """Match the normalization used by the original RNN translator notebook."""
    text = text.lower().strip()
    text = re.sub(r'([.,!?;:\'"\-()%$])', r" \1", text)
    text = re.sub(r'[^a-zA-Z0-9.,!?;:\'"\-()%$]+', r" ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode_chinese(
    text: str,
    word2index: dict[str, int],
    max_length: int = 32,
) -> list[int]:
    token_ids = [word2index.get(word, UNK_TOKEN) for word in jieba.lcut(text)]
    token_ids = token_ids[: max(1, max_length - 1)]
    return token_ids + [EOS_TOKEN]


def load_chinese_vocabulary(path: str | Path) -> tuple[dict[str, int], int]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    word2index = state["chinese_word2index"]
    return word2index, int(state["chinese_word_count"])


def load_english_bpe(
    vocab_path: str | Path,
    merges_path: str | Path,
) -> Tokenizer:
    return Tokenizer.from_files(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=ENGLISH_SPECIAL_TOKENS,
    )


def special_id(tokenizer: Tokenizer, token: str) -> int:
    return tokenizer.token_to_id[token.encode("utf-8")]


def encode_english(
    text: str,
    tokenizer: Tokenizer,
    max_length: int = 64,
) -> list[int]:
    eos_id = special_id(tokenizer, "<eos>")
    token_ids = tokenizer.encode(normalize_english(text))
    token_ids = token_ids[: max(1, max_length - 1)]
    return token_ids + [eos_id]


def decode_english(
    token_ids: Iterable[int],
    tokenizer: Tokenizer,
) -> str:
    special_ids = {
        special_id(tokenizer, "<pad>"),
        special_id(tokenizer, "<sos>"),
        special_id(tokenizer, "<eos>"),
    }
    clean_ids = [int(token_id) for token_id in token_ids if int(token_id) not in special_ids]
    text = tokenizer.decode(clean_ids)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text
