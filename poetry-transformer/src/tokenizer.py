from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def jieba_lcut(text: str) -> list[str]:
    """Match the tokenization used by the original RNN preprocessing."""
    import jieba

    return jieba.lcut(text.strip())


class WordTokenizer:
    def __init__(self, tokens: Sequence[str], source: str = "rnn_preprocessing"):
        unique_tokens: list[str] = []
        seen = set(SPECIAL_TOKENS)
        for token in tokens:
            if token not in seen:
                unique_tokens.append(token)
                seen.add(token)
        self.tokens = list(SPECIAL_TOKENS) + unique_tokens
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.source = source

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode_tokens(self, tokens: Iterable[str], char_fallback: bool = False) -> list[int]:
        token_ids: list[int] = []
        for token in tokens:
            token_id = self.stoi.get(token)
            if token_id is not None:
                token_ids.append(token_id)
                continue
            if char_fallback:
                for char in token:
                    token_ids.append(self.stoi.get(char, UNK_ID))
            else:
                token_ids.append(UNK_ID)
        return token_ids

    def encode_prompt(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for line in text.splitlines() or [text]:
            line = line.strip()
            if line:
                token_ids.extend(
                    self.encode_tokens(jieba_lcut(line), char_fallback=True)
                )
        return token_ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if not 0 <= int(token_id) < len(self.tokens):
                continue
            token = self.tokens[int(token_id)]
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces)

    def to_dict(self) -> dict:
        return {
            "type": "rnn_word_frequency",
            "source": self.source,
            "tokens": self.tokens[len(SPECIAL_TOKENS) :],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WordTokenizer":
        return cls(data["tokens"], source=data.get("source", "unknown"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def load_rnn_preprocessing(path: str | Path) -> dict:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"word2idx", "unique_words", "corpus_idx"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"preprocessing.pt is missing fields: {sorted(missing)}")
    return state


def build_tokenizer_from_rnn(
    path: str | Path,
    vocab_size: int,
) -> tuple[WordTokenizer, np.ndarray]:
    state = load_rnn_preprocessing(path)
    corpus_idx = np.asarray(state["corpus_idx"], dtype=np.int64)
    keep_count = vocab_size - len(SPECIAL_TOKENS)
    if keep_count <= 0:
        raise ValueError("vocab_size must be greater than the number of special tokens")

    counts = Counter(corpus_idx.tolist())
    selected = [source_id for source_id, _ in counts.most_common(keep_count)]
    unique_words = state["unique_words"]
    tokens = [str(unique_words[source_id]) for source_id in selected]
    tokenizer = WordTokenizer(tokens, source=str(Path(path)))

    source_to_new = np.full(len(unique_words), UNK_ID, dtype=np.uint16)
    for compact_id, source_id in enumerate(selected, start=len(SPECIAL_TOKENS)):
        source_to_new[source_id] = compact_id
    compact_corpus = source_to_new[corpus_idx]
    return tokenizer, compact_corpus


def count_jieba_tokens(corpus_path: str | Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with Path(corpus_path).open("r", encoding="utf-8") as corpus:
        for line in corpus:
            stripped = line.strip()
            if stripped:
                counts.update(jieba_lcut(stripped))
    return counts


def build_tokenizer_from_corpus(
    corpus_path: str | Path,
    vocab_size: int,
) -> WordTokenizer:
    counts = count_jieba_tokens(corpus_path)
    keep_count = vocab_size - len(SPECIAL_TOKENS)
    tokens = [token for token, _ in counts.most_common(keep_count)]
    return WordTokenizer(tokens, source=f"retokenized:{Path(corpus_path)}")


def encode_corpus_with_jieba(
    corpus_path: str | Path,
    tokenizer: WordTokenizer,
) -> np.ndarray:
    encoded: list[int] = []
    with Path(corpus_path).open("r", encoding="utf-8") as corpus:
        for line in corpus:
            stripped = line.strip()
            if stripped:
                encoded.extend(tokenizer.encode_tokens(jieba_lcut(stripped)))
    return np.asarray(encoded, dtype=np.uint16)
