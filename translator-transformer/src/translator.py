from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.nmt_model import EncoderDecoderTransformer
from src.text import (
    SOS_TOKEN,
    decode_english,
    encode_chinese,
    load_english_bpe,
    special_id,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "translator_transformer.pt"


class TransformerTranslator:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        bpe_vocab_path: str | Path = ROOT / "models" / "bpe_vocab.json",
        bpe_merges_path: str | Path = ROOT / "models" / "bpe_merges.txt",
        device: Optional[str] = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Run `python -m src.prepare_data` and `python -m src.train` first."
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model = EncoderDecoderTransformer(
            **checkpoint["model_config"],
            device=self.device,
            dtype=torch.float32,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.chinese_word2index = checkpoint["chinese_word2index"]
        self.english_tokenizer = load_english_bpe(bpe_vocab_path, bpe_merges_path)
        self.pad_id = special_id(self.english_tokenizer, "<pad>")
        # Training used the translator-level SOS id, not the unused BPE special token.
        self.sos_id = SOS_TOKEN
        self.eos_id = special_id(self.english_tokenizer, "<eos>")
        self.endoftext_id = self.english_tokenizer.token_to_id[b"<|endoftext|>"]

    @torch.no_grad()
    def translate(
        self,
        chinese_sentence: str,
        max_new_tokens: int = 64,
        repetition_penalty: float = 2.0,
    ) -> str:
        if not chinese_sentence.strip():
            return ""
        source_ids = encode_chinese(chinese_sentence, self.chinese_word2index)
        source_tensor = torch.tensor(
            [source_ids],
            dtype=torch.long,
            device=self.device,
        )
        source_mask = source_tensor != 0
        output_ids = self.model.generate(
            source_tensor,
            source_mask,
            bos_id=self.sos_id,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            blocked_ids=[self.endoftext_id],
        )
        return decode_english(output_ids, self.english_tokenizer)


_DEFAULT_TRANSLATOR: Optional[TransformerTranslator] = None


def translate(
    chinese_sentence: str,
    max_new_tokens: int = 64,
    repetition_penalty: float = 2.0,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: Optional[str] = None,
) -> str:
    global _DEFAULT_TRANSLATOR
    resolved_path = Path(checkpoint_path)
    if (
        _DEFAULT_TRANSLATOR is None
        or getattr(_DEFAULT_TRANSLATOR, "_checkpoint_path", None) != resolved_path
        or (device is not None and str(_DEFAULT_TRANSLATOR.device) != device)
    ):
        _DEFAULT_TRANSLATOR = TransformerTranslator(resolved_path, device=device)
        _DEFAULT_TRANSLATOR._checkpoint_path = resolved_path
    return _DEFAULT_TRANSLATOR.translate(
        chinese_sentence,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
