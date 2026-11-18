from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, WordTokenizer
from src.transformer_model import TransformerLM, sample_next_token


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "poetry_transformer.pt"
LINE_PUNCTUATION = set("，。！？；：")


def is_finished_sentence(
    generated_token_count: int,
    required_token_count: int,
    token_text: str,
) -> bool:
    return generated_token_count >= required_token_count and token_text.endswith("。")


def format_poem(text: str, line_width: int = 12) -> str:
    text = text.strip()
    if not text or line_width <= 0 or "\n" in text:
        return text

    lines: list[str] = []
    remaining = text
    while len(remaining) > line_width:
        break_at = line_width
        lower = max(1, line_width - 4)
        upper = min(len(remaining), line_width + 2)
        for index in range(upper - 1, lower - 1, -1):
            if remaining[index] in LINE_PUNCTUATION:
                break_at = index + 1
                break
        lines.append(remaining[:break_at].strip())
        remaining = remaining[break_at:].strip()
    if remaining:
        lines.append(remaining)
    return "\n".join(line for line in lines if line)


class TransformerPoetryGenerator:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
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
        model_config = checkpoint["model_config"]
        self.model = TransformerLM(
            **model_config,
            device=self.device,
            dtype=torch.float32,
        ).to(self.device)
        state_dict = checkpoint.get("model", checkpoint.get("model_state_dict"))
        if state_dict is None:
            raise ValueError("checkpoint does not contain model weights")
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if "vocab" in checkpoint:
            self.tokenizer = WordTokenizer.from_dict(checkpoint["vocab"])
        else:
            self.tokenizer = WordTokenizer.load(ROOT / "data" / "vocab.json")
        self.context_length = model_config["context_length"]

    @torch.no_grad()
    def generate(
        self,
        start_text: str,
        poem_length: int,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 2.0,
        repetition_window: int = 20,
        line_width: int = 12,
        max_extra_tokens: int = 32,
    ) -> str:
        if poem_length <= 0:
            raise ValueError("poem_length must be positive")
        if repetition_window < 0:
            raise ValueError("repetition_window cannot be negative")
        if max_extra_tokens < 0:
            raise ValueError("max_extra_tokens cannot be negative")

        prompt_ids = self.tokenizer.encode_prompt(start_text)
        if not prompt_ids:
            raise ValueError("start_text must contain at least one non-space token")

        generated_ids = list(prompt_ids)
        generated_token_count = 0
        max_new_tokens = poem_length + max_extra_tokens
        while generated_token_count < max_new_tokens:
            context = torch.tensor(
                [generated_ids[-self.context_length :]],
                dtype=torch.long,
                device=self.device,
            )
            logits = self.model(context)[0, -1, :].float().clone()

            if temperature <= 0:
                logits[PAD_ID] = float("-inf")
                logits[BOS_ID] = float("-inf")
                logits[EOS_ID] = float("-inf")
                logits[UNK_ID] = float("-inf")
                next_id = int(torch.argmax(logits).item())
            else:
                logits /= temperature
                if repetition_window > 0 and repetition_penalty != 0:
                    for token_id in set(generated_ids[-repetition_window:]):
                        logits[token_id] -= repetition_penalty

                logits[PAD_ID] = float("-inf")
                logits[BOS_ID] = float("-inf")
                logits[EOS_ID] = float("-inf")
                logits[UNK_ID] = float("-inf")
                if top_k > 0:
                    kth_value = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                    logits = logits.masked_fill(logits < kth_value, float("-inf"))

                sampled = sample_next_token(
                    logits.unsqueeze(0),
                    temperature=1.0,
                    top_p=top_p,
                )
                next_id = int(sampled.item())

            generated_ids.append(next_id)
            generated_token_count += 1
            if is_finished_sentence(
                generated_token_count,
                poem_length,
                self.tokenizer.tokens[next_id],
            ):
                break

        decoded = self.tokenizer.decode(generated_ids)
        if not decoded.rstrip().endswith("。"):
            decoded = decoded.rstrip() + "。"
        return format_poem(decoded, line_width=line_width)


_DEFAULT_GENERATOR: Optional[TransformerPoetryGenerator] = None


def Generating_Poetry(
    start_word: str,
    poem_length: int,
    temperature: float = 0.8,
    top_k: int = 0,
    repetition_penalty: float = 2.0,
    repetition_window: int = 20,
    top_p: float = 0.9,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: Optional[str] = None,
    max_extra_tokens: int = 32,
) -> str:
    """Compatibility wrapper matching the original RNN project's entry point."""
    global _DEFAULT_GENERATOR
    resolved_path = Path(checkpoint_path)
    if (
        _DEFAULT_GENERATOR is None
        or getattr(_DEFAULT_GENERATOR, "_checkpoint_path", None) != resolved_path
        or (device is not None and str(_DEFAULT_GENERATOR.device) != device)
    ):
        _DEFAULT_GENERATOR = TransformerPoetryGenerator(resolved_path, device=device)
        _DEFAULT_GENERATOR._checkpoint_path = resolved_path
    return _DEFAULT_GENERATOR.generate(
        start_text=start_word,
        poem_length=poem_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        max_extra_tokens=max_extra_tokens,
    )
