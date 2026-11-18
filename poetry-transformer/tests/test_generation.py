import tempfile
import unittest
from pathlib import Path

import torch

from src.poetry_generator import (
    TransformerPoetryGenerator,
    format_poem,
    is_finished_sentence,
)
from src.tokenizer import WordTokenizer
from src.transformer_model import TransformerLM


class GenerationTest(unittest.TestCase):
    def test_sentence_requires_length_and_full_stop(self) -> None:
        self.assertFalse(is_finished_sentence(4, 5, "。"))
        self.assertFalse(is_finished_sentence(5, 5, "，"))
        self.assertTrue(is_finished_sentence(5, 5, "。"))

    def test_format_poem_breaks_long_text(self) -> None:
        text = "春风吹柳绿，江水映花红。山月照人去，孤舟入远空。"
        formatted = format_poem(text, line_width=12)
        self.assertIn("\n", formatted)
        self.assertTrue(all(len(line) <= 14 for line in formatted.splitlines()))

    def test_checkpoint_can_generate(self) -> None:
        tokenizer = WordTokenizer(["春", "风", "花", "月", "，", "。"])
        config = {
            "vocab_size": tokenizer.vocab_size,
            "context_length": 8,
            "num_layers": 1,
            "d_model": 32,
            "num_heads": 4,
            "d_ff": 64,
            "use_rope": True,
            "theta": 10000.0,
        }
        model = TransformerLM(**config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "tiny.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": config,
                    "vocab": tokenizer.to_dict(),
                },
                checkpoint_path,
            )
            generator = TransformerPoetryGenerator(checkpoint_path, device="cpu")
            result = generator.generate(
                start_text="春",
                poem_length=3,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=0.0,
                max_extra_tokens=1,
            )
        self.assertIn("春", result)
        self.assertTrue(result.rstrip().endswith("。"))

    def test_generation_waits_for_full_stop_after_target(self) -> None:
        tokenizer = WordTokenizer(["春", "花", "。"])
        flower_id = tokenizer.stoi["花"]
        full_stop_id = tokenizer.stoi["。"]

        class FixedModel:
            def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
                logits = torch.full(
                    (inputs.shape[0], inputs.shape[1], tokenizer.vocab_size),
                    -100.0,
                )
                target_id = flower_id if inputs.shape[1] < 4 else full_stop_id
                logits[..., target_id] = 100.0
                return logits

        generator = TransformerPoetryGenerator.__new__(TransformerPoetryGenerator)
        generator.device = torch.device("cpu")
        generator.context_length = 16
        generator.tokenizer = tokenizer
        generator.model = FixedModel()

        result = generator.generate(
            start_text="春",
            poem_length=3,
            temperature=0.0,
            max_extra_tokens=5,
            line_width=0,
        )
        self.assertEqual(result, "春花花花。")


if __name__ == "__main__":
    unittest.main()
