import unittest

import torch

from src.transformer_model import TransformerLM


class TransformerModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TransformerLM(
            vocab_size=32,
            context_length=8,
            num_layers=2,
            d_model=32,
            num_heads=4,
            d_ff=64,
            use_rope=True,
            theta=10000.0,
        )

    def test_forward_shape(self) -> None:
        inputs = torch.randint(0, 32, (2, 8))
        logits = self.model(inputs)
        self.assertEqual(logits.shape, (2, 8, 32))

    def test_attention_is_causal(self) -> None:
        self.model.eval()
        inputs = torch.tensor([[1, 2, 3, 4]])
        changed = inputs.clone()
        changed[0, -1] = 9
        with torch.no_grad():
            original_logits = self.model(inputs)
            changed_logits = self.model(changed)
        self.assertTrue(torch.allclose(original_logits[:, :-1], changed_logits[:, :-1]))


if __name__ == "__main__":
    unittest.main()
