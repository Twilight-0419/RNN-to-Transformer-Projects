import unittest

import torch

from src.nmt_model import EncoderDecoderTransformer


class NMTModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = EncoderDecoderTransformer(
            source_vocab_size=32,
            target_vocab_size=40,
            d_model=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            d_ff=64,
            max_seq_len=16,
        )

    def test_forward_shape(self) -> None:
        source = torch.tensor([[4, 5, 2, 0], [6, 7, 8, 2]])
        target = torch.tensor([[1, 9, 10], [1, 11, 2]])
        logits = self.model(
            source,
            target,
            source != 0,
            target != 0,
        )
        self.assertEqual(tuple(logits.shape), (2, 3, 40))

    def test_generation_returns_token_ids(self) -> None:
        source = torch.tensor([[4, 5, 2]])
        output = self.model.generate(
            source,
            source != 0,
            bos_id=1,
            eos_id=2,
            pad_id=0,
            max_new_tokens=4,
        )
        self.assertIsInstance(output, list)
        self.assertLessEqual(len(output), 4)


if __name__ == "__main__":
    unittest.main()
