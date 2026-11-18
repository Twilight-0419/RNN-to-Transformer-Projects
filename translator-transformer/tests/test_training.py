import unittest

import torch

from src.text import PAD_TOKEN
from src.train import masked_cross_entropy


class TrainingLossTest(unittest.TestCase):
    def test_padding_is_ignored(self) -> None:
        logits = torch.tensor(
            [
                [[0.0, 10.0], [10.0, 0.0]],
                [[0.0, 10.0], [0.0, 10.0]],
            ]
        )
        labels = torch.tensor([[1, 0], [1, 0]])
        loss = masked_cross_entropy(logits, labels)
        self.assertLess(loss.item(), 1e-4)
        self.assertEqual(PAD_TOKEN, 0)


if __name__ == "__main__":
    unittest.main()
