import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data import PackedTranslationDataset, collate_translation_batch
from src.text import PAD_TOKEN, SOS_TOKEN


class PackedDataTest(unittest.TestCase):
    def test_dataset_and_collate(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            np.save(root / "train_source.npy", np.array([4, 5, 2, 6, 2], dtype=np.uint16))
            np.save(root / "train_source_offsets.npy", np.array([0, 3, 5], dtype=np.int64))
            np.save(root / "train_target.npy", np.array([20, 21, 2, 22, 2], dtype=np.uint16))
            np.save(root / "train_target_offsets.npy", np.array([0, 3, 5], dtype=np.int64))
            dataset = PackedTranslationDataset(
                root / "train_source.npy",
                root / "train_target.npy",
            )
            batch = collate_translation_batch([dataset[0], dataset[1]])
            dataset.close()

        self.assertEqual(len(dataset), 2)
        self.assertEqual(tuple(batch["source_ids"].shape), (2, 3))
        self.assertEqual(tuple(batch["decoder_input_ids"].shape), (2, 3))
        self.assertEqual(batch["decoder_input_ids"][0, 0].item(), SOS_TOKEN)
        self.assertEqual(batch["source_ids"][0, -1].item(), 2)
        self.assertEqual(batch["source_ids"][1, -1].item(), PAD_TOKEN)
        self.assertTrue(torch.equal(batch["source_mask"][0], torch.tensor([True, True, True])))
        self.assertTrue(torch.equal(batch["source_mask"][1], torch.tensor([True, True, False])))


if __name__ == "__main__":
    unittest.main()
