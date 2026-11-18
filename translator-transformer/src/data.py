from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.text import PAD_TOKEN, SOS_TOKEN


class PackedTranslationDataset(Dataset):
    def __init__(self, source_path: str | Path, target_path: str | Path):
        self.source = np.load(source_path, mmap_mode="r")
        self.target = np.load(target_path, mmap_mode="r")
        self.source_offsets = np.load(
            Path(source_path).with_name(Path(source_path).stem + "_offsets.npy"),
            mmap_mode="r",
        )
        self.target_offsets = np.load(
            Path(target_path).with_name(Path(target_path).stem + "_offsets.npy"),
            mmap_mode="r",
        )
        if len(self.source_offsets) != len(self.target_offsets):
            raise ValueError("source and target offset files do not match")
        self.length = len(self.source_offsets) - 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_start = int(self.source_offsets[index])
        source_end = int(self.source_offsets[index + 1])
        target_start = int(self.target_offsets[index])
        target_end = int(self.target_offsets[index + 1])
        source = torch.as_tensor(
            np.asarray(self.source[source_start:source_end], dtype=np.int64),
            dtype=torch.long,
        )
        target = torch.as_tensor(
            np.asarray(self.target[target_start:target_end], dtype=np.int64),
            dtype=torch.long,
        )
        return source, target

    def close(self) -> None:
        for array in (
            self.source,
            self.target,
            self.source_offsets,
            self.target_offsets,
        ):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


def collate_translation_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    sources, targets = zip(*batch)
    batch_size = len(batch)
    max_source = max(len(item) for item in sources)
    max_target = max(len(item) for item in targets)
    source_ids = torch.full((batch_size, max_source), PAD_TOKEN, dtype=torch.long)
    target_labels = torch.full((batch_size, max_target), PAD_TOKEN, dtype=torch.long)
    decoder_input_ids = torch.full((batch_size, max_target), PAD_TOKEN, dtype=torch.long)
    decoder_input_ids[:, 0] = SOS_TOKEN

    for index, (source, target) in enumerate(batch):
        source_ids[index, : len(source)] = source
        target_labels[index, : len(target)] = target
        if len(target) > 1:
            decoder_input_ids[index, 1 : len(target)] = target[:-1]

    return {
        "source_ids": source_ids,
        "decoder_input_ids": decoder_input_ids,
        "target_labels": target_labels,
        "source_mask": source_ids != PAD_TOKEN,
        "target_mask": decoder_input_ids != PAD_TOKEN,
    }
