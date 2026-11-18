from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_starts = len(data) - context_length
    if num_starts <= 0:
        raise ValueError("dataset must be longer than context_length")
    starts = np.random.randint(0, num_starts, size=batch_size)
    indices = starts[:, None] + np.arange(context_length)[None, :]
    inputs = torch.as_tensor(
        data[indices].astype(np.int64),
        dtype=torch.long,
        device=device,
    )
    targets = torch.as_tensor(
        data[indices + 1].astype(np.int64),
        dtype=torch.long,
        device=device,
    )
    return inputs, targets


def load_token_file(path: str | Path, dtype: str = "uint16") -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Token file not found: {path}. Run `python -m src.prepare_data` first."
        )
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")
