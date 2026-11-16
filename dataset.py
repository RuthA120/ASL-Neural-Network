"""PyTorch dataset reader for processed image files and split manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class ASLAlphabetDataset(Dataset[tuple[Tensor, int]]):
    """Load one named split from ``data/splits/manifest.csv``.

    Images are read as grayscale uint8 PNGs and converted to float32 tensors
    shaped (1, 400, 400) with values in [0, 1]. The label is the integer
    ``label_id`` in the manifest, using the class ordering in dataset_report.json.
    An optional transform is applied after conversion. Supply augmentation only
    for the training instance; validation and test instances should use the
    deterministic default preprocessing.
    """

    def __init__(self, manifest: str | Path = "data/splits/manifest.csv",
                 split: str = "train", transform: Callable[[Tensor], Tensor] | None = None):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        frame = pd.read_csv(manifest)
        self.rows = frame.loc[frame["split"] == split, ["image_path", "label_id"]].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows for split {split!r} in {manifest}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.rows.iloc[index]
        with Image.open(row["image_path"]) as source:
            gray = source.convert("L")
            if gray.size != (400, 400):
                raise ValueError(f"Expected 400x400 processed image, got {gray.size}: {row['image_path']}")
            pixels = np.asarray(gray, dtype=np.float32) / 255.0
        image = torch.from_numpy(pixels).unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_id"])
