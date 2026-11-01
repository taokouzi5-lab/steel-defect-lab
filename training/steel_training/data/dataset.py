"""PyTorch Dataset for the NEU-DET steel surface defect images."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from steel_training.data.manifest import Manifest

VALID_SPLITS = ("train", "val", "test")


class SteelDefectDataset(Dataset):
    """Dataset over a manifest subset (split) of the NEU-DET images.

    Each item is ``(image, label_index)`` where ``label_index`` follows the
    order of ``manifest["class_names"]``. The manifest entry order within a
    split is preserved, so ``dataset[index]`` corresponds to the ``index``-th
    manifest entry of that split.
    """

    def __init__(
        self,
        image_root: Path,
        manifest: Manifest,
        split: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"unknown split '{split}'; expected one of {VALID_SPLITS}")
        self._image_root = image_root.resolve()
        self._transform = transform

        class_names = manifest["class_names"]
        self._label_to_index = {name: index for index, name in enumerate(class_names)}
        self._items = [
            (entry["path"], self._label_to_index[entry["label"]])
            for entry in manifest["entries"]
            if entry["split"] == split
        ]
        if not self._items:
            raise ValueError(f"manifest contains no entries for split '{split}'")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        relative_path, label = self._items[index]
        image_path = (self._image_root / relative_path).resolve()
        if not image_path.is_relative_to(self._image_root):
            raise ValueError(f"manifest path escapes image root: {relative_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"dataset image is missing: {image_path}")

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image, label
