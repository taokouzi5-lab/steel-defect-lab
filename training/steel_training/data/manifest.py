"""Dataset manifest builder for the NEU-DET steel surface defect dataset.

Creates a reproducible stratified 80/10/10 (train/val/test) split of the
image directories below ``image_root`` and serializes it as ``manifest.json``.
The manifest is stored outside the dataset tree so the raw dataset stays
read-only and the split is tracked in version control.

Schema of the produced manifest::

    {
        "seed": 42,
        "class_names": ["crazing", "inclusion", ...],
        "split_counts": {"train": 1440, "val": 180, "test": 180},
        "entries": [
            {"path": "crazing/crazing_1.jpg", "label": "crazing", "split": "train"},
            ...
        ]
    }

Entry paths are POSIX-style relative paths (``class_dir/file_name``).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TypedDict

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_RANK = {split: rank for rank, split in enumerate(SPLIT_NAMES)}


class ManifestEntry(TypedDict):
    path: str
    label: str
    split: str


class Manifest(TypedDict):
    seed: int
    class_names: list[str]
    split_counts: dict[str, int]
    entries: list[ManifestEntry]


def build_manifest(image_root: Path, output_path: Path, seed: int = 42) -> Manifest:
    """Build a stratified 80/10/10 manifest from the class folders of ``image_root``.

    Raises:
        FileNotFoundError: if ``image_root`` does not exist.
        ValueError: if no class folders or not enough images are found.

    The generated manifest is written to ``output_path`` and also returned.
    """
    image_root = image_root.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")

    class_names = sorted(entry.name for entry in image_root.iterdir() if entry.is_dir())
    if not class_names:
        raise ValueError(f"no class directories found under {image_root}")

    rng = random.Random(seed)
    entries: list[ManifestEntry] = []
    split_counts: dict[str, int] = {split: 0 for split in SPLIT_NAMES}

    for class_name in class_names:
        files = _list_images(image_root / class_name)
        split_points = _split_points(len(files))
        if min(split_points) < 1:
            raise ValueError(
                f"class '{class_name}' has {len(files)} images; "
                f"need enough images for an 80/10/10 split"
            )
        rng.shuffle(files)
        for split, (start, stop) in zip(SPLIT_NAMES, _ranges(split_points)):
            for file_name in files[start:stop]:
                entries.append(
                    {
                        "path": f"{class_name}/{file_name}",
                        "label": class_name,
                        "split": split,
                    }
                )
                split_counts[split] += 1

    # Canonical, deterministic ordering: by split rank, then by relative path.
    entries.sort(key=lambda entry: (SPLIT_RANK[entry["split"]], entry["path"]))

    manifest: Manifest = {
        "seed": seed,
        "class_names": class_names,
        "split_counts": split_counts,
        "entries": entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def load_manifest(path: Path) -> Manifest:
    """Load and validate a manifest previously written by :func:`build_manifest`.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is not a valid manifest.
    """
    manifest_path = path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    required_keys = ("seed", "class_names", "split_counts", "entries")
    missing = [key for key in required_keys if key not in manifest]
    if missing:
        raise ValueError(f"invalid manifest {manifest_path}: missing keys {missing}")
    for entry in manifest["entries"]:
        if not {"path", "label", "split"} <= set(entry):
            raise ValueError(f"invalid manifest entry in {manifest_path}: {entry!r}")
    return manifest


def _list_images(class_dir: Path) -> list[str]:
    """Return the sorted file names of supported images in ``class_dir``."""
    return sorted(
        entry.name
        for entry in class_dir.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    )


def _split_points(total: int) -> tuple[int, int, int]:
    """Return (train_count, val_count, test_count) for the 80/10/10 split."""
    train = int(total * SPLIT_RATIOS[0])
    val = int(total * SPLIT_RATIOS[1])
    return train, val, total - train - val


def _ranges(split_points: tuple[int, int, int]) -> list[tuple[int, int]]:
    """Return half-open index ranges for train, val and test slices."""
    train, val, _ = split_points
    return [(0, train), (train, train + val), (train + val, sum(split_points))]
