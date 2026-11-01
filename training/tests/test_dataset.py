"""Tests for the steel dataset class and augmentation pipelines (T2.2)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from steel_training.data.augmentation import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
)
from steel_training.data.dataset import SteelDefectDataset
from steel_training.data.manifest import Manifest

EXPECTED_SPLIT_LENGTHS = {"train": 1440, "val": 180, "test": 180}


@pytest.fixture()
def train_dataset(built_manifest: tuple[Manifest, Path], image_root: Path):
    manifest, _ = built_manifest
    return SteelDefectDataset(
        image_root, manifest, "train", transform=build_train_transform()
    )


@pytest.fixture()
def val_dataset(built_manifest: tuple[Manifest, Path], image_root: Path):
    manifest, _ = built_manifest
    return SteelDefectDataset(
        image_root, manifest, "val", transform=build_eval_transform()
    )


@pytest.fixture()
def eval_transform():
    return build_eval_transform()


def test_train_transform_pipeline_stages() -> None:
    # Arrange
    # Act
    transform = build_train_transform()

    # Assert
    stage_names = [type(stage).__name__ for stage in transform.transforms]
    assert stage_names == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomRotation",
        "ColorJitter",
        "ToTensor",
        "Normalize",
    ]
    crop = transform.transforms[0]
    assert crop.size == (224, 224)
    assert crop.scale == (0.8, 1.0)


def test_eval_transform_pipeline_stages() -> None:
    # Arrange
    # Act
    transform = build_eval_transform()

    # Assert
    stage_names = [type(stage).__name__ for stage in transform.transforms]
    assert stage_names == ["Resize", "ToTensor", "Normalize"]


def test_dataset_item_shape_and_dtype(train_dataset: SteelDefectDataset) -> None:
    # Arrange
    # Act
    image, label = train_dataset[0]

    # Assert
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label <= 5


def test_dataset_split_lengths(built_manifest: tuple[Manifest, Path], image_root: Path) -> None:
    # Arrange
    manifest, _ = built_manifest

    # Act / Assert
    for split, expected in EXPECTED_SPLIT_LENGTHS.items():
        dataset = SteelDefectDataset(
            image_root, manifest, split, transform=build_eval_transform()
        )
        assert len(dataset) == expected


def test_augmentation_deterministic_given_seed(train_dataset: SteelDefectDataset) -> None:
    # Arrange
    index = 42

    # Act
    torch.manual_seed(7)
    image_a, label_a = train_dataset[index]
    torch.manual_seed(7)
    image_b, label_b = train_dataset[index]

    # Assert
    assert torch.equal(image_a, image_b)
    assert label_a == label_b


def test_augmentation_preserves_label(
    built_manifest: tuple[Manifest, Path], image_root: Path
) -> None:
    # Arrange
    manifest, _ = built_manifest
    train_entries = [entry for entry in manifest["entries"] if entry["split"] == "train"]
    dataset = SteelDefectDataset(
        image_root, manifest, "train", transform=build_train_transform()
    )

    # Act / Assert (dataset preserves manifest entry order within a split)
    for index in (0, 42, 137):
        _, label = dataset[index]
        expected = manifest["class_names"].index(train_entries[index]["label"])
        assert label == expected


def test_val_transform_has_no_randomness(val_dataset: SteelDefectDataset) -> None:
    # Arrange
    index = 5

    # Act
    image_a, label_a = val_dataset[index]
    image_b, label_b = val_dataset[index]

    # Assert
    assert torch.equal(image_a, image_b)
    assert label_a == label_b


def test_eval_transform_resizes_200_to_224(eval_transform) -> None:
    # Arrange
    image = Image.new("RGB", (200, 200), color=(128, 128, 128))

    # Act
    output = eval_transform(image)

    # Assert
    assert output.shape == (3, 224, 224)
    assert output.dtype == torch.float32


def test_eval_transform_applies_imagenet_normalize(eval_transform) -> None:
    # Arrange
    image = Image.new("RGB", (200, 200), color=(128, 128, 128))
    pixel = 128.0 / 255.0
    expected_channels = [
        (pixel - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)
    ]

    # Act
    output = eval_transform(image)

    # Assert
    for channel, expected in zip(output, expected_channels):
        assert torch.allclose(channel, torch.full((224, 224), expected), atol=1e-4)


def test_dataloader_smoke(train_dataset: SteelDefectDataset) -> None:
    # Arrange
    loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Act
    start = time.perf_counter()
    images, labels = next(iter(loader))
    elapsed = time.perf_counter() - start

    # Assert
    assert images.shape == (8, 3, 224, 224)
    assert images.dtype == torch.float32
    assert labels.shape == (8,)
    assert labels.dtype == torch.int64
    assert bool(((labels >= 0) & (labels <= 5)).all())
    assert elapsed < 1.0


def test_label_index_matches_class_order(tmp_path: Path, eval_transform) -> None:
    # Arrange
    class_names = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]
    entries = []
    for class_name in class_names:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        Image.new("RGB", (24, 24), color=(64, 64, 64)).save(class_dir / "img.png")
        entries.append({"path": f"{class_name}/img.png", "label": class_name, "split": "train"})
    manifest: Manifest = {
        "seed": 0,
        "class_names": class_names,
        "split_counts": {"train": 6, "val": 0, "test": 0},
        "entries": entries,
    }

    # Act
    dataset = SteelDefectDataset(tmp_path, manifest, "train", transform=eval_transform)

    # Assert
    assert len(dataset) == 6
    for index in range(6):
        _, label = dataset[index]
        assert label == index


def test_dataset_unknown_split_raises(
    built_manifest: tuple[Manifest, Path], image_root: Path, eval_transform
) -> None:
    # Arrange
    manifest, _ = built_manifest

    # Act / Assert
    with pytest.raises(ValueError, match="split"):
        SteelDefectDataset(image_root, manifest, "nonsense", transform=eval_transform)


def test_dataset_missing_image_raises(tmp_path: Path, eval_transform) -> None:
    # Arrange
    manifest: Manifest = {
        "seed": 0,
        "class_names": ["crazing"],
        "split_counts": {"train": 1, "val": 0, "test": 0},
        "entries": [{"path": "crazing/missing.jpg", "label": "crazing", "split": "train"}],
    }
    dataset = SteelDefectDataset(tmp_path, manifest, "train", transform=eval_transform)

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_dataset_empty_split_raises(tmp_path: Path, eval_transform) -> None:
    # Arrange
    manifest: Manifest = {
        "seed": 0,
        "class_names": ["crazing"],
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "entries": [],
    }

    # Act / Assert
    with pytest.raises(ValueError, match="no entries"):
        SteelDefectDataset(tmp_path, manifest, "val", transform=eval_transform)


def test_dataset_rejects_path_escaping_root(tmp_path: Path, eval_transform) -> None:
    # Arrange
    manifest: Manifest = {
        "seed": 0,
        "class_names": ["crazing"],
        "split_counts": {"train": 1, "val": 0, "test": 0},
        "entries": [{"path": "../outside.jpg", "label": "crazing", "split": "train"}],
    }
    dataset = SteelDefectDataset(tmp_path, manifest, "train", transform=eval_transform)

    # Act / Assert
    with pytest.raises(ValueError, match="escape"):
        dataset[0]
