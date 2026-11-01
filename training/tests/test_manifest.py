"""Tests for the NEU-DET stratified manifest builder (T2.1).

These tests read the real NEU-DET dataset through the read-only
``training/data`` junction (1800 images, 6 classes x 300) and never
write into it; all outputs go to pytest's ``tmp_path``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from steel_training.data.manifest import Manifest, build_manifest, load_manifest

EXPECTED_CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# 80/10/10 of 300 images per class -> 240/30/30 per class,
# 1440/180/180 across the full 1800-image dataset.
EXPECTED_SPLIT_COUNTS = {"train": 1440, "val": 180, "test": 180}
EXPECTED_PER_CLASS_COUNTS = {"train": 240, "val": 30, "test": 30}


def test_manifest_stratified_balance(built_manifest: tuple[Manifest, Path]) -> None:
    # Arrange
    manifest, _ = built_manifest

    # Act
    split_totals = Counter(entry["split"] for entry in manifest["entries"])
    per_class = Counter((entry["label"], entry["split"]) for entry in manifest["entries"])

    # Assert
    assert len(manifest["entries"]) == 1800
    assert dict(split_totals) == EXPECTED_SPLIT_COUNTS
    assert manifest["split_counts"] == EXPECTED_SPLIT_COUNTS
    for class_name in EXPECTED_CLASS_NAMES:
        for split, expected in EXPECTED_PER_CLASS_COUNTS.items():
            assert per_class[(class_name, split)] == expected, (
                f"{class_name}/{split} count mismatch"
            )


def test_manifest_class_names_sorted_and_fixed(built_manifest: tuple[Manifest, Path]) -> None:
    # Arrange
    manifest, _ = built_manifest

    # Act
    class_names = manifest["class_names"]

    # Assert
    assert class_names == sorted(class_names)
    assert class_names == EXPECTED_CLASS_NAMES


def test_manifest_reproducible(tmp_path: Path, image_root: Path) -> None:
    # Arrange
    out_a = tmp_path / "a" / "manifest.json"
    out_b = tmp_path / "b" / "manifest.json"
    out_a.parent.mkdir()
    out_b.parent.mkdir()

    # Act
    build_manifest(image_root, out_a, seed=42)
    build_manifest(image_root, out_b, seed=42)

    # Assert
    assert out_a.read_bytes() == out_b.read_bytes()


def test_no_leak_between_splits(built_manifest: tuple[Manifest, Path]) -> None:
    # Arrange
    manifest, _ = built_manifest

    # Act
    by_split = {
        split: {entry["path"] for entry in manifest["entries"] if entry["split"] == split}
        for split in ("train", "val", "test")
    }

    # Assert
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])


def test_manifest_covers_all_images(
    built_manifest: tuple[Manifest, Path], dataset_files: set[str]
) -> None:
    # Arrange
    manifest, _ = built_manifest

    # Act
    manifest_paths = {entry["path"] for entry in manifest["entries"]}

    # Assert
    assert len(dataset_files) == 1800
    assert manifest_paths == dataset_files


def test_manifest_entry_schema(
    built_manifest: tuple[Manifest, Path], image_root: Path
) -> None:
    # Arrange
    manifest, _ = built_manifest
    class_names = set(manifest["class_names"])
    valid_splits = {"train", "val", "test"}

    # Act / Assert
    for entry in manifest["entries"]:
        assert set(entry.keys()) == {"path", "label", "split"}
        assert entry["label"] in class_names
        assert entry["split"] in valid_splits
        assert entry["path"].startswith(f"{entry['label']}/")
        assert (image_root / entry["path"]).is_file()


def test_load_manifest_roundtrip(built_manifest: tuple[Manifest, Path]) -> None:
    # Arrange
    manifest, output_path = built_manifest

    # Act
    loaded = load_manifest(output_path)

    # Assert
    assert loaded == manifest
    assert loaded["class_names"] == EXPECTED_CLASS_NAMES
    assert loaded["split_counts"] == EXPECTED_SPLIT_COUNTS


def test_manifest_json_is_valid_json(built_manifest: tuple[Manifest, Path]) -> None:
    # Arrange
    _, output_path = built_manifest

    # Act
    with output_path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)

    # Assert
    assert isinstance(parsed, dict)
    assert len(parsed["entries"]) == 1800


def test_load_manifest_missing_file_raises(tmp_path: Path) -> None:
    # Arrange
    missing = tmp_path / "does-not-exist.json"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        load_manifest(missing)


def test_load_manifest_missing_keys_raises(tmp_path: Path) -> None:
    # Arrange
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(json.dumps({"entries": []}), encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match="missing keys"):
        load_manifest(bad_manifest)


def test_load_manifest_bad_entry_raises(tmp_path: Path) -> None:
    # Arrange
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(
        json.dumps(
            {
                "seed": 42,
                "class_names": [],
                "split_counts": {},
                "entries": [{"path": "x.jpg"}],
            }
        ),
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(ValueError, match="invalid manifest entry"):
        load_manifest(bad_manifest)


def test_build_manifest_missing_root_raises(tmp_path: Path) -> None:
    # Arrange
    missing_root = tmp_path / "no-such-root"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        build_manifest(missing_root, tmp_path / "manifest.json", seed=42)


def test_build_manifest_empty_root_raises(tmp_path: Path) -> None:
    # Arrange
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    # Act / Assert
    with pytest.raises(ValueError, match="class"):
        build_manifest(empty_root, tmp_path / "manifest.json", seed=42)


def test_build_manifest_too_few_images_raises(tmp_path: Path) -> None:
    # Arrange
    small_root = tmp_path / "small"
    class_dir = small_root / "patches"
    class_dir.mkdir(parents=True)
    for i in range(5):
        (class_dir / f"img_{i}.jpg").touch()

    # Act / Assert
    with pytest.raises(ValueError, match="enough images"):
        build_manifest(small_root, tmp_path / "manifest.json", seed=42)


def test_build_manifest_different_seed_differs(tmp_path: Path, image_root: Path) -> None:
    # Arrange
    out_a = tmp_path / "seed42.json"
    out_b = tmp_path / "seed43.json"

    # Act
    manifest_a = build_manifest(image_root, out_a, seed=42)
    manifest_b = build_manifest(image_root, out_b, seed=43)

    # Assert
    split_a = {entry["path"]: entry["split"] for entry in manifest_a["entries"]}
    split_b = {entry["path"]: entry["split"] for entry in manifest_b["entries"]}
    assert split_a != split_b
