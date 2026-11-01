"""Shared pytest fixtures for the steel_training test suite.

All tests that need the real NEU-DET images read them through the read-only
``training/data`` junction; tests never write into it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from steel_training.data.manifest import Manifest, build_manifest

NEU_DET_ROOT = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture()
def image_root() -> Path:
    """Path to the NEU-DET dataset junction (read-only)."""
    if not NEU_DET_ROOT.is_dir():
        pytest.skip("NEU-DET junction (training/data) is not present in this checkout")
    return NEU_DET_ROOT


@pytest.fixture()
def built_manifest(tmp_path: Path, image_root: Path) -> tuple[Manifest, Path]:
    """A freshly built manifest plus the path of its JSON file."""
    output_path = tmp_path / "manifest.json"
    manifest = build_manifest(image_root, output_path, seed=42)
    return manifest, output_path


@pytest.fixture()
def dataset_files(image_root: Path) -> set[str]:
    """Relative POSIX paths of every supported image in the dataset."""
    files: set[str] = set()
    for class_dir in image_root.iterdir():
        if class_dir.is_dir():
            for image in class_dir.iterdir():
                if image.is_file() and image.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    files.add(f"{class_dir.name}/{image.name}")
    return files
