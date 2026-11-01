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


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def repo_root() -> Path:
    """Repository root (parent of the ``training`` package)."""
    return REPO_ROOT


@pytest.fixture()
def manifest_path(repo_root: Path) -> Path:
    """Path of the committed training manifest (training/manifest.json)."""
    path = repo_root / "training" / "manifest.json"
    if not path.is_file():
        pytest.skip("training/manifest.json is not present in this checkout")
    return path


@pytest.fixture()
def pretrained_dir(repo_root: Path) -> Path:
    """Directory holding locally cached ImageNet weights (artifacts)."""
    return repo_root / "artifacts" / "checkpoints" / "pretrained"


@pytest.fixture()
def configs_dir(repo_root: Path) -> Path:
    """Directory holding the committed training configs (training/configs)."""
    return repo_root / "training" / "configs"


class FakeCarbonTracker:
    """Stand-in for CarbonTracker that never touches CodeCarbon.

    Records every ``start``/``stop`` call so tests can assert how the
    training engine interacts with the tracker. ``is_active`` mirrors the
    state a real tracker has while running inside its context manager.
    """

    def __init__(self, active: bool = False) -> None:
        self.calls: list[str] = []
        self._active = active
        self.result: dict[str, float] | None = {"co2eq_g": 1.5}

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")
        self.result = {"co2eq_g": 1.5}


@pytest.fixture()
def fake_carbon_tracker() -> type[FakeCarbonTracker]:
    """Factory fixture returning the :class:`FakeCarbonTracker` class."""
    return FakeCarbonTracker
