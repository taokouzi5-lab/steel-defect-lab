"""Shared pytest fixtures for the backend test suite.

Backend tests read artifacts that live elsewhere in the repository (the
``training/data`` NEU-DET junction and the trainer checkpoints) or in the
sibling mace-f benchmark checkout, all read-only. Every such fixture skips
gracefully when the source is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

# Make `app` importable no matter which ini file governs the run: when the
# combined suite (`pytest training/tests backend/tests`) is executed from the
# repository root, pytest 7.4 applies the training pyproject (rootdir=training)
# to every argument, so backend's own pythonpath setting is not in effect.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[2]


class TinyGradCNN(nn.Module):
    """Two-conv CNN whose ``conv2`` output is the Grad-CAM target layer."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.fc(torch.flatten(self.pool(x), 1))


@pytest.fixture()
def tiny_model() -> TinyGradCNN:
    """A small random-weight CNN in eval mode (seed fixed)."""
    torch.manual_seed(0)
    model = TinyGradCNN()
    model.eval()
    return model


class BiasFreeCNN(nn.Module):
    """Like :class:`TinyGradCNN` but with bias-free convolutions.

    Zero input then yields exactly zero activations at the target layer,
    which is the precondition for the "all-black image -> zero heatmap"
    Grad-CAM property.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.fc(torch.flatten(self.pool(x), 1))


@pytest.fixture()
def bias_free_model() -> BiasFreeCNN:
    """A small bias-free random-weight CNN in eval mode (seed fixed)."""
    torch.manual_seed(2)
    model = BiasFreeCNN()
    model.eval()
    return model


@pytest.fixture()
def sample_input() -> torch.Tensor:
    """A random 1x3x8x8 image tensor (seed fixed)."""
    torch.manual_seed(1)
    return torch.rand(1, 3, 8, 8)


@pytest.fixture()
def data_junction() -> Path:
    """Path of the read-only NEU-DET junction used by the training suite."""
    path = REPO_ROOT / "training" / "data"
    if not path.is_dir():
        pytest.skip("NEU-DET junction (training/data) is not present in this checkout")
    return path


@pytest.fixture()
def resnet18_checkpoint() -> Path:
    """Path of the trainer-format resnet18 best checkpoint."""
    path = REPO_ROOT / "artifacts" / "checkpoints" / "resnet18_best.pth"
    if not path.is_file():
        pytest.skip("resnet18_best.pth is not present in this checkout")
    return path


@pytest.fixture()
def macef_results_dir() -> Path:
    """Path of the mace-f benchmark results directory (sibling checkout)."""
    path = REPO_ROOT.parent / "carbon_benchmark" / "results"
    if not path.is_dir():
        pytest.skip("mace-f benchmark results are not present on this machine")
    return path


# ---------------------------------------------------------------------------
# T3.5/T3.6 FastAPI service fixtures
# ---------------------------------------------------------------------------

TEST_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _fastapi_env_defaults(monkeypatch):
    """Keep tests fast and deterministic: no model preload, no rate limiting.

    The frontend dist is pointed at a missing path so API tests run in
    API-only mode regardless of whether a real frontend build exists on the
    machine; SPA hosting behaviour is covered explicitly by
    test_spa_static.py with an injected dist directory.

    Individual tests that exercise preloading or rate limiting re-enable the
    feature by setting the env var again (the same function-scoped
    ``monkeypatch`` instance is used, so the later ``setenv`` wins) before
    creating their own app.
    """
    monkeypatch.setenv("PRELOAD_MODELS", "0")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "0")
    monkeypatch.setenv("FRONTEND_DIST_DIR", str(TEST_FIXTURES_DIR / "no-dist"))


@pytest.fixture()
def fixtures_dir() -> Path:
    return TEST_FIXTURES_DIR


@pytest.fixture()
def client():
    """A TestClient around a fresh app (no preload, rate limiting disabled)."""
    from app.config import Settings
    from app.main import create_app

    app = create_app(Settings.from_env())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def make_client():
    """Factory: build a TestClient from an explicit Settings instance."""

    def _make(settings):
        from app.main import create_app

        app = create_app(settings)
        return TestClient(app)

    return _make


@pytest.fixture()
def sample_assets_dir() -> Path:
    """Directory of the committed demo samples (12 NEU-DET test images)."""
    path = BACKEND_ROOT / "app" / "assets" / "samples"
    if not path.is_dir():
        pytest.skip("backend/app/assets/samples is not present in this checkout")
    return path


@pytest.fixture()
def sample_crazing_bytes(sample_assets_dir) -> bytes:
    image = sample_assets_dir / "crazing" / "crazing_101.jpg"
    if not image.is_file():
        pytest.skip(f"sample image missing: {image}")
    return image.read_bytes()


@pytest.fixture()
def real_onnx_dir() -> Path:
    """Repository ONNX artifacts (generated, not tracked — skip if absent)."""
    path = REPO_ROOT / "artifacts" / "onnx"
    if not path.is_dir() or not (path / "resnet18_int8_static.onnx").is_file():
        pytest.skip("ONNX artifacts are not present in this checkout")
    return path


@pytest.fixture()
def real_metrics_path() -> Path:
    """Repository metrics.json (generated, not tracked — skip if absent)."""
    path = REPO_ROOT / "artifacts" / "metrics.json"
    if not path.is_file():
        pytest.skip("artifacts/metrics.json is not present in this checkout")
    return path
