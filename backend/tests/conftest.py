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
