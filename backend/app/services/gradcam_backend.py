"""PyTorch Grad-CAM backend for the predict service (T3.6).

Loads the trainer-format checkpoints (``<arch>_best.pth``, key
``model_state_dict``) once per architecture and produces an alpha-blended
base64 PNG overlay for the *ONNX-predicted* class, so the explanation always
matches the served prediction even when int8 quantization flips a near-tie.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from time import perf_counter

import torch
import torchvision.models as models
from PIL import Image
from torch import nn

from app.core.errors import ModelNotReadyError
from app.explain.gradcam import GradCAM, heatmap_to_base64_png
from app.inference.onnx_engine import OnnxEngine

logger = logging.getLogger(__name__)

_NUM_CLASSES = 6


class GradCAMBackend:
    """Thread-safe, checkpoint-backed Grad-CAM explainer.

    Grad-CAM autograd (``zero_grad`` + ``backward``) on a shared model is
    not safe under concurrency — concurrent calls corrupt autograd state and
    have been observed to crash the process — so each ``explain()`` runs
    inside a per-architecture lock (C1).
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._models: dict[str, nn.Module] = {}
        self._lock = threading.Lock()
        self._compute_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _get_model(self, arch: str) -> nn.Module:
        model = self._models.get(arch)
        if model is not None:
            return model
        with self._lock:
            model = self._models.get(arch)
            if model is not None:
                return model
            model = self._load_model(arch)
            self._models[arch] = model
            return model

    def _compute_lock(self, arch: str) -> threading.Lock:
        """Return the per-architecture compute lock, creating it lazily."""
        lock = self._compute_locks.get(arch)
        if lock is None:
            with self._locks_guard:
                lock = self._compute_locks.get(arch)
                if lock is None:
                    lock = threading.Lock()
                    self._compute_locks[arch] = lock
        return lock

    def _load_model(self, arch: str) -> nn.Module:
        checkpoint_path = self._checkpoint_dir / f"{arch}_best.pth"
        if not checkpoint_path.is_file():
            # Full path goes to the server log only; the client message is
            # generic so no filesystem layout leaks (M1).
            logger.warning(
                "Grad-CAM checkpoint %s is missing from %s",
                checkpoint_path.name,
                self._checkpoint_dir,
            )
            raise ModelNotReadyError("model artifact not ready")
        try:
            model = _build_resnet(arch)
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except Exception as exc:
            logger.exception("failed to load Grad-CAM checkpoint %s", checkpoint_path)
            raise ModelNotReadyError(f"failed to load the {arch} Grad-CAM checkpoint") from exc
        model.eval()
        logger.info("loaded Grad-CAM checkpoint %s", checkpoint_path)
        return model

    def explain(self, arch: str, image_rgb: Image.Image, target_class: int) -> tuple[str, float]:
        """Return ``(base64_png, elapsed_ms)`` for one image and class.

        Model fetch and compute run inside the per-architecture lock: the
        shared model must never see concurrent ``zero_grad``/``backward``
        calls (C1), and different architectures do not block each other.
        """
        with self._compute_lock(arch):
            model = self._get_model(arch)
            tensor = torch.from_numpy(OnnxEngine.preprocess(image_rgb))
            cam = GradCAM(model, model.layer4)
            started = perf_counter()
            heatmap = cam.compute(tensor, target_class=target_class)
            elapsed_ms = (perf_counter() - started) * 1000.0
            encoded = heatmap_to_base64_png(heatmap, image_rgb)
            return encoded, elapsed_ms


def _build_resnet(arch: str) -> nn.Module:
    """Build a torchvision ResNet with a 6-class head (trainer format)."""
    builders = {"resnet18": models.resnet18, "resnet50": models.resnet50}
    if arch not in builders:
        raise ModelNotReadyError(f"unsupported Grad-CAM architecture '{arch}'")
    model = builders[arch](weights=None)
    model.fc = nn.Linear(model.fc.in_features, _NUM_CLASSES)
    return model
