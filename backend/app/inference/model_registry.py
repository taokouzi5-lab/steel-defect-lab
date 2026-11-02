"""Model registry: model id -> serving metadata (T3.6).

Reads the training pipeline's ``artifacts/metrics.json`` and combines it with
the carbon estimator so ``GET /models`` can report per-model accuracy,
latency, and carbon benchmarks. Class names are derived from the
``per_class_f1`` keys of the metrics file — which the training pipeline
orders exactly like ``training/manifest.json`` (alphabetical for the six
NEU-DET classes) — so no class list is hardcoded here.

The registry is read-only metadata; a missing or malformed metrics file
degrades to an empty registry (models are then reported as unavailable, and
``/predict`` answers 503).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.carbon.estimator import estimate
from app.inference.onnx_engine import OnnxEngine

logger = logging.getLogger(__name__)

# Carbon estimator model keys: resnet18 has no mace-f method-D measurement and
# intentionally falls back to the T3-tier representative; resnet50 is measured.
CARBON_MODEL_NAMES = {"resnet18": "resnet18", "resnet50": "resnet_50"}


@dataclass(frozen=True)
class ModelMeta:
    """Immutable serving metadata for one model id."""

    id: str
    arch: str
    quantization: str
    params: int
    test_acc: float
    test_macro_f1: float
    onnx_size_mb: float
    cpu_latency_ms: dict[str, float]
    training_carbon_co2eq_g: float
    inference_carbon_co2eq_g: float
    inference_carbon_method: str
    carbon_model_name: str
    class_names: tuple[str, ...]
    available: bool


class ModelRegistry:
    """Serves :class:`ModelMeta` objects; empty when metrics are unavailable."""

    def __init__(self, metrics_path: Path, model_dir: Path, country: str = "China") -> None:
        self._metrics_path = Path(metrics_path)
        self._model_dir = Path(model_dir)
        self._country = country
        self._models: list[ModelMeta] = self._load()

    @property
    def class_names(self) -> list[str]:
        """Class names in training order (derived from the metrics file)."""
        for model in self._models:
            if model.class_names:
                return list(model.class_names)
        return []

    # -- loading ------------------------------------------------------------

    def _load(self) -> list[ModelMeta]:
        if not self._metrics_path.is_file():
            logger.warning(
                "metrics file %s not found; serving an empty model registry",
                self._metrics_path,
            )
            return []
        try:
            payload = json.loads(self._metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("could not read %s: %s", self._metrics_path, exc)
            return []
        if not isinstance(payload, dict):
            logger.warning("metrics file %s has an unexpected shape", self._metrics_path)
            return []

        models: list[ModelMeta] = []
        for arch, metrics in (payload.get("models") or {}).items():
            for quantization in ("fp32", "int8"):
                model_id = f"{arch}-{quantization}"
                try:
                    models.append(self._build_meta(arch, quantization, metrics))
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("skipping malformed metrics entry %s: %s", model_id, exc)
        return models

    def _build_meta(self, arch: str, quantization: str, metrics: dict) -> ModelMeta:
        latency = metrics["cpu_latency_ms"][quantization]
        size_key = f"onnx_{quantization}_size_mb"
        filename = OnnxEngine.model_filename(f"{arch}-{quantization}")
        carbon = estimate(CARBON_MODEL_NAMES[arch], country=self._country)
        # Sorted per_class_f1 keys reproduce the training class order exactly.
        class_names = tuple(sorted(metrics["per_class_f1"]))
        return ModelMeta(
            id=f"{arch}-{quantization}",
            arch=arch,
            quantization=quantization,
            params=int(metrics["params"]),
            test_acc=float(metrics["test_acc"]),
            test_macro_f1=float(metrics["test_macro_f1"]),
            onnx_size_mb=float(metrics[size_key]),
            cpu_latency_ms={"p50": float(latency["p50"]), "p95": float(latency["p95"])},
            training_carbon_co2eq_g=float(metrics["carbon_co2eq_g"]),
            inference_carbon_co2eq_g=carbon.co2eq_g,
            inference_carbon_method=carbon.method,
            carbon_model_name=CARBON_MODEL_NAMES[arch],
            class_names=class_names,
            available=(self._model_dir / filename).is_file(),
        )

    # -- queries ------------------------------------------------------------

    def list(self) -> list[ModelMeta]:
        return list(self._models)

    def get(self, model_id: str) -> ModelMeta | None:
        for model in self._models:
            if model.id == model_id:
                return model
        return None

    def is_empty(self) -> bool:
        return not self._models
