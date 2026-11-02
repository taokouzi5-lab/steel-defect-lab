"""Predict service: orchestrates validation -> ONNX inference ->
optional Grad-CAM -> carbon estimation (T3.6).

The upload and sample channels share one code path: both call
:meth:`PredictService.predict` with in-memory bytes, so behavior (including
all validation and error mapping) is identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.carbon.estimator import estimate
from app.config import Settings
from app.core.errors import ModelNotFoundError, ModelNotReadyError, SampleNotFoundError
from app.core.sanitize import sanitize_text
from app.core.validation import validate_upload
from app.inference.model_registry import ModelMeta, ModelRegistry
from app.inference.onnx_engine import OnnxEngine
from app.services.sample_catalog import SampleCatalog

logger = logging.getLogger(__name__)

TOP_K = 3


@dataclass(frozen=True)
class PredictResult:
    """The raw data payload of a predict response (pre-envelope)."""

    data: dict


class PredictService:
    """Stateless orchestrator; engine/registry/catalog are injected."""

    def __init__(
        self,
        engine: OnnxEngine,
        registry: ModelRegistry,
        settings: Settings,
        catalog: SampleCatalog | None = None,
        gradcam_backend=None,
    ) -> None:
        self._engine = engine
        self._registry = registry
        self._settings = settings
        self._catalog = catalog
        self._gradcam_backend = gradcam_backend

    # -- public entry points -------------------------------------------------

    def predict(
        self,
        image_bytes: bytes,
        model_id: str,
        want_gradcam: bool = False,
    ) -> PredictResult:
        """Predict on an in-memory upload."""
        validated = validate_upload(image_bytes, self._settings.max_upload_bytes())
        return self._infer(validated.image, model_id, want_gradcam)

    def predict_sample(
        self,
        sample_id: str,
        model_id: str,
        want_gradcam: bool = False,
    ) -> PredictResult:
        """Predict on a whitelisted demo sample (same inference path)."""
        if self._catalog is None:
            raise SampleNotFoundError(sample_id)
        info = self._catalog.get(sample_id)
        if info is None:
            raise SampleNotFoundError(f"no sample with id '{sanitize_text(sample_id)}'")
        path = self._catalog.path_of(info)
        try:
            image_bytes = path.read_bytes()
        except OSError as exc:
            logger.exception("could not read sample %s", path)
            raise SampleNotFoundError(sample_id) from exc
        validated = validate_upload(image_bytes, self._settings.max_upload_bytes())
        return self._infer(validated.image, model_id, want_gradcam)

    # -- orchestration -------------------------------------------------------

    def _resolve_model(self, model_id: str) -> ModelMeta:
        meta = self._registry.get(model_id)
        if meta is None:
            if self._registry.is_empty():
                raise ModelNotReadyError(
                    "no models are registered; check the metrics file configuration"
                )
            raise ModelNotFoundError(f"unknown model id '{sanitize_text(model_id)}'")
        return meta

    def _infer(self, image, model_id: str, want_gradcam: bool) -> PredictResult:
        meta = self._resolve_model(model_id)
        class_names = self._registry.class_names

        prediction = self._engine.predict(meta.id, image, top_k=TOP_K)
        predicted_class = prediction.class_id

        gradcam_payload: dict | None = None
        gradcam_ms: float | None = None
        if want_gradcam:
            if self._gradcam_backend is None:
                raise ModelNotReadyError("Grad-CAM is not available in this deployment")
            encoded, gradcam_ms = self._gradcam_backend.explain(meta.arch, image, predicted_class)
            gradcam_payload = {"base64": encoded, "format": "png"}

        carbon = estimate(
            meta.carbon_model_name,
            country=self._settings.carbon_country,
        )

        return PredictResult(
            data={
                "class_id": predicted_class,
                "class_name": _class_name(class_names, predicted_class),
                "confidence": prediction.top_k[0].confidence,
                "top_k": [
                    {
                        "class_id": item.class_id,
                        "class_name": _class_name(class_names, item.class_id),
                        "confidence": item.confidence,
                    }
                    for item in prediction.top_k
                ],
                "latency_ms": prediction.latency_ms,
                "gradcam_ms": gradcam_ms,
                "gradcam": gradcam_payload,
                "carbon": {
                    "co2eq_g": carbon.co2eq_g,
                    "energy_kwh": carbon.energy_kwh,
                    "method": carbon.method,
                    "equivalent": carbon.equivalent,
                },
            }
        )


def _class_name(class_names: list[str], class_id: int) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"
