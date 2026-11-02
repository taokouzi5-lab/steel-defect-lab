"""REST endpoints under ``/api/v1`` (T3.6).

Thin handlers only: validation, orchestration and metadata live in
``app.core``, ``app.services`` and ``app.inference``. Endpoints are sync
``def`` on purpose — inference is CPU-bound and FastAPI runs sync handlers
in the threadpool.

Every route is rate-limited (per-minute budget from settings; disabled in
the test environment). ``/health`` is the only endpoint without the JSON
envelope — it is the infrastructure probe.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.api.deps import (
    get_catalog,
    get_engine,
    get_predict_service,
    get_registry,
    get_settings,
)
from app.carbon.meta import build_carbon_meta
from app.core import envelope
from app.core.rate_limit import dynamic_limit_value, limiter
from app.core.validation import read_upload_stream
from app.schemas.predict import (
    HealthResponse,
    MetaResponse,
    ModelsResponse,
    PredictResponse,
    SamplesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", envelope.new_request_id())


# ---------------------------------------------------------------------------
# Health / metadata
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
@limiter.limit(dynamic_limit_value)
def health(request: Request, engine=Depends(get_engine)) -> dict:
    """Infrastructure probe; deliberately plain (no envelope)."""
    return {"status": "ok", "models_loaded": engine.loaded_ids()}


@router.get("/models", response_model=ModelsResponse)
@limiter.limit(dynamic_limit_value)
def list_models(request: Request, registry=Depends(get_registry)) -> dict:
    """Model metadata with per-model latency and carbon benchmarks."""
    models = registry.list()
    return envelope.success(
        [
            {
                "id": model.id,
                "arch": model.arch,
                "quantization": model.quantization,
                "params": model.params,
                "test_acc": model.test_acc,
                "test_macro_f1": model.test_macro_f1,
                "onnx_size_mb": model.onnx_size_mb,
                "cpu_latency_ms": model.cpu_latency_ms,
                "training_carbon_co2eq_g": model.training_carbon_co2eq_g,
                "inference_carbon_co2eq_g": model.inference_carbon_co2eq_g,
                "inference_carbon_method": model.inference_carbon_method,
                "available": model.available,
            }
            for model in models
        ],
        {
            "count": len(models),
            "class_names": registry.class_names,
            "request_id": _request_id(request),
        },
    )


@router.get("/samples", response_model=SamplesResponse)
@limiter.limit(dynamic_limit_value)
def list_samples(request: Request, catalog=Depends(get_catalog)) -> dict:
    """Whitelisted demo samples (id / filename / ground-truth label)."""
    samples = catalog.list()
    return envelope.success(
        [{"id": sample.id, "filename": sample.name, "label": sample.label} for sample in samples],
        {"count": len(samples), "request_id": _request_id(request)},
    )


@router.get("/meta", response_model=MetaResponse)
@limiter.limit(dynamic_limit_value)
def carbon_method_meta(request: Request, settings=Depends(get_settings)) -> dict:
    """Carbon estimation method disclosure (formula / constants / boundaries)."""
    return envelope.success(
        build_carbon_meta(country=settings.carbon_country),
        {"request_id": _request_id(request)},
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@router.post("/predict", response_model=PredictResponse)
@limiter.limit(dynamic_limit_value)
def predict(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("resnet18-int8"),
    gradcam: bool = Form(False),
    service=Depends(get_predict_service),
) -> dict:
    """Classify an uploaded image (multipart, in-memory only)."""
    settings = request.app.state.settings
    image_bytes = read_upload_stream(file.file, settings.max_upload_bytes())
    result = service.predict(image_bytes, model_id=model, want_gradcam=gradcam)
    return envelope.success(
        result.data,
        {"model": model, "request_id": _request_id(request)},
    )


@router.post("/predict/sample/{sample_id}", response_model=PredictResponse)
@limiter.limit(dynamic_limit_value)
def predict_sample(
    request: Request,
    sample_id: str,
    model: str = Form("resnet18-int8"),
    gradcam: bool = Form(False),
    service=Depends(get_predict_service),
) -> dict:
    """Upload-free channel: classify a whitelisted demo sample by id."""
    result = service.predict_sample(sample_id, model_id=model, want_gradcam=gradcam)
    return envelope.success(
        result.data,
        {"model": model, "request_id": _request_id(request)},
    )
