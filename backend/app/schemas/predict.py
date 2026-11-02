"""Pydantic response DTOs for the DefectLens API (T3.6)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# -- shared envelope pieces --------------------------------------------------


class RequestMeta(BaseModel):
    request_id: str


# -- predict -----------------------------------------------------------------


class TopKItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class GradCAMOut(BaseModel):
    base64: str
    format: str


class CarbonOut(BaseModel):
    co2eq_g: float
    energy_kwh: float
    method: str
    equivalent: str


class PredictData(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    top_k: list[TopKItem]
    latency_ms: float
    gradcam_ms: float | None
    gradcam: GradCAMOut | None
    carbon: CarbonOut


class PredictMeta(BaseModel):
    model: str
    request_id: str


class PredictResponse(BaseModel):
    success: bool
    data: PredictData
    error: None
    meta: PredictMeta


# -- models / samples / meta -------------------------------------------------


class ModelSummary(BaseModel):
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
    available: bool


class ModelsMeta(BaseModel):
    count: int
    class_names: list[str]
    request_id: str


class ModelsResponse(BaseModel):
    success: bool
    data: list[ModelSummary]
    error: None
    meta: ModelsMeta


class SampleOut(BaseModel):
    id: str
    filename: str
    label: str


class SamplesMeta(BaseModel):
    count: int
    request_id: str


class SamplesResponse(BaseModel):
    success: bool
    data: list[SampleOut]
    error: None
    meta: SamplesMeta


class MetaResponse(BaseModel):
    success: bool
    data: dict[str, Any]
    error: None
    meta: RequestMeta


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
