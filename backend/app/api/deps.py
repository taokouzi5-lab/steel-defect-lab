"""FastAPI dependency providers (T3.6).

State is assembled once in ``create_app`` and stored on ``app.state``; the
providers below just look it up per request, which keeps the test overrides
trivial (``app.dependency_overrides`` or monkeypatching ``app.state``).
"""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.inference.model_registry import ModelRegistry
from app.inference.onnx_engine import OnnxEngine
from app.services.predict_service import PredictService
from app.services.sample_catalog import SampleCatalog


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_engine(request: Request) -> OnnxEngine:
    return request.app.state.onnx_engine


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


def get_catalog(request: Request) -> SampleCatalog:
    return request.app.state.sample_catalog


def get_predict_service(request: Request) -> PredictService:
    return request.app.state.predict_service
