"""Unit tests for predict-service error hygiene (L1 log injection).

Client-controlled identifiers (sample ids, model ids) are echoed into error
messages; control characters must be stripped before interpolation so error
payloads and logs cannot be injected with newlines or terminal escapes.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.config import Settings
from app.core.errors import ModelNotFoundError, SampleNotFoundError
from app.services.predict_service import PredictService

pytestmark = pytest.mark.unit


class _NoModelRegistry:
    def get(self, model_id):
        return None

    def is_empty(self):
        return False


class _MissingCatalog:
    def get(self, sample_id):
        return None


def _service(tmp_path) -> PredictService:
    settings = Settings.from_env(model_dir=tmp_path, metrics_path=tmp_path / "missing.json")
    return PredictService(
        engine=None, registry=_NoModelRegistry(), settings=settings, catalog=_MissingCatalog()
    )


def _valid_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_predict_sample_error_sanitizes_sample_id(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(SampleNotFoundError) as excinfo:
        service.predict_sample("evil\nid", "resnet18-int8")
    message = str(excinfo.value)
    assert "\n" not in message
    assert "evilid" in message


def test_unknown_model_error_sanitizes_model_id(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(ModelNotFoundError) as excinfo:
        service.predict(_valid_png_bytes(), "resnet18-fp32\ninjected")
    message = str(excinfo.value)
    assert "\n" not in message
    assert "resnet18-fp32injected" in message
