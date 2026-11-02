"""Unit tests for the model registry (T3.6).

The registry turns artifacts/metrics.json plus the carbon estimator into the
read-only metadata served by ``GET /models``. Tests use a synthetic metrics
dict so they run without the (gitignored, generated) real artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.inference.model_registry import ModelRegistry

pytestmark = pytest.mark.unit

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def _synthetic_metrics() -> dict:
    model = {
        "params": 11179590,
        "test_acc": 0.9944,
        "test_macro_f1": 0.9944,
        "per_class_f1": {name: 0.99 for name in CLASS_NAMES},
        "training_epochs": 8,
        "carbon_co2eq_g": 0.3085,
        "onnx_fp32_size_mb": 42.6377,
        "onnx_int8_size_mb": 10.7735,
        "cpu_latency_ms": {"fp32": {"p50": 9.3, "p95": 10.3}, "int8": {"p50": 10.1, "p95": 11.4}},
        "quantization": "static",
    }
    return {"models": {"resnet18": dict(model), "resnet50": {**dict(model), "params": 23520326}}}


def _write_metrics(tmp_path: Path, payload) -> Path:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _registry(tmp_path: Path, metrics: dict | None = None) -> ModelRegistry:
    metrics_path = _write_metrics(
        tmp_path, metrics if metrics is not None else _synthetic_metrics()
    )
    return ModelRegistry(metrics_path=metrics_path, model_dir=tmp_path / "onnx")


@pytest.fixture()
def registry(tmp_path) -> ModelRegistry:
    return _registry(tmp_path)


# ---------------------------------------------------------------------------
# Listing and metadata
# ---------------------------------------------------------------------------


def test_lists_four_model_ids(registry):
    ids = [model.id for model in registry.list()]
    assert ids == ["resnet18-fp32", "resnet18-int8", "resnet50-fp32", "resnet50-int8"]


def test_metadata_fields_from_metrics(registry):
    resnet18 = registry.get("resnet18-fp32")
    assert resnet18 is not None
    assert resnet18.arch == "resnet18"
    assert resnet18.quantization == "fp32"
    assert resnet18.params == 11179590
    assert resnet18.test_acc == 0.9944
    assert resnet18.test_macro_f1 == 0.9944
    assert resnet18.cpu_latency_ms == {"p50": 9.3, "p95": 10.3}
    assert resnet18.training_carbon_co2eq_g == 0.3085
    assert resnet18.onnx_size_mb == 42.6377


def test_int8_metadata_uses_int8_latency_and_size(registry):
    resnet50_int8 = registry.get("resnet50-int8")
    assert resnet50_int8 is not None
    assert resnet50_int8.params == 23520326
    assert resnet50_int8.cpu_latency_ms == {"p50": 10.1, "p95": 11.4}
    assert resnet50_int8.onnx_size_mb == 10.7735


def test_class_names_follow_training_order(registry):
    # Training's manifest.json lists classes in this exact (alphabetical)
    # order; the registry derives the same order from per_class_f1 keys.
    assert registry.class_names == CLASS_NAMES


# ---------------------------------------------------------------------------
# Carbon benchmarks
# ---------------------------------------------------------------------------


def test_inference_carbon_resnet50_is_measured(registry):
    meta = registry.get("resnet50-fp32")
    assert meta.inference_carbon_co2eq_g > 0.0
    assert meta.inference_carbon_method == "mace-f-methodD-measured"


def test_inference_carbon_resnet18_falls_back_to_t3_representative(registry):
    meta = registry.get("resnet18-fp32")
    assert meta.inference_carbon_co2eq_g > 0.0
    assert meta.inference_carbon_method == "tier-representative-fallback"


# ---------------------------------------------------------------------------
# Availability (onnx artifact presence)
# ---------------------------------------------------------------------------


def test_availability_false_when_artifact_missing(registry, tmp_path):
    assert all(not model.available for model in registry.list())


def test_availability_true_when_artifact_present(tmp_path):
    # Availability is evaluated when the registry is built, so the artifacts
    # must exist before construction.
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "resnet18_fp32.onnx").write_bytes(b"x")
    (tmp_path / "onnx" / "resnet50_int8_static.onnx").write_bytes(b"x")
    registry = ModelRegistry(
        metrics_path=_write_metrics(tmp_path, _synthetic_metrics()), model_dir=tmp_path / "onnx"
    )

    assert registry.get("resnet18-fp32").available
    assert registry.get("resnet50-int8").available
    assert not registry.get("resnet18-int8").available


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


def test_missing_metrics_file_yields_empty_registry(tmp_path):
    registry = ModelRegistry(metrics_path=tmp_path / "nope.json", model_dir=tmp_path)
    assert registry.is_empty()
    assert registry.list() == []


def test_malformed_metrics_yields_empty_registry(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("this is not json at all", encoding="utf-8")
    registry = ModelRegistry(metrics_path=path, model_dir=tmp_path)
    assert registry.is_empty()
    assert registry.list() == []


def test_metrics_that_is_valid_json_but_wrong_shape_yields_empty(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text('"just a string"', encoding="utf-8")
    registry = ModelRegistry(metrics_path=path, model_dir=tmp_path)
    assert registry.is_empty()
    assert registry.list() == []


def test_metrics_without_models_key_yields_empty(tmp_path):
    registry = ModelRegistry(metrics_path=_write_metrics(tmp_path, {}), model_dir=tmp_path)
    assert registry.is_empty()


def test_get_unknown_model_returns_none(registry):
    assert registry.get("resnet18-fp16") is None
    assert registry.get("") is None


def test_real_metrics_round_trip(real_metrics_path, tmp_path):
    registry = ModelRegistry(metrics_path=real_metrics_path, model_dir=tmp_path)
    assert [model.id for model in registry.list()] == [
        "resnet18-fp32",
        "resnet18-int8",
        "resnet50-fp32",
        "resnet50-int8",
    ]
    assert registry.class_names == CLASS_NAMES
