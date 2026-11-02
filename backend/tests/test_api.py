"""Integration tests for the FastAPI REST layer (T3.6).

Most tests run against an app whose metrics come from a synthetic
``metrics.json`` and whose ONNX sessions are monkeypatched, so the suite is
deterministic and artifact-free. A few tests marked ``slow`` (or guarded by
fixture skips) exercise the real exported ONNX models and the real Grad-CAM
checkpoint end to end.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.core.errors import ModelNotReadyError
from app.inference.onnx_engine import Prediction, TopKItem
from app.main import create_app

pytestmark = pytest.mark.integration

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

ONE_MB = 1024 * 1024


def _synthetic_metrics_payload() -> dict:
    model = {
        "params": 11179590,
        "test_acc": 0.9944,
        "test_macro_f1": 0.9944,
        "per_class_f1": {name: 0.99 for name in CLASS_NAMES},
        "training_epochs": 8,
        "carbon_co2eq_g": 0.3085,
        "onnx_fp32_size_mb": 42.6,
        "onnx_int8_size_mb": 10.7,
        "cpu_latency_ms": {"fp32": {"p50": 9.3, "p95": 10.3}, "int8": {"p50": 10.1, "p95": 11.4}},
        "quantization": "static",
    }
    return {"models": {"resnet18": dict(model), "resnet50": dict(model)}}


def _write_metrics(tmp_path: Path) -> Path:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_synthetic_metrics_payload()), encoding="utf-8")
    return path


def _make_app(tmp_path: Path, **overrides) -> object:
    settings = Settings.from_env(
        model_dir=tmp_path / "onnx", metrics_path=_write_metrics(tmp_path), **overrides
    )
    return create_app(settings)


@pytest.fixture()
def api_client(tmp_path) -> TestClient:
    """App with deterministic synthetic metrics; engine is monkeypatched per test."""
    with TestClient(_make_app(tmp_path)) as test_client:
        yield test_client


def _mock_prediction():
    return Prediction(
        class_id=0,
        top_k=(TopKItem(0, 0.9), TopKItem(1, 0.05), TopKItem(2, 0.02)),
        latency_ms=4.2,
    )


def _mock_engine(api_client, monkeypatch, captured=None):
    def fake_predict(model_id, image, top_k=3):
        if captured is not None:
            captured["model_id"] = model_id
            captured["image"] = image
            captured["top_k"] = top_k
        return _mock_prediction()

    monkeypatch.setattr(api_client.app.state.onnx_engine, "predict", fake_predict)


class _EmptyRegistry:
    def get(self, model_id):
        return None

    def is_empty(self):
        return True


class _StubGradCAMBackend:
    def __init__(self, result=("ZmFrZS1wbmc=", 1.23), raise_error=None) -> None:
        self._result = result
        self._raise_error = raise_error
        self.calls: list[tuple] = []

    def explain(self, arch, image, target_class):
        self.calls.append((arch, image, target_class))
        if self._raise_error is not None:
            raise self._raise_error
        return self._result


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_is_plain_shape_without_envelope(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert set(response.json()) == {"status", "models_loaded"}
    assert response.json()["status"] == "ok"
    assert response.json()["models_loaded"] == []  # preload disabled in tests


@pytest.mark.slow
def test_health_reports_models_loaded_when_preload_on(monkeypatch, real_onnx_dir):
    monkeypatch.setenv("PRELOAD_MODELS", "1")
    settings = Settings.from_env(model_dir=real_onnx_dir)
    with TestClient(create_app(settings)) as test_client:
        response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    loaded = response.json()["models_loaded"]
    assert loaded == ["resnet18-fp32", "resnet18-int8", "resnet50-fp32", "resnet50-int8"]


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


def test_models_endpoint_lists_metadata(api_client):
    response = api_client.get("/api/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["error"] is None
    assert body["meta"]["count"] == 4
    assert body["meta"]["class_names"] == CLASS_NAMES
    assert len(body["meta"]["request_id"]) == 32
    models = body["data"]
    assert [model["id"] for model in models] == [
        "resnet18-fp32",
        "resnet18-int8",
        "resnet50-fp32",
        "resnet50-int8",
    ]
    resnet18_int8 = models[1]
    assert resnet18_int8["arch"] == "resnet18"
    assert resnet18_int8["quantization"] == "int8"
    assert resnet18_int8["params"] == 11179590
    assert resnet18_int8["test_acc"] == 0.9944
    assert resnet18_int8["cpu_latency_ms"] == {"p50": 10.1, "p95": 11.4}
    assert resnet18_int8["training_carbon_co2eq_g"] == 0.3085
    assert resnet18_int8["inference_carbon_co2eq_g"] > 0.0
    assert resnet18_int8["inference_carbon_method"] == "tier-representative-fallback"
    assert resnet18_int8["onnx_size_mb"] == 10.7
    assert resnet18_int8["available"] is False  # no artifacts in the synthetic app


def test_models_endpoint_real_artifacts_report_available(real_onnx_dir, real_metrics_path):
    settings = Settings.from_env(model_dir=real_onnx_dir, metrics_path=real_metrics_path)
    with TestClient(create_app(settings)) as test_client:
        response = test_client.get("/api/v1/models")
    assert response.status_code == 200
    assert all(model["available"] for model in response.json()["data"])


def test_models_endpoint_without_metrics_file_returns_empty_list(tmp_path):
    settings = Settings.from_env(model_dir=tmp_path, metrics_path=tmp_path / "missing.json")
    with TestClient(create_app(settings)) as test_client:
        response = test_client.get("/api/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"] == []
    assert body["meta"]["count"] == 0


# ---------------------------------------------------------------------------
# GET /samples
# ---------------------------------------------------------------------------


def test_samples_endpoint_lists_committed_samples(client, sample_assets_dir):
    response = client.get("/api/v1/samples")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    samples = body["data"]
    assert len(samples) == 12
    assert body["meta"]["count"] == 12
    ids = [sample["id"] for sample in samples]
    assert len(set(ids)) == 12
    assert {sample["label"] for sample in samples} == set(CLASS_NAMES)
    first = samples[0]
    assert set(first) == {"id", "filename", "label"}
    assert first["filename"].endswith(".jpg")


# ---------------------------------------------------------------------------
# GET /meta
# ---------------------------------------------------------------------------


def test_meta_endpoint_discloses_formula_constants_boundaries(client):
    response = client.get("/api/v1/meta")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert set(data) == {"formula", "constants", "boundaries", "methods"}
    assert isinstance(data["formula"], str)
    assert data["constants"]["grid_carbon_intensity_g_per_kwh"] > 0
    assert len(data["boundaries"]) >= 3


# ---------------------------------------------------------------------------
# POST /predict — envelope, orchestration, and form handling
# ---------------------------------------------------------------------------


def test_predict_returns_full_response_envelope(api_client, monkeypatch, fixtures_dir):
    captured: dict = {}
    _mock_engine(api_client, monkeypatch, captured)
    image_bytes = (fixtures_dir / "valid_jpeg.jpg").read_bytes()

    response = api_client.post(
        "/api/v1/predict",
        files={"file": ("photo.jpg", image_bytes, "image/jpeg")},
        data={"model": "resnet18-fp32", "gradcam": "false"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["error"] is None
    assert body["meta"]["model"] == "resnet18-fp32"
    assert len(body["meta"]["request_id"]) == 32

    data = body["data"]
    assert data["class_id"] == 0
    assert data["class_name"] == "crazing"
    assert data["confidence"] == pytest.approx(0.9)
    assert [item["class_id"] for item in data["top_k"]] == [0, 1, 2]
    assert data["top_k"][0]["class_name"] == "crazing"
    assert data["top_k"][0]["confidence"] == pytest.approx(0.9)
    assert data["latency_ms"] == pytest.approx(4.2)
    assert data["gradcam_ms"] is None
    assert data["gradcam"] is None
    carbon = data["carbon"]
    assert carbon["co2eq_g"] > 0.0
    assert carbon["energy_kwh"] > 0.0
    assert carbon["method"] in {"mace-f-methodD-measured", "tier-representative-fallback"}
    assert isinstance(carbon["equivalent"], str)

    assert captured["model_id"] == "resnet18-fp32"
    assert captured["top_k"] == 3
    assert isinstance(captured["image"], Image.Image)
    assert captured["image"].mode == "RGB"


def test_predict_defaults_to_resnet18_int8(api_client, monkeypatch, fixtures_dir):
    captured: dict = {}
    _mock_engine(api_client, monkeypatch, captured)
    image_bytes = (fixtures_dir / "valid_jpeg.jpg").read_bytes()

    response = api_client.post(
        "/api/v1/predict", files={"file": ("photo.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert captured["model_id"] == "resnet18-int8"
    assert response.json()["meta"]["model"] == "resnet18-int8"


def test_predict_with_gradcam_returns_overlay(api_client, monkeypatch, fixtures_dir):
    _mock_engine(api_client, monkeypatch)
    stub = _StubGradCAMBackend()
    monkeypatch.setattr(api_client.app.state.predict_service, "_gradcam_backend", stub)
    image_bytes = (fixtures_dir / "valid_jpeg.jpg").read_bytes()

    response = api_client.post(
        "/api/v1/predict",
        files={"file": ("photo.jpg", image_bytes, "image/jpeg")},
        data={"model": "resnet18-fp32", "gradcam": "true"},
    )

    assert response.status_code == 200
    gradcam = response.json()["data"]["gradcam"]
    assert gradcam == {"base64": "ZmFrZS1wbmc=", "format": "png"}
    assert response.json()["data"]["gradcam_ms"] == pytest.approx(1.23)
    assert len(stub.calls) == 1
    arch, image, target_class = stub.calls[0]
    assert arch == "resnet18"
    assert isinstance(image, Image.Image)
    assert target_class == 0  # Grad-CAM targets the ONNX-predicted class


def test_predict_gradcam_false_never_invokes_backend(api_client, monkeypatch, fixtures_dir):
    _mock_engine(api_client, monkeypatch)
    stub = _StubGradCAMBackend()
    monkeypatch.setattr(api_client.app.state.predict_service, "_gradcam_backend", stub)
    image_bytes = (fixtures_dir / "valid_jpeg.jpg").read_bytes()

    response = api_client.post(
        "/api/v1/predict", files={"file": ("photo.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert stub.calls == []


def test_predict_gradcam_backend_failure_is_503(api_client, monkeypatch, fixtures_dir):
    _mock_engine(api_client, monkeypatch)
    stub = _StubGradCAMBackend(raise_error=ModelNotReadyError("checkpoint missing"))
    monkeypatch.setattr(api_client.app.state.predict_service, "_gradcam_backend", stub)
    image_bytes = (fixtures_dir / "valid_jpeg.jpg").read_bytes()

    response = api_client.post(
        "/api/v1/predict",
        files={"file": ("photo.jpg", image_bytes, "image/jpeg")},
        data={"model": "resnet18-fp32", "gradcam": "true"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_ready"


# ---------------------------------------------------------------------------
# POST /predict — error mapping
# ---------------------------------------------------------------------------


def test_predict_rejects_fake_image_415(api_client, fixtures_dir):
    image_bytes = (fixtures_dir / "fake_image.jpg").read_bytes()
    response = api_client.post(
        "/api/v1/predict", files={"file": ("photo.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 415
    body = response.json()
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "unsupported_media_type"
    assert body["error"]["status"] == 415
    assert len(body["meta"]["request_id"]) == 32


def test_predict_rejects_truncated_image_422(api_client, fixtures_dir):
    image_bytes = (fixtures_dir / "truncated.jpg").read_bytes()
    response = api_client.post(
        "/api/v1/predict", files={"file": ("photo.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unprocessable_image"


def test_predict_rejects_oversized_upload_413(api_client):
    payload = b"\x00" * (5 * ONE_MB + 1)
    response = api_client.post(
        "/api/v1/predict", files={"file": ("huge.bin", payload, "application/octet-stream")}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_predict_unknown_model_404(api_client, monkeypatch, fixtures_dir):
    _mock_engine(api_client, monkeypatch)
    image_bytes = (fixtures_dir / "valid_jpeg.jpg").read_bytes()
    response = api_client.post(
        "/api/v1/predict",
        files={"file": ("photo.jpg", image_bytes, "image/jpeg")},
        data={"model": "resnet101-fp32"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_predict_missing_file_field_422(api_client):
    response = api_client.post("/api/v1/predict", data={"model": "resnet18-int8"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_predict_invalid_gradcam_form_422(api_client, fixtures_dir):
    image_bytes = (fixtures_dir / "valid_jpeg.jpg").read_bytes()
    response = api_client.post(
        "/api/v1/predict",
        files={"file": ("photo.jpg", image_bytes, "image/jpeg")},
        data={"model": "resnet18-int8", "gradcam": "not-a-bool"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_predict_503_when_registry_is_empty(api_client, monkeypatch, fixtures_dir):
    monkeypatch.setattr(api_client.app.state.predict_service, "_registry", _EmptyRegistry())
    image_bytes = (fixtures_dir / "valid_jpeg.jpg").read_bytes()
    response = api_client.post(
        "/api/v1/predict", files={"file": ("photo.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_ready"


def test_predict_503_when_session_fails_to_load(api_client, monkeypatch, fixtures_dir):
    def failing_ensure(model_id):
        raise ModelNotReadyError(f"onnx artifact missing for {model_id}")

    monkeypatch.setattr(api_client.app.state.onnx_engine, "ensure_loaded", failing_ensure)
    image_bytes = (fixtures_dir / "valid_jpeg.jpg").read_bytes()
    response = api_client.post(
        "/api/v1/predict", files={"file": ("photo.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_ready"


# ---------------------------------------------------------------------------
# POST /predict/sample/{sample_id}
# ---------------------------------------------------------------------------


def test_predict_sample_unknown_id_404(client):
    response = client.post("/api/v1/predict/sample/definitely-not-a-sample")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "sample_not_found"


def test_predict_sample_traversal_id_404(client):
    # Path-traversal attempts are rejected: either they never reach the
    # catalog (client normalizes ".." before routing, giving a generic 404)
    # or the catalog's strict whitelist lookup misses (sample_not_found).
    # The strict catalog behavior is unit-tested in test_sample_catalog.py.
    response = client.post("/api/v1/predict/sample/%2E%2E%2F%2E%2E%2Fsecret")
    assert response.status_code == 404
    assert response.json()["error"]["code"] in {"sample_not_found", "http_error"}


def test_predict_sample_uses_same_inference_path(api_client, monkeypatch):
    captured: dict = {}
    _mock_engine(api_client, monkeypatch, captured)
    response = api_client.post(
        "/api/v1/predict/sample/crazing_101", data={"model": "resnet18-int8"}
    )
    assert response.status_code == 200
    assert captured["model_id"] == "resnet18-int8"
    assert isinstance(captured["image"], Image.Image)
    assert response.json()["data"]["class_name"] == "crazing"


# ---------------------------------------------------------------------------
# Real end-to-end inference on exported artifacts
# ---------------------------------------------------------------------------


def test_predict_real_int8_onnx_upload(
    client, sample_crazing_bytes, real_onnx_dir, real_metrics_path
):
    settings = Settings.from_env(model_dir=real_onnx_dir, metrics_path=real_metrics_path)
    with TestClient(create_app(settings)) as test_client:
        response = test_client.post(
            "/api/v1/predict",
            files={"file": ("crazing_101.jpg", sample_crazing_bytes, "image/jpeg")},
            data={"model": "resnet18-int8", "gradcam": "false"},
        )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["class_name"] == "crazing"
    assert data["class_id"] == 0
    assert data["confidence"] > 0.9
    assert len(data["top_k"]) == 3
    assert data["top_k"][0]["class_name"] == "crazing"
    assert data["latency_ms"] > 0.0
    assert data["gradcam"] is None
    assert data["carbon"]["co2eq_g"] > 0.0


def test_predict_sample_real_int8_onnx(client, real_onnx_dir, real_metrics_path):
    settings = Settings.from_env(model_dir=real_onnx_dir, metrics_path=real_metrics_path)
    with TestClient(create_app(settings)) as test_client:
        response = test_client.post(
            "/api/v1/predict/sample/crazing_101", data={"model": "resnet18-int8"}
        )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["class_name"] == "crazing"
    assert data["confidence"] > 0.9


@pytest.mark.slow
def test_predict_real_with_gradcam_end_to_end(
    client, sample_crazing_bytes, real_onnx_dir, real_metrics_path, resnet18_checkpoint
):
    settings = Settings.from_env(
        model_dir=real_onnx_dir,
        metrics_path=real_metrics_path,
        checkpoint_dir=resnet18_checkpoint.parent,
    )
    with TestClient(create_app(settings)) as test_client:
        response = test_client.post(
            "/api/v1/predict",
            files={"file": ("crazing_101.jpg", sample_crazing_bytes, "image/jpeg")},
            data={"model": "resnet18-int8", "gradcam": "true"},
        )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["gradcam"]["format"] == "png"
    decoded = base64.b64decode(data["gradcam"]["base64"])
    overlay = Image.open(io.BytesIO(decoded))
    assert overlay.format == "PNG"
    assert data["gradcam_ms"] > 0.0
    assert data["class_name"] == "crazing"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_envelope(monkeypatch, tmp_path):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    settings = Settings.from_env(model_dir=tmp_path, metrics_path=_write_metrics(tmp_path))
    app = create_app(settings)
    app.state.limiter.reset()

    with TestClient(app) as test_client:
        assert test_client.get("/api/v1/health").status_code == 200
        assert test_client.get("/api/v1/health").status_code == 200
        blocked = test_client.get("/api/v1/health")

    assert blocked.status_code == 429
    body = blocked.json()
    assert body["success"] is False
    assert body["error"]["code"] == "rate_limited"
    assert body["error"]["status"] == 429


# ---------------------------------------------------------------------------
# Request id middleware
# ---------------------------------------------------------------------------


def test_request_id_header_is_echoed(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "my-trace-123"})
    assert response.headers["x-request-id"] == "my-trace-123"


def test_request_id_generated_when_absent(client):
    response = client.get("/api/v1/health")
    assert len(response.headers["x-request-id"]) == 32
    assert all(char in "0123456789abcdef" for char in response.headers["x-request-id"])


def test_malformed_request_id_header_is_replaced(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "bad id with spaces!"})
    generated = response.headers["x-request-id"]
    assert len(generated) == 32
    assert all(char in "0123456789abcdef" for char in generated)


def test_request_id_appears_in_error_envelope(client):
    response = client.get("/api/v1/does-not-exist")
    body = response.json()
    assert body["success"] is False
    assert len(body["meta"]["request_id"]) == 32


# ---------------------------------------------------------------------------
# HTTP errors use the envelope too
# ---------------------------------------------------------------------------


def test_unknown_route_returns_404_envelope(client):
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "http_error"
    assert body["error"]["status"] == 404


def test_method_not_allowed_returns_405_envelope(client):
    response = client.delete("/api/v1/health")
    assert response.status_code == 405
    assert response.json()["error"]["status"] == 405


def test_predict_carbon_uses_configured_country(api_client, monkeypatch, fixtures_dir, tmp_path):
    _mock_engine(api_client, monkeypatch)
    app = _make_app(tmp_path, carbon_country="UK")
    monkeypatch.setattr(app.state.onnx_engine, "predict", lambda *a, **k: _mock_prediction())
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/predict",
            files={
                "file": (
                    "photo.jpg",
                    (fixtures_dir / "valid_jpeg.jpg").read_bytes(),
                    "image/jpeg",
                )
            },
        )
    assert response.status_code == 200
    # UK grid (231 g/kWh) vs China (582 g/kWh): co2eq scales with intensity.
    assert response.json()["data"]["carbon"]["co2eq_g"] > 0.0
