"""Unit tests for the onnxruntime wrapper (T3.5/T3.6).

Uses injected fake sessions for behavioral tests (fast, artifact-free) and
compares the numpy preprocessor against the torchvision training pipeline
for exact parity.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from PIL import Image

from app.core.errors import InferenceTimeoutError, ModelNotFoundError, ModelNotReadyError
from app.inference import onnx_engine
from app.inference.onnx_engine import IMAGENET_MEAN, IMAGENET_STD, OnnxEngine

pytestmark = pytest.mark.unit


def _rgb_image(width: int = 200, height: int = 200, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = (rng.integers(0, 256, size=(height, width, 3))).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class FakeSession:
    """Mimics the subset of InferenceSession the engine touches."""

    def __init__(self, logits: np.ndarray, input_name: str = "input"):
        self._logits = logits
        self._input_name = input_name
        self.calls: list[dict] = []

    def get_inputs(self):
        class NodeArg:
            def __init__(self, name: str) -> None:
                self.name = name

        return [NodeArg(self._input_name)]

    def run(self, _output_names, feed: dict):
        self.calls.append(feed)
        return [self._logits]


class SlowSession(FakeSession):
    def run(self, _output_names, feed: dict):
        time.sleep(0.4)
        return [self._logits]


# ---------------------------------------------------------------------------
# Model id parsing (whitelist)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("resnet18-fp32", ("resnet18", "fp32")),
        ("resnet18-int8", ("resnet18", "int8")),
        ("resnet50-fp32", ("resnet50", "fp32")),
        ("resnet50-int8", ("resnet50", "int8")),
    ],
)
def test_parse_model_id_whitelist(model_id, expected):
    assert OnnxEngine.parse_model_id(model_id) == expected


@pytest.mark.parametrize(
    "model_id",
    ["resnet18-fp16", "resnet101-int8", "banana", "resnet18", "", "resnet18-int8-extra"],
)
def test_parse_model_id_rejects_unknown_404(model_id):
    with pytest.raises(ModelNotFoundError) as excinfo:
        OnnxEngine.parse_model_id(model_id)
    assert excinfo.value.status_code == 404


def test_int8_id_maps_to_int8_static_file():
    assert OnnxEngine.model_filename("resnet18-int8") == "resnet18_int8_static.onnx"
    assert OnnxEngine.model_filename("resnet18-fp32") == "resnet18_fp32.onnx"


# ---------------------------------------------------------------------------
# Preprocessing parity with the training eval pipeline
# ---------------------------------------------------------------------------


def test_preprocess_output_shape_and_dtype():
    array = OnnxEngine.preprocess(_rgb_image(200, 200))
    assert array.shape == (1, 3, 224, 224)
    assert array.dtype == np.float32


def test_preprocess_matches_torchvision_on_square_image():
    import torchvision.transforms as transforms

    image = _rgb_image(200, 200)
    expected = (
        transforms.Compose(
            [
                transforms.Resize(224, antialias=False),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )(image)
        .unsqueeze(0)
        .numpy()
    )
    actual = OnnxEngine.preprocess(image)
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_preprocess_matches_torchvision_on_non_square_image():
    import torchvision.transforms as transforms

    image = _rgb_image(300, 200)  # non-square: resize then center crop
    expected = (
        transforms.Compose(
            [
                transforms.Resize(224, antialias=False),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )(image)
        .unsqueeze(0)
        .numpy()
    )
    actual = OnnxEngine.preprocess(image)
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_preprocess_preserves_square_images_exactly():
    # A 200x200 image resizes to exactly 224x224 — the center crop must be a
    # no-op, matching the training transform byte for byte.
    import torchvision.transforms as transforms

    image = _rgb_image(200, 200)
    expected = transforms.Compose([transforms.Resize(224, antialias=False), transforms.ToTensor()])(
        image
    )
    actual = OnnxEngine.preprocess(image)[0]
    actual_unnormalized = (
        actual * np.asarray(IMAGENET_STD, np.float32)[:, None, None]
        + np.asarray(IMAGENET_MEAN, np.float32)[:, None, None]
    )
    np.testing.assert_allclose(actual_unnormalized, expected.numpy(), atol=1e-3)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def test_engine_starts_with_no_loaded_models(tmp_path):
    engine = OnnxEngine(tmp_path)
    assert engine.loaded_ids() == []


def test_ensure_loaded_missing_artifact_raises_503(tmp_path):
    engine = OnnxEngine(tmp_path)
    with pytest.raises(ModelNotReadyError) as excinfo:
        engine.ensure_loaded("resnet18-fp32")
    assert excinfo.value.status_code == 503


def test_ensure_loaded_unknown_id_raises_404(tmp_path):
    engine = OnnxEngine(tmp_path)
    with pytest.raises(ModelNotFoundError):
        engine.ensure_loaded("resnet18-fp16")


def test_ensure_loaded_caches_session(monkeypatch, tmp_path):
    calls = []
    (tmp_path / "resnet18_fp32.onnx").write_bytes(b"placeholder")

    class CountingSession:
        def __init__(self, *args, **kwargs) -> None:
            calls.append(args)

    monkeypatch.setattr(onnx_engine.ort, "InferenceSession", CountingSession)
    engine = OnnxEngine(tmp_path)
    engine.ensure_loaded("resnet18-fp32")
    engine.ensure_loaded("resnet18-fp32")
    assert len(calls) == 1
    assert engine.loaded_ids() == ["resnet18-fp32"]


def test_ensure_loaded_wraps_runtime_errors_as_503(monkeypatch, tmp_path):
    (tmp_path / "resnet18_fp32.onnx").write_bytes(b"not a real onnx file")

    engine = OnnxEngine(tmp_path)
    with pytest.raises(ModelNotReadyError) as excinfo:
        engine.ensure_loaded("resnet18-fp32")
    assert excinfo.value.status_code == 503


def test_load_all_skips_unavailable_models(tmp_path):
    engine = OnnxEngine(tmp_path)
    loaded = engine.load_all(["resnet18-fp32", "resnet18-int8", "unknown-model"])
    assert loaded == []


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_softmax_top_k_and_inputs(monkeypatch, tmp_path):
    logits = np.array([[2.0, 1.0, 0.5, 0.2, 0.1, -1.0]], dtype=np.float32)
    fake = FakeSession(logits)
    engine = OnnxEngine(tmp_path)
    monkeypatch.setattr(engine, "ensure_loaded", lambda model_id: fake)
    monkeypatch.setattr(engine, "_sessions", {"resnet18-fp32": fake})

    prediction = engine.predict("resnet18-fp32", _rgb_image())

    assert prediction.class_id == 0
    assert [item.class_id for item in prediction.top_k] == [0, 1, 2]
    confidences = [item.confidence for item in prediction.top_k]
    assert confidences == sorted(confidences, reverse=True)
    # The full 6-class softmax sums to 1; the top-3 subset is strictly less.
    assert 0.0 < sum(confidences) < 1.0
    # Exact softmax value for the top class: e^2 / sum(e^logits).
    denominator = sum(np.exp(v) for v in [2.0, 1.0, 0.5, 0.2, 0.1, -1.0])
    assert confidences[0] == pytest.approx(np.exp(2.0) / denominator, abs=1e-6)
    assert prediction.latency_ms >= 0.0

    assert len(fake.calls) == 1
    (feed,) = fake.calls
    assert list(feed) == ["input"]
    assert feed["input"].shape == (1, 3, 224, 224)
    assert feed["input"].dtype == np.float32


def test_predict_top_k_limit(monkeypatch, tmp_path):
    fake = FakeSession(np.array([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0]], dtype=np.float32))
    engine = OnnxEngine(tmp_path)
    monkeypatch.setattr(engine, "ensure_loaded", lambda model_id: fake)
    monkeypatch.setattr(engine, "_sessions", {"resnet18-fp32": fake})

    prediction = engine.predict("resnet18-fp32", _rgb_image(), top_k=5)
    assert len(prediction.top_k) == 5


def test_predict_timeout_raises_503(monkeypatch, tmp_path):
    slow = SlowSession(np.zeros((1, 6), np.float32))
    engine = OnnxEngine(tmp_path, timeout_s=0.05)
    monkeypatch.setattr(engine, "ensure_loaded", lambda model_id: slow)
    monkeypatch.setattr(engine, "_sessions", {"resnet18-fp32": slow})

    with pytest.raises(InferenceTimeoutError) as excinfo:
        engine.predict("resnet18-fp32", _rgb_image())
    assert excinfo.value.status_code == 503


def test_predict_rejects_non_pil_image(tmp_path):
    engine = OnnxEngine(tmp_path)
    with pytest.raises(TypeError):
        engine.preprocess(np.zeros((224, 224, 3), dtype=np.uint8))


def test_engine_never_writes_disk(monkeypatch, tmp_path):
    # The whole inference path (preprocess + session run) is memory-only.
    import tempfile

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(scratch))

    logits = np.array([[2.0, 1.0, 0.5, 0.2, 0.1, -1.0]], dtype=np.float32)
    fake = FakeSession(logits)
    engine = OnnxEngine(tmp_path)
    monkeypatch.setattr(engine, "ensure_loaded", lambda model_id: fake)
    monkeypatch.setattr(engine, "_sessions", {"resnet18-fp32": fake})
    engine.predict("resnet18-fp32", _rgb_image())

    assert list(scratch.iterdir()) == []


def test_imagenet_constants_match_training():
    # The training suite normalizes with torchvision's ImageNet constants.
    assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert IMAGENET_STD == [0.229, 0.224, 0.225]


def test_preprocess_output_is_float32_contiguous():
    array = OnnxEngine.preprocess(_rgb_image())
    assert array.flags["C_CONTIGUOUS"]
    assert array.dtype == np.float32


def test_preprocess_accepts_bytes_io_roundtrip():
    # Pillow images opened from BytesIO (validation output) preprocess fine.
    buffer = io.BytesIO()
    _rgb_image().save(buffer, format="JPEG")
    buffer.seek(0)
    image = Image.open(buffer)
    image.load()
    assert OnnxEngine.preprocess(image.convert("RGB")).shape == (1, 3, 224, 224)


def test_onnx_artifact_files_have_expected_input_output_names(real_onnx_dir):
    import onnxruntime as ort

    for filename in ("resnet18_fp32.onnx", "resnet18_int8_static.onnx"):
        session = ort.InferenceSession(str(real_onnx_dir / filename))
        assert [node.name for node in session.get_inputs()] == ["input"]
        assert [node.name for node in session.get_outputs()] == ["logits"]
        assert list(session.get_inputs()[0].shape) == [1, 3, 224, 224]


def test_real_int8_session_runs_on_real_artifact(real_onnx_dir):
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(real_onnx_dir / "resnet18_int8_static.onnx"), providers=["CPUExecutionProvider"]
    )
    output = session.run(None, {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})[0]
    assert output.shape == (1, 6)
    assert np.isfinite(output).all()
