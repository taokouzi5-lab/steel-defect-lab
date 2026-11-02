"""onnxruntime wrapper for the exported ResNet models (T3.5/T3.6).

- Whitelist: resnet18/resnet50 x fp32/int8 — anything else is a 404.
- Lazy session loading; a model whose artifact is missing or that fails to
  initialize surfaces as 503 (model known, not serving) with a generic
  client message — the full path stays in the server log.
- Preprocessing is pure numpy+PIL and mirrors the training eval pipeline
  (``Resize(224)`` = shorter side to 224, truncated to int exactly like
  torchvision, then a center crop to the fixed 224x224 graph input).
- Session.run is executed with a configurable timeout, and concurrent
  inferences are bounded by a process-lifetime executor + semaphore.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.core.errors import InferenceTimeoutError, ModelNotFoundError, ModelNotReadyError
from app.core.sanitize import sanitize_text

logger = logging.getLogger(__name__)

ARCHS = ("resnet18", "resnet50")
QUANTS = ("fp32", "int8")

# int8 ids map to the exporter's *_int8_static.onnx filenames.
_QUANT_FILE_SUFFIX = {"fp32": "fp32", "int8": "int8_static"}

INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -- bounded inference execution (M5) ----------------------------------------
# session.run is CPU-bound but releases the GIL, so concurrent requests can
# oversubscribe the host. Bound them: one process-lifetime executor whose
# worker count equals the semaphore, admitting at most 2 in-flight
# inferences at any time. The executor is never shut down — a timeout cannot
# cancel an in-flight session.run (threads are not cooperatively
# cancellable), so the bounded singleton caps how many threads stuck calls
# can occupy for the life of the process.
_MAX_CONCURRENT_INFERENCES = 2
_INFERENCE_EXECUTOR = ThreadPoolExecutor(
    max_workers=_MAX_CONCURRENT_INFERENCES, thread_name_prefix="defectlens-inference"
)
_INFERENCE_SLOTS = threading.Semaphore(_MAX_CONCURRENT_INFERENCES)


@dataclass(frozen=True)
class TopKItem:
    """A single top-k entry: class index and softmax confidence."""

    class_id: int
    confidence: float


@dataclass(frozen=True)
class Prediction:
    """Engine output: predicted class plus top-k list and wall latency."""

    class_id: int
    top_k: tuple[TopKItem, ...]
    latency_ms: float


class OnnxEngine:
    """Loads and runs ONNX sessions; all computation is in-memory."""

    def __init__(self, model_dir: Path, timeout_s: float = 10.0) -> None:
        self._model_dir = Path(model_dir)
        self._timeout_s = float(timeout_s)
        self._sessions: dict[str, ort.InferenceSession] = {}

    # -- model id whitelist ------------------------------------------------

    @staticmethod
    def parse_model_id(model_id: str) -> tuple[str, str]:
        """Split ``"<arch>-<quant>"``; anything outside the whitelist is 404."""
        parts = model_id.split("-") if isinstance(model_id, str) else []
        if len(parts) != 2 or parts[0] not in ARCHS or parts[1] not in QUANTS:
            raise ModelNotFoundError(
                f"unknown model id '{sanitize_text(model_id)}'; expected <arch>-<quant> "
                f"with arch in {ARCHS} and quant in {QUANTS}"
            )
        return parts[0], parts[1]

    @staticmethod
    def model_filename(model_id: str) -> str:
        arch, quant = OnnxEngine.parse_model_id(model_id)
        return f"{arch}_{_QUANT_FILE_SUFFIX[quant]}.onnx"

    # -- session management ------------------------------------------------

    def ensure_loaded(self, model_id: str) -> ort.InferenceSession:
        """Return the session for ``model_id``, loading it on first use."""
        self.parse_model_id(model_id)  # whitelist check -> 404
        session = self._sessions.get(model_id)
        if session is not None:
            return session

        path = self._model_dir / self.model_filename(model_id)
        if not path.is_file():
            # Full path goes to the server log only; the client message is
            # generic so no filesystem layout leaks (M1).
            logger.warning("onnx artifact %s is missing from %s", path.name, self._model_dir)
            raise ModelNotReadyError("model artifact not ready")
        try:
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as exc:
            logger.exception("failed to initialize onnx session for %s", model_id)
            raise ModelNotReadyError(
                f"failed to initialize the {model_id} inference session"
            ) from exc
        self._sessions[model_id] = session
        logger.info("loaded onnx session %s from %s", model_id, path)
        return session

    def loaded_ids(self) -> list[str]:
        return list(self._sessions)

    def load_all(self, model_ids: list[str]) -> list[str]:
        """Best-effort eager load; unavailable models are logged and skipped."""
        loaded: list[str] = []
        for model_id in model_ids:
            try:
                self.ensure_loaded(model_id)
                loaded.append(model_id)
            except (ModelNotFoundError, ModelNotReadyError) as exc:
                logger.warning("preload skipped %s: %s", model_id, exc)
        return loaded

    # -- preprocessing ------------------------------------------------------

    @staticmethod
    def preprocess(image: Image.Image) -> np.ndarray:
        """Convert an RGB PIL image to a float32 ``[1, 3, 224, 224]`` tensor.

        Mirrors the training eval transform (Resize(224) = shorter side to
        224). The scaled size is truncated to int exactly like torchvision
        (``int(scale * side)``, not rounding), and the result is center
        cropped to the graph's fixed 224x224 input.

        Alignment contract: in the square NEU-DET domain the crop is a no-op
        and results are bit-exact with the training pipeline. For non-square
        inputs this is a conventional resize+crop handling, not a
        bit-for-bit guarantee.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"image must be a PIL Image, got {type(image).__name__}")
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = INPUT_SIZE / min(width, height)
        new_width = max(INPUT_SIZE, int(width * scale))
        new_height = max(INPUT_SIZE, int(height * scale))
        resized = rgb.resize((new_width, new_height), Image.Resampling.BILINEAR)
        left = (new_width - INPUT_SIZE) // 2
        top = (new_height - INPUT_SIZE) // 2
        cropped = resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

        array = np.asarray(cropped, dtype=np.float32) / 255.0
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(IMAGENET_STD, dtype=np.float32)
        array = (array - mean) / std
        return np.ascontiguousarray(array.transpose(2, 0, 1))[None, ...]

    # -- inference ----------------------------------------------------------

    def predict(self, model_id: str, image: Image.Image, top_k: int = 3) -> Prediction:
        """Run one inference and return the predicted class with top-k."""
        session = self.ensure_loaded(model_id)
        inputs = {session.get_inputs()[0].name: self.preprocess(image)}
        started = perf_counter()
        logits = self._run_with_timeout(session, inputs)
        latency_ms = (perf_counter() - started) * 1000.0
        probabilities = _softmax(np.asarray(logits, dtype=np.float64)[0])
        order = np.argsort(probabilities)[::-1][:top_k]
        top = tuple(
            TopKItem(class_id=int(index), confidence=float(probabilities[index])) for index in order
        )
        return Prediction(class_id=top[0].class_id, top_k=top, latency_ms=latency_ms)

    def _run_with_timeout(
        self, session: ort.InferenceSession, inputs: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Run ``session.run`` with a wall-clock timeout and bounded concurrency.

        The module-level semaphore admits at most
        ``_MAX_CONCURRENT_INFERENCES`` in-flight inferences; the executor is
        a process-lifetime singleton (never shut down) whose worker count
        caps the threads that timed-out, uncancellable ``session.run`` calls
        can occupy — a timeout only abandons the wait, it cannot stop the
        run, and the slot is released in ``finally`` so a stuck call cannot
        exhaust admission.
        """
        if not _INFERENCE_SLOTS.acquire(timeout=self._timeout_s):
            raise InferenceTimeoutError(
                f"no inference slot became available within {self._timeout_s}s"
            )
        try:
            future = _INFERENCE_EXECUTOR.submit(session.run, None, inputs)
            try:
                outputs = future.result(timeout=self._timeout_s)
            except TimeoutError as exc:
                future.cancel()
                raise InferenceTimeoutError(
                    f"inference exceeded the {self._timeout_s}s timeout"
                ) from exc
            return np.asarray(outputs[0])
        finally:
            _INFERENCE_SLOTS.release()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
