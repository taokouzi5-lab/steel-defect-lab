"""CLI entry point for INT8 quantization with accuracy gate (T3.2).

Quantizes an exported FP32 ONNX model (``scripts/export_onnx.py``) to INT8
using onnxruntime static quantization (per-channel weights, reduced range)
calibrated on a fixed-seed sample of the training set. The INT8 model is
then evaluated on the full 180-image test set against the FP32 ONNX model:
the top-1 drop must stay within 1.5 percentage points, otherwise the script
automatically degrades to dynamic quantization and re-tests; if both INT8
variants miss the gate the FP32 model is kept and the failure is logged
honestly (the FP32 artifact is never deleted).

Finally a CPU latency benchmark (500 timed forwards after 20 warmups, fixed
thread count) compares FP32 and INT8, and the results — model sizes, gate
outcome, top-1 drop, p50/p95 latencies — are merged into the model's entry
in ``artifacts/metrics.json`` (all other entries and fields are preserved).

Usage:
    python training/scripts/quantize.py --model resnet18
    python training/scripts/quantize.py --model resnet50 --calibration-size 100
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the sibling steel_training package importable when this script is run
# directly (e.g. `python training/scripts/quantize.py`).
TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import argparse
import logging
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import quantize_dynamic, quantize_static
from torch.utils.data import DataLoader

from scripts import evaluate, export_onnx
from steel_training.data.augmentation import build_eval_transform
from steel_training.data.dataset import SteelDefectDataset
from steel_training.data.manifest import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "training" / "manifest.json"
DATA_ROOT = REPO_ROOT / "training" / "data"
ONNX_DIR = REPO_ROOT / "artifacts" / "onnx"
METRICS_PATH = REPO_ROOT / "artifacts" / "metrics.json"

CALIBRATION_SIZE = 200
CALIBRATION_SEED = 42
GATE_DROP_LIMIT_PT = 1.5
GATE_EPSILON_PT = 1e-9
# The exported ONNX models have a static batch-1 input, so accuracy
# evaluation feeds one image per forward.
EVAL_BATCH_SIZE = 1
BENCHMARK_ITERATIONS = 500
BENCHMARK_WARMUP = 20
BENCHMARK_THREADS = 4

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments; derive the FP32 ONNX default path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet50"),
        required=True,
        help="model architecture of the FP32 ONNX model to quantize",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=None,
        help="FP32 ONNX model (default: artifacts/onnx/<model>_fp32.onnx)",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=CALIBRATION_SIZE,
        help="number of training images used for static-quantization calibration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ONNX_DIR,
        help="directory for the quantized models (default: artifacts/onnx)",
    )
    args = parser.parse_args(argv)
    if args.onnx_path is None:
        args.onnx_path = ONNX_DIR / f"{args.model}_fp32.onnx"
    return args


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (0 = success)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:  # CLI boundary: report clearly and exit non-zero.
        log.error("quantization failed: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


class CalibrationDataReader:
    """Feeds fixed ``(input,)`` batches to onnxruntime static quantization.

    Implements the ``get_next`` protocol expected by
    ``onnxruntime.quantization.quantize_static``: each call returns the next
    ``{input_name: batch}`` dict (batches cast to float32), then ``None``
    once every batch has been consumed.
    """

    def __init__(self, input_name: str, batches: list[np.ndarray]) -> None:
        self._input_name = input_name
        self._batches = [np.asarray(batch, dtype=np.float32) for batch in batches]
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._batches):
            return None
        batch = self._batches[self._index]
        self._index += 1
        return {self._input_name: batch}


def top1_drop_pt(fp32_top1: float, quantized_top1: float) -> float:
    """Top-1 drop of the quantized model in percentage points.

    Positive means the quantized model is worse than FP32; negative means
    it is (unusually) better.
    """
    return (float(fp32_top1) - float(quantized_top1)) * 100.0


def decide_quantization(
    fp32_top1: float,
    static_top1: float | None,
    dynamic_top1: float | None = None,
    drop_limit_pt: float = GATE_DROP_LIMIT_PT,
) -> dict[str, Any]:
    """Pure accuracy-gate decision for INT8 quantization.

    Returns ``{"kind": "static" | "dynamic" | "none", "drop_pt": float | None}``:
    static is accepted when its top-1 drop is within ``drop_limit_pt``
    percentage points; otherwise dynamic is tried and accepted under the
    same limit; when neither passes, ``kind`` is ``"none"`` (FP32 is kept)
    and ``drop_pt`` reports the best attempt's drop for the record.
    """
    if static_top1 is not None and top1_drop_pt(fp32_top1, static_top1) <= drop_limit_pt + GATE_EPSILON_PT:
        return {
            "kind": "static",
            "drop_pt": round(top1_drop_pt(fp32_top1, static_top1), 4),
        }
    if dynamic_top1 is not None and top1_drop_pt(fp32_top1, dynamic_top1) <= drop_limit_pt + GATE_EPSILON_PT:
        return {
            "kind": "dynamic",
            "drop_pt": round(top1_drop_pt(fp32_top1, dynamic_top1), 4),
        }
    attempts = [top1 for top1 in (static_top1, dynamic_top1) if top1 is not None]
    best = min(attempts, key=lambda top1: top1_drop_pt(fp32_top1, top1)) if attempts else None
    return {
        "kind": "none",
        "drop_pt": round(top1_drop_pt(fp32_top1, best), 4) if best is not None else None,
    }


def merge_onnx_metrics(
    existing: dict[str, Any] | None, model_name: str, fields: dict[str, Any]
) -> dict[str, Any]:
    """Merge ONNX/quantization ``fields`` into ``model_name``'s metrics entry.

    Returns a NEW document; ``existing`` is never mutated. Entries of other
    models, unrelated top-level keys, and pre-existing fields of the target
    entry are all preserved (the update is additive, unlike evaluate.py's
    wholesale entry replacement).
    """
    base = {} if existing is None else dict(existing)
    models = dict(base.get("models", {}))
    entry = dict(models.get(model_name, {}))
    entry.update(dict(fields))
    models[model_name] = entry
    base["models"] = models
    return base


def compute_latency_stats(timings_s: list[float]) -> dict[str, float]:
    """p50/p95 latencies in milliseconds from per-iteration timings (seconds)."""
    if not timings_s:
        raise ValueError("no timings recorded; cannot compute latency statistics")
    values = np.asarray(timings_s, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50.0) * 1000.0),
        "p95_ms": float(np.percentile(values, 95.0) * 1000.0),
    }


def run_latency_benchmark(
    session: ort.InferenceSession,
    input_array: np.ndarray,
    iterations: int = BENCHMARK_ITERATIONS,
    warmup: int = BENCHMARK_WARMUP,
) -> dict[str, float]:
    """Benchmark CPU inference: ``warmup`` untimed runs, then timed runs."""
    input_name = session.get_inputs()[0].name
    feed = {input_name: np.asarray(input_array, dtype=np.float32)}
    for _ in range(warmup):
        session.run(None, feed)
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, feed)
        timings.append(time.perf_counter() - start)
    return compute_latency_stats(timings)


# ---------------------------------------------------------------------------
# Quantization / evaluation plumbing
# ---------------------------------------------------------------------------


def quantize_static_to_file(
    fp32_path: Path, output_path: Path, reader: CalibrationDataReader
) -> Path:
    """Static INT8 quantization (per-channel weights, reduced range)."""
    quantize_static(
        str(fp32_path),
        str(output_path),
        reader,
        per_channel=True,
        reduce_range=True,
    )
    return output_path


def quantize_dynamic_to_file(fp32_path: Path, output_path: Path) -> Path:
    """Dynamic INT8 quantization (fallback when static misses the gate).

    The output is post-processed with :func:`fix_conv_integer_zero_points`
    so that onnxruntime 1.20.x CPU sessions can execute the quantized
    ConvInteger nodes (see that function for the compatibility rationale).
    """
    quantize_dynamic(str(fp32_path), str(output_path))
    return fix_conv_integer_zero_points(output_path)


def _plan_conv_integer_replacements(
    graph: onnx.GraphProto,
) -> dict[int, tuple[onnx.NodeProto, onnx.NodeProto]]:
    """Locate 4-input ConvInteger nodes followed by a Cast node.

    Returns ``{id(conv_integer_node): (conv_integer_node, cast_node)}``.
    Raises ValueError when the surrounding graph does not match the pattern
    produced by onnxruntime's dynamic quantizer (single Cast consumer whose
    output feeds a single Mul node), so silent mis-rewrites are impossible.
    """
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    plans: dict[int, tuple[onnx.NodeProto, onnx.NodeProto]] = {}
    for node in graph.node:
        if node.op_type != "ConvInteger" or len(node.input) != 4:
            continue
        cast_nodes = consumers.get(node.output[0], [])
        if len(cast_nodes) != 1 or cast_nodes[0].op_type != "Cast":
            raise ValueError(
                f"unexpected dynamic-quantization graph around ConvInteger "
                f"'{node.name or node.output[0]}'"
            )
        cast_node = cast_nodes[0]
        mul_nodes = consumers.get(cast_node.output[0], [])
        if len(mul_nodes) != 1 or mul_nodes[0].op_type != "Mul":
            raise ValueError(
                f"unexpected dynamic-quantization graph around Cast "
                f"'{cast_node.name or cast_node.output[0]}'"
            )
        plans[id(node)] = (node, cast_node)
    return plans


def fix_conv_integer_zero_points(model_path: Path) -> Path:
    """Make ORT<=1.20 dynamic-quantized ConvInteger nodes executable on CPU.

    onnxruntime 1.20.x's dynamic quantizer emits ``ConvInteger`` with the
    zero points as extra inputs, but its CPU runtime only implements the
    opset-11 attribute form, so such models fail at inference. Every
    4-input ``ConvInteger`` is rewritten into an equivalent float ``Conv``
    chain: cast ``x`` and ``x_zero_point`` to float, subtract, and convolve
    with a statically dequantized float weight; the downstream scale
    multiplication and bias add are left untouched. With onnxruntime
    >= 1.21 (which emits the attribute form for opset-17 models) there is
    nothing to rewrite and the file is returned unchanged.
    """
    model = onnx.load(str(model_path))
    plans = _plan_conv_integer_replacements(model.graph)
    if not plans:
        return model_path
    _apply_conv_integer_rewrites(model.graph, plans)
    onnx.save(model, str(model_path))
    return model_path


def _apply_conv_integer_rewrites(
    graph: onnx.GraphProto,
    plans: dict[int, tuple[onnx.NodeProto, onnx.NodeProto]],
) -> None:
    """Rewrite the graph in place: replace ConvInteger nodes, drop old Casts."""
    initializers = {init.name: init for init in graph.initializer}
    removed_cast_ids = {id(cast_node) for _, cast_node in plans.values()}
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    for node in graph.node:
        plan = plans.get(id(node))
        if plan is None:
            if id(node) not in removed_cast_ids:
                new_nodes.append(node)
            continue
        nodes, dequantized = _rewrite_conv_integer(node, plan[1], initializers)
        new_nodes.extend(nodes)
        new_initializers.append(dequantized)
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)


def _rewrite_conv_integer(
    conv_integer_node: onnx.NodeProto,
    cast_node: onnx.NodeProto,
    initializers: dict[str, onnx.TensorProto],
) -> tuple[list[onnx.NodeProto], onnx.TensorProto]:
    """Build the float Conv replacement chain for one 4-input ConvInteger.

    Returns the replacement nodes (two Casts, a Sub, and a Conv writing to
    the removed Cast's output name) plus the statically dequantized float
    weight initializer.
    """
    x_q_name, w_q_name, x_zp_name, w_zp_name = conv_integer_node.input[:4]
    weights = numpy_helper.to_array(initializers[w_q_name]).astype(np.float32)
    zero_points = numpy_helper.to_array(initializers[w_zp_name]).astype(np.float32)
    if zero_points.ndim == 1:  # per-channel zero points must broadcast per filter
        zero_points = zero_points.reshape((-1,) + (1,) * (weights.ndim - 1))
    dequantized = numpy_helper.from_array(
        weights - zero_points, name=f"{conv_integer_node.name}_weight_dequantized"
    )
    base = conv_integer_node.name or conv_integer_node.output[0]
    cast_x_name = f"{base}_x_cast"
    cast_zp_name = f"{base}_zp_cast"
    sub_name = f"{base}_zp_sub"
    attributes = {
        attr.name: onnx.helper.get_attribute_value(attr) for attr in conv_integer_node.attribute
    }
    nodes = [
        onnx.helper.make_node(
            "Cast", [x_q_name], [cast_x_name], name=cast_x_name, to=onnx.TensorProto.FLOAT
        ),
        onnx.helper.make_node(
            "Cast", [x_zp_name], [cast_zp_name], name=cast_zp_name, to=onnx.TensorProto.FLOAT
        ),
        onnx.helper.make_node("Sub", [cast_x_name, cast_zp_name], [sub_name], name=sub_name),
        onnx.helper.make_node(
            "Conv",
            [sub_name, dequantized.name],
            [cast_node.output[0]],
            name=conv_integer_node.name,
            **attributes,
        ),
    ]
    return nodes, dequantized


def _make_session(path: Path, threads: int) -> ort.InferenceSession:
    """Build a CPU-only inference session with a fixed thread count."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def evaluate_onnx_top1(
    session: ort.InferenceSession, loader: Any, output_name: str = export_onnx.OUTPUT_NAME
) -> float:
    """Top-1 accuracy fraction over an iterable of ``(inputs, labels)`` batches.

    Raises ValueError when the loader produces no samples.
    """
    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0
    for inputs, labels in loader:
        logits = session.run([output_name], {input_name: inputs.numpy()})[0]
        predictions = np.asarray(logits).argmax(axis=1)
        correct += int((predictions == np.asarray(labels.numpy())).sum())
        total += len(labels)
    if total == 0:
        raise ValueError("loader produced no samples; cannot compute ONNX accuracy")
    return correct / total


def _build_split_loader(
    manifest: dict[str, Any], data_root: Path, split: str, batch_size: int
) -> DataLoader:
    """Deterministic eval-transform DataLoader for one manifest split."""
    dataset = SteelDefectDataset(
        data_root, manifest, split, transform=build_eval_transform()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _calibration_batches(
    manifest: dict[str, Any], data_root: Path, size: int, seed: int
) -> list[np.ndarray]:
    """Deterministic sample of ``size`` training images as (1,3,224,224) batches."""
    samples = export_onnx.sample_split_images(manifest, data_root, "train", size, seed)
    return [image.unsqueeze(0).numpy() for image, _ in samples]


def _benchmark_input(manifest: dict[str, Any]) -> np.ndarray:
    """One real preprocessed test image (1,3,224,224) as the latency dummy."""
    dataset = SteelDefectDataset(
        DATA_ROOT, manifest, "test", transform=build_eval_transform()
    )
    image, _ = dataset[0]
    return image.unsqueeze(0).numpy()


def _select_int8_path(
    decision: dict[str, Any],
    static_path: Path,
    static_top1: float | None,
    dynamic_path: Path,
    dynamic_top1: float | None,
) -> Path | None:
    """Pick the INT8 artifact to ship/benchmark from the gate decision.

    For ``"none"`` (every variant rejected) the best attempt is still
    returned so the latency comparison can be reported honestly.
    """
    if decision["kind"] == "static":
        return static_path
    if decision["kind"] == "dynamic":
        return dynamic_path
    candidates = [
        (path, top1)
        for path, top1 in ((static_path, static_top1), (dynamic_path, dynamic_top1))
        if top1 is not None and path.is_file()
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair[1])[0]


# ---------------------------------------------------------------------------
# Metrics persistence
# ---------------------------------------------------------------------------


def _latency_entry(stats: dict[str, float] | None) -> dict[str, float] | None:
    """Round latency stats into the metrics.json ``{p50, p95}`` shape."""
    if stats is None:
        return None
    return {"p50": round(stats["p50_ms"], 3), "p95": round(stats["p95_ms"], 3)}


def _update_metrics_file(model_name: str, result: dict[str, Any]) -> None:
    """Merge the quantization result into artifacts/metrics.json (atomically)."""
    fields = {
        "onnx_fp32_size_mb": result["fp32_size_mb"],
        "onnx_int8_size_mb": result["int8_size_mb"],
        "int8_top1_drop_pt": result["drop_pt"],
        "cpu_latency_ms": {
            "fp32": _latency_entry(result["fp32_latency"]),
            "int8": _latency_entry(result["int8_latency"]),
        },
        "quantization": result["kind"],
        "quantized_at": evaluate._iso_timestamp(time.time()),
    }
    updated = merge_onnx_metrics(
        evaluate._read_metrics_json(METRICS_PATH), model_name, fields
    )
    evaluate._write_metrics_json(METRICS_PATH, updated)
    log.info("updated metrics file: %s", METRICS_PATH)


def _log_summary(model_name: str, result: dict[str, Any]) -> None:
    """Log the quantization gate outcome and the latency comparison."""
    log.info("=================== quantization report (%s) ===================", model_name)
    log.info("fp32 top-1      : %.4f", result["fp32_top1"])
    log.info(
        "static/dynamic  : %s / %s",
        "%.4f" % result["static_top1"] if result["static_top1"] is not None else "n/a",
        "%.4f" % result["dynamic_top1"] if result["dynamic_top1"] is not None else "n/a",
    )
    log.info(
        "gate decision   : %s (int8 top-1 drop %s pt)",
        result["kind"],
        result["drop_pt"] if result["drop_pt"] is not None else "n/a",
    )
    log.info(
        "model sizes     : fp32 %.2f MB, int8 %s MB",
        result["fp32_size_mb"],
        "%.2f" % result["int8_size_mb"] if result["int8_size_mb"] is not None else "n/a",
    )
    log.info(
        "cpu latency p50 : fp32 %.2f ms, int8 %s ms",
        result["fp32_latency"]["p50_ms"],
        "%.2f" % result["int8_latency"]["p50_ms"]
        if result["int8_latency"] is not None
        else "n/a",
    )
    log.info(
        "cpu latency p95 : fp32 %.2f ms, int8 %s ms",
        result["fp32_latency"]["p95_ms"],
        "%.2f" % result["int8_latency"]["p95_ms"]
        if result["int8_latency"] is not None
        else "n/a",
    )
    log.info("speedup (p50)   : %s", result["speedup"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _try_static_quantize(
    model_name: str,
    fp32_path: Path,
    output_dir: Path,
    calibration: list[np.ndarray],
    test_loader: DataLoader,
) -> tuple[Path, float | None]:
    """Static-quantize and evaluate top-1; failures degrade to ``(path, None)``."""
    path = output_dir / f"{model_name}_int8_static.onnx"
    try:
        reader = CalibrationDataReader(export_onnx.INPUT_NAME, calibration)
        quantize_static_to_file(fp32_path, path, reader)
        top1 = evaluate_onnx_top1(_make_session(path, BENCHMARK_THREADS), test_loader)
        log.info("static INT8 top-1: %.4f", top1)
        return path, top1
    except Exception as exc:
        log.warning("static quantization failed (%s); degrading to dynamic", exc)
        return path, None


def _try_dynamic_quantize(
    model_name: str, fp32_path: Path, output_dir: Path, test_loader: DataLoader
) -> tuple[Path, float | None]:
    """Dynamic-quantize and evaluate top-1; failures degrade to ``(path, None)``."""
    path = output_dir / f"{model_name}_int8_dynamic.onnx"
    try:
        quantize_dynamic_to_file(fp32_path, path)
        top1 = evaluate_onnx_top1(_make_session(path, BENCHMARK_THREADS), test_loader)
        log.info("dynamic INT8 top-1: %.4f", top1)
        return path, top1
    except Exception as exc:
        log.warning("dynamic quantization failed: %s", exc)
        return path, None


def _build_quantization_result(
    fp32_path: Path,
    fp32_top1: float,
    static_top1: float | None,
    dynamic_top1: float | None,
    decision: dict[str, Any],
    int8_path: Path | None,
    fp32_latency: dict[str, float],
    int8_latency: dict[str, float] | None,
    speedup: float | None,
) -> dict[str, Any]:
    """Assemble the result dict consumed by the metrics writer and logger."""
    int8_size_mb = (
        round(int8_path.stat().st_size / (1024 ** 2), 4) if int8_path is not None else None
    )
    return {
        "fp32_top1": fp32_top1,
        "static_top1": static_top1,
        "dynamic_top1": dynamic_top1,
        "kind": decision["kind"],
        "drop_pt": decision["drop_pt"],
        "fp32_size_mb": round(fp32_path.stat().st_size / (1024 ** 2), 4),
        "int8_size_mb": int8_size_mb,
        "fp32_latency": fp32_latency,
        "int8_latency": int8_latency,
        "speedup": speedup,
        "int8_path": str(int8_path) if int8_path is not None else None,
    }


def _benchmark_latency_pair(
    fp32_session: ort.InferenceSession, int8_path: Path | None, dummy_input: np.ndarray
) -> tuple[dict[str, float], dict[str, float] | None, float | None]:
    """Benchmark FP32 and INT8 sessions; returns (fp32, int8, speedup)."""
    fp32_latency = run_latency_benchmark(
        fp32_session, dummy_input, BENCHMARK_ITERATIONS, BENCHMARK_WARMUP
    )
    if int8_path is None:
        return fp32_latency, None, None
    int8_session = _make_session(int8_path, BENCHMARK_THREADS)
    int8_latency = run_latency_benchmark(
        int8_session, dummy_input, BENCHMARK_ITERATIONS, BENCHMARK_WARMUP
    )
    speedup = round(fp32_latency["p50_ms"] / int8_latency["p50_ms"], 3)
    return fp32_latency, int8_latency, speedup


def _evaluate_fp32_baseline(
    manifest: dict[str, Any], fp32_path: Path, calibration_size: int
) -> tuple[DataLoader, ort.InferenceSession, float, list[np.ndarray]]:
    """Build the test loader, evaluate FP32 top-1, and sample the calibration set."""
    test_loader = _build_split_loader(manifest, DATA_ROOT, "test", EVAL_BATCH_SIZE)
    fp32_session = _make_session(fp32_path, BENCHMARK_THREADS)
    fp32_top1 = evaluate_onnx_top1(fp32_session, test_loader)
    log.info("fp32 top-1 on %d test images: %.4f", len(test_loader.dataset), fp32_top1)
    calibration = _calibration_batches(manifest, DATA_ROOT, calibration_size, CALIBRATION_SEED)
    log.info("static-quantization calibration set: %d images", len(calibration))
    return test_loader, fp32_session, fp32_top1, calibration


def _run_accuracy_gate(
    model_name: str,
    fp32_path: Path,
    output_dir: Path,
    fp32_top1: float,
    calibration: list[np.ndarray],
    test_loader: DataLoader,
) -> tuple[dict[str, Any], Path, Path, float | None, float | None]:
    """Static first, dynamic fallback; returns (decision, static, dynamic...).

    The tuple is ``(decision, static_path, dynamic_path, static_top1,
    dynamic_top1)``; a failed variant yields ``top1=None`` and the gate
    automatically degrades to the next option (or ``"none"``).
    """
    static_path, static_top1 = _try_static_quantize(
        model_name, fp32_path, output_dir, calibration, test_loader
    )
    decision = decide_quantization(fp32_top1, static_top1, None, GATE_DROP_LIMIT_PT)
    dynamic_path = output_dir / f"{model_name}_int8_dynamic.onnx"
    dynamic_top1 = None
    if decision["kind"] == "none":
        dynamic_path, dynamic_top1 = _try_dynamic_quantize(
            model_name, fp32_path, output_dir, test_loader
        )
        decision = decide_quantization(
            fp32_top1, static_top1, dynamic_top1, GATE_DROP_LIMIT_PT
        )
    log.info(
        "accuracy gate: kind=%s drop=%s pt (limit %.1f pt)",
        decision["kind"],
        decision["drop_pt"],
        GATE_DROP_LIMIT_PT,
    )
    return decision, static_path, dynamic_path, static_top1, dynamic_top1


def _quantize_and_gate(
    model_name: str,
    fp32_path: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    calibration_size: int,
) -> dict[str, Any]:
    """Quantize, evaluate, gate, and benchmark; returns the full result dict."""
    test_loader, fp32_session, fp32_top1, calibration = _evaluate_fp32_baseline(
        manifest, fp32_path, calibration_size
    )
    decision, static_path, dynamic_path, static_top1, dynamic_top1 = _run_accuracy_gate(
        model_name, fp32_path, output_dir, fp32_top1, calibration, test_loader
    )
    int8_path = _select_int8_path(
        decision, static_path, static_top1, dynamic_path, dynamic_top1
    )
    dummy_input = _benchmark_input(manifest)
    log.info(
        "CPU latency benchmark: %d iterations after %d warmups (threads=%d)",
        BENCHMARK_ITERATIONS,
        BENCHMARK_WARMUP,
        BENCHMARK_THREADS,
    )
    fp32_latency, int8_latency, speedup = _benchmark_latency_pair(
        fp32_session, int8_path, dummy_input
    )
    return _build_quantization_result(
        fp32_path,
        fp32_top1,
        static_top1,
        dynamic_top1,
        decision,
        int8_path,
        fp32_latency,
        int8_latency,
        speedup,
    )


def _run(args: argparse.Namespace) -> int:
    """Execute quantization for parsed arguments (happy path only).

    Any failure raises; :func:`main` turns exceptions into a non-zero exit
    code with a clear error message.
    """
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"training manifest is missing: {MANIFEST_PATH} (run the T2.1 manifest builder)"
        )
    manifest = load_manifest(MANIFEST_PATH)
    fp32_path = args.onnx_path.resolve()
    if not fp32_path.is_file():
        raise FileNotFoundError(
            f"FP32 ONNX model not found: {fp32_path} (run scripts/export_onnx.py first)"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = _quantize_and_gate(
        args.model, fp32_path, args.output_dir, manifest, args.calibration_size
    )
    _update_metrics_file(args.model, result)
    _log_summary(args.model, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
