"""Tests for the INT8 quantization script (T3.2).

The accuracy-gate decision function is a pure function and is covered for
every branch (accept / degrade to dynamic / reject) including the 1.5 pt
boundary and the drop calculation. ``CalibrationDataReader`` and the
quantize wrappers are smoke-tested against the tiny ONNX model exported by
the production export path; latency percentiles are verified against a
known numeric sequence and a call-counting mock session.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from scripts import quantize


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_args_derives_default_onnx_path() -> None:
    args = quantize.parse_args(["--model", "resnet18"])
    assert args.onnx_path == quantize.ONNX_DIR / "resnet18_fp32.onnx"
    assert args.calibration_size == 200
    assert args.output_dir == quantize.ONNX_DIR


def test_parse_args_custom_values(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model_fp32.onnx"
    args = quantize.parse_args(
        [
            "--model",
            "resnet50",
            "--onnx-path",
            str(onnx_path),
            "--calibration-size",
            "32",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert args.onnx_path == onnx_path
    assert args.calibration_size == 32
    assert args.output_dir == tmp_path / "out"


def test_parse_args_rejects_unknown_model() -> None:
    with pytest.raises(SystemExit):
        quantize.parse_args(["--model", "resnet101"])


# ---------------------------------------------------------------------------
# top1_drop_pt
# ---------------------------------------------------------------------------


def test_top1_drop_pt_positive_drop() -> None:
    assert quantize.top1_drop_pt(0.9944, 0.9889) == pytest.approx(0.55)


def test_top1_drop_pt_quantized_better_is_negative() -> None:
    assert quantize.top1_drop_pt(0.9, 0.91) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# decide_quantization (accuracy gate)
# ---------------------------------------------------------------------------


def test_decide_accepts_static_within_limit() -> None:
    decision = quantize.decide_quantization(fp32_top1=0.99, static_top1=0.985)
    assert decision["kind"] == "static"
    assert decision["drop_pt"] == pytest.approx(0.5)


def test_decide_accepts_static_exactly_at_boundary() -> None:
    decision = quantize.decide_quantization(fp32_top1=0.99, static_top1=0.975)
    assert decision["kind"] == "static"
    assert decision["drop_pt"] == pytest.approx(1.5)


def test_decide_degrades_to_dynamic_just_above_boundary() -> None:
    decision = quantize.decide_quantization(
        fp32_top1=1.0, static_top1=0.9849, dynamic_top1=0.99
    )
    assert decision["kind"] == "dynamic"
    assert decision["drop_pt"] == pytest.approx(1.0)


def test_decide_accepts_dynamic_exactly_at_boundary() -> None:
    decision = quantize.decide_quantization(
        fp32_top1=0.99, static_top1=0.97, dynamic_top1=0.975
    )
    assert decision["kind"] == "dynamic"
    assert decision["drop_pt"] == pytest.approx(1.5)


def test_decide_rejects_both_variants() -> None:
    decision = quantize.decide_quantization(
        fp32_top1=0.99, static_top1=0.97, dynamic_top1=0.965
    )
    assert decision["kind"] == "none"
    assert decision["drop_pt"] == pytest.approx(2.0)  # best attempt (static)


def test_decide_without_static_result_uses_dynamic() -> None:
    decision = quantize.decide_quantization(
        fp32_top1=0.99, static_top1=None, dynamic_top1=0.985
    )
    assert decision["kind"] == "dynamic"


def test_decide_none_when_no_variant_succeeded() -> None:
    decision = quantize.decide_quantization(
        fp32_top1=0.99, static_top1=None, dynamic_top1=None
    )
    assert decision["kind"] == "none"
    assert decision["drop_pt"] is None


def test_decide_honors_custom_drop_limit() -> None:
    decision = quantize.decide_quantization(
        fp32_top1=0.99, static_top1=0.972, drop_limit_pt=2.0
    )
    assert decision["kind"] == "static"
    assert decision["drop_pt"] == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# CalibrationDataReader
# ---------------------------------------------------------------------------


def test_calibration_data_reader_feeds_batches_then_none() -> None:
    batches = [
        np.zeros((1, 3, 224, 224), dtype=np.float32),
        np.ones((1, 3, 224, 224), dtype=np.float32),
    ]
    reader = quantize.CalibrationDataReader("input", batches)
    first = reader.get_next()
    second = reader.get_next()
    assert first is not None and first["input"] is batches[0]
    assert second is not None and second["input"] is batches[1]
    assert reader.get_next() is None
    assert reader.get_next() is None


def test_calibration_data_reader_casts_to_float32() -> None:
    reader = quantize.CalibrationDataReader(
        "input", [np.zeros((1, 3, 224, 224), dtype=np.float64)]
    )
    batch = reader.get_next()
    assert batch is not None
    assert batch["input"].dtype == np.float32


# ---------------------------------------------------------------------------
# quantize wrappers (tiny ONNX model)
# ---------------------------------------------------------------------------


def test_quantize_static_smoke_tiny(tiny_fp32_onnx: Path, tmp_path: Path) -> None:
    batches = [np.zeros((1, 3, 224, 224), dtype=np.float32) for _ in range(4)]
    reader = quantize.CalibrationDataReader("input", batches)
    out_path = tmp_path / "tiny_int8_static.onnx"
    result = quantize.quantize_static_to_file(tiny_fp32_onnx, out_path, reader)
    assert result == out_path
    assert out_path.is_file()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    outputs = session.run(
        ["logits"], {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}
    )
    assert outputs[0].shape == (1, 6)


def test_quantize_dynamic_smoke_tiny(tiny_fp32_onnx: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "tiny_int8_dynamic.onnx"
    result = quantize.quantize_dynamic_to_file(tiny_fp32_onnx, out_path)
    assert result == out_path
    assert out_path.is_file()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    outputs = session.run(
        ["logits"], {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}
    )
    assert outputs[0].shape == (1, 6)


def test_fix_conv_integer_rewrites_and_runs(tiny_fp32_onnx: Path, tmp_path: Path) -> None:
    raw_path = tmp_path / "tiny_raw_dynamic.onnx"
    quantize.quantize_dynamic(str(tiny_fp32_onnx), str(raw_path))
    raw_model = onnx.load(str(raw_path))
    raw_nodes = list(raw_model.graph.node)
    conv_integers = [n for n in raw_nodes if n.op_type == "ConvInteger"]
    assert conv_integers and all(len(n.input) == 4 for n in conv_integers)
    matmul_integers_before = [n for n in raw_nodes if n.op_type == "MatMulInteger"]
    w_q = onnx.numpy_helper.to_array(
        next(i for i in raw_model.graph.initializer if i.name == conv_integers[0].input[1])
    )
    w_zp = onnx.numpy_helper.to_array(
        next(i for i in raw_model.graph.initializer if i.name == conv_integers[0].input[3])
    )

    result = quantize.fix_conv_integer_zero_points(raw_path)
    assert result == raw_path
    fixed_model = onnx.load(str(raw_path))
    fixed_nodes = list(fixed_model.graph.node)
    assert not [n for n in fixed_nodes if n.op_type == "ConvInteger"]
    assert [n.op_type for n in fixed_nodes].count("Conv") == len(conv_integers)
    assert len([n for n in fixed_nodes if n.op_type == "MatMulInteger"]) == len(
        matmul_integers_before
    )
    dequantized = onnx.numpy_helper.to_array(
        next(
            i
            for i in fixed_model.graph.initializer
            if i.name == f"{conv_integers[0].name}_weight_dequantized"
        )
    )
    assert np.allclose(dequantized, w_q.astype(np.float32) - w_zp.astype(np.float32))
    session = ort.InferenceSession(str(raw_path), providers=["CPUExecutionProvider"])
    outputs = session.run(
        ["logits"], {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}
    )
    assert outputs[0].shape == (1, 6)


def test_fix_conv_integer_idempotent(tiny_fp32_onnx: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "tiny_dynamic.onnx"
    quantize.quantize_dynamic_to_file(tiny_fp32_onnx, out_path)
    node_count = len(onnx.load(str(out_path)).graph.node)
    assert quantize.fix_conv_integer_zero_points(out_path) == out_path
    assert len(onnx.load(str(out_path)).graph.node) == node_count


# ---------------------------------------------------------------------------
# evaluate_onnx_top1
# ---------------------------------------------------------------------------


class _FakeInputInfo:
    name = "input"


class _FakeSession:
    def __init__(self, logits: np.ndarray) -> None:
        self._logits = logits

    def get_inputs(self) -> list[_FakeInputInfo]:
        return [_FakeInputInfo()]

    def run(self, output_names, input_feed) -> list[np.ndarray]:
        return [self._logits]


def _dummy_loader(labels: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    inputs = torch.zeros((len(labels), 3, 224, 224), dtype=torch.float32)
    return [(inputs, torch.tensor(labels))]


def test_evaluate_onnx_top1_all_correct() -> None:
    logits = np.zeros((4, 6), dtype=np.float32)
    for row, label in enumerate((0, 1, 0, 1)):
        logits[row, label] = 1.0
    result = quantize.evaluate_onnx_top1(_FakeSession(logits), _dummy_loader([0, 1, 0, 1]))
    assert result == pytest.approx(1.0)


def test_evaluate_onnx_top1_all_wrong() -> None:
    logits = np.zeros((4, 6), dtype=np.float32)
    for row, label in enumerate((1, 0, 1, 0)):
        logits[row, label] = 1.0
    result = quantize.evaluate_onnx_top1(_FakeSession(logits), _dummy_loader([0, 1, 0, 1]))
    assert result == pytest.approx(0.0)


def test_evaluate_onnx_top1_empty_loader_raises() -> None:
    session = _FakeSession(np.zeros((1, 6), dtype=np.float32))
    with pytest.raises(ValueError):
        quantize.evaluate_onnx_top1(session, [])


def test_evaluate_onnx_top1_real_loader_batch1(tiny_fp32_onnx: Path, tmp_path: Path) -> None:
    # Regression test for the static batch-1 constraint: the loader built by
    # _build_split_loader must feed one image per forward.
    from PIL import Image

    data_root = tmp_path / "data"
    for class_name in ("a", "b"):
        class_dir = data_root / class_name
        class_dir.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (32, 32), color=(10 * index, 20 * index, 30 * index)).save(
                class_dir / f"img_{index}.jpg"
            )
    manifest = {
        "class_names": ["a", "b"],
        "split_counts": {"train": 4, "val": 0, "test": 0},
        "entries": [
            {"path": "a/img_0.jpg", "label": "a", "split": "train"},
            {"path": "a/img_1.jpg", "label": "a", "split": "train"},
            {"path": "b/img_0.jpg", "label": "b", "split": "train"},
            {"path": "b/img_1.jpg", "label": "b", "split": "train"},
        ],
    }
    loader = quantize._build_split_loader(manifest, data_root, "train", batch_size=32)
    session = ort.InferenceSession(str(tiny_fp32_onnx), providers=["CPUExecutionProvider"])
    assert len(loader) == 1  # one batch holding all 4 samples
    # Feeding batch 4 to the static batch-1 model raises; the gate path uses
    # EVAL_BATCH_SIZE == 1, so evaluate through the production loader shape.
    with pytest.raises(Exception, match="dimension"):
        for inputs, labels in loader:
            session.run(["logits"], {"input": inputs.numpy()})
    single_loader = quantize._build_split_loader(manifest, data_root, "train", batch_size=1)
    top1 = quantize.evaluate_onnx_top1(session, single_loader)
    assert 0.0 <= top1 <= 1.0


# ---------------------------------------------------------------------------
# latency benchmark
# ---------------------------------------------------------------------------


def test_compute_latency_stats_known_sequence() -> None:
    # numpy linear interpolation: rank = (n - 1) * p / 100.
    timings_s = [i / 1000.0 for i in range(1, 101)]  # 1 ms .. 100 ms
    stats = quantize.compute_latency_stats(timings_s)
    assert stats["p50_ms"] == pytest.approx(50.5)  # rank 49.5 -> between 50 and 51
    assert stats["p95_ms"] == pytest.approx(95.05)  # rank 94.05 -> 95 + 0.05


def test_compute_latency_stats_empty_raises() -> None:
    with pytest.raises(ValueError):
        quantize.compute_latency_stats([])


def test_run_latency_benchmark_mock_session_counts_runs() -> None:
    class _CountingSession:
        def __init__(self) -> None:
            self.calls = 0

        def get_inputs(self) -> list[_FakeInputInfo]:
            return [_FakeInputInfo()]

        def run(self, output_names, input_feed) -> list[np.ndarray]:
            self.calls += 1
            return []

    session = _CountingSession()
    stats = quantize.run_latency_benchmark(
        session, np.zeros((1, 3, 224, 224), dtype=np.float32), iterations=5, warmup=2
    )
    assert session.calls == 7
    assert set(stats) == {"p50_ms", "p95_ms"}


def test_run_latency_benchmark_tiny_session(tiny_fp32_onnx: Path) -> None:
    session = ort.InferenceSession(str(tiny_fp32_onnx), providers=["CPUExecutionProvider"])
    stats = quantize.run_latency_benchmark(
        session, np.zeros((1, 3, 224, 224), dtype=np.float32), iterations=3, warmup=1
    )
    assert stats["p50_ms"] >= 0.0
    assert stats["p95_ms"] >= stats["p50_ms"]


# ---------------------------------------------------------------------------
# merge_onnx_metrics
# ---------------------------------------------------------------------------


def test_merge_onnx_metrics_adds_fields_and_preserves_everything() -> None:
    existing = {
        "models": {
            "resnet18": {"test_acc": 0.9944},
            "resnet50": {"test_acc": 0.9889},
        },
        "generated_at": "yesterday",
    }
    fields = {"onnx_fp32_size_mb": 44.7, "quantization": "static"}
    merged = quantize.merge_onnx_metrics(existing, "resnet18", fields)
    assert merged["models"]["resnet18"]["test_acc"] == 0.9944
    assert merged["models"]["resnet18"]["onnx_fp32_size_mb"] == 44.7
    assert merged["models"]["resnet18"]["quantization"] == "static"
    assert merged["models"]["resnet50"] == {"test_acc": 0.9889}
    assert merged["generated_at"] == "yesterday"
    assert existing["models"]["resnet18"] == {"test_acc": 0.9944}  # input not mutated


def test_merge_onnx_metrics_without_existing_document() -> None:
    merged = quantize.merge_onnx_metrics(None, "resnet18", {"quantization": "dynamic"})
    assert merged == {"models": {"resnet18": {"quantization": "dynamic"}}}


# ---------------------------------------------------------------------------
# _run smoke (full metrics write path)
# ---------------------------------------------------------------------------


def test_run_smoke_writes_metrics(monkeypatch, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"models": {"resnet18": {"test_acc": 0.9944}}}), encoding="utf-8"
    )
    monkeypatch.setattr(quantize, "METRICS_PATH", metrics_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")  # _run checks existence first
    monkeypatch.setattr(quantize, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(
        quantize,
        "load_manifest",
        lambda path: {
            "class_names": ["a", "b", "c", "d", "e", "f"],
            "split_counts": {"train": 0, "val": 0, "test": 0},
            "entries": [],
        },
    )
    monkeypatch.setattr(
        quantize,
        "_quantize_and_gate",
        lambda *a, **k: {
            "fp32_top1": 0.99,
            "static_top1": 0.985,
            "dynamic_top1": None,
            "kind": "static",
            "drop_pt": 0.5,
            "fp32_size_mb": 44.7,
            "int8_size_mb": 12.3,
            "fp32_latency": {"p50_ms": 10.0, "p95_ms": 12.0},
            "int8_latency": {"p50_ms": 5.0, "p95_ms": 6.0},
            "speedup": 2.0,
            "int8_path": None,
        },
    )
    fp32_path = tmp_path / "model_fp32.onnx"
    fp32_path.write_bytes(b"fake onnx")
    args = quantize.parse_args(
        [
            "--model",
            "resnet18",
            "--onnx-path",
            str(fp32_path),
            "--output-dir",
            str(tmp_path / "onnx"),
        ]
    )
    assert quantize._run(args) == 0
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    entry = data["models"]["resnet18"]
    assert entry["test_acc"] == 0.9944  # pre-existing field preserved
    assert entry["onnx_fp32_size_mb"] == 44.7
    assert entry["onnx_int8_size_mb"] == 12.3
    assert entry["int8_top1_drop_pt"] == 0.5
    assert entry["quantization"] == "static"
    assert entry["cpu_latency_ms"] == {
        "fp32": {"p50": 10.0, "p95": 12.0},
        "int8": {"p50": 5.0, "p95": 6.0},
    }
    assert isinstance(entry["quantized_at"], str) and entry["quantized_at"]
