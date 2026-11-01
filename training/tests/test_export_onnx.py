"""Tests for the ONNX export script (T3.1).

Fast unit tests export a tiny random-weight CNN through the production
export path and verify model structure (opset 17, static batch-1 input,
``input``/``logits`` names), static-shape enforcement by onnxruntime, and
the parity metric helpers. The numerical parity check against the real
trained checkpoints (64 fixed-seed test images, max abs diff < 1e-3,
identical top-1) is marked slow and runs on CPU so CI can cover it too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from scripts import export_onnx
from steel_training.data.manifest import load_manifest
from steel_training.models.factory import create_resnet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_args_derives_default_paths() -> None:
    args = export_onnx.parse_args(["--model", "resnet50"])
    assert args.model == "resnet50"
    assert args.checkpoint == export_onnx.CHECKPOINT_DIR / "resnet50_best.pth"
    assert args.output_dir == export_onnx.ONNX_DIR


def test_parse_args_custom_paths(tmp_path: Path) -> None:
    checkpoint = tmp_path / "custom.pth"
    output_dir = tmp_path / "out"
    args = export_onnx.parse_args(
        [
            "--model",
            "resnet18",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert args.checkpoint == checkpoint
    assert args.output_dir == output_dir


def test_parse_args_rejects_unknown_model() -> None:
    with pytest.raises(SystemExit):
        export_onnx.parse_args(["--model", "resnet101"])


# ---------------------------------------------------------------------------
# sample_indices
# ---------------------------------------------------------------------------


def test_sample_indices_deterministic_within_range() -> None:
    first = export_onnx.sample_indices(total=180, count=64, seed=42)
    second = export_onnx.sample_indices(total=180, count=64, seed=42)
    assert first == second
    assert len(first) == 64
    assert len(set(first)) == 64
    assert min(first) >= 0
    assert max(first) < 180


def test_sample_indices_rejects_count_larger_than_total() -> None:
    with pytest.raises(ValueError):
        export_onnx.sample_indices(total=10, count=11, seed=42)


# ---------------------------------------------------------------------------
# compute_parity_metrics / parity_passed
# ---------------------------------------------------------------------------


def test_compute_parity_metrics_identical_logits() -> None:
    logits = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    result = export_onnx.compute_parity_metrics(logits, logits.copy())
    assert result["max_abs_diff"] == 0.0
    assert result["top1_matched"] == 2
    assert result["total"] == 2


def test_compute_parity_metrics_known_difference() -> None:
    torch_logits = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    ort_logits = np.array([[0.1, 0.9], [0.2, 0.3]], dtype=np.float32)
    result = export_onnx.compute_parity_metrics(torch_logits, ort_logits)
    assert result["max_abs_diff"] == pytest.approx(0.8)
    assert result["top1_matched"] == 1
    assert result["total"] == 2


def test_compute_parity_metrics_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        export_onnx.compute_parity_metrics(np.zeros((2, 2)), np.zeros((3, 2)))


def test_parity_passed_true_for_good_metrics() -> None:
    metrics = {"max_abs_diff": 1e-5, "top1_matched": 64, "total": 64}
    assert export_onnx.parity_passed(metrics)


def test_parity_passed_false_for_high_diff() -> None:
    metrics = {"max_abs_diff": 0.01, "top1_matched": 64, "total": 64}
    assert not export_onnx.parity_passed(metrics)


def test_parity_passed_false_for_top1_mismatch() -> None:
    metrics = {"max_abs_diff": 1e-5, "top1_matched": 63, "total": 64}
    assert not export_onnx.parity_passed(metrics)


# ---------------------------------------------------------------------------
# export_to_onnx (tiny model)
# ---------------------------------------------------------------------------


def test_export_to_onnx_structure_and_checker(tmp_path: Path, tiny_cnn_model) -> None:
    out_path = tmp_path / "tiny_fp32.onnx"
    result = export_onnx.export_to_onnx(tiny_cnn_model, out_path)
    assert result == out_path
    assert out_path.is_file()
    model_proto = onnx.load(str(out_path))
    onnx.checker.check_model(model_proto)
    input_proto = next(item for item in model_proto.graph.input if item.name == "input")
    dims = [dim.dim_value for dim in input_proto.type.tensor_type.shape.dim]
    assert dims == [1, 3, 224, 224]
    assert "logits" in {item.name for item in model_proto.graph.output}
    opsets = {opset.domain: opset.version for opset in model_proto.opset_import}
    assert opsets[""] == 17


def test_exported_model_rejects_batch_size_4(tiny_fp32_onnx: Path) -> None:
    session = ort.InferenceSession(str(tiny_fp32_onnx), providers=["CPUExecutionProvider"])
    batch4 = np.zeros((4, 3, 224, 224), dtype=np.float32)
    with pytest.raises(Exception, match="dimension"):
        session.run(["logits"], {"input": batch4})


def test_exported_model_accepts_batch_size_1(tiny_fp32_onnx: Path) -> None:
    session = ort.InferenceSession(str(tiny_fp32_onnx), providers=["CPUExecutionProvider"])
    outputs = session.run(
        ["logits"], {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}
    )
    assert outputs[0].shape == (1, 6)


# ---------------------------------------------------------------------------
# _load_export_model
# ---------------------------------------------------------------------------


def test_load_export_model_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_onnx._load_export_model("resnet18", tmp_path / "nope.pth", num_classes=6)


def test_load_export_model_wrong_checkpoint_format(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.pth"
    torch.save({"epoch": 1}, bad_path)
    with pytest.raises(ValueError, match="model_state_dict"):
        export_onnx._load_export_model("resnet18", bad_path, num_classes=6)


def test_load_export_model_state_dict_mismatch(tmp_path: Path) -> None:
    wrong_path = tmp_path / "wrong.pth"
    torch.save({"model_state_dict": {"bogus.weight": torch.zeros(1)}}, wrong_path)
    with pytest.raises(ValueError, match="does not match"):
        export_onnx._load_export_model("resnet18", wrong_path, num_classes=6)


def test_load_export_model_success_sets_eval_mode(tmp_path: Path) -> None:
    source = create_resnet("resnet18", num_classes=6)
    source_state = source.state_dict()
    checkpoint_path = tmp_path / "ok.pth"
    torch.save({"model_state_dict": source_state}, checkpoint_path)
    model, checkpoint = export_onnx._load_export_model(
        "resnet18", checkpoint_path, num_classes=6
    )
    assert not model.training
    assert set(checkpoint["model_state_dict"]) == set(source_state)
    assert torch.equal(checkpoint["model_state_dict"]["fc.weight"], source_state["fc.weight"])


# ---------------------------------------------------------------------------
# _run smoke
# ---------------------------------------------------------------------------


def test_run_smoke(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        export_onnx,
        "load_manifest",
        lambda path: {
            "class_names": ["a", "b", "c", "d", "e", "f"],
            "split_counts": {"train": 0, "val": 0, "test": 0},
            "entries": [],
        },
    )
    fake_model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        export_onnx, "_load_export_model", lambda *a, **k: (fake_model, {})
    )
    out_path = tmp_path / "fake.onnx"
    monkeypatch.setattr(export_onnx, "export_to_onnx", lambda *a, **k: out_path)
    monkeypatch.setattr(export_onnx, "sample_split_images", lambda *a, **k: [])
    monkeypatch.setattr(
        export_onnx,
        "run_parity_check",
        lambda *a, **k: {"max_abs_diff": 0.0, "top1_matched": 0, "total": 0},
    )
    monkeypatch.setattr(export_onnx.ort, "InferenceSession", lambda *a, **k: None)
    args = export_onnx.parse_args(
        ["--model", "resnet18", "--output-dir", str(tmp_path / "onnx")]
    )
    assert export_onnx._run(args) == 0


# ---------------------------------------------------------------------------
# Numerical parity against the real checkpoints (slow, CPU)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["resnet18", "resnet50"])
def test_parity_real_checkpoint(
    repo_root: Path,
    manifest_path: Path,
    image_root: Path,
    model_name: str,
    tmp_path: Path,
) -> None:
    checkpoint = repo_root / "artifacts" / "checkpoints" / f"{model_name}_best.pth"
    if not checkpoint.is_file():
        pytest.skip(f"checkpoint not present: {checkpoint}")
    manifest = load_manifest(manifest_path)
    model, _ = export_onnx._load_export_model(
        model_name, checkpoint, len(manifest["class_names"])
    )
    out_path = tmp_path / f"{model_name}_fp32.onnx"
    export_onnx.export_to_onnx(model, out_path)
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    samples = export_onnx.sample_split_images(
        manifest, image_root, "test", export_onnx.PARITY_IMAGE_COUNT, export_onnx.PARITY_SEED
    )
    metrics = export_onnx.run_parity_check(model, session, samples)
    assert metrics["total"] == export_onnx.PARITY_IMAGE_COUNT
    assert metrics["max_abs_diff"] < export_onnx.PARITY_MAX_ABS_DIFF
    assert metrics["top1_matched"] == metrics["total"]
