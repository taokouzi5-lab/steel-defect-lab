"""CLI entry point for exporting a trained checkpoint to ONNX (T3.1).

Builds the model via :func:`steel_training.models.factory.create_resnet`
(``weights=None``), strictly loads a trainer-format checkpoint written by
``scripts/train.py``, switches the model to eval mode, and exports it with
``torch.onnx.export`` using opset 17 and a static batch-1 input
(``input`` -> ``logits``). The produced ``<model>_fp32.onnx`` file is
validated with ``onnx.checker.check_model``, then a numerical parity check
runs on a fixed-seed sample of 64 test images: PyTorch logits vs
onnxruntime logits must agree within 1e-3 max abs diff with identical
top-1 predictions, otherwise the script exits non-zero.

Usage:
    python training/scripts/export_onnx.py --model resnet18
    python training/scripts/export_onnx.py --model resnet50 --output-dir out
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling steel_training package importable when this script is run
# directly (e.g. `python training/scripts/export_onnx.py`).
TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import argparse
import logging
import random
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from steel_training.data.augmentation import build_eval_transform
from steel_training.data.dataset import SteelDefectDataset
from steel_training.data.manifest import load_manifest
from steel_training.models.factory import create_resnet

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "training" / "manifest.json"
DATA_ROOT = REPO_ROOT / "training" / "data"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
ONNX_DIR = REPO_ROOT / "artifacts" / "onnx"

EXPORT_OPSET = 17
INPUT_NAME = "input"
OUTPUT_NAME = "logits"
INPUT_SHAPE = (1, 3, 224, 224)
PARITY_IMAGE_COUNT = 64
PARITY_SEED = 42
PARITY_MAX_ABS_DIFF = 1e-3

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments; derive checkpoint/output defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet50"),
        required=True,
        help="model architecture of the checkpoint to export",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "checkpoint to export "
            "(default: artifacts/checkpoints/<model>_best.pth under the repo root)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ONNX_DIR,
        help="directory for the exported <model>_fp32.onnx (default: artifacts/onnx)",
    )
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = CHECKPOINT_DIR / f"{args.model}_best.pth"
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
        log.error("ONNX export failed: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def sample_indices(total: int, count: int, seed: int) -> list[int]:
    """Deterministic sorted sample of ``count`` indices from ``range(total)``.

    Raises ValueError when ``count`` exceeds ``total`` (or is negative).
    """
    if count < 0 or count > total:
        raise ValueError(f"cannot sample {count} indices from a population of {total}")
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), count))


def compute_parity_metrics(
    torch_logits: np.ndarray, ort_logits: np.ndarray
) -> dict[str, Any]:
    """Compare PyTorch and onnxruntime logits; returns parity metrics.

    ``max_abs_diff`` is computed in float64 for stability; ``top1_matched``
    counts rows whose argmax agrees. Raises ValueError on a shape mismatch.
    """
    torch_logits = np.asarray(torch_logits)
    ort_logits = np.asarray(ort_logits)
    if torch_logits.shape != ort_logits.shape:
        raise ValueError(
            f"logits shape mismatch: torch {torch_logits.shape} vs ort {ort_logits.shape}"
        )
    if torch_logits.size == 0:
        raise ValueError("cannot compute parity metrics on empty logits")
    diff = np.abs(torch_logits.astype(np.float64) - ort_logits.astype(np.float64))
    torch_top1 = torch_logits.argmax(axis=1)
    ort_top1 = ort_logits.argmax(axis=1)
    return {
        "max_abs_diff": float(diff.max()),
        "top1_matched": int((torch_top1 == ort_top1).sum()),
        "total": int(len(torch_top1)),
    }


def parity_passed(metrics: dict[str, Any]) -> bool:
    """True when parity metrics meet the export acceptance thresholds."""
    return (
        metrics["max_abs_diff"] < PARITY_MAX_ABS_DIFF
        and metrics["top1_matched"] == metrics["total"]
    )


# ---------------------------------------------------------------------------
# Model / export plumbing
# ---------------------------------------------------------------------------


def _load_export_model(
    model_name: str, checkpoint_path: Path, num_classes: int
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the model, strictly load a trainer checkpoint, set eval mode.

    Raises FileNotFoundError/ValueError with clear messages on failure; a
    state-dict mismatch (``load_state_dict`` RuntimeError) is wrapped into
    a ValueError naming the checkpoint path.
    """
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"checkpoint is not a steel-defect-lab training checkpoint "
            f"(missing 'model_state_dict'): {checkpoint_path}"
        )
    model = create_resnet(model_name, num_classes=num_classes)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"model_state_dict in {checkpoint_path} does not match the "
            f"{model_name} architecture: {exc}"
        ) from exc
    model.eval()
    return model, checkpoint


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple[int, int, int, int] = INPUT_SHAPE,
) -> Path:
    """Export an eval-mode model to FP32 ONNX and validate the result.

    Uses opset 17, a static batch-1 ``input`` and an ``output`` named
    ``logits`` (no dynamic axes). The written file is checked with
    ``onnx.checker.check_model`` before being returned.
    """
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=EXPORT_OPSET,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=None,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    return output_path


def sample_split_images(
    manifest: dict[str, Any],
    data_root: Path,
    split: str,
    count: int,
    seed: int,
) -> list[tuple[torch.Tensor, int]]:
    """Deterministic sample of preprocessed ``(image, label)`` pairs."""
    dataset = SteelDefectDataset(
        data_root, manifest, split, transform=build_eval_transform()
    )
    indices = sample_indices(len(dataset), count, seed)
    return [dataset[index] for index in indices]


def run_parity_check(
    model: nn.Module,
    session: ort.InferenceSession,
    samples: list[tuple[torch.Tensor, int]],
) -> dict[str, Any]:
    """Compare PyTorch and onnxruntime logits over ``samples`` (CPU).

    Every sample is fed one at a time so both engines see the exported
    static batch-1 input shape. The model must already be in eval mode;
    inference runs under ``torch.inference_mode``. Raises ValueError when
    ``samples`` is empty.
    """
    if not samples:
        raise ValueError("no samples provided for parity check")
    model.eval()
    input_name = session.get_inputs()[0].name
    torch_parts: list[np.ndarray] = []
    ort_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for image, _ in samples:
            inputs = image.unsqueeze(0)
            torch_parts.append(model(inputs).numpy())
            ort_parts.append(session.run([OUTPUT_NAME], {input_name: inputs.numpy()})[0])
    return compute_parity_metrics(
        np.concatenate(torch_parts, axis=0), np.concatenate(ort_parts, axis=0)
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _log_parity(metrics: dict[str, Any]) -> None:
    """Log the parity comparison results."""
    log.info(
        "parity over %d samples: max abs diff %.6f, top-1 matched %d/%d",
        metrics["total"],
        metrics["max_abs_diff"],
        metrics["top1_matched"],
        metrics["total"],
    )


def _run(args: argparse.Namespace) -> int:
    """Execute the export for parsed arguments (happy path only).

    Any failure raises; :func:`main` turns exceptions into a non-zero exit
    code with a clear error message. Parity failure raises RuntimeError.
    """
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"training manifest is missing: {MANIFEST_PATH} (run the T2.1 manifest builder)"
        )
    manifest = load_manifest(MANIFEST_PATH)
    num_classes = len(manifest["class_names"])
    model, checkpoint = _load_export_model(
        args.model, args.checkpoint, num_classes=num_classes
    )
    log.info("loaded checkpoint %s (epoch %s)", args.checkpoint, checkpoint.get("epoch"))

    output_path = export_to_onnx(model, args.output_dir / f"{args.model}_fp32.onnx")
    log.info("exported FP32 ONNX model: %s", output_path)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    samples = sample_split_images(
        manifest, DATA_ROOT, "test", PARITY_IMAGE_COUNT, PARITY_SEED
    )
    metrics = run_parity_check(model, session, samples)
    _log_parity(metrics)
    if not parity_passed(metrics):
        raise RuntimeError(f"parity check failed: {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
