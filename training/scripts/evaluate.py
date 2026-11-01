"""CLI entry point for test-set evaluation of a trained checkpoint.

Loads a best checkpoint (written by ``scripts/train.py`` via
``steel_training.engine.trainer.train_model``), builds the test split
DataLoader from ``training/manifest.json`` and the ``training/data``
junction, runs the model in eval mode over the test set, computes
accuracy / macro-F1 / confusion matrix / per-class precision, recall, F1,
and updates ``artifacts/metrics.json`` atomically (existing model entries
are preserved).

Convention: a class with no true and no predicted samples has
precision = recall = F1 = 0.0 (not NaN), matching
``steel_training.eval.metrics.macro_f1`` and keeping the JSON output valid.

Usage:
    python training/scripts/evaluate.py --model resnet18
    python training/scripts/evaluate.py --model resnet50 --config other.yaml
    python training/scripts/evaluate.py --model resnet18 --checkpoint ckpt.pth
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling steel_training package importable when this script is run
# directly (e.g. `python training/scripts/evaluate.py`).
TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import argparse
import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from steel_training.data.augmentation import build_eval_transform
from steel_training.data.dataset import SteelDefectDataset
from steel_training.data.manifest import load_manifest
from steel_training.engine.config import load_config
from steel_training.engine.trainer import resolve_device
from steel_training.eval.metrics import accuracy, confusion_matrix_counts, macro_f1
from steel_training.models.factory import create_resnet

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "training" / "configs"
MANIFEST_PATH = REPO_ROOT / "training" / "manifest.json"
DATA_ROOT = REPO_ROOT / "training" / "data"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
CARBON_DIR = REPO_ROOT / "artifacts" / "carbon"
METRICS_PATH = REPO_ROOT / "artifacts" / "metrics.json"

EVAL_BATCH_SIZE = 32
METRIC_DECIMALS = 4

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments; derive config/checkpoint defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet50"),
        required=True,
        help="model architecture of the checkpoint to evaluate",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"YAML config (default: training/configs/<model>.yaml under {REPO_ROOT})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"checkpoint to evaluate (default: artifacts/checkpoints/<model>_best.pth under {REPO_ROOT})",
    )
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = CONFIG_DIR / f"{args.model}.yaml"
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
        log.error("evaluation failed: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def _per_class_metrics(
    y_true: Any, y_pred: Any, num_classes: int
) -> list[dict[str, float | int]]:
    """Per-class precision / recall / F1 / support, in class-index order.

    Input validation is delegated to
    :func:`steel_training.eval.metrics.confusion_matrix_counts`, so the
    semantics (empty input, dtype, label range, length mismatch) match the
    existing metrics. A class with no predicted positives has precision
    0.0; a class with no true positives has recall 0.0; a class with
    neither (zero-sample class) has F1 = 0.0, not NaN.
    """
    matrix = confusion_matrix_counts(y_true, y_pred, num_classes)
    per_class: list[dict[str, float | int]] = []
    for class_index in range(num_classes):
        true_positive = int(matrix[class_index, class_index])
        false_positive = int(matrix[:, class_index].sum()) - true_positive
        false_negative = int(matrix[class_index, :].sum()) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": true_positive + false_negative,
            }
        )
    return per_class


def _update_metrics_json(
    existing: dict[str, Any] | None, model_name: str, entry: dict[str, Any]
) -> dict[str, Any]:
    """Merge ``entry`` for ``model_name`` into a metrics document.

    Returns a NEW document (the input is never mutated). Entries of other
    models and unrelated top-level keys are preserved; ``model_name``'s
    entry is replaced wholesale.
    """
    base = {} if existing is None else dict(existing)
    models = dict(base.get("models", {}))
    models[model_name] = dict(entry)
    base["models"] = models
    return base


def _write_metrics_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically persist a metrics document (tmp file, then ``os.replace``)."""
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_run_meta(path: Path) -> dict[str, Any]:
    """Read the training run-metadata sidecar written by train.py (or ``{}``).

    The sidecar attributes carbon/energy/duration/epoch totals to the exact
    training session that produced the checkpoint; it supersedes the
    best-effort carbon-CSV heuristic (whose "last row" assumption breaks when
    the same model is trained more than once).
    """
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        log.warning("could not parse run metadata %s; falling back", path)
        return {}


def _read_carbon_csv(path: Path) -> float | None:
    """Emissions (g CO2eq) from a CodeCarbon emissions CSV, or None.

    CodeCarbon appends one row per tracker session; the LAST data row
    belongs to the session that produced the evaluated checkpoint, so its
    ``emissions`` value (kg) is converted to grams and returned. Any parse
    problem or a missing file yields None (carbon is best-effort metadata).
    """
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "emissions" not in reader.fieldnames:
                return None
            emissions_kg: float | None = None
            for row in reader:
                value = (row.get("emissions") or "").strip()
                if not value:
                    continue
                emissions_kg = float(value)
            if emissions_kg is None:
                return None
            return emissions_kg * 1000.0
    except (OSError, ValueError, csv.Error):
        log.warning("could not parse carbon CSV %s; carbon will be null", path)
        return None


def _extract_training_duration(checkpoint: dict[str, Any]) -> float | None:
    """Best-effort training duration (seconds) from a checkpoint; None if absent.

    The current trainer checkpoint format does not store a duration, so this
    returns None until the trainer records one (spec allows null).
    """
    metrics = checkpoint.get("metrics")
    if isinstance(metrics, dict):
        for key in ("training_duration_s", "duration", "duration_s"):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    for key in ("training_duration_s", "duration", "duration_s"):
        value = checkpoint.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _iso_timestamp(epoch_seconds: float) -> str:
    """ISO-8601 UTC timestamp string for a POSIX epoch value (e.g. mtime)."""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _read_metrics_json(path: Path) -> dict[str, Any]:
    """Load an existing metrics document; a missing file yields ``{}``.

    Raises ValueError when the file exists but is not a valid
    ``{"models": {...}}`` document (refusing to silently overwrite it).
    """
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"metrics file exists but is not valid JSON: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise ValueError(
            f"metrics file has an unexpected structure (expected a 'models' mapping): {path}"
        )
    return data


def _build_entry(
    num_params: int,
    test_metrics: dict[str, Any],
    checkpoint: dict[str, Any],
    carbon_co2eq_g: float | None,
    training_duration_s: float | None,
    trained_at: str,
    training_epochs: int | None = None,
) -> dict[str, Any]:
    """Assemble the per-model entry persisted in artifacts/metrics.json."""
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    best_val_macro_f1 = metrics.get("val_macro_f1")
    return {
        "params": int(num_params),
        "test_acc": round(float(test_metrics["accuracy"]), METRIC_DECIMALS),
        "test_macro_f1": round(float(test_metrics["macro_f1"]), METRIC_DECIMALS),
        "per_class_f1": {
            name: round(float(value), METRIC_DECIMALS)
            for name, value in test_metrics["per_class_f1"].items()
        },
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "training_epochs": training_epochs
        if training_epochs is not None
        else checkpoint.get("epoch"),
        "best_val_macro_f1": round(float(best_val_macro_f1), METRIC_DECIMALS)
        if best_val_macro_f1 is not None
        else None,
        "carbon_co2eq_g": round(float(carbon_co2eq_g), METRIC_DECIMALS)
        if carbon_co2eq_g is not None
        else None,
        "training_duration_s": round(float(training_duration_s), 2)
        if training_duration_s is not None
        else None,
        "trained_at": trained_at,
    }


# ---------------------------------------------------------------------------
# Model / data plumbing
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path, model: torch.nn.Module) -> dict[str, Any]:
    """Load a trainer-format checkpoint and strictly apply its state dict.

    Raises FileNotFoundError/ValueError with clear messages on failure;
    a state-dict mismatch (``load_state_dict`` RuntimeError) is wrapped
    into a ValueError naming the checkpoint path.
    """
    checkpoint_path = path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"checkpoint is not a steel-defect-lab training checkpoint "
            f"(missing 'model_state_dict'): {checkpoint_path}"
        )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"model_state_dict in {checkpoint_path} does not match the "
            f"{type(model).__name__} architecture: {exc}"
        ) from exc
    return checkpoint


def _count_params(model: torch.nn.Module) -> int:
    """Total number of trainable parameters in the model."""
    return sum(param.numel() for param in model.parameters())


def _build_test_loader(
    manifest: dict[str, Any], data_root: Path, num_workers: int = 0
) -> DataLoader:
    """Test-split DataLoader: eval transform, batch 32, no shuffling."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"dataset junction is missing: {data_root}")
    dataset = SteelDefectDataset(
        data_root, manifest, "test", transform=build_eval_transform()
    )
    return DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _run_inference(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run eval-mode inference; return ``(y_true, y_pred)`` label arrays."""
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    if not targets:
        raise ValueError("test loader produced no samples; cannot compute metrics")
    return np.concatenate(targets), np.concatenate(predictions)


def _compute_report_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    """Compute the metric set persisted in metrics.json plus per-class rows."""
    num_classes = len(class_names)
    per_class = _per_class_metrics(y_true, y_pred, num_classes)
    per_class_f1 = {class_names[i]: per_class[i]["f1"] for i in range(num_classes)}
    return (
        {
            "accuracy": accuracy(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred, num_classes),
            "per_class_f1": per_class_f1,
            "confusion_matrix": confusion_matrix_counts(y_true, y_pred, num_classes),
        },
        per_class,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report(
    model_name: str,
    config_path: Path,
    checkpoint_path: Path,
    entry: dict[str, Any],
    class_names: list[str],
    per_class: list[dict[str, float | int]],
    matrix: np.ndarray,
) -> None:
    """Log a human-readable evaluation report with a per-class F1 table."""
    log.info("=================== test-set evaluation ===================")
    log.info("model            : %s", model_name)
    log.info("config           : %s", config_path)
    log.info("checkpoint       : %s", checkpoint_path)
    log.info("parameters       : %d", entry["params"])
    log.info("test accuracy    : %.4f", entry["test_acc"])
    log.info("test macro-F1    : %.4f", entry["test_macro_f1"])
    log.info("best val macro-F1: %s", entry["best_val_macro_f1"])
    log.info("carbon co2eq     : %s g", entry["carbon_co2eq_g"])
    log.info("training epochs  : %s", entry["training_epochs"])
    log.info("trained at       : %s", entry["trained_at"])
    log.info("")
    log.info("per-class results (precision / recall / F1 / support):")
    for name, row in zip(class_names, per_class):
        log.info(
            "  %-16s %8.4f %8.4f %8.4f %8d",
            name,
            row["precision"],
            row["recall"],
            row["f1"],
            row["support"],
        )
    log.info("confusion matrix (rows = true class, columns = predicted):")
    log.info("  %s", "".join(f"{name[:5]:>6s}" for name in class_names))
    for name, row in zip(class_names, matrix):
        log.info("  %-16s %s", name, " ".join(f"{value:6d}" for value in row))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> int:
    """Execute the evaluation for parsed arguments (happy path only).

    Any failure raises; :func:`main` turns exceptions into a non-zero exit
    code with a clear error message.
    """
    config = load_config(args.config)
    log.info("loaded config: %s", args.config)

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"training manifest is missing: {MANIFEST_PATH} (run the T2.1 manifest builder)"
        )
    manifest = load_manifest(MANIFEST_PATH)
    class_names = list(manifest["class_names"])
    num_classes = len(class_names)
    log.info(
        "test set: %s images across %d classes",
        manifest["split_counts"].get("test", "?"),
        num_classes,
    )

    model = create_resnet(args.model, num_classes=num_classes)
    checkpoint = _load_checkpoint(args.checkpoint, model)
    log.info("loaded checkpoint %s (epoch %s)", args.checkpoint, checkpoint.get("epoch"))

    device, _ = resolve_device(use_amp=True)
    model = model.to(device)
    loader = _build_test_loader(manifest, DATA_ROOT, config.num_workers)
    y_true, y_pred = _run_inference(model, loader, device)
    log.info("evaluated %d test samples on %s", y_true.size, device.type)

    test_metrics, per_class = _compute_report_metrics(y_true, y_pred, class_names)
    run_meta = _read_run_meta(Path(args.checkpoint).parent / f"{args.model}_run.json")
    carbon_g = run_meta.get("carbon_co2eq_g")
    if carbon_g is None:
        carbon_g = _read_carbon_csv(CARBON_DIR / f"B_defectlens_{args.model}_emissions.csv")
    duration_s = run_meta.get("training_duration_s")
    if duration_s is None:
        duration_s = _extract_training_duration(checkpoint)
    training_epochs = run_meta.get("total_epochs")
    if training_epochs is None:
        training_epochs = checkpoint.get("epoch")
    trained_at = _iso_timestamp(args.checkpoint.stat().st_mtime)

    entry = _build_entry(
        num_params=_count_params(model),
        test_metrics=test_metrics,
        checkpoint=checkpoint,
        carbon_co2eq_g=carbon_g,
        training_duration_s=duration_s,
        trained_at=trained_at,
        training_epochs=training_epochs,
    )

    updated = _update_metrics_json(_read_metrics_json(METRICS_PATH), args.model, entry)
    _write_metrics_json(METRICS_PATH, updated)
    log.info("updated metrics file: %s", METRICS_PATH)

    _print_report(
        args.model,
        args.config,
        args.checkpoint,
        entry,
        class_names,
        per_class,
        test_metrics["confusion_matrix"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
