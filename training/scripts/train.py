"""CLI entry point for training the steel defect classifier.

Builds the train/val/test DataLoaders from ``training/manifest.json`` and the
``training/data`` junction, loads ImageNet-pretrained ResNet weights from the
local artifacts cache, trains with AMP + early stopping while measuring
carbon emissions with the carbon_benchmark CarbonTracker, evaluates the best
checkpoint on the test split, and prints a summary table.

Usage:
    python training/scripts/train.py --model resnet18
    python training/scripts/train.py --model resnet50 --epochs 5
    python training/scripts/train.py --model resnet18 --config other.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling steel_training package importable when this script is run
# directly (e.g. `python training/scripts/train.py`).
TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import argparse
import logging
import time
from dataclasses import replace
from typing import Optional

import torch
from torch.utils.data import DataLoader

from steel_training.data.augmentation import build_eval_transform, build_train_transform
from steel_training.data.dataset import SteelDefectDataset
from steel_training.data.manifest import load_manifest
from steel_training.engine.config import load_config
from steel_training.engine.trainer import evaluate, set_seed, train_model
from steel_training.models.factory import create_resnet

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "training" / "configs"
MANIFEST_PATH = REPO_ROOT / "training" / "manifest.json"
DATA_ROOT = REPO_ROOT / "training" / "data"
PRETRAINED_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "pretrained"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
CARBON_DIR = REPO_ROOT / "artifacts" / "carbon"

log = logging.getLogger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet50"),
        required=True,
        help="model architecture to train",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"YAML config (default: training/configs/<model>.yaml under {REPO_ROOT})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override the epoch count from the config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help=f"checkpoint directory (default: {CHECKPOINT_DIR})",
    )
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be >= 1")
    return args


def build_data_loaders(
    config: object,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Build train/val/test loaders from the committed manifest and junction.

    Returns:
        ``(train_loader, val_loader, test_loader, num_classes)``.
    """
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"training manifest is missing: {MANIFEST_PATH} (run the T2.1 manifest builder)"
        )
    manifest = load_manifest(MANIFEST_PATH)
    num_classes = len(manifest["class_names"])
    generator = torch.Generator().manual_seed(config.seed)  # type: ignore[attr-defined]
    loader_kwargs = {
        "num_workers": config.num_workers,  # type: ignore[attr-defined]
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,  # type: ignore[attr-defined]
    }
    train_loader = DataLoader(
        SteelDefectDataset(DATA_ROOT, manifest, "train", transform=build_train_transform()),
        batch_size=config.batch_size,  # type: ignore[attr-defined]
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        SteelDefectDataset(DATA_ROOT, manifest, "val", transform=build_eval_transform()),
        batch_size=config.batch_size,  # type: ignore[attr-defined]
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        SteelDefectDataset(DATA_ROOT, manifest, "test", transform=build_eval_transform()),
        batch_size=config.batch_size,  # type: ignore[attr-defined]
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader, num_classes


def _model_device(model: torch.nn.Module) -> torch.device:
    """Device of a model's parameters (after the engine moved it)."""
    return next(model.parameters()).device


def main(argv: Optional[list[str]] = None) -> int:
    """Run a full training job; return the process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    config_path = args.config if args.config is not None else CONFIG_DIR / f"{args.model}.yaml"
    config = load_config(config_path)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    set_seed(config.seed)
    log.info("loaded config: %s (%d epochs, batch %d)", config_path, config.epochs, config.batch_size)

    pretrained_path = PRETRAINED_DIR / f"{args.model}.pth"
    if not pretrained_path.is_file():
        log.error(
            "pretrained weights not found: %s (run training/scripts/download_weights.py --model %s)",
            pretrained_path,
            args.model,
        )
        return 1

    try:
        from carbon_benchmark.measurement.tracker import CarbonTracker
    except ImportError:
        log.error("carbon_benchmark is not installed; cannot measure carbon emissions")
        return 1

    train_loader, val_loader, test_loader, num_classes = build_data_loaders(config)
    model = create_resnet(args.model, num_classes=num_classes, pretrained_path=pretrained_path)
    checkpoint_path = Path(args.output_dir) / f"{args.model}_best.pth"

    started_at = time.perf_counter()
    with CarbonTracker(
        f"B_defectlens_{args.model}", args.model, "A", output_dir=CARBON_DIR
    ) as tracker:
        history = train_model(
            model,
            train_loader,
            val_loader,
            config,
            checkpoint_path=checkpoint_path,
            carbon_tracker=tracker,
        )
    duration = time.perf_counter() - started_at

    # Final evaluation of the best checkpoint on the held-out test split.
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = _model_device(model)
    test_metrics = evaluate(model, test_loader, device)

    log.info("=================== training summary ===================")
    log.info("model            : %s", args.model)
    log.info("best val macro-F1: %.4f @ epoch %d", history.best_val_macro_f1, history.best_epoch)
    log.info(
        "epochs run       : %d of %d (early stop: %s)",
        history.epochs_run,
        config.epochs,
        history.stopped_early,
    )
    log.info("total duration   : %.1fs", duration)
    log.info(
        "test metrics     : loss %.4f acc %.4f macro-F1 %.4f",
        test_metrics["loss"],
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
    )
    result = getattr(tracker, "result", None)
    if result is not None:
        log.info(
            "carbon           : co2eq %.2fg (corrected %.2fg, tier %s), %.6fkWh over %.1fs",
            result.co2eq_g,
            result.co2eq_corrected_g,
            result.tier,
            result.energy_kwh,
            result.duration_seconds,
        )
    log.info("checkpoint       : %s", checkpoint_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
