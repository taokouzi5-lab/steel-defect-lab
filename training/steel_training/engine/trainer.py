"""Training engine: AMP loop, early stopping, checkpointing, carbon hook.

Trains a classifier with AdamW differential learning rates (backbone vs
``fc`` head, see :func:`steel_training.models.factory.split_params`), a
warmup + cosine LR schedule, AMP on CUDA with graceful CPU fallback,
per-epoch train/val metrics (loss, accuracy, macro-F1, confusion matrix),
early stopping on ``val_macro_f1`` with best-checkpoint persistence, and one
logging line per epoch.

Carbon hook: ``train_model`` accepts an optional CodeCarbon-style tracker.
If the tracker is not already active the engine owns its lifecycle
(``start()`` before the loop, ``stop()`` afterwards, ``result`` exposed in
the history). If it is already active (e.g. inside its own context manager,
as in ``scripts/train.py``), the engine leaves it alone and the caller reads
``tracker.result`` after the context exits.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from steel_training.engine.config import TrainConfig
from steel_training.engine.early_stopping import EarlyStopping
from steel_training.eval.metrics import accuracy, confusion_matrix_counts, macro_f1
from steel_training.models.factory import split_params

logger = logging.getLogger(__name__)


class CarbonTrackerLike(Protocol):
    """Minimal protocol of a CodeCarbon-style tracker."""

    def start(self) -> None: ...

    def stop(self) -> object: ...


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch for reproducible runs.

    Pins cuDNN to deterministic algorithms, while
    ``use_deterministic_algorithms`` stays disabled because AMP is not
    deterministic-algorithm compatible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)


def resolve_device(
    device: Optional[torch.device] = None, use_amp: bool = True
) -> tuple[torch.device, bool]:
    """Resolve the training device and whether AMP should be enabled.

    A requested CUDA device degrades gracefully to CPU (with AMP disabled)
    when CUDA is unavailable; AMP is only ever enabled on CUDA.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU (AMP off)")
        device = torch.device("cpu")
    amp_enabled = bool(use_amp) and device.type == "cuda"
    return device, amp_enabled


def _build_param_groups(model: nn.Module, config: TrainConfig) -> list[dict[str, Any]]:
    """AdamW groups: backbone (lr_backbone) and fc head (lr_fc)."""
    backbone_params, head_params = split_params(model)
    groups: list[dict[str, Any]] = []
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": config.lr_backbone,
                "weight_decay": config.weight_decay,
            }
        )
    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": config.lr_fc,
                "weight_decay": config.weight_decay,
            }
        )
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup followed by cosine annealing (stepped once per epoch)."""
    if config.warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, config.epochs), eta_min=0.0)
    warmup = LinearLR(
        optimizer,
        start_factor=0.2,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs - config.warmup_epochs),
        eta_min=0.0,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs])


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Run one training epoch; return ``(mean_loss, accuracy)``."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = targets.numel()
        total_loss += loss.detach().item() * batch_size
        correct += (logits.detach().argmax(dim=1) == targets).sum().item()
        total += batch_size
    return total_loss / total, correct / total


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    num_classes: int,
) -> dict[str, Any]:
    """Run validation; return loss, accuracy, macro F1 and confusion matrix."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)
            batch_size = targets.numel()
            total_loss += loss.detach().item() * batch_size
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += batch_size
            all_targets.append(targets.cpu())
            all_preds.append(predictions.cpu())
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    return {
        "loss": total_loss / total,
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "confusion_matrix": confusion_matrix_counts(y_true, y_pred, num_classes),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: Optional[int] = None,
    use_amp: bool = True,
    criterion: Optional[nn.Module] = None,
) -> dict[str, Any]:
    """Evaluate ``model`` on ``loader``; see :func:`_evaluate` for the result."""
    if num_classes is None:
        num_classes = _num_outputs(model)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    return _evaluate(model, loader, criterion, device, use_amp, num_classes)


def _num_outputs(model: nn.Module) -> int:
    """Number of outputs of the classification head (``fc.out_features``)."""
    head = getattr(model, "fc", None)
    if head is None or not hasattr(head, "out_features"):
        raise ValueError(
            "model has no 'fc' head; pass num_classes explicitly for "
            "models with a different head layout"
        )
    return int(head.out_features)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_metrics: dict[str, Any],
    config: TrainConfig,
) -> None:
    """Persist the current best checkpoint (state dict + epoch + metrics)."""
    checkpoint_path = path.resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        },
        "config": asdict(config),
    }
    torch.save(checkpoint, checkpoint_path)


def _tracker_is_active(tracker: object) -> bool:
    """Best-effort check whether a tracker is already measuring.

    Trackers may expose an ``is_active`` attribute/property; CodeCarbon-style
    trackers create their underlying tracker in ``start()``, so a truthy
    ``_tracker`` means the context manager already owns the lifecycle.
    """
    if hasattr(tracker, "is_active"):
        return bool(getattr(tracker, "is_active"))
    return getattr(tracker, "_tracker", None) is not None


@dataclass(frozen=True)
class TrainHistory:
    """Immutable summary of one training run."""

    train_losses: tuple[float, ...]
    train_accuracies: tuple[float, ...]
    val_losses: tuple[float, ...]
    val_accuracies: tuple[float, ...]
    val_macro_f1s: tuple[float, ...]
    val_confusion_matrices: tuple[np.ndarray, ...]
    best_epoch: int
    best_val_macro_f1: float
    epochs_run: int
    stopped_early: bool
    learning_rates: tuple[float, ...]
    carbon_result: object = None


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    *,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Path] = None,
    carbon_tracker: Optional[CarbonTrackerLike] = None,
    num_classes: Optional[int] = None,
) -> TrainHistory:
    """Train ``model`` for ``config.epochs`` epochs and return a history.

    Args:
        model: classification model (moved to ``device`` by this function).
        train_loader/val_loader: training and validation data loaders.
        config: :class:`TrainConfig` controlling the run.
        device: target device; defaults to CUDA when available, else CPU.
        checkpoint_path: best checkpoint destination; None writes nothing.
        carbon_tracker: optional tracker; see the module docstring for the
            ownership contract.
        num_classes: classes for metric computation; defaults to
            ``model.fc.out_features``.
    """
    if num_classes is None:
        num_classes = _num_outputs(model)
    device, use_amp = resolve_device(device, config.amp)
    set_seed(config.seed)
    model = model.to(device)

    logger.info(
        "starting training: model=%s epochs=%d batch=%d device=%s amp=%s",
        config.model_name,
        config.epochs,
        config.batch_size,
        device.type,
        use_amp,
    )
    started_at = time.perf_counter()

    optimizer = torch.optim.AdamW(_build_param_groups(model, config))
    scheduler = _build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=config.patience, mode="max")

    manage_tracker = carbon_tracker is not None and not _tracker_is_active(carbon_tracker)
    if manage_tracker:
        carbon_tracker.start()  # type: ignore[union-attr]

    train_losses: list[float] = []
    train_accuracies: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    val_macro_f1s: list[float] = []
    val_confusion_matrices: list[np.ndarray] = []
    learning_rates: list[float] = []
    stopped_early = False
    try:
        for epoch in range(1, config.epochs + 1):
            train_loss, train_accuracy = _train_one_epoch(
                model, train_loader, optimizer, scaler, criterion, device, use_amp
            )
            val_metrics = _evaluate(model, val_loader, criterion, device, use_amp, num_classes)
            learning_rates.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_metrics["loss"])
            val_accuracies.append(val_metrics["accuracy"])
            val_macro_f1s.append(val_metrics["macro_f1"])
            val_confusion_matrices.append(val_metrics["confusion_matrix"])

            should_stop = stopper.update(val_metrics["macro_f1"], epoch)
            if stopper.is_best and checkpoint_path is not None:
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    epoch,
                    train_loss,
                    train_accuracy,
                    val_metrics,
                    config,
                )
            logger.info(
                "epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f f1 %.4f"
                " | lr %.2e | best %.4f @ %s%s",
                epoch,
                config.epochs,
                train_loss,
                train_accuracy,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                learning_rates[-1],
                stopper.best_score,
                stopper.best_epoch,
                " (early stop)" if should_stop else "",
            )
            if should_stop:
                stopped_early = True
                logger.info("early stopping after %d epochs without F1 improvement", config.patience)
                break
    finally:
        if manage_tracker:
            carbon_tracker.stop()  # type: ignore[union-attr]

    duration = time.perf_counter() - started_at
    best_score = stopper.best_score if stopper.best_score is not None else 0.0
    best_epoch = int(stopper.best_epoch) if stopper.best_epoch is not None else 0
    logger.info(
        "training finished: %d epochs in %.1fs | best val macro-F1 %.4f @ epoch %d",
        len(train_losses),
        duration,
        best_score,
        best_epoch,
    )

    return TrainHistory(
        train_losses=tuple(train_losses),
        train_accuracies=tuple(train_accuracies),
        val_losses=tuple(val_losses),
        val_accuracies=tuple(val_accuracies),
        val_macro_f1s=tuple(val_macro_f1s),
        val_confusion_matrices=tuple(val_confusion_matrices),
        best_epoch=best_epoch,
        best_val_macro_f1=best_score,
        epochs_run=len(train_losses),
        stopped_early=stopped_early,
        learning_rates=tuple(learning_rates),
        # The result is only authoritative when the engine owns the tracker
        # lifecycle; for an already-active tracker the caller reads
        # tracker.result after its context manager exits.
        carbon_result=getattr(carbon_tracker, "result", None) if manage_tracker else None,
    )
