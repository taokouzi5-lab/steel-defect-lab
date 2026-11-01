"""Tests for the training engine: config, early stopping, device/AMP
resolution, carbon tracker hook, and a GPU overfit smoke test (T2.3).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from steel_training.data.augmentation import build_eval_transform
from steel_training.engine.config import TrainConfig, load_config
from steel_training.engine.early_stopping import EarlyStopping
from steel_training.engine.trainer import resolve_device, train_model
from steel_training.models.factory import create_resnet

# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


def test_early_stopping_stops_after_patience_non_improvements() -> None:
    # Arrange
    stopper = EarlyStopping(patience=4)

    # Act
    verdicts = [stopper.update(0.5, epoch) for epoch in range(1, 6)]

    # Assert: first call sets the best; the next four non-improvements stop.
    assert verdicts == [False, False, False, False, True]
    assert stopper.best_score == 0.5
    assert stopper.best_epoch == 1
    assert stopper.counter == 4


def test_early_stopping_improvement_resets_counter() -> None:
    # Arrange
    stopper = EarlyStopping(patience=4)

    # Act
    first = stopper.update(0.5, 1)
    improved = stopper.update(0.6, 2)
    verdicts = [stopper.update(0.5, epoch) for epoch in range(3, 7)]

    # Assert: the improvement at epoch 2 resets the counter, so four more
    # non-improvements (epochs 3-6) are needed before stopping.
    assert first is False
    assert improved is False
    assert stopper.is_best is False  # last update was a non-improvement
    assert stopper.best_score == 0.6
    assert stopper.best_epoch == 2
    assert verdicts == [False, False, False, True]


def test_early_stopping_tracks_is_best_flag() -> None:
    # Arrange
    stopper = EarlyStopping(patience=2)

    # Act
    stopper.update(0.5, 1)

    # Assert
    assert stopper.is_best is True


def test_early_stopping_mode_min() -> None:
    # Arrange
    stopper = EarlyStopping(patience=2, mode="min")

    # Act
    first = stopper.update(1.0, 1)
    improved = stopper.update(0.9, 2)
    improved_is_best = stopper.is_best
    stale = stopper.update(0.95, 3)
    stopped = stopper.update(0.95, 4)

    # Assert: the drop to 0.9 established a new best; two later stale updates
    # trigger the stop. is_best only reflects the most recent update.
    assert first is False and improved is False and improved_is_best is True
    assert stale is False
    assert stopped is True and stopper.is_best is False
    assert stopper.best_score == 0.9
    assert stopper.best_epoch == 2


def test_early_stopping_invalid_mode_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="mode"):
        EarlyStopping(patience=2, mode="median")


def test_early_stopping_negative_patience_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=-1)


def test_early_stopping_patience_zero_never_stops() -> None:
    # Arrange
    stopper = EarlyStopping(patience=0)

    # Act
    verdicts = [stopper.update(0.5, epoch) for epoch in range(1, 6)]

    # Assert: disabled early stopping still tracks the best score.
    assert verdicts == [False] * 5
    assert stopper.best_score == 0.5


def test_early_stopping_reset_clears_state() -> None:
    # Arrange
    stopper = EarlyStopping(patience=2)
    stopper.update(0.5, 1)
    stopper.update(0.4, 2)
    stopper.update(0.4, 3)

    # Act
    stopper.reset()

    # Assert
    assert stopper.best_score is None
    assert stopper.best_epoch is None
    assert stopper.counter == 0
    assert stopper.is_best is False


# ---------------------------------------------------------------------------
# TrainConfig / load_config
# ---------------------------------------------------------------------------


def test_train_config_defaults() -> None:
    # Arrange / Act
    config = TrainConfig.from_dict({"model_name": "resnet18"})

    # Assert
    assert config.batch_size == 32
    assert config.lr_fc == 1e-3
    assert config.lr_backbone == 1e-4
    assert config.epochs == 20
    assert config.warmup_epochs == 2
    assert config.patience == 4
    assert config.amp is True
    assert config.weight_decay == 1e-4
    assert config.num_workers == 2
    assert config.seed == 42
    assert config.monitor == "val_macro_f1"


def test_train_config_from_full_dict() -> None:
    # Arrange / Act
    config = TrainConfig.from_dict(
        {
            "model_name": "resnet50",
            "batch_size": 16,
            "epochs": 5,
            "warmup_epochs": 0,
            "patience": 0,
            "amp": False,
        }
    )

    # Assert
    assert config.model_name == "resnet50"
    assert config.batch_size == 16
    assert config.epochs == 5
    assert config.warmup_epochs == 0
    assert config.patience == 0
    assert config.amp is False


def test_train_config_unknown_key_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_dict({"model_name": "resnet18", "batch_szie": 32})


def test_train_config_invalid_monitor_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="monitor"):
        TrainConfig.from_dict({"model_name": "resnet18", "monitor": "val_loss"})


def test_train_config_invalid_values_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig.from_dict({"model_name": "resnet18", "epochs": 0})
    with pytest.raises(ValueError, match="lr_fc"):
        TrainConfig.from_dict({"model_name": "resnet18", "lr_fc": -0.1})


def test_load_config_reads_yaml(configs_dir: Path) -> None:
    # Arrange
    resnet18_yaml = configs_dir / "resnet18.yaml"
    resnet50_yaml = configs_dir / "resnet50.yaml"

    # Act
    config18 = load_config(resnet18_yaml)
    config50 = load_config(resnet50_yaml)

    # Assert
    assert config18.model_name == "resnet18"
    assert config18.batch_size == 32
    assert config18.lr_fc == 1e-3
    assert config18.lr_backbone == 1e-4
    assert config18.epochs == 20
    assert config18.warmup_epochs == 2
    assert config18.patience == 4
    assert config18.amp is True
    assert config18.monitor == "val_macro_f1"
    assert config50.model_name == "resnet50"
    assert config50.batch_size == 16


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    # Arrange / Act / Assert
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


# ---------------------------------------------------------------------------
# Device / AMP resolution
# ---------------------------------------------------------------------------


def test_resolve_device_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # Act
    device, use_amp = resolve_device(torch.device("cuda"), use_amp=True)

    # Assert
    assert device.type == "cuda"
    assert use_amp is True


def test_resolve_device_falls_back_to_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Act
    device, use_amp = resolve_device(torch.device("cuda"), use_amp=True)

    # Assert
    assert device.type == "cpu"
    assert use_amp is False


def test_resolve_device_cpu_disables_amp(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # Act
    device, use_amp = resolve_device(torch.device("cpu"), use_amp=True)

    # Assert
    assert device.type == "cpu"
    assert use_amp is False


def test_resolve_device_default_detects_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Act
    device, use_amp = resolve_device(None, use_amp=True)

    # Assert
    assert device.type == "cpu"
    assert use_amp is False


# ---------------------------------------------------------------------------
# Carbon tracker hook
# ---------------------------------------------------------------------------


def _tiny_loaders(
    n_samples: int = 8, num_classes: int = 2, batch_size: int = 4
) -> tuple[DataLoader, DataLoader]:
    """Deterministic random data loaders for a trivial classification task."""
    rng = torch.Generator().manual_seed(0)
    inputs = torch.randn(n_samples, 8, generator=rng)
    labels = torch.randint(0, num_classes, (n_samples,), generator=rng)
    train_loader = DataLoader(
        TensorDataset(inputs, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(7),
    )
    val_loader = DataLoader(
        TensorDataset(inputs, labels), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def _tiny_config(**overrides: object) -> TrainConfig:
    """One-epoch CPU config for a tiny model, immune to real CodeCarbon."""
    defaults = {
        "model_name": "tiny",
        "epochs": 1,
        "warmup_epochs": 0,
        "batch_size": 4,
        "amp": False,
        "seed": 0,
        "num_workers": 0,
    }
    defaults.update(overrides)
    return TrainConfig.from_dict(defaults)


def test_train_model_manages_inactive_carbon_tracker(
    fake_carbon_tracker: type,
) -> None:
    # Arrange: an inactive tracker must be started/stopped by the engine.
    train_loader, val_loader = _tiny_loaders()
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    tracker = fake_carbon_tracker(active=False)

    # Act
    history = train_model(
        model,
        train_loader,
        val_loader,
        _tiny_config(),
        device=torch.device("cpu"),
        num_classes=2,
        carbon_tracker=tracker,
    )

    # Assert
    assert tracker.calls == ["start", "stop"]
    assert history.carbon_result is tracker.result
    assert history.epochs_run == 1


def test_train_model_leaves_active_carbon_tracker_alone(
    fake_carbon_tracker: type,
) -> None:
    # Arrange: a tracker already running inside its context manager owns its
    # own lifecycle; the engine must not touch it.
    train_loader, val_loader = _tiny_loaders()
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    tracker = fake_carbon_tracker(active=True)

    # Act
    history = train_model(
        model,
        train_loader,
        val_loader,
        _tiny_config(),
        device=torch.device("cpu"),
        num_classes=2,
        carbon_tracker=tracker,
    )

    # Assert
    assert tracker.calls == []
    assert history.carbon_result is None


def test_train_model_runs_without_carbon_tracker() -> None:
    # Arrange
    train_loader, val_loader = _tiny_loaders()
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))

    # Act
    history = train_model(
        model,
        train_loader,
        val_loader,
        _tiny_config(),
        device=torch.device("cpu"),
        num_classes=2,
        carbon_tracker=None,
    )

    # Assert
    assert history.epochs_run == 1
    assert history.carbon_result is None
    assert len(history.train_losses) == 1
    assert len(history.val_macro_f1s) == 1


# ---------------------------------------------------------------------------
# GPU overfit smoke (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_overfit_smoke_resnet18_random_labels(
    tmp_path: Path, image_root: Path, dataset_files: set[str]
) -> None:
    """End-to-end trainability check: resnet18 (no ImageNet weights) on 8 real
    images with random 2-class labels must memorize the train set in 3 epochs.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for the overfit smoke test")

    # Arrange: 8 real NEU-DET images resized to 224 with random binary labels.
    transform = build_eval_transform()
    tensors: list[torch.Tensor] = []
    for relative in sorted(dataset_files)[:8]:
        with Image.open(image_root / relative) as handle:
            tensors.append(transform(handle.convert("RGB")))
    inputs = torch.stack(tensors)
    labels = torch.randint(0, 2, (8,), generator=torch.Generator().manual_seed(42))
    # Fixed batch order (no shuffle): at this scale the 512-d backbone
    # features make the 8 random-label points linearly separable for the fc
    # head, so a modest lr gives a decisive, monotone loss decrease each
    # epoch instead of batch-to-batch oscillation.
    train_loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
    val_loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)

    torch.manual_seed(42)  # deterministic random-init weights for the smoke
    model = create_resnet("resnet18", num_classes=2)  # random init, fc replaced
    config = TrainConfig.from_dict(
        {
            "model_name": "resnet18",
            "epochs": 3,
            "warmup_epochs": 0,
            "batch_size": 4,
            "amp": True,
            "patience": 0,
            "lr_fc": 1e-3,
            "lr_backbone": 1e-4,
            "weight_decay": 0.0,
            "num_workers": 0,
            "seed": 42,
        }
    )
    checkpoint_path = tmp_path / "smoke_best.pth"

    # Act
    history = train_model(
        model,
        train_loader,
        val_loader,
        config,
        device=torch.device("cuda"),
        checkpoint_path=checkpoint_path,
        num_classes=2,
    )

    # Assert: train loss strictly decreases and drops well below its start.
    losses = history.train_losses
    assert len(losses) == 3
    assert all(
        current < previous for previous, current in zip(losses, losses[1:])
    ), f"train loss is not strictly decreasing: {losses}"
    assert losses[-1] < 0.75 * losses[0], f"loss barely moved: {losses}"
    assert 0.0 <= history.best_val_macro_f1 <= 1.0

    # Assert: the best checkpoint exists and loads back into a fresh model.
    assert checkpoint_path.is_file()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    assert set(checkpoint) >= {"model_state_dict", "epoch", "metrics"}
    fresh = create_resnet("resnet18", num_classes=2)
    missing, unexpected = fresh.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    assert not missing, f"checkpoint lacks keys: {missing}"
    assert not unexpected, f"checkpoint has extra keys: {unexpected}"
    assert checkpoint["epoch"] == history.best_epoch
    assert checkpoint["metrics"]["val_macro_f1"] == history.best_val_macro_f1
