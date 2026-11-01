"""Engine subpackage: config, early stopping, training/validation loops."""

from steel_training.engine.config import TrainConfig, load_config
from steel_training.engine.early_stopping import EarlyStopping
from steel_training.engine.trainer import (
    TrainHistory,
    evaluate,
    resolve_device,
    set_seed,
    train_model,
)

__all__ = [
    "EarlyStopping",
    "TrainConfig",
    "TrainHistory",
    "evaluate",
    "load_config",
    "resolve_device",
    "set_seed",
    "train_model",
]
