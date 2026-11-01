"""Training configuration: immutable TrainConfig and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_MONITORS = ("val_macro_f1",)


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training configuration (loadable from dict or YAML)."""

    model_name: str = "resnet18"
    batch_size: int = 32
    lr_fc: float = 1e-3
    lr_backbone: float = 1e-4
    epochs: int = 20
    warmup_epochs: int = 2
    patience: int = 4
    amp: bool = True
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    monitor: str = "val_macro_f1"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "TrainConfig":
        """Build a config from a dict, rejecting unknown keys.

        Raises:
            ValueError: for unknown keys or invalid field values.
        """
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(config) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}; expected one of {sorted(known)}")
        merged = {field.name: config.get(field.name, field.default) for field in fields(cls)}
        config_obj = cls(**merged)
        config_obj.validate()
        return config_obj

    def validate(self) -> None:
        """Fail fast on semantically invalid values."""
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.lr_fc <= 0:
            raise ValueError(f"lr_fc must be > 0, got {self.lr_fc}")
        if self.lr_backbone <= 0:
            raise ValueError(f"lr_backbone must be > 0, got {self.lr_backbone}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.monitor not in SUPPORTED_MONITORS:
            raise ValueError(
                f"monitor must be one of {SUPPORTED_MONITORS}, got {self.monitor!r}"
            )


def load_config(path: Path) -> TrainConfig:
    """Load a :class:`TrainConfig` from a YAML file.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file content is not a mapping.
    """
    config_path = path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a YAML mapping, got {type(data).__name__}")
    return TrainConfig.from_dict(data)
