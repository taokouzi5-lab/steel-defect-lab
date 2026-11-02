"""Application configuration, read from environment variables only.

No hardcoded runtime paths: every directory/file location has an env var with
a default computed relative to the repository root (``backend/app/config.py``
walks up two levels). Boolean flags accept 1/0/true/false.

Environment variables
---------------------
MODEL_DIR            ONNX artifact directory (default: <repo>/artifacts/onnx)
CHECKPOINT_DIR       PyTorch checkpoint directory for Grad-CAM
                     (default: <repo>/artifacts/checkpoints)
METRICS_PATH         training metrics.json for the model registry
                     (default: <repo>/artifacts/metrics.json)
SAMPLES_DIR          demo sample images + manifest
                     (default: <repo>/backend/app/assets/samples)
MAX_UPLOAD_MB        upload size cap in MB (default: 5)
RATE_LIMIT_PER_MIN   per-client request budget (default: 20)
RATE_LIMIT_ENABLED   enable slowapi limiting (default: 1)
REQUEST_TIMEOUT_S    per-inference timeout in seconds (default: 10)
PRELOAD_MODELS       load all ONNX sessions at startup (default: 1)
CORS_ORIGINS         comma-separated allowed origins (dev default: localhost:5173)
CARBON_COUNTRY       grid intensity region for carbon estimates (default: China)
DEFECTLENS_ENABLE_DOCS  serve /docs, /redoc and /openapi.json (default: 1;
                     set 0 in deployed containers — see docs/deployment.md)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.carbon import constants as carbon_constants

REPO_ROOT = Path(__file__).resolve().parents[2]

_DEV_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings; build via :meth:`from_env`."""

    model_dir: Path
    checkpoint_dir: Path
    metrics_path: Path
    samples_dir: Path
    max_upload_mb: int = 5
    rate_limit_per_min: int = 20
    rate_limit_enabled: bool = True
    request_timeout_s: float = 10.0
    preload_models: bool = True
    cors_origins: tuple[str, ...] = _DEV_CORS_ORIGINS
    carbon_country: str = carbon_constants.DEFAULT_COUNTRY
    enable_docs: bool = True

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Read settings from the environment (``overrides`` win, for tests)."""

        def path_env(name: str, default: Path) -> Path:
            raw = os.environ.get(name)
            if not raw:
                return default
            path = Path(raw.strip())
            return path if path.is_absolute() else (REPO_ROOT / path).resolve()

        def bool_env(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None or raw.strip() == "":
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        def int_env(name: str, default: int) -> int:
            raw = os.environ.get(name)
            return int(raw) if raw and raw.strip() else default

        def float_env(name: str, default: float) -> float:
            raw = os.environ.get(name)
            return float(raw) if raw and raw.strip() else default

        values: dict = {
            "model_dir": path_env("MODEL_DIR", REPO_ROOT / "artifacts" / "onnx"),
            "checkpoint_dir": path_env("CHECKPOINT_DIR", REPO_ROOT / "artifacts" / "checkpoints"),
            "metrics_path": path_env("METRICS_PATH", REPO_ROOT / "artifacts" / "metrics.json"),
            "samples_dir": path_env(
                "SAMPLES_DIR", REPO_ROOT / "backend" / "app" / "assets" / "samples"
            ),
            "max_upload_mb": int_env("MAX_UPLOAD_MB", 5),
            "rate_limit_per_min": int_env("RATE_LIMIT_PER_MIN", 20),
            "rate_limit_enabled": bool_env("RATE_LIMIT_ENABLED", True),
            "request_timeout_s": float_env("REQUEST_TIMEOUT_S", 10.0),
            "preload_models": bool_env("PRELOAD_MODELS", True),
            "carbon_country": os.environ.get("CARBON_COUNTRY", carbon_constants.DEFAULT_COUNTRY),
            "enable_docs": bool_env("DEFECTLENS_ENABLE_DOCS", True),
        }
        raw_origins = os.environ.get("CORS_ORIGINS")
        values["cors_origins"] = (
            tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
            if raw_origins
            else _DEV_CORS_ORIGINS
        )
        values.update(overrides)
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        """Fail fast on nonsensical configuration."""
        if self.max_upload_mb < 1:
            raise ValueError(f"MAX_UPLOAD_MB must be >= 1, got {self.max_upload_mb}")
        if self.rate_limit_per_min < 1:
            raise ValueError(f"RATE_LIMIT_PER_MIN must be >= 1, got {self.rate_limit_per_min}")
        if self.request_timeout_s <= 0:
            raise ValueError(f"REQUEST_TIMEOUT_S must be positive, got {self.request_timeout_s}")
        if self.carbon_country not in carbon_constants.GRID_CARBON_INTENSITY:
            raise ValueError(
                f"CARBON_COUNTRY '{self.carbon_country}' is unknown; expected one of "
                f"{sorted(carbon_constants.GRID_CARBON_INTENSITY)}"
            )

    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


_active_settings: Settings | None = None


def set_active_settings(settings: Settings) -> None:
    """Register the settings of the currently serving app.

    The rate limiter's dynamic limit callable reads this at request time so
    per-app (test) configurations take effect without rebuilding decorators.
    """
    global _active_settings
    _active_settings = settings


def get_active_settings() -> Settings:
    if _active_settings is None:
        set_active_settings(Settings.from_env())
    return _active_settings
