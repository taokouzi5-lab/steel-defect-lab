"""Carbon estimation for model inference (T3.4).

``estimate()`` converts a measured per-inference energy into grams of
CO2-equivalent:

    gCO2eq = energy_per_inference (kWh) * grid_carbon_intensity (g/kWh)
             * PUE * calibration_factor

Models without a mace-f method-D measurement fall back to the T3-tier
representative (resnet_50) with a logged warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.carbon import constants

logger = logging.getLogger(__name__)

METHOD_MEASURED = "mace-f-methodD-measured"
METHOD_TIER_FALLBACK = "tier-representative-fallback"

# Process-level dedup for the fallback warning (LOW-2): the model table is
# static, so warn once per model name instead of on every request.
_FALLBACK_WARNED_ONCE: set[str] = set()


@dataclass(frozen=True)
class CarbonEstimate:
    """Immutable carbon estimate for a single model inference."""

    co2eq_g: float
    energy_kwh: float
    method: str
    equivalent: str


def _normalize_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(f"model_name must be a non-empty string, got {model_name!r}")
    return model_name.strip().lower().replace("-", "_")


def _resolve_energy(model_name: str) -> tuple[float, str]:
    measured = constants.INFERENCE_ENERGY_KWH_PER_1000.get(model_name)
    if measured is not None:
        return measured, METHOD_MEASURED
    representative = constants.T3_REPRESENTATIVE_MODEL
    if model_name not in _FALLBACK_WARNED_ONCE:
        _FALLBACK_WARNED_ONCE.add(model_name)
        logger.warning(
            "model '%s' has no mace-f method-D measurement; falling back to the "
            "T3-tier representative '%s'",
            model_name,
            representative,
        )
    return constants.INFERENCE_ENERGY_KWH_PER_1000[representative], METHOD_TIER_FALLBACK


def _resolve_calibration_factor(model_name: str, calibration_factor: float | None) -> float:
    if calibration_factor is not None:
        return float(calibration_factor)
    tier = constants.MODEL_TIERS.get(model_name, constants.DEFAULT_TIER)
    return constants.TIER_CALIBRATION_FACTORS.get(tier, constants.DEFAULT_CALIBRATION_FACTOR)


def _format_equivalent(co2eq_g: float) -> str:
    seconds = co2eq_g / constants.PHONE_CHARGE_GCO2EQ_PER_SECOND
    if seconds >= 100.0:
        text = f"{seconds:.0f} 秒"
    elif seconds >= 1.0:
        text = f"{seconds:.1f} 秒"
    elif seconds >= 0.01:
        text = f"{seconds:.2f} 秒"
    else:
        # Sub-10 ms equivalents are below the measurement's meaningful
        # precision: report milliseconds instead of sub-millisecond
        # "seconds" (LOW-1).
        text = f"{seconds * 1000.0:.3g} 毫秒"
    return f"≈ 手机充电 {text}"


def estimate(
    model_name: str,
    country: str = constants.DEFAULT_COUNTRY,
    pue: float = constants.DEFAULT_PUE,
    calibration_factor: float | None = None,
) -> CarbonEstimate:
    """Estimate the carbon footprint of a single inference.

    Args:
        model_name: model key such as ``"resnet_50"`` (case-insensitive,
            ``-`` normalized to ``_``). Unknown models fall back to the
            T3-tier representative energy with a logged warning.
        country: key of :data:`constants.GRID_CARBON_INTENSITY`.
        pue: power usage effectiveness of the deployment topology.
        calibration_factor: explicit Fischer-2025 correction; ``None`` picks
            the factor of the model's mace-f tier (T3 default).

    Returns:
        A :class:`CarbonEstimate` with ``co2eq_g``, the per-inference energy
        in kWh, the estimation method, and a phone-charging equivalent text.

    Raises:
        ValueError: for unknown countries, non-positive PUE/calibration
            factor, empty model names, or non-positive measured energy.
    """
    name = _normalize_model_name(model_name)
    if country not in constants.GRID_CARBON_INTENSITY:
        raise ValueError(
            f"unknown country '{country}'; expected one of "
            f"{sorted(constants.GRID_CARBON_INTENSITY)}"
        )
    if float(pue) <= 0.0:
        raise ValueError(f"pue must be positive, got {pue!r}")
    if calibration_factor is not None and float(calibration_factor) <= 0.0:
        raise ValueError(f"calibration_factor must be positive, got {calibration_factor!r}")

    energy_per_1000, method = _resolve_energy(name)
    if energy_per_1000 <= 0.0:
        raise ValueError(f"measured energy for '{name}' must be positive, got {energy_per_1000}")

    energy_per_inference = energy_per_1000 / 1000.0
    factor = _resolve_calibration_factor(name, calibration_factor)
    co2eq_g = energy_per_inference * constants.GRID_CARBON_INTENSITY[country] * float(pue) * factor
    return CarbonEstimate(
        co2eq_g=co2eq_g,
        energy_kwh=energy_per_inference,
        method=method,
        equivalent=_format_equivalent(co2eq_g),
    )
