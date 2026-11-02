"""Carbon estimation method disclosure (T3.6, ``GET /meta``).

Builds the three declarations the About page needs — formula, constants,
boundaries — straight from :mod:`app.carbon.constants` so the disclosure can
never drift from the estimator implementation.
"""

from __future__ import annotations

from typing import Any

from app.carbon import constants
from app.carbon.estimator import METHOD_MEASURED, METHOD_TIER_FALLBACK

_BOUNDARIES = (
    "inference energy is a CodeCarbon method-D measurement on the development GPU "
    "(1000 inferences per model); resnet18 falls back to the T3-tier representative "
    "resnet_50 because it was not in the measurement matrix",
    "covers inference electricity only; training carbon (see /models) and embodied "
    "hardware carbon are excluded",
    "grid carbon intensity is a regional annual average (CO2 Signal + IPCC), not a "
    "real-time reading",
    "PUE assumes a factory-floor deployment topology (2.00)",
    "calibration factors follow Fischer 2025, correcting CodeCarbon's 25-37 % "
    "systematic underestimation",
)


def build_carbon_meta(country: str = constants.DEFAULT_COUNTRY) -> dict[str, Any]:
    """Return the formula/constants/boundaries/methods disclosure dict."""
    if country not in constants.GRID_CARBON_INTENSITY:
        raise ValueError(
            f"unknown country '{country}'; expected one of "
            f"{sorted(constants.GRID_CARBON_INTENSITY)}"
        )
    return {
        "formula": (
            "co2eq_g = energy_kwh_per_inference * grid_carbon_intensity_g_per_kwh "
            "* pue * calibration_factor"
        ),
        "constants": {
            "country": country,
            "grid_carbon_intensity_g_per_kwh": constants.GRID_CARBON_INTENSITY[country],
            "pue": constants.DEFAULT_PUE,
            "default_calibration_factor": constants.DEFAULT_CALIBRATION_FACTOR,
            "tier_calibration_factors": dict(constants.TIER_CALIBRATION_FACTORS),
            "phone_charge_gco2eq_per_second": constants.PHONE_CHARGE_GCO2EQ_PER_SECOND,
            "measured_inference_energy_kwh_per_1000": dict(constants.INFERENCE_ENERGY_KWH_PER_1000),
        },
        "boundaries": list(_BOUNDARIES),
        "methods": [METHOD_MEASURED, METHOD_TIER_FALLBACK],
    }
