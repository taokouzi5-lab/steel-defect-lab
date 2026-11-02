"""TDD tests for the MACE-F carbon estimation module (T3.4).

Covers the gCO2eq formula against hand-computed values, tier-based
calibration defaults, the unknown-model fallback with warning, defensive
validation (zero/negative energy, unknown country, bad PUE/calibration),
equivalent-text formatting, frozen-dataclass immutability, and live
consistency between the hardcoded constants and the mace-f E_methodD CSVs.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import re

import pytest

from app.carbon import constants
from app.carbon import estimator
from app.carbon.estimator import (
    METHOD_MEASURED,
    METHOD_TIER_FALLBACK,
    CarbonEstimate,
    estimate,
)

CHINA_CI = 582.0


# ---------------------------------------------------------------------------
# Formula correctness
# ---------------------------------------------------------------------------


def test_formula_matches_hand_calculation():
    result = estimate("resnet_50", country="China", pue=2.0, calibration_factor=1.39)
    energy_per_inference = constants.INFERENCE_ENERGY_KWH_PER_1000["resnet_50"] / 1000.0
    expected = energy_per_inference * CHINA_CI * 2.0 * 1.39
    assert isinstance(result, CarbonEstimate)
    assert result.co2eq_g == pytest.approx(expected, rel=1e-12)
    assert result.energy_kwh == pytest.approx(energy_per_inference, rel=1e-12)
    assert result.method == METHOD_MEASURED


def test_defaults_apply_china_factory_pue_and_t3_calibration():
    result = estimate("resnet_50")
    energy_per_inference = constants.INFERENCE_ENERGY_KWH_PER_1000["resnet_50"] / 1000.0
    expected = energy_per_inference * CHINA_CI * constants.DEFAULT_PUE * 1.39
    assert result.co2eq_g == pytest.approx(expected, rel=1e-12)


def test_calibration_factor_defaults_to_model_tier():
    cases = {
        "linear_regression": 1.58,  # T1
        "xgboost": 1.43,  # T2
        "resnet_50": 1.39,  # T3
        "lstm": 1.39,  # T3
        "vit_b_16": 1.34,  # T4
    }
    for model, factor in cases.items():
        result = estimate(model, country="China", pue=1.0)
        energy_per_inference = constants.INFERENCE_ENERGY_KWH_PER_1000[model] / 1000.0
        assert result.co2eq_g == pytest.approx(energy_per_inference * CHINA_CI * factor, rel=1e-12)


def test_explicit_calibration_factor_overrides_tier_default():
    base = estimate("vit_b_16", country="China", pue=1.0)
    explicit = estimate("vit_b_16", country="China", pue=1.0, calibration_factor=2.5)
    assert explicit.co2eq_g == pytest.approx(base.co2eq_g / 1.34 * 2.5, rel=1e-12)


def test_model_name_is_normalized():
    result_a = estimate("ResNet-50", country="China", pue=1.0, calibration_factor=1.0)
    result_b = estimate("  resnet_50  ", country="China", pue=1.0, calibration_factor=1.0)
    assert result_a == result_b
    assert result_b.method == METHOD_MEASURED


# ---------------------------------------------------------------------------
# Unknown-model fallback (resnet18 -> resnet_50 T3 representative)
# ---------------------------------------------------------------------------


def test_unknown_model_falls_back_to_resnet50_with_warning(caplog, monkeypatch):
    monkeypatch.setattr(estimator, "_FALLBACK_WARNED_ONCE", set())
    with caplog.at_level(logging.WARNING):
        result = estimate("resnet18", country="China", pue=1.0, calibration_factor=1.0)
    assert "resnet_18" in caplog.text or "resnet18" in caplog.text
    assert constants.T3_REPRESENTATIVE_MODEL in caplog.text
    energy_per_inference = (
        constants.INFERENCE_ENERGY_KWH_PER_1000[constants.T3_REPRESENTATIVE_MODEL] / 1000.0
    )
    assert result.co2eq_g == pytest.approx(energy_per_inference * CHINA_CI, rel=1e-12)
    assert result.method == METHOD_TIER_FALLBACK


def test_fallback_warning_is_logged_once_per_process(caplog, monkeypatch):
    # LOW-2: repeated unknown-model requests must not warn on every call —
    # the fallback warning is emitted once per model name.
    monkeypatch.setattr(estimator, "_FALLBACK_WARNED_ONCE", set())
    with caplog.at_level(logging.WARNING):
        estimate("resnet18", country="China", pue=1.0, calibration_factor=1.0)
        estimate("resnet18", country="China", pue=1.0, calibration_factor=1.0)
        estimate("resnet18", country="China", pue=1.0, calibration_factor=1.0)
    assert caplog.text.count("falling back") == 1


def test_resnet18_is_not_in_measured_table():
    assert "resnet18" not in constants.INFERENCE_ENERGY_KWH_PER_1000


# ---------------------------------------------------------------------------
# Defensive validation
# ---------------------------------------------------------------------------


def test_zero_energy_raises(monkeypatch):
    monkeypatch.setitem(constants.INFERENCE_ENERGY_KWH_PER_1000, "zero_model", 0.0)
    with pytest.raises(ValueError, match="energy"):
        estimate("zero_model")


def test_negative_energy_raises(monkeypatch):
    monkeypatch.setitem(constants.INFERENCE_ENERGY_KWH_PER_1000, "negative_model", -1e-6)
    with pytest.raises(ValueError, match="energy"):
        estimate("negative_model")


def test_unknown_country_raises():
    with pytest.raises(ValueError, match="country"):
        estimate("resnet_50", country="Atlantis")


def test_nonpositive_pue_raises():
    with pytest.raises(ValueError, match="pue"):
        estimate("resnet_50", pue=0.0)
    with pytest.raises(ValueError, match="pue"):
        estimate("resnet_50", pue=-1.5)


def test_nonpositive_calibration_factor_raises():
    with pytest.raises(ValueError, match="calibration"):
        estimate("resnet_50", calibration_factor=0.0)
    with pytest.raises(ValueError, match="calibration"):
        estimate("resnet_50", calibration_factor=-0.5)


def test_invalid_model_name_raises():
    for bad in ("", "   ", 42):
        with pytest.raises(ValueError, match="model_name"):
            estimate(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Equivalent text
# ---------------------------------------------------------------------------


def test_equivalent_text_formats_across_magnitudes(monkeypatch):
    # With China CI, pue=1 and calibration=1, co2eq_g == energy_per_inf * 582,
    # and the phone-charge equivalent is co2eq_g / 0.1 seconds. Sub-10 ms
    # equivalents switch to milliseconds (sub-millisecond "seconds" are
    # meaningless precision theater).
    monkeypatch.setitem(constants.INFERENCE_ENERGY_KWH_PER_1000, "heavy", 17.2)
    monkeypatch.setitem(constants.INFERENCE_ENERGY_KWH_PER_1000, "medium", 1.0)
    monkeypatch.setitem(constants.INFERENCE_ENERGY_KWH_PER_1000, "light", 0.01)
    monkeypatch.setitem(constants.INFERENCE_ENERGY_KWH_PER_1000, "tiny", 0.001)
    cases = {
        "heavy": "≈ 手机充电 100 秒",  # 10.0104 g -> 100.1 s
        "medium": "≈ 手机充电 5.8 秒",  # 0.582 g -> 5.82 s
        "light": "≈ 手机充电 0.06 秒",  # 0.00582 g -> 0.0582 s
        "tiny": "≈ 手机充电 5.82 毫秒",  # 0.000582 g -> 0.00582 s -> 5.82 ms
    }
    for model, expected in cases.items():
        result = estimate(model, country="China", pue=1.0, calibration_factor=1.0)
        assert result.equivalent == expected


def test_equivalent_text_matches_expected_pattern():
    result = estimate("resnet_50")
    assert re.fullmatch(r"≈ 手机充电 [0-9.]+ (秒|毫秒)", result.equivalent)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_carbon_estimate_is_frozen():
    result = estimate("resnet_50")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.co2eq_g = 123.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.equivalent = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Constants sanity and live consistency with the mace-f CSVs
# ---------------------------------------------------------------------------


def test_default_constants_values():
    assert constants.DEFAULT_COUNTRY == "China"
    assert constants.GRID_CARBON_INTENSITY["China"] == 582.0
    assert constants.DEFAULT_PUE == 2.0
    assert constants.DEFAULT_CALIBRATION_FACTOR == 1.39
    assert constants.T3_REPRESENTATIVE_MODEL in constants.INFERENCE_ENERGY_KWH_PER_1000


def test_all_energy_constants_are_positive():
    assert len(constants.INFERENCE_ENERGY_KWH_PER_1000) >= 8
    for model, value in constants.INFERENCE_ENERGY_KWH_PER_1000.items():
        assert value > 0.0, model


def test_constants_match_macef_methodd_csvs(macef_results_dir):
    for model in ("resnet_50", "vit_b_16", "mobilenet_v2"):
        csv_path = macef_results_dir / f"E_methodD_{model}_emissions.csv"
        assert csv_path.is_file(), csv_path
        with open(csv_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, f"CSV has no data rows: {csv_path}"
        last_energy_kwh = float(rows[-1]["energy_consumed"])
        last_emissions_kg = float(rows[-1]["emissions"])
        # Hardcoded constant == last cumulative CodeCarbon row (kWh per 1000
        # inferences, method D uses n_samples=1000).
        assert constants.INFERENCE_ENERGY_KWH_PER_1000[model] == pytest.approx(
            last_energy_kwh, rel=1e-9
        )
        # Sanity: CodeCarbon emissions (kg) == energy (kWh) x 231 g/kWh (UK grid
        # intensity configured for the mace-f runs, PUE 1.0), divided by 1000.
        assert last_emissions_kg == pytest.approx(last_energy_kwh * 231.0 / 1000.0, rel=1e-6)
