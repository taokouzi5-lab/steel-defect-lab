"""Carbon constants for the DefectLens backend (T3.4).

All values are sourced from the mace-f benchmark (the sibling checkout at
``D:/claudecodetest/carbon_benchmark``, published as "MACE-F: Manufacturing
AI Carbon Estimation Framework"):

- Inference energy: CodeCarbon 3.2.7 CSV measurements under
  ``carbon_benchmark/results/E_methodD_<model>_emissions.csv`` (method D =
  baseline inference, 1000 samples per run, RTX 3070 Ti Laptop GPU).
- Grid carbon intensity / PUE / calibration factors / model tiers: copied
  from ``carbon_benchmark.config`` (sources: CO2 Signal API + IPCC regional
  averages; Lacoste et al. 2019; Patterson et al. 2022; Fischer 2025
  external-power-meter calibration of CodeCarbon's 25-37 % underestimation).
"""

# ---------------------------------------------------------------------------
# Measured inference energy (kWh per 1000 inferences)
# ---------------------------------------------------------------------------
# Extraction recipe (one-off, not shipped): for each E_methodD CSV, read the
# LAST data row's ``energy_consumed`` column — CodeCarbon rows are
# cumulative, so the last row is the total machine energy of the whole
# 1000-inference run, in kWh. (The ``emissions`` column is kg and satisfies
# emissions_kg == energy_kwh * 231 g/kWh / 1000, i.e. the UK grid intensity
# configured for those runs at PUE 1.0.)
INFERENCE_ENERGY_KWH_PER_1000: dict[str, float] = {
    "gpt_4o_mini": 2.2053048200960164e-05,  # local machine only; server-side cost excluded
    "linear_regression": 4.308263782690969e-05,
    "lstm": 3.254848103632711e-05,
    "mobilenet_v2": 2.3515822304850935e-05,
    "random_forest": 4.972218711017473e-05,
    "resnet_152": 9.250281779984237e-05,
    "resnet_50": 4.574800494862325e-05,
    "stable_diffusion": 0.028940927701536747,
    "vit_b_16": 0.00013036173725915169,
    "xgboost": 4.3941923591524906e-05,
}

# resnet18 was not part of the mace-f measurement matrix. It sits in the same
# T3 tier (standard DL, 10M-50M params) as resnet_50, so resnet_50's measured
# value is used as the T3-tier representative whenever a model has no
# measurement of its own (see estimator.estimate).
T3_REPRESENTATIVE_MODEL: str = "resnet_50"

# ---------------------------------------------------------------------------
# Grid carbon intensity by country/region (gCO2eq/kWh)
# ---------------------------------------------------------------------------
# Copied verbatim from carbon_benchmark.config.CARBON_INTENSITY.
GRID_CARBON_INTENSITY: dict[str, float] = {
    "Switzerland": 16.0,
    "Norway": 20.0,
    "France": 56.0,
    "Sweden": 13.0,
    "Finland": 54.0,
    "UK": 231.0,
    "Germany": 338.0,
    "Japan": 472.0,
    "South_Korea": 416.0,
    "USA_California": 204.0,
    "USA_Texas": 427.0,
    "USA_Iowa": 737.0,
    "China": 582.0,
    "India": 708.0,
    "Vietnam": 432.0,
    "Mexico": 431.0,
    "Serbia": 671.0,
    "South_Africa": 1009.0,
    "Ireland": 331.0,
    "Netherlands": 397.0,
}

# Deployment default: China grid (mace-f measured under the UK grid; the
# DefectLens manufacturing use case defaults to China).
DEFAULT_COUNTRY: str = "China"

# ---------------------------------------------------------------------------
# PUE and calibration factors
# ---------------------------------------------------------------------------
# carbon_benchmark.config.PUE["factory_floor"] = 2.00 (Lacoste 2019,
# Patterson 2022). Factory-floor is the deployment topology for this product.
DEFAULT_PUE: float = 2.0

# carbon_benchmark.config.CALIBRATION_FACTORS (Fischer 2025 validation
# methodology; corrects CodeCarbon's 25-37 % systematic underestimation).
TIER_CALIBRATION_FACTORS: dict[str, float] = {
    "T1": 1.58,
    "T2": 1.43,
    "T3": 1.39,
    "T4": 1.34,
    "T5": 1.25,
}

# carbon_benchmark.config.MODEL_TIERS.
MODEL_TIERS: dict[str, str] = {
    "linear_regression": "T1",
    "decision_tree": "T1",
    "random_forest": "T2",
    "xgboost": "T2",
    "mobilenet_v2": "T2",
    "resnet_50": "T3",
    "lstm": "T3",
    "resnet_152": "T4",
    "vit_b_16": "T4",
    "stable_diffusion": "T5",
    "gpt_4o_mini": "T5",
}

# T3-tier calibration factor; used when a model's tier is unknown.
DEFAULT_CALIBRATION_FACTOR: float = 1.39
DEFAULT_TIER: str = "T3"

# ---------------------------------------------------------------------------
# Equivalent-consumption constant
# ---------------------------------------------------------------------------
# Derived from the equivalence table in
# carbon_benchmark/results/analysis_summary.md, which consistently maps
# ~0.1 g CO2eq to one second of smartphone charging (0.01 g ≈ 0.1 s,
# 0.86 g ≈ 8 s, 7.66 g ≈ 75 s), i.e. ≈ 0.1 g CO2eq/s. We follow the mace-f
# 口径 for consistency with the published analysis; note this nominal rate
# (≈ 1.5 kW at UK grid intensity) is larger than a physical 5-20 W phone
# charger and should be read as the paper's illustrative benchmark.
PHONE_CHARGE_GCO2EQ_PER_SECOND: float = 0.1
