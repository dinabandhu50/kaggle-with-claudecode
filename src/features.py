import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual feature-group builders
# All functions receive a copy of X with sanitized column names (spaces → _).
# ---------------------------------------------------------------------------

def add_flag_features(X: pd.DataFrame) -> pd.DataFrame:
    """Binary flags from strong categorical predictors."""
    # Thallium==7 → 81.5% Presence rate
    X["high_risk_thallium"] = (X["Thallium"] == 7).astype(np.int8)
    # Thallium==3 → 19.8% Presence (low risk anchor)
    X["normal_thallium"] = (X["Thallium"] == 3).astype(np.int8)
    # Chest_pain_type==4 (typical angina) → 69.7% Presence
    X["typical_chest_pain"] = (X["Chest_pain_type"] == 4).astype(np.int8)
    # Multiple vessels → ≥90% Presence
    X["multiple_vessels"] = (X["Number_of_vessels_fluro"] > 1).astype(np.int8)
    # Any vessel involvement
    X["any_vessel"] = (X["Number_of_vessels_fluro"] > 0).astype(np.int8)
    # Slope of ST ≥ 2 → 69-72% Presence
    X["high_slope_risk"] = (X["Slope_of_ST"] >= 2).astype(np.int8)
    # EKG == 2
    X["abnormal_ekg"] = (X["EKG_results"] == 2).astype(np.int8)
    return X


def add_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative interactions between the strongest predictors."""
    # Vessels × Angina — both near-ceiling predictors, combined even stronger
    X["vessels_x_angina"] = X["Number_of_vessels_fluro"] * X["Exercise_angina"]
    # Vessels × high-risk Thallium
    X["vessels_x_thallium_risk"] = X["Number_of_vessels_fluro"] * X["high_risk_thallium"]
    # Angina × Chest pain type
    X["angina_x_chest_pain"] = X["Exercise_angina"] * X["Chest_pain_type"]
    # Chest pain × Sex
    X["chest_pain_x_sex"] = X["Chest_pain_type"] * X["Sex"]
    # Vessels × Slope
    X["vessels_x_slope"] = X["Number_of_vessels_fluro"] * X["Slope_of_ST"]
    # Thallium risk × Angina
    X["thallium_risk_x_angina"] = X["high_risk_thallium"] * X["Exercise_angina"]
    # Typical chest pain × Angina
    X["typical_pain_x_angina"] = X["typical_chest_pain"] * X["Exercise_angina"]
    return X


def add_risk_score(X: pd.DataFrame) -> pd.DataFrame:
    """Composite cardiac risk score from top predictors.

    Weights derived from approximate log-odds of each binary feature
    against the ~45% base rate of Presence.
    """
    X["cardiac_risk_score"] = (
        X["Number_of_vessels_fluro"] * 1.0   # strong gradient 0→3
        + X["Exercise_angina"] * 1.8          # angina is very discriminative
        + X["high_risk_thallium"] * 1.5       # thallium 7
        + X["typical_chest_pain"] * 1.2       # chest pain type 4
        + X["Sex"] * 0.8                      # males at higher risk
        + X["high_slope_risk"] * 0.6
        + X["abnormal_ekg"] * 0.3
    )
    return X


def add_continuous_features(X: pd.DataFrame) -> pd.DataFrame:
    """Ratios and products from continuous variables."""
    # Age / Max_HR: higher = heart under strain relative to age
    X["age_hr_ratio"] = X["Age"] / (X["Max_HR"] + 1e-6)
    # ST depression amplified by slope severity
    X["st_x_slope"] = X["ST_depression"] * X["Slope_of_ST"]
    # BP × Age interaction (older hypertensives at higher risk)
    X["bp_age"] = X["BP"] * X["Age"] / 1000.0
    # Max HR deficit from age-predicted max (220 - Age rule)
    X["hr_reserve"] = (220 - X["Age"]) - X["Max_HR"]
    # ST depression × vessels
    X["st_x_vessels"] = X["ST_depression"] * X["Number_of_vessels_fluro"]
    return X


def add_polynomial_features(X: pd.DataFrame) -> pd.DataFrame:
    """Squared terms to capture non-linear effects."""
    X["vessels_sq"] = X["Number_of_vessels_fluro"] ** 2
    X["st_sq"] = X["ST_depression"] ** 2
    X["age_sq"] = (X["Age"] / 100.0) ** 2   # scaled to avoid large values
    return X


def add_bin_features(X: pd.DataFrame) -> pd.DataFrame:
    """Discretize continuous variables into clinically meaningful buckets."""
    X["age_bin"] = pd.cut(
        X["Age"],
        bins=[0, 45, 55, 65, 100],
        labels=[0, 1, 2, 3],
    ).astype(np.int8)

    X["bp_bin"] = pd.cut(
        X["BP"],
        bins=[0, 120, 130, 140, 300],
        labels=[0, 1, 2, 3],
    ).astype(np.int8)

    X["hr_bin"] = pd.cut(
        X["Max_HR"],
        bins=[0, 100, 130, 160, 250],
        labels=[0, 1, 2, 3],
    ).astype(np.int8)

    X["st_bin"] = pd.cut(
        X["ST_depression"],
        bins=[-0.1, 0.0, 1.0, 2.5, 10.0],
        labels=[0, 1, 2, 3],
    ).astype(np.int8)

    return X


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_features(X: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline (v1 — all groups, for exploration).

    Adds ~30 engineered features on top of the 13 raw features.
    All groups are always applied; call individual add_* functions
    directly if you need a subset.
    """
    X = X.copy()
    # Order matters: flags must come before interactions and risk score
    X = add_flag_features(X)
    X = add_interaction_features(X)
    X = add_risk_score(X)
    X = add_continuous_features(X)
    X = add_polynomial_features(X)
    X = add_bin_features(X)
    return X


# Top features selected from importance analysis of xgb_fe_v1
# Cutoff: importance > 0.003 (drops 22 low-signal features)
_TOP_FEATURES_V2 = [
    # Raw features (all retained — XGBoost is regularised)
    "Age", "Sex", "Chest_pain_type", "BP", "Cholesterol",
    "FBS_over_120", "EKG_results", "Max_HR", "Exercise_angina",
    "ST_depression", "Slope_of_ST", "Number_of_vessels_fluro", "Thallium",
    # Engineered: Thallium flags (top-2 by importance)
    "high_risk_thallium", "normal_thallium",
    # Composite risk score
    "cardiac_risk_score",
    # Chest-pain flags and interactions
    "typical_chest_pain", "chest_pain_x_sex",
    # HR binning
    "hr_bin",
    # Slope interactions
    "vessels_x_slope", "st_x_slope", "high_slope_risk",
    # Continuous ratios
    "age_hr_ratio",
    # Pain × angina interaction
    "typical_pain_x_angina",
    # Vessel presence
    "any_vessel",
    # Polynomial
    "st_sq",
]


def get_features_v2(X: pd.DataFrame) -> pd.DataFrame:
    """Pruned feature pipeline (v2 — top features only).

    Runs the full v1 pipeline then drops features with importance < 0.003
    as measured on the xgb_fe_v1 run. Reduces feature count from 41 → 27,
    improving the effective colsample_bytree ratio.
    """
    X = get_features(X)
    return X[[c for c in _TOP_FEATURES_V2 if c in X.columns]]
