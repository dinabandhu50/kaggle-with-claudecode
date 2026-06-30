# Experiment Plan — Playground Series S6E2 (Heart Disease)

## ROI Priority

| Phase | Expected gain | Status |
|---|---|---|
| **1. Feature Engineering** | +0.01 – 0.03 AUC | In progress |
| **2. Hyperparameter Tuning (Optuna)** | +0.003 – 0.008 AUC | Pending |
| **3. Model Ensembling (XGB + LGB + CatBoost)** | +0.001 – 0.003 AUC | Pending |

Baseline OOF AUC: **0.95526**  
Baseline public score: **0.95343**

---

## Phase 1 — Feature Engineering

All FE experiments use the same XGB baseline hyperparams.  
OOF predictions saved as `submissions/oof_<experiment_name>.npy` for future ensembling.

### Feature Groups

**Flags** (strong categorical signals → binary)
- `high_risk_thallium`: Thallium == 7 (81.5% Presence rate)
- `typical_chest_pain`: Chest_pain_type == 4 (69.7%)
- `multiple_vessels`: Number_of_vessels_fluro > 1 (≥90%)
- `high_slope_risk`: Slope_of_ST >= 2 (69-72%)

**Interactions** (strong × strong)
- `vessels_x_angina`: Number_of_vessels_fluro × Exercise_angina
- `vessels_x_thallium_risk`: Number_of_vessels_fluro × high_risk_thallium
- `angina_x_chest_pain`: Exercise_angina × Chest_pain_type
- `chest_pain_x_sex`: Chest_pain_type × Sex

**Composite risk score**
- `cardiac_risk_score`: weighted sum of top 5 predictors

**Continuous engineering**
- `age_hr_ratio`: Age / Max_HR (aging heart under stress)
- `st_x_slope`: ST_depression × Slope_of_ST
- `bp_age`: BP × Age / 1000

**Polynomial**
- `vessels_sq`: Number_of_vessels_fluro² (amplifies near-linear gradient 0→3)
- `st_sq`: ST_depression²

**Binning**
- `age_bin`: 4 buckets (<45, 45-55, 55-65, 65+)
- `bp_bin`: 4 buckets (normal/elevated/stage1/stage2)
- `hr_bin`: 4 buckets by max HR ranges

### FE Experiments

| Config | Description | OOF AUC |
|---|---|---|
| `baseline_xgb` | Raw features, no FE | 0.95526 |
| `xgb_fe_v1` | All feature groups above | TBD |
| `xgb_fe_v2` | Pruned to top features by importance | TBD |

---

## Phase 2 — Hyperparameter Tuning (Optuna)

Run per model on the **best FE config** from Phase 1.  
100 Optuna trials, pruning with MedianPruner.

Search space (XGBoost):
- `n_estimators`: 300–2000
- `learning_rate`: 0.005–0.3 (log)
- `max_depth`: 3–10
- `subsample`: 0.5–1.0
- `colsample_bytree`: 0.4–1.0
- `min_child_weight`: 1–10
- `gamma`: 0–5
- `reg_alpha`: 1e-8–10 (log)
- `reg_lambda`: 1e-8–10 (log)

| Config | Description | OOF AUC |
|---|---|---|
| `xgb_hp_v1` | Optuna-tuned on best FE | TBD |

---

## Phase 3 — Hyperparameter Tuning: LightGBM & CatBoost

Same process as Phase 2 but for LGB and CatBoost.  
Each model saves its own OOF predictions.

| Config | Description | OOF AUC |
|---|---|---|
| `lgb_fe_best` | Best FE on LGB | TBD |
| `lgb_hp_v1` | Optuna-tuned LGB | TBD |
| `cat_fe_best` | Best FE on CatBoost | TBD |
| `cat_hp_v1` | Optuna-tuned CatBoost | TBD |

---

## Phase 4 — Model Ensembling

Only after all three best models are in hand.

Strategies to try:
1. **Simple average**: mean of 3 OOF probability arrays
2. **Weighted average**: optimize weights on OOF AUC
3. **Stacking**: logistic regression meta-learner on OOF preds

| Config | Description | OOF AUC |
|---|---|---|
| `ensemble_avg` | Simple average XGB+LGB+CAT | TBD |
| `ensemble_weighted` | Optuna-optimized weights | TBD |
| `ensemble_stack` | Logistic stacker | TBD |

---

## OOF Files (for ensembling)

All saved to `submissions/`:
- `oof_<experiment_name>.npy` — OOF predictions array (630,000 floats)
- `oof_train_ids.npy` — train IDs in same row order (saved once)
