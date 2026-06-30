# Kaggle Playground Series S6E2 — Heart Disease Classification

Binary classification competition on a synthetic heart disease dataset.  
**Best public score: 0.95361 (XGB) / LGB submitted**

## Competition

[Playground Series S6E2 — Predicting Heart Disease](https://www.kaggle.com/competitions/playground-series-s6e2)

- Task: Binary classification (`Absence` / `Presence` of heart disease)
- Metric: ROC-AUC
- Train: 630,000 rows × 13 features | Test: 270,000 rows

## Results

| Experiment | OOF AUC | Public AUC | Notes |
|---|---|---|---|
| `baseline_xgb` | 0.95526 | 0.95343 | Raw features, CPU |
| `xgb_fe_v1` | 0.95511 | — | All 41 engineered features |
| `xgb_fe_v2` | 0.95517 | — | Pruned top-27 features |
| `xgb_hp_v1` | 0.95540 | 0.95361 | XGBoost CUDA + Optuna 50 trials |
| `lgb_fe_v1` | 0.95501 | — | LightGBM CUDA baseline |
| `lgb_hp_v1` | **0.95542** | submitted | LightGBM CUDA + Optuna 50 trials |

## Experiment Roadmap

```
Phase 1 — Feature Engineering  (highest ROI: +0.01–0.03)
Phase 2 — Hyperparameter Tuning via Optuna  (+0.003–0.008)
Phase 3 — CatBoost (pending)
Phase 4 — Model Ensembling XGB + LGB + CatBoost  (+0.001–0.003)
```

See [`docs/experiment-plan.md`](docs/experiment-plan.md) for full details.

## Project Layout

```
├── src/
│   ├── data.py        # load_train / load_test / preprocess
│   ├── features.py    # feature engineering pipeline (v1 full, v2 pruned)
│   ├── models.py      # cross_validate — XGBoost CUDA + LightGBM CUDA
│   ├── train.py       # CLI training entry point, saves OOF predictions
│   ├── tune.py        # Optuna HPT — model-agnostic, early-stopping ceiling
│   └── predict.py     # generate_submission — averages fold model predictions
├── configs/           # one YAML per experiment
├── notebooks/         # 01 EDA → 02 features → 03 modeling
├── scripts/           # data download
├── docs/              # experiment plan and design specs
├── data/raw/          # competition CSVs (gitignored)
└── submissions/       # timestamped CSVs + OOF .npy files (gitignored)
```

## Setup

Requires [uv](https://github.com/astral-sh/uv) and [just](https://github.com/casey/just).

```bash
# 1. Clone
git clone https://github.com/dinabandhu50/kaggle-with-claudecode
cd kaggle-with-claudecode

# 2. Install dependencies
just sync

# 3. Set credentials
cp .envrc.example .envrc
# fill in KAGGLE_USERNAME, KAGGLE_KEY, WANDB_API_KEY
direnv allow

# 4. Download competition data
just download

# 5. Run baseline training
just train

# 6. Run a specific experiment
just train configs/xgb_hp_v1.yaml
```

## Running Optuna Hyperparameter Tuning

```bash
PYTHONPATH=. uv run python src/tune.py configs/lgb_hp_v1.yaml
```

Runs 50 Optuna trials (3-fold CV on 300k subsample for speed), then retrains the best params on full 5-fold CV. Saves OOF predictions to `submissions/oof_<name>.npy` for future ensembling.

## Feature Engineering

`src/features.py` implements two pipelines:

- `get_features(X)` — full v1 pipeline (41 features): flags, interactions, composite risk score, ratios, polynomial, bins
- `get_features_v2(X)` — pruned v2 (27 features): top features by XGBoost importance (cutoff > 0.003)

Key engineered features:
- `high_risk_thallium` — Thallium == 7 (81.5% Presence rate, importance 0.41)
- `cardiac_risk_score` — weighted sum of top 5 clinical predictors
- `vessels_x_slope`, `chest_pain_x_sex` — strong interaction features

## GPU Acceleration

Both models run on CUDA:
- **XGBoost**: `device="cuda"`
- **LightGBM**: `device="cuda"` (built from source with `-DUSE_CUDA=1`)

Optuna speed: ~24s/trial on GTX 1660 Ti (6GB).

## Experiment Tracking

All runs logged to [W&B project `playground-series-s6e2`](https://wandb.ai/dbweapon/playground-series-s6e2).  
OOF predictions saved as `submissions/oof_<experiment_name>.npy` for ensembling.
