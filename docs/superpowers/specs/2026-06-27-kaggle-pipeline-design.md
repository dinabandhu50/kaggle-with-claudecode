# Kaggle Competition Pipeline Design

**Competition:** Playground Series S6E2 (tabular classification)
**Date:** 2026-06-27

## Overview

A modular, notebook-first pipeline for a Kaggle tabular classification competition. Exploration lives in notebooks; reusable logic lives in `src/`; experiments are tracked via Weights & Biases; `uv` manages the Python environment.

## Folder Structure

```
kaggle-with-claudecode/
├── data/
│   ├── raw/          # original competition files — gitignored
│   ├── processed/    # cleaned/merged data
│   └── features/     # engineered feature sets (saved as parquet)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data.py       # load and preprocess raw data
│   ├── features.py   # feature engineering functions
│   ├── models.py     # model definitions and wrappers
│   ├── train.py      # training loop with W&B logging
│   └── predict.py    # inference and submission generation
├── configs/
│   └── baseline.yaml # experiment config (model type, features, hyperparams)
├── submissions/       # timestamped submission CSVs — gitignored
├── docs/
│   └── superpowers/specs/
├── CLAUDE.md
├── pyproject.toml    # uv-managed dependencies
└── .gitignore
```

## Key Conventions

### Notebook ↔ src Sync

`src/features.py` is the single source of truth for all feature engineering logic.

Workflow:
1. Prototype a feature idea freely in `02_feature_engineering.ipynb`
2. Once stable, extract the logic into a named function in `src/features.py`
3. Replace the notebook cell with an import: `from src.features import <function>`
4. `train.py` imports the same functions — no drift possible

Notebooks import `src/` via `sys.path.append('..')` at the top.

### Experiment Config

Each experiment run is driven by a YAML config in `configs/`. Example:

```yaml
experiment_name: baseline_xgb
model: xgboost
hyperparams:
  n_estimators: 500
  learning_rate: 0.05
  max_depth: 6
```

`train.py` reads the config, trains the model, logs everything to W&B, and saves the submission CSV to `submissions/<timestamp>_<experiment_name>.csv`.

Feature engineering is out of scope for the initial pipeline — `src/features.py` is a placeholder for future additions.

### W&B Integration

- W&B project name: `playground-series-s6e2`
- Credentials provided separately via environment variable (`WANDB_API_KEY`)
- Each run logs: config params, CV scores per fold, final OOF score

### Data Flow

```
data/raw/ → src/data.py → data/processed/ → src/features.py → data/features/ → src/train.py → submissions/
```

### Package Management

`uv` manages the virtualenv. Core dependencies: `xgboost`, `scikit-learn`, `pandas`, `numpy`, `wandb`, `jupyter`, `pyyaml`, `pyarrow`.

## What Is Not In Scope Now

- Ensembling / stacking (added later as needed)
- AutoML or hyperparameter search automation
- Docker / containerization
