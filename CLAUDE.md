# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Competition

Kaggle Playground Series S6E2 — tabular binary classification.
Competition data goes in `data/raw/` (gitignored — cannot be redistributed).

## Environment

Managed with `uv`. Use `just` for common tasks (requires [just](https://github.com/casey/just)).

```bash
just sync        # install all dependencies
just download    # download competition data from Kaggle
just train       # run baseline training experiment
just train configs/my_experiment.yaml   # run a specific config
just test        # run tests
just notebook    # launch Jupyter Notebook
just lab         # launch JupyterLab
```

Credentials are loaded via [direnv](https://direnv.net/) from `.envrc` (gitignored). Copy `.envrc.example` → `.envrc` and fill in your keys. Run `direnv allow` once after creating the file.

## Project Layout

- `src/` — reusable Python modules (data loading, features, model, train, predict)
- `notebooks/` — numbered exploration notebooks (01 EDA → 02 features → 03 modeling)
- `configs/` — one YAML per experiment
- `scripts/` — utility scripts (data download)
- `submissions/` — timestamped output CSVs (gitignored)
- `data/raw/` — original competition CSVs (gitignored)

## Notebook ↔ src Convention

Prototype features in `notebooks/02_feature_engineering.ipynb`, then extract stable functions into `src/features.py`. Replace notebook cells with imports. `src/features.py` is always the source of truth — `train.py` and notebooks import the same functions.

## Experiment Tracking

W&B project: `playground-series-s6e2`. `WANDB_API_KEY` is set via `.envrc`. Each run logs fold AUC scores and OOF AUC.

## Before Adding a New Experiment

1. Copy `configs/baseline.yaml` → `configs/<name>.yaml`
2. Edit hyperparams / features
3. Run: `uv run python src/train.py configs/<name>.yaml`
