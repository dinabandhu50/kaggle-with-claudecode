# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Competition

Kaggle Playground Series S6E2 — tabular binary classification.
Competition data goes in `data/raw/` (gitignored — cannot be redistributed).

## Environment

Managed with `uv`. Always prefix Python/pytest commands with `uv run`.

```bash
uv sync          # install all dependencies
uv run pytest    # run tests
uv run python src/train.py configs/baseline.yaml   # run a training experiment
```

## Project Layout

- `src/` — reusable Python modules (data loading, features, model, train, predict)
- `notebooks/` — numbered exploration notebooks (01 EDA → 02 features → 03 modeling)
- `configs/` — one YAML per experiment
- `submissions/` — timestamped output CSVs (gitignored)
- `data/raw/` — original competition CSVs (gitignored)

## Notebook ↔ src Convention

Prototype features in `notebooks/02_feature_engineering.ipynb`, then extract stable functions into `src/features.py`. Replace notebook cells with imports. `src/features.py` is always the source of truth — `train.py` and notebooks import the same functions.

## Experiment Tracking

W&B project: `playground-series-s6e2`. Set `WANDB_API_KEY` in `.env` (see `.env.example`). Each run logs fold AUC scores and OOF AUC.

## Before Adding a New Experiment

1. Copy `configs/baseline.yaml` → `configs/<name>.yaml`
2. Edit hyperparams / features
3. Run: `uv run python src/train.py configs/<name>.yaml`
