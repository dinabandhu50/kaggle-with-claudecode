# Kaggle E2E Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold a complete train-to-submission pipeline for Kaggle Playground Series S6E2 (tabular classification) using XGBoost and W&B experiment tracking.

**Architecture:** Raw CSVs in `data/raw/` are loaded and preprocessed by `src/data.py`, passed through a pass-through `src/features.py`, then trained via stratified K-fold in `src/models.py`. `src/train.py` orchestrates everything and logs to W&B. `src/predict.py` averages fold predictions and writes a timestamped submission CSV.

**Tech Stack:** Python 3.11+, uv, xgboost, scikit-learn, wandb, pandas, numpy, pyyaml, jupyter, pyarrow, pytest

---

## File Map

| File | Role |
|------|------|
| `pyproject.toml` | uv project + dependencies |
| `.gitignore` | Ignore data, submissions, venv, secrets |
| `.env.example` | WANDB_API_KEY placeholder |
| `configs/baseline.yaml` | Experiment config (model, hyperparams, CV, target column) |
| `src/__init__.py` | Empty — makes src a package |
| `src/data.py` | `load_train()`, `load_test()`, `preprocess()` |
| `src/features.py` | `get_features()` — pass-through for now |
| `src/models.py` | `cross_validate()` — stratified K-fold with XGBoost |
| `src/predict.py` | `generate_submission()` — averages fold models, writes CSV |
| `src/train.py` | CLI entry point — reads config, orchestrates pipeline, logs to W&B |
| `tests/test_data.py` | Tests for preprocess() |
| `tests/test_models.py` | Tests for cross_validate() on synthetic data |
| `tests/test_predict.py` | Tests for submission CSV shape and columns |
| `notebooks/01_eda.ipynb` | EDA template |
| `notebooks/02_feature_engineering.ipynb` | Feature engineering scratchpad (placeholder) |
| `notebooks/03_modeling.ipynb` | Model exploration scratchpad |

---

## Task 1: Initialize uv project and folder structure

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize uv project**

```bash
cd /media/db/DATA/Work/code_db/2026/kaggle-with-claudecode
uv init --no-readme
```

Expected: creates `pyproject.toml` and `.python-version`

- [ ] **Step 2: Add dependencies**

```bash
uv add xgboost scikit-learn pandas numpy wandb pyyaml pyarrow ipykernel
uv add --dev pytest jupyter
```

Expected: `uv.lock` created, `.venv/` created

- [ ] **Step 3: Create folder structure**

```bash
mkdir -p data/raw data/processed data/features
mkdir -p notebooks
mkdir -p src
mkdir -p configs
mkdir -p submissions
mkdir -p tests
```

- [ ] **Step 4: Create `src/__init__.py` and `tests/__init__.py`**

`src/__init__.py` — empty file
`tests/__init__.py` — empty file

- [ ] **Step 5: Create `.gitignore`**

```
# Python
.venv/
__pycache__/
*.pyc
.python-version

# Data — Kaggle data cannot be redistributed
data/raw/
data/processed/
data/features/

# Submissions
submissions/

# Secrets
.env

# Jupyter
.ipynb_checkpoints/

# uv
uv.lock
```

- [ ] **Step 6: Create `.env.example`**

```
WANDB_API_KEY=your_api_key_here
```

- [ ] **Step 7: Commit**

```bash
git init
git add pyproject.toml src/__init__.py tests/__init__.py .gitignore .env.example
git commit -m "feat: initialize uv project with folder structure"
```

---

## Task 2: Implement `src/data.py`

**Files:**
- Create: `src/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data.py`:

```python
import pandas as pd
import pytest
from src.data import preprocess


def test_preprocess_train_splits_target():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "feat_a": [0.1, 0.2, 0.3],
        "feat_b": [10, 20, 30],
        "target": [0, 1, 0],
    })
    X, y = preprocess(df, target_col="target", drop_cols=["id"])
    assert list(X.columns) == ["feat_a", "feat_b"]
    assert list(y) == [0, 1, 0]


def test_preprocess_test_returns_none_for_y():
    df = pd.DataFrame({
        "id": [1, 2],
        "feat_a": [0.1, 0.2],
        "feat_b": [10, 20],
    })
    X, y = preprocess(df, target_col="target", drop_cols=["id"])
    assert list(X.columns) == ["feat_a", "feat_b"]
    assert y is None


def test_preprocess_drops_specified_cols():
    df = pd.DataFrame({
        "id": [1],
        "extra": [99],
        "feat": [0.5],
        "target": [1],
    })
    X, y = preprocess(df, target_col="target", drop_cols=["id", "extra"])
    assert "id" not in X.columns
    assert "extra" not in X.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_data.py -v
```

Expected: `ImportError` — `src.data` not found

- [ ] **Step 3: Implement `src/data.py`**

```python
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "raw" / "train.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "raw" / "test.csv")


def preprocess(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series | None]:
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if target_col in df.columns:
        y = df[target_col].reset_index(drop=True)
        X = df.drop(columns=cols_to_drop + [target_col])
    else:
        y = None
        X = df.drop(columns=cols_to_drop)
    return X.reset_index(drop=True), y
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_data.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: add data loading and preprocessing"
```

---

## Task 3: Implement `src/features.py`

**Files:**
- Create: `src/features.py`

- [ ] **Step 1: Create pass-through feature pipeline**

```python
import pandas as pd


def get_features(X: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering pipeline. Pass-through until features are added."""
    return X
```

- [ ] **Step 2: Commit**

```bash
git add src/features.py
git commit -m "feat: add placeholder feature engineering pipeline"
```

---

## Task 4: Implement `src/models.py`

**Files:**
- Create: `src/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.models import cross_validate


def make_synthetic(n=100, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def test_cross_validate_returns_correct_shapes():
    X, y = make_synthetic()
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    models, oof_preds, fold_scores = cross_validate(X, y, params, n_splits=3)
    assert len(models) == 3
    assert oof_preds.shape == (100,)
    assert len(fold_scores) == 3


def test_cross_validate_oof_preds_in_unit_interval():
    X, y = make_synthetic()
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    _, oof_preds, _ = cross_validate(X, y, params, n_splits=3)
    assert np.all(oof_preds >= 0) and np.all(oof_preds <= 1)


def test_fold_scores_are_valid_auc():
    X, y = make_synthetic()
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    _, _, fold_scores = cross_validate(X, y, params, n_splits=3)
    assert all(0 <= s <= 1 for s in fold_scores)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_models.py -v
```

Expected: `ImportError` — `src.models` not found

- [ ] **Step 3: Implement `src/models.py`**

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_splits: int = 5,
) -> tuple[list[xgb.XGBClassifier], np.ndarray, list[float]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_scores = []
    models = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss",
            early_stopping_rounds=50,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_scores.append(roc_auc_score(y_val, oof_preds[val_idx]))
        models.append(model)

    return models, oof_preds, fold_scores
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_models.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: add XGBoost cross-validation"
```

---

## Task 5: Implement `src/predict.py`

**Files:**
- Create: `src/predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_predict.py`:

```python
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from src.predict import build_submission


def test_build_submission_shape():
    ids = pd.Series([101, 102, 103])
    preds = np.array([0.1, 0.9, 0.5])
    df = build_submission(ids, preds, target_col="target")
    assert list(df.columns) == ["id", "target"]
    assert len(df) == 3


def test_build_submission_values():
    ids = pd.Series([1, 2])
    preds = np.array([0.3, 0.7])
    df = build_submission(ids, preds, target_col="label")
    assert list(df["label"]) == [0.3, 0.7]
    assert list(df["id"]) == [1, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_predict.py -v
```

Expected: `ImportError` — `src.predict` not found

- [ ] **Step 3: Implement `src/predict.py`**

```python
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"


def build_submission(
    ids: pd.Series,
    preds: np.ndarray,
    target_col: str,
) -> pd.DataFrame:
    return pd.DataFrame({"id": ids, target_col: preds})


def generate_submission(
    models: list,
    X_test: pd.DataFrame,
    ids: pd.Series,
    target_col: str,
    experiment_name: str,
) -> Path:
    preds = np.mean(
        [m.predict_proba(X_test)[:, 1] for m in models], axis=0
    )
    submission = build_submission(ids, preds, target_col)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SUBMISSIONS_DIR / f"{timestamp}_{experiment_name}.csv"
    submission.to_csv(path, index=False)
    print(f"Submission saved → {path}")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_predict.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: add submission generation"
```

---

## Task 6: Create `configs/baseline.yaml`

**Files:**
- Create: `configs/baseline.yaml`

- [ ] **Step 1: Create baseline config**

```yaml
experiment_name: baseline_xgb

data:
  target_col: target      # update to actual competition target column name after downloading data
  drop_cols: [id]

model: xgboost
hyperparams:
  n_estimators: 500
  learning_rate: 0.05
  max_depth: 6
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42

cv:
  n_splits: 5

wandb:
  project: playground-series-s6e2
```

- [ ] **Step 2: Commit**

```bash
git add configs/baseline.yaml
git commit -m "feat: add baseline experiment config"
```

---

## Task 7: Implement `src/train.py`

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Implement `src/train.py`**

```python
import sys
from pathlib import Path

import wandb
import yaml
from sklearn.metrics import roc_auc_score

from src.data import load_train, load_test, preprocess
from src.features import get_features
from src.models import cross_validate
from src.predict import generate_submission


def run(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["experiment_name"],
        config=cfg,
    )

    # Load and preprocess
    train_df = load_train()
    X, y = preprocess(train_df, cfg["data"]["target_col"], cfg["data"]["drop_cols"])
    X = get_features(X)

    # Train
    models, oof_preds, fold_scores = cross_validate(
        X, y, cfg["hyperparams"], n_splits=cfg["cv"]["n_splits"]
    )

    # Log to W&B
    for i, score in enumerate(fold_scores):
        wandb.log({f"fold_{i + 1}_auc": score})
    oof_score = roc_auc_score(y, oof_preds)
    wandb.log({"oof_auc": oof_score})
    print(f"OOF AUC: {oof_score:.5f}")

    # Generate submission
    test_df = load_test()
    ids = test_df["id"]
    X_test, _ = preprocess(test_df, cfg["data"]["target_col"], cfg["data"]["drop_cols"])
    X_test = get_features(X_test)

    path = generate_submission(
        models, X_test, ids, cfg["data"]["target_col"], cfg["experiment_name"]
    )
    wandb.save(str(path))
    wandb.finish()


if __name__ == "__main__":
    run(sys.argv[1])
```

- [ ] **Step 2: Dry-run import check (no data needed)**

```bash
uv run python -c "from src.train import run; print('import OK')"
```

Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: add training orchestration with W&B logging"
```

---

## Task 8: Create notebook templates

**Files:**
- Create: `notebooks/01_eda.ipynb`
- Create: `notebooks/02_feature_engineering.ipynb`
- Create: `notebooks/03_modeling.ipynb`

- [ ] **Step 1: Create `notebooks/01_eda.ipynb`**

```json
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "matplotlib.rcParams['figure.figsize'] = (12, 5)\n",
    "\n",
    "from src.data import load_train, load_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train = load_train()\n",
    "test = load_test()\n",
    "print(f'Train: {train.shape}, Test: {test.shape}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Missing values\n",
    "train.isnull().sum()[train.isnull().sum() > 0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Target distribution — update 'target' to actual column name\n",
    "train['target'].value_counts(normalize=True)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Create `notebooks/02_feature_engineering.ipynb`**

```json
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from src.data import load_train, preprocess\n",
    "from src.features import get_features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Feature Engineering Scratchpad\n\nPrototype features here. Once stable, move them into `src/features.py` as named functions and replace cells with imports."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 3: Create `notebooks/03_modeling.ipynb`**

```json
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import yaml\n",
    "from sklearn.metrics import roc_auc_score\n",
    "\n",
    "from src.data import load_train, preprocess\n",
    "from src.features import get_features\n",
    "from src.models import cross_validate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cfg = yaml.safe_load(open('../configs/baseline.yaml'))\n",
    "train = load_train()\n",
    "X, y = preprocess(train, cfg['data']['target_col'], cfg['data']['drop_cols'])\n",
    "X = get_features(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "models, oof_preds, fold_scores = cross_validate(X, y, cfg['hyperparams'], n_splits=cfg['cv']['n_splits'])\n",
    "print(f'Fold AUCs: {[f\"{s:.4f}\" for s in fold_scores]}')\n",
    "print(f'OOF AUC:  {roc_auc_score(y, oof_preds):.5f}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/
git commit -m "feat: add EDA, feature engineering, and modeling notebook templates"
```

---

## Task 9: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Rewrite CLAUDE.md**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with project conventions and commands"
```

---

## Task 10: Run full test suite

- [ ] **Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: 8 tests PASS (3 data + 3 models + 2 predict)

- [ ] **Step 2: Verify imports from project root**

```bash
uv run python -c "from src.data import preprocess; from src.features import get_features; from src.models import cross_validate; from src.predict import generate_submission; from src.train import run; print('all imports OK')"
```

Expected: `all imports OK`

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: verify full test suite passes"
```
