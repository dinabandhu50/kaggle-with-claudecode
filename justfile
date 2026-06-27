# Kaggle Playground Series S6E2 — common commands

# Install / sync all dependencies
sync:
    uv sync

# Run test suite
test:
    uv run pytest tests/ -v

# Run a training experiment (default: baseline config)
train config="configs/baseline.yaml":
    uv run python src/train.py {{config}}

# Download competition data from Kaggle
download:
    uv run python scripts/download_data.py

# Launch Jupyter Notebook
notebook:
    uv run jupyter notebook notebooks/

# Launch JupyterLab
lab:
    uv run jupyter lab notebooks/
