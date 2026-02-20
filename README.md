# Music Emotion Predictor

Machine-learning pipeline for predicting music emotion classes (`Calm`, `Sad`, `Energetic`, `Happy`) from audio features.

The original notebook-based workflow is preserved in `music_emotion_prediction.ipynb`. This repository also includes a production-ready Python package and CLI for reproducible runs, robust evaluation, and artifact tracking.

## What is included

- Supervised models:
  - Decision Tree Classifier
  - K-Nearest Neighbors (KNN)
- Unsupervised model:
  - KMeans clustering (evaluated with ARI)
- Evaluation hardening:
  - Holdout metrics: accuracy, macro-F1, weighted-F1
  - Stratified cross-validation for Decision Tree and KNN
- Artifact persistence:
  - Saved scaler and trained models (`joblib`)
  - Run metadata and metrics JSON
  - Top feature importances and energy-cluster summary CSVs
- CI quality gate:
  - Lint (`ruff`), tests (`pytest`), package build

## Project structure

- `music_emotion_prediction.ipynb`: original exploratory notebook
- `src/music_emotion_predictor/`: reusable package and CLI
- `tests/`: automated tests
- `data/sample_music_dataset.csv`: tiny bundled sample for quick validation
- `.github/workflows/ci.yml`: CI pipeline
- `artifacts/`: generated run outputs (created at runtime)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For contributor tooling (lint + tests via extras):

```bash
pip install -e ".[dev]"
```

## Run the pipeline

Quick smoke run with bundled sample data:

```bash
music-emotion-predict --dataset data/sample_music_dataset.csv
```

Run with your full dataset export:

```bash
music-emotion-predict --dataset /path/to/278k_labelled_uri.csv --test-size 0.2 --random-state 42 --cv-folds 5
```

Disable artifact writes:

```bash
music-emotion-predict --no-save-artifacts
```

Print full metrics JSON in terminal:

```bash
music-emotion-predict --json
```

## Tests and quality

```bash
python3 -m pytest
ruff check src tests
```

## Artifact outputs

By default, each run writes a timestamped directory under `artifacts/` containing:

- `scaler.joblib`
- `decision_tree.joblib`
- `knn.joblib`
- `metrics.json`
- `metadata.json`
- `top_features.csv`
- `energy_cluster_summary.csv`

## Dataset notes

- Source (Kaggle): https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset
- Expected requirements:
  - CSV file must be non-empty
  - `labels` column must exist
  - feature columns must be numeric
- Optional columns like `uri`, `Unnamed: 0`, and `Unnamed: 0.1` are dropped automatically if present.
