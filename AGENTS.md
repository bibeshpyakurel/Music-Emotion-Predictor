# AGENTS

## Project Overview
- Purpose: reproducible ML pipeline to predict music emotion classes from tabular audio features.
- Core package: `src/music_emotion_predictor/`.
- Legacy notebook: `music_emotion_prediction.ipynb` (not the source of truth for production behavior).

## Conventions (Strict)
- Keep production code under `src/music_emotion_predictor/`; tests under `tests/`.
- Use explicit type hints for public functions, dataclasses, and return values.
- Keep functions focused and deterministic where possible (`random_state` must be threaded through model/eval code).
- Lint standard is `ruff` configured in `pyproject.toml`; do not merge lint violations.
- Test standard is `pytest`; every behavior change must include or update tests.

## Run Pipeline
- Install:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -e ".[dev]"`
- CLI smoke run:
  - `music-emotion-predict --dataset data/sample_music_dataset.csv`
- Optional:
  - `music-emotion-predict --dataset data/sample_music_dataset.csv --json`
  - `music-emotion-predict --dataset data/sample_music_dataset.csv --no-save-artifacts`

## Artifact Writing
- Default output root: `artifacts/`.
- Each run writes a timestamped folder `artifacts/run_<UTC timestamp>/`.
- Expected files:
  - `scaler.joblib`
  - `decision_tree.joblib`
  - `knn.joblib`
  - `metrics.json`
  - `metadata.json`
  - `top_features.csv`
  - `energy_cluster_summary.csv`

## Required Pre-Finish Commands (Run Before Completing Any Change)
1. `python3 -m pytest -q`
2. `ruff check src tests`
3. `python3 -m build`

If any command fails, do not finalize silently: report failure and root cause.
