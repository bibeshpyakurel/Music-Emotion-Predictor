from pathlib import Path

import pandas as pd
import pytest

from music_emotion_predictor.data import DatasetError, prepare_supervised_data
from music_emotion_predictor.pipeline import run_pipeline


def test_prepare_supervised_data_requires_labels() -> None:
    df = pd.DataFrame({"energy": [0.1, 0.2], "tempo": [100, 110]})
    with pytest.raises(DatasetError):
        prepare_supervised_data(df)


def test_run_pipeline_smoke() -> None:
    result = run_pipeline(
        Path("data/sample_music_dataset.csv"),
        test_size=0.25,
        random_state=42,
        cv_folds=3,
        artifacts_dir=None,
    )

    assert 0.0 <= result.decision_tree_accuracy <= 1.0
    assert 0.0 <= result.knn_accuracy <= 1.0
    assert -1.0 <= result.kmeans_ari <= 1.0
    assert len(result.top_features) > 0
    assert set(result.energy_cluster_summary.columns) == {
        "energy",
        "valence",
        "tempo",
        "danceability",
    }
    assert "decision_tree" in result.cross_validation
    assert "knn" in result.cross_validation


def test_run_pipeline_saves_artifacts(tmp_path: Path) -> None:
    result = run_pipeline(
        Path("data/sample_music_dataset.csv"),
        test_size=0.25,
        random_state=42,
        cv_folds=3,
        artifacts_dir=tmp_path,
    )

    assert result.artifact_dir is not None
    run_dir = Path(result.artifact_dir)
    assert run_dir.exists()
    assert (run_dir / "scaler.joblib").exists()
    assert (run_dir / "decision_tree.joblib").exists()
    assert (run_dir / "knn.joblib").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "metrics.json").exists()


def test_run_pipeline_rejects_invalid_cv_folds() -> None:
    with pytest.raises(ValueError):
        run_pipeline(
            Path("data/sample_music_dataset.csv"),
            test_size=0.25,
            random_state=42,
            cv_folds=999,
            artifacts_dir=None,
        )
