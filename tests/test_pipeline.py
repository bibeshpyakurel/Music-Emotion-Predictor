import json
from pathlib import Path

import pandas as pd
import pytest

from music_emotion_predictor import cli
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

    assert 0.0 <= result.majority_baseline_metrics.accuracy <= 1.0
    assert 0.0 <= result.decision_tree_accuracy <= 1.0
    assert 0.0 <= result.knn_accuracy <= 1.0
    assert 0.0 <= result.logistic_regression_metrics.accuracy <= 1.0
    assert -1.0 <= result.kmeans_ari <= 1.0
    assert len(result.top_features) > 0
    assert set(result.energy_cluster_summary.columns) == {
        "energy",
        "valence",
        "tempo",
        "danceability",
    }
    assert "majority_baseline" in result.cross_validation
    assert "decision_tree" in result.cross_validation
    assert "knn" in result.cross_validation
    assert "logistic_regression" in result.cross_validation


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
    assert (run_dir / "majority_baseline.joblib").exists()
    assert (run_dir / "decision_tree.joblib").exists()
    assert (run_dir / "knn.joblib").exists()
    assert (run_dir / "logistic_regression.joblib").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "confusion_matrix.csv").exists()
    assert (run_dir / "classification_report.json").exists()
    assert (run_dir / "run_summary.md").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "majority_baseline" in metrics
    assert "decision_tree" in metrics
    assert "knn" in metrics
    assert "logistic_regression" in metrics

    report = json.loads((run_dir / "classification_report.json").read_text(encoding="utf-8"))
    assert "majority_baseline" in report
    assert "decision_tree" in report
    assert "knn" in report
    assert "logistic_regression" in report


def test_run_pipeline_rejects_invalid_cv_folds() -> None:
    with pytest.raises(ValueError):
        run_pipeline(
            Path("data/sample_music_dataset.csv"),
            test_size=0.25,
            random_state=42,
            cv_folds=999,
            artifacts_dir=None,
        )


def test_run_pipeline_no_artifacts_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Path("data/sample_music_dataset.csv").resolve()
    monkeypatch.chdir(tmp_path)

    result = run_pipeline(
        dataset,
        test_size=0.25,
        random_state=42,
        cv_folds=3,
        artifacts_dir=None,
    )

    assert result.artifact_dir is None
    assert list(tmp_path.iterdir()) == []


def test_run_registry_appends_one_line_per_run(tmp_path: Path) -> None:
    run_pipeline(
        Path("data/sample_music_dataset.csv"),
        test_size=0.25,
        random_state=42,
        cv_folds=3,
        artifacts_dir=tmp_path,
    )
    run_pipeline(
        Path("data/sample_music_dataset.csv"),
        test_size=0.25,
        random_state=7,
        cv_folds=3,
        artifacts_dir=tmp_path,
    )

    index_path = tmp_path / "index.jsonl"
    assert index_path.exists()
    lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]
    for record in records:
        assert "timestamp_utc" in record
        assert "git_commit_hash" in record
        assert "dataset_path_hash" in record
        assert "parameters" in record
        assert "metrics" in record
        assert set(record["parameters"]) == {"test_size", "random_state", "cv_folds"}
        assert "majority_baseline" in record["metrics"]
        assert "decision_tree" in record["metrics"]
        assert "knn" in record["metrics"]
        assert "logistic_regression" in record["metrics"]
        assert "kmeans_ari" in record["metrics"]

    assert records[0]["parameters"]["random_state"] == 42
    assert records[1]["parameters"]["random_state"] == 7


def test_cli_no_save_artifacts_passes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _Metrics:
        accuracy = 0.5
        macro_f1 = 0.5
        weighted_f1 = 0.5

    class _Result:
        majority_baseline_metrics = _Metrics()
        decision_tree_metrics = _Metrics()
        knn_metrics = _Metrics()
        logistic_regression_metrics = _Metrics()
        kmeans_ari = 0.0
        top_features = pd.Series({"energy": 1.0})
        energy_cluster_summary = pd.DataFrame(
            [{"energy": 0.5, "valence": 0.5, "tempo": 100.0, "danceability": 0.5}]
        )
        cross_validation = {
            "majority_baseline": {
                "accuracy_mean": 0.5,
                "accuracy_std": 0.0,
                "macro_f1_mean": 0.5,
                "macro_f1_std": 0.0,
            },
            "decision_tree": {
                "accuracy_mean": 0.5,
                "accuracy_std": 0.0,
                "macro_f1_mean": 0.5,
                "macro_f1_std": 0.0,
            },
            "knn": {
                "accuracy_mean": 0.5,
                "accuracy_std": 0.0,
                "macro_f1_mean": 0.5,
                "macro_f1_std": 0.0,
            },
            "logistic_regression": {
                "accuracy_mean": 0.5,
                "accuracy_std": 0.0,
                "macro_f1_mean": 0.5,
                "macro_f1_std": 0.0,
            },
        }
        artifact_dir = None

        @staticmethod
        def as_dict() -> dict[str, object]:
            return {}

    def fake_run_pipeline(**kwargs: object) -> _Result:
        calls.update(kwargs)
        return _Result()

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "music-emotion-predict",
            "--dataset",
            "data/sample_music_dataset.csv",
            "--no-save-artifacts",
        ],
    )

    exit_code = cli.main()
    assert exit_code == 0
    assert calls["artifacts_dir"] is None
