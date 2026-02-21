from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .validation import (
    DatasetBundle,
    DatasetError,
    load_and_validate_dataset,
    validate_supervised_dataframe,
)

ENERGY_FEATURES = ("energy", "valence", "tempo", "danceability")


@dataclass(frozen=True)
class ModelMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    classification_report: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "classification_report": self.classification_report,
        }


@dataclass(frozen=True)
class TrainedModels:
    scaler: StandardScaler
    majority_baseline: DummyClassifier
    decision_tree: DecisionTreeClassifier
    knn: KNeighborsClassifier
    logistic_regression: LogisticRegression
    feature_names: list[str]


@dataclass(frozen=True)
class HoldoutPredictions:
    y_true: pd.Series
    majority_baseline: pd.Series
    decision_tree: pd.Series
    knn: pd.Series
    logistic_regression: pd.Series


@dataclass(frozen=True)
class PipelineResult:
    majority_baseline_metrics: ModelMetrics
    decision_tree_metrics: ModelMetrics
    knn_metrics: ModelMetrics
    logistic_regression_metrics: ModelMetrics
    kmeans_ari: float
    top_features: pd.Series
    energy_cluster_summary: pd.DataFrame
    cross_validation: dict[str, dict[str, float]]
    artifact_dir: str | None = None

    @property
    def decision_tree_accuracy(self) -> float:
        return self.decision_tree_metrics.accuracy

    @property
    def knn_accuracy(self) -> float:
        return self.knn_metrics.accuracy

    def as_dict(self) -> dict[str, Any]:
        return {
            "majority_baseline": self.majority_baseline_metrics.as_dict(),
            "decision_tree": self.decision_tree_metrics.as_dict(),
            "knn": self.knn_metrics.as_dict(),
            "logistic_regression": self.logistic_regression_metrics.as_dict(),
            "kmeans_ari": self.kmeans_ari,
            "cross_validation": self.cross_validation,
            "top_features": self.top_features.to_dict(),
            "energy_cluster_summary": self.energy_cluster_summary.to_dict(orient="index"),
            "artifact_dir": self.artifact_dir,
        }


def _evaluate_classifier(y_true: pd.Series, y_pred: pd.Series) -> ModelMetrics:
    return ModelMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        classification_report=classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    )


def _validate_cv_folds(labels: pd.Series, cv_folds: int) -> None:
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    min_class_count = int(labels.value_counts().min())
    if cv_folds > min_class_count:
        raise ValueError(
            "cv_folds cannot be larger than the smallest class size "
            f"({min_class_count})."
        )


def _cross_validate_models(
    features: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
    cv_folds: int,
) -> dict[str, dict[str, float]]:
    _validate_cv_folds(labels, cv_folds)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = ("accuracy", "f1_macro", "f1_weighted")

    decision_tree_pipeline = make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier(random_state=random_state),
    )
    knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    logistic_regression_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state),
    )
    majority_baseline_pipeline = make_pipeline(
        StandardScaler(),
        DummyClassifier(strategy="most_frequent"),
    )

    dt_cv = cross_validate(decision_tree_pipeline, features, labels, scoring=scoring, cv=cv)
    knn_cv = cross_validate(knn_pipeline, features, labels, scoring=scoring, cv=cv)
    lr_cv = cross_validate(logistic_regression_pipeline, features, labels, scoring=scoring, cv=cv)
    majority_cv = cross_validate(
        majority_baseline_pipeline,
        features,
        labels,
        scoring=scoring,
        cv=cv,
    )

    return {
        "majority_baseline": {
            "accuracy_mean": float(majority_cv["test_accuracy"].mean()),
            "accuracy_std": float(majority_cv["test_accuracy"].std()),
            "macro_f1_mean": float(majority_cv["test_f1_macro"].mean()),
            "macro_f1_std": float(majority_cv["test_f1_macro"].std()),
            "weighted_f1_mean": float(majority_cv["test_f1_weighted"].mean()),
            "weighted_f1_std": float(majority_cv["test_f1_weighted"].std()),
        },
        "decision_tree": {
            "accuracy_mean": float(dt_cv["test_accuracy"].mean()),
            "accuracy_std": float(dt_cv["test_accuracy"].std()),
            "macro_f1_mean": float(dt_cv["test_f1_macro"].mean()),
            "macro_f1_std": float(dt_cv["test_f1_macro"].std()),
            "weighted_f1_mean": float(dt_cv["test_f1_weighted"].mean()),
            "weighted_f1_std": float(dt_cv["test_f1_weighted"].std()),
        },
        "knn": {
            "accuracy_mean": float(knn_cv["test_accuracy"].mean()),
            "accuracy_std": float(knn_cv["test_accuracy"].std()),
            "macro_f1_mean": float(knn_cv["test_f1_macro"].mean()),
            "macro_f1_std": float(knn_cv["test_f1_macro"].std()),
            "weighted_f1_mean": float(knn_cv["test_f1_weighted"].mean()),
            "weighted_f1_std": float(knn_cv["test_f1_weighted"].std()),
        },
        "logistic_regression": {
            "accuracy_mean": float(lr_cv["test_accuracy"].mean()),
            "accuracy_std": float(lr_cv["test_accuracy"].std()),
            "macro_f1_mean": float(lr_cv["test_f1_macro"].mean()),
            "macro_f1_std": float(lr_cv["test_f1_macro"].std()),
            "weighted_f1_mean": float(lr_cv["test_f1_weighted"].mean()),
            "weighted_f1_std": float(lr_cv["test_f1_weighted"].std()),
        },
    }


def _fit_models(
    features: pd.DataFrame,
    labels: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[
    ModelMetrics,
    ModelMetrics,
    ModelMetrics,
    ModelMetrics,
    float,
    TrainedModels,
    HoldoutPredictions,
]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    majority_baseline = DummyClassifier(strategy="most_frequent")
    majority_baseline.fit(x_train_scaled, y_train)
    majority_predictions = majority_baseline.predict(x_test_scaled)

    decision_tree = DecisionTreeClassifier(random_state=random_state)
    decision_tree.fit(x_train_scaled, y_train)
    dt_predictions = decision_tree.predict(x_test_scaled)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_scaled, y_train)
    knn_predictions = knn.predict(x_test_scaled)

    logistic_regression = LogisticRegression(max_iter=1000, random_state=random_state)
    logistic_regression.fit(x_train_scaled, y_train)
    lr_predictions = logistic_regression.predict(x_test_scaled)

    kmeans = KMeans(n_clusters=labels.nunique(), random_state=random_state, n_init=10)
    kmeans_predictions = kmeans.fit_predict(x_test_scaled)

    majority_metrics = _evaluate_classifier(y_test, majority_predictions)
    dt_metrics = _evaluate_classifier(y_test, dt_predictions)
    knn_metrics = _evaluate_classifier(y_test, knn_predictions)
    logistic_regression_metrics = _evaluate_classifier(y_test, lr_predictions)
    kmeans_ari = adjusted_rand_score(y_test, kmeans_predictions)

    trained = TrainedModels(
        scaler=scaler,
        majority_baseline=majority_baseline,
        decision_tree=decision_tree,
        knn=knn,
        logistic_regression=logistic_regression,
        feature_names=list(features.columns),
    )

    predictions = HoldoutPredictions(
        y_true=y_test.reset_index(drop=True),
        majority_baseline=pd.Series(majority_predictions).reset_index(drop=True),
        decision_tree=pd.Series(dt_predictions).reset_index(drop=True),
        knn=pd.Series(knn_predictions).reset_index(drop=True),
        logistic_regression=pd.Series(lr_predictions).reset_index(drop=True),
    )

    return (
        majority_metrics,
        dt_metrics,
        knn_metrics,
        logistic_regression_metrics,
        kmeans_ari,
        trained,
        predictions,
    )


def _per_class_report(classification_report_dict: dict[str, Any]) -> dict[str, dict[str, float]]:
    aggregate_keys = {"accuracy", "macro avg", "weighted avg"}
    return {
        label: {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1-score"]),
            "support": float(metrics["support"]),
        }
        for label, metrics in classification_report_dict.items()
        if label not in aggregate_keys
    }


def _build_confusion_matrix_rows(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    all_labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=all_labels)
    records: list[dict[str, Any]] = []
    for i, actual in enumerate(all_labels):
        for j, predicted in enumerate(all_labels):
            records.append(
                {
                    "model": model_name,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(matrix[i, j]),
                }
            )
    return pd.DataFrame.from_records(records)


def _build_run_summary_markdown(result: PipelineResult) -> str:
    rows = [
        ("Majority Baseline", result.majority_baseline_metrics),
        ("Decision Tree", result.decision_tree_metrics),
        ("KNN", result.knn_metrics),
        ("Logistic Regression", result.logistic_regression_metrics),
    ]
    winner = max(rows, key=lambda entry: entry[1].accuracy)[0]

    return "\n".join(
        [
            "# Run Summary",
            "",
            "## Holdout Comparison",
            "| Model | Accuracy | Macro F1 | Weighted F1 |",
            "| --- | ---: | ---: | ---: |",
            *[
                (
                    f"| {name} | {metrics.accuracy:.4f} | {metrics.macro_f1:.4f} | "
                    f"{metrics.weighted_f1:.4f} |"
                )
                for name, metrics in rows
            ],
            "",
            f"KMeans ARI (holdout): **{result.kmeans_ari:.4f}**",
            "",
            f"Best supervised model by holdout accuracy: **{winner}**",
        ]
    )


def _dataset_path_hash(dataset: Path) -> str:
    return hashlib.sha256(str(dataset).encode("utf-8")).hexdigest()


def _git_commit_hash() -> str | None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return commit or None


def _build_registry_record(
    *,
    timestamp_utc: str,
    dataset: Path,
    test_size: float,
    random_state: int,
    cv_folds: int,
    result: PipelineResult,
) -> dict[str, Any]:
    return {
        "timestamp_utc": timestamp_utc,
        "git_commit_hash": _git_commit_hash(),
        "dataset_path_hash": _dataset_path_hash(dataset),
        "parameters": {
            "test_size": test_size,
            "random_state": random_state,
            "cv_folds": cv_folds,
        },
        "metrics": {
            "majority_baseline": {
                "accuracy": result.majority_baseline_metrics.accuracy,
                "macro_f1": result.majority_baseline_metrics.macro_f1,
                "weighted_f1": result.majority_baseline_metrics.weighted_f1,
            },
            "decision_tree": {
                "accuracy": result.decision_tree_metrics.accuracy,
                "macro_f1": result.decision_tree_metrics.macro_f1,
                "weighted_f1": result.decision_tree_metrics.weighted_f1,
            },
            "knn": {
                "accuracy": result.knn_metrics.accuracy,
                "macro_f1": result.knn_metrics.macro_f1,
                "weighted_f1": result.knn_metrics.weighted_f1,
            },
            "logistic_regression": {
                "accuracy": result.logistic_regression_metrics.accuracy,
                "macro_f1": result.logistic_regression_metrics.macro_f1,
                "weighted_f1": result.logistic_regression_metrics.weighted_f1,
            },
            "kmeans_ari": result.kmeans_ari,
        },
    }


def _append_registry_record(index_path: Path, record: dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":")) + "\n"
    fd = os.open(index_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _energy_clustering(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    missing = [col for col in ENERGY_FEATURES if col not in df.columns]
    if missing:
        raise DatasetError(
            "Missing required energy-clustering columns: " + ", ".join(missing)
        )

    energy_df = df[list(ENERGY_FEATURES)]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(energy_df)

    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(scaled)

    reduced = PCA(n_components=2, random_state=random_state).fit_transform(scaled)
    clustered = df.copy()
    clustered["energy_cluster"] = clusters
    clustered["pca_1"] = reduced[:, 0]
    clustered["pca_2"] = reduced[:, 1]

    return (
        clustered.groupby("energy_cluster")[list(ENERGY_FEATURES)]
        .mean()
        .sort_index()
    )


def _save_artifacts(
    result: PipelineResult,
    trained: TrainedModels,
    predictions: HoldoutPredictions,
    dataset: Path,
    test_size: float,
    random_state: int,
    cv_folds: int,
    artifacts_dir: Path,
) -> str:
    generated_at = datetime.now(timezone.utc)
    run_id = generated_at.strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifacts_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(trained.scaler, run_dir / "scaler.joblib")
    joblib.dump(trained.majority_baseline, run_dir / "majority_baseline.joblib")
    joblib.dump(trained.decision_tree, run_dir / "decision_tree.joblib")
    joblib.dump(trained.knn, run_dir / "knn.joblib")
    joblib.dump(trained.logistic_regression, run_dir / "logistic_regression.joblib")

    result.top_features.to_csv(run_dir / "top_features.csv", header=True)
    result.energy_cluster_summary.to_csv(run_dir / "energy_cluster_summary.csv")
    confusion_matrix_df = pd.concat(
        [
            _build_confusion_matrix_rows(
                predictions.y_true,
                predictions.majority_baseline,
                "majority_baseline",
            ),
            _build_confusion_matrix_rows(
                predictions.y_true,
                predictions.decision_tree,
                "decision_tree",
            ),
            _build_confusion_matrix_rows(
                predictions.y_true,
                predictions.knn,
                "knn",
            ),
            _build_confusion_matrix_rows(
                predictions.y_true,
                predictions.logistic_regression,
                "logistic_regression",
            ),
        ],
        ignore_index=True,
    )
    confusion_matrix_df.to_csv(run_dir / "confusion_matrix.csv", index=False)

    classification_report_payload = {
        "majority_baseline": _per_class_report(
            result.majority_baseline_metrics.classification_report
        ),
        "decision_tree": _per_class_report(result.decision_tree_metrics.classification_report),
        "knn": _per_class_report(result.knn_metrics.classification_report),
        "logistic_regression": _per_class_report(
            result.logistic_regression_metrics.classification_report
        ),
    }
    with (run_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(classification_report_payload, f, indent=2)

    with (run_dir / "run_summary.md").open("w", encoding="utf-8") as f:
        f.write(_build_run_summary_markdown(result) + "\n")

    metadata = {
        "dataset": str(dataset),
        "test_size": test_size,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "feature_names": trained.feature_names,
        "generated_at_utc": generated_at.isoformat(),
    }

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result.as_dict(), f, indent=2)

    _append_registry_record(
        artifacts_dir / "index.jsonl",
        _build_registry_record(
            timestamp_utc=generated_at.isoformat(),
            dataset=dataset,
            test_size=test_size,
            random_state=random_state,
            cv_folds=cv_folds,
            result=result,
        ),
    )

    return str(run_dir)


def run_pipeline(
    dataset: str | Path | pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    artifacts_dir: str | Path | None = None,
    nan_strategy: Literal["drop", "impute"] = "drop",
    logger: Callable[[str], None] | None = None,
) -> PipelineResult:
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")

    prepared: DatasetBundle
    dataset_ref: Path | None
    if isinstance(dataset, pd.DataFrame):
        prepared = validate_supervised_dataframe(
            dataset,
            nan_strategy=nan_strategy,
            logger=logger,
        )
        dataset_ref = None
    else:
        dataset_ref = Path(dataset)
        prepared = load_and_validate_dataset(
            dataset_ref,
            nan_strategy=nan_strategy,
            logger=logger,
        )

    cross_validation = _cross_validate_models(
        prepared.features,
        prepared.labels,
        random_state=random_state,
        cv_folds=cv_folds,
    )

    (
        majority_metrics,
        dt_metrics,
        knn_metrics,
        logistic_regression_metrics,
        kmeans_ari,
        trained,
        predictions,
    ) = _fit_models(
        prepared.features,
        prepared.labels,
        test_size=test_size,
        random_state=random_state,
    )

    top_features = pd.Series(
        trained.decision_tree.feature_importances_,
        index=prepared.features.columns,
        name="feature_importance",
    ).sort_values(ascending=False)

    energy_summary = _energy_clustering(prepared.raw, random_state=random_state)

    result = PipelineResult(
        majority_baseline_metrics=majority_metrics,
        decision_tree_metrics=dt_metrics,
        knn_metrics=knn_metrics,
        logistic_regression_metrics=logistic_regression_metrics,
        kmeans_ari=kmeans_ari,
        top_features=top_features,
        energy_cluster_summary=energy_summary,
        cross_validation=cross_validation,
    )

    if artifacts_dir is None:
        return result

    dataset_for_metadata = dataset_ref if dataset_ref is not None else Path("<in-memory-dataframe>")
    artifact_path = _save_artifacts(
        result=result,
        trained=trained,
        predictions=predictions,
        dataset=dataset_for_metadata,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
        artifacts_dir=Path(artifacts_dir),
    )

    return PipelineResult(
        majority_baseline_metrics=result.majority_baseline_metrics,
        decision_tree_metrics=result.decision_tree_metrics,
        knn_metrics=result.knn_metrics,
        logistic_regression_metrics=result.logistic_regression_metrics,
        kmeans_ari=result.kmeans_ari,
        top_features=result.top_features,
        energy_cluster_summary=result.energy_cluster_summary,
        cross_validation=result.cross_validation,
        artifact_dir=artifact_path,
    )
