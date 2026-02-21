from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import DatasetError, load_dataset, prepare_supervised_data

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
    decision_tree: DecisionTreeClassifier
    knn: KNeighborsClassifier
    feature_names: list[str]


@dataclass(frozen=True)
class PipelineResult:
    decision_tree_metrics: ModelMetrics
    knn_metrics: ModelMetrics
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
            "decision_tree": self.decision_tree_metrics.as_dict(),
            "knn": self.knn_metrics.as_dict(),
            "kmeans_ari": self.kmeans_ari,
            "cross_validation": self.cross_validation,
            "top_features": self.top_features.to_dict(),
            "energy_cluster_summary": self.energy_cluster_summary.to_dict(orient="index"),
            "artifact_dir": self.artifact_dir,
        }


def _evaluate_classifier(y_true: pd.Series, y_pred: pd.Series) -> ModelMetrics:
    return ModelMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro"),
        weighted_f1=f1_score(y_true, y_pred, average="weighted"),
        classification_report=classification_report(y_true, y_pred, output_dict=True),
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

    dt_cv = cross_validate(decision_tree_pipeline, features, labels, scoring=scoring, cv=cv)
    knn_cv = cross_validate(knn_pipeline, features, labels, scoring=scoring, cv=cv)

    return {
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
    }


def _fit_models(
    features: pd.DataFrame,
    labels: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[ModelMetrics, ModelMetrics, float, TrainedModels]:
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

    decision_tree = DecisionTreeClassifier(random_state=random_state)
    decision_tree.fit(x_train_scaled, y_train)
    dt_predictions = decision_tree.predict(x_test_scaled)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_scaled, y_train)
    knn_predictions = knn.predict(x_test_scaled)

    kmeans = KMeans(n_clusters=labels.nunique(), random_state=random_state, n_init=10)
    kmeans_predictions = kmeans.fit_predict(x_test_scaled)

    dt_metrics = _evaluate_classifier(y_test, dt_predictions)
    knn_metrics = _evaluate_classifier(y_test, knn_predictions)
    kmeans_ari = adjusted_rand_score(y_test, kmeans_predictions)

    trained = TrainedModels(
        scaler=scaler,
        decision_tree=decision_tree,
        knn=knn,
        feature_names=list(features.columns),
    )

    return dt_metrics, knn_metrics, kmeans_ari, trained


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
    dataset: Path,
    test_size: float,
    random_state: int,
    cv_folds: int,
    artifacts_dir: Path,
) -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifacts_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(trained.scaler, run_dir / "scaler.joblib")
    joblib.dump(trained.decision_tree, run_dir / "decision_tree.joblib")
    joblib.dump(trained.knn, run_dir / "knn.joblib")

    result.top_features.to_csv(run_dir / "top_features.csv", header=True)
    result.energy_cluster_summary.to_csv(run_dir / "energy_cluster_summary.csv")

    metadata = {
        "dataset": str(dataset),
        "test_size": test_size,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "feature_names": trained.feature_names,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result.as_dict(), f, indent=2)

    return str(run_dir)


def run_pipeline(
    dataset: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    artifacts_dir: str | Path | None = None,
) -> PipelineResult:
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")

    dataset_path = Path(dataset)
    df = load_dataset(dataset_path)
    prepared = prepare_supervised_data(df)

    cross_validation = _cross_validate_models(
        prepared.features,
        prepared.labels,
        random_state=random_state,
        cv_folds=cv_folds,
    )

    dt_metrics, knn_metrics, kmeans_ari, trained = _fit_models(
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
        decision_tree_metrics=dt_metrics,
        knn_metrics=knn_metrics,
        kmeans_ari=kmeans_ari,
        top_features=top_features,
        energy_cluster_summary=energy_summary,
        cross_validation=cross_validation,
    )

    if artifacts_dir is None:
        return result

    artifact_path = _save_artifacts(
        result=result,
        trained=trained,
        dataset=dataset_path,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
        artifacts_dir=Path(artifacts_dir),
    )

    return PipelineResult(
        decision_tree_metrics=result.decision_tree_metrics,
        knn_metrics=result.knn_metrics,
        kmeans_ari=result.kmeans_ari,
        top_features=result.top_features,
        energy_cluster_summary=result.energy_cluster_summary,
        cross_validation=result.cross_validation,
        artifact_dir=artifact_path,
    )
