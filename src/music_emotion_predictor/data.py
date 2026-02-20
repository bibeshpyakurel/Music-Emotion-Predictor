from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DROP_COLUMNS = ("Unnamed: 0", "Unnamed: 0.1", "uri")


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or malformed."""


@dataclass(frozen=True)
class DatasetBundle:
    raw: pd.DataFrame
    features: pd.DataFrame
    labels: pd.Series


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetError(
            f"Dataset not found at {dataset_path}. "
            "Download the Kaggle dataset and pass its CSV path via --dataset."
        )
    if dataset_path.stat().st_size == 0:
        raise DatasetError(
            f"Dataset at {dataset_path} is empty. "
            "Provide a non-empty CSV exported from the Moodify dataset."
        )

    try:
        df = pd.read_csv(dataset_path)
    except Exception as exc:  # pragma: no cover - depends on pandas parser internals
        raise DatasetError(f"Failed to parse CSV at {dataset_path}: {exc}") from exc

    if df.empty:
        raise DatasetError(f"Dataset at {dataset_path} has zero rows.")
    return df


def prepare_supervised_data(df: pd.DataFrame) -> DatasetBundle:
    drop_cols = [c for c in DEFAULT_DROP_COLUMNS if c in df.columns]
    cleaned = df.drop(columns=drop_cols)

    if "labels" not in cleaned.columns:
        raise DatasetError("Expected a 'labels' column for supervised training.")

    features = cleaned.drop(columns=["labels"])
    if features.empty:
        raise DatasetError("No feature columns remain after preprocessing.")

    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise DatasetError(
            "All feature columns must be numeric. "
            f"Found non-numeric columns: {', '.join(non_numeric)}"
        )

    labels = cleaned["labels"]
    if labels.nunique() < 2:
        raise DatasetError("Need at least two label classes for training.")

    return DatasetBundle(raw=df, features=features, labels=labels)
