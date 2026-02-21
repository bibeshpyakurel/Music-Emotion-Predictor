from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

ALLOWED_LABELS = frozenset({"Calm", "Sad", "Energetic", "Happy"})
NAN_STRATEGIES = ("drop", "impute")


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or malformed."""


@dataclass(frozen=True)
class DatasetBundle:
    raw: pd.DataFrame
    features: pd.DataFrame
    labels: pd.Series
    dropped_columns: tuple[str, ...]


def _emit(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _columns_to_drop(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col == "uri" or col.startswith("Unnamed")]


def validate_supervised_dataframe(
    df: pd.DataFrame,
    *,
    nan_strategy: Literal["drop", "impute"] = "drop",
    logger: Callable[[str], None] | None = None,
) -> DatasetBundle:
    if nan_strategy not in NAN_STRATEGIES:
        raise ValueError(f"nan_strategy must be one of {NAN_STRATEGIES}.")

    if df.empty:
        raise DatasetError("Dataset has zero rows.")

    dropped_columns = _columns_to_drop(df)
    cleaned = df.drop(columns=dropped_columns).copy()
    if dropped_columns:
        _emit(logger, "Dropped columns: " + ", ".join(dropped_columns))

    if "labels" not in cleaned.columns:
        raise DatasetError("Expected a 'labels' column for supervised training.")

    labels = cleaned["labels"]
    if labels.isna().any():
        missing_label_rows = int(labels.isna().sum())
        cleaned = cleaned.loc[~labels.isna()].copy()
        labels = cleaned["labels"]
        _emit(
            logger,
            f"Dropped {missing_label_rows} row(s) with missing labels.",
        )

    labels = labels.astype(str).str.strip()
    invalid_labels = sorted(set(labels.unique()) - ALLOWED_LABELS)
    if invalid_labels:
        raise DatasetError(
            "Invalid label values found: "
            f"{', '.join(invalid_labels)}. "
            f"Allowed labels: {', '.join(sorted(ALLOWED_LABELS))}."
        )

    features = cleaned.drop(columns=["labels"]).copy()
    if features.empty:
        raise DatasetError("No feature columns remain after preprocessing.")

    non_numeric_columns: list[str] = []
    for column in features.columns:
        try:
            features[column] = pd.to_numeric(features[column], errors="raise")
        except (TypeError, ValueError):
            non_numeric_columns.append(column)
    if non_numeric_columns:
        raise DatasetError(
            "All feature columns must be numeric after cleaning. "
            f"Found non-numeric columns: {', '.join(non_numeric_columns)}"
        )

    missing_feature_cells = int(features.isna().sum().sum())
    if missing_feature_cells:
        if nan_strategy == "drop":
            keep_rows = features.notna().all(axis=1)
            dropped_rows = int((~keep_rows).sum())
            features = features.loc[keep_rows].copy()
            labels = labels.loc[keep_rows].copy()
            cleaned = cleaned.loc[keep_rows].copy()
            _emit(
                logger,
                "NaN handling strategy=drop: "
                f"dropped {dropped_rows} row(s) containing missing feature values.",
            )
        else:
            imputed_columns: list[str] = []
            for column in features.columns:
                if not features[column].isna().any():
                    continue
                median_value = features[column].median()
                if pd.isna(median_value):
                    raise DatasetError(
                        f"Cannot impute column '{column}' because it contains only NaN values."
                    )
                features[column] = features[column].fillna(median_value)
                imputed_columns.append(column)
            cleaned.loc[:, features.columns] = features
            _emit(
                logger,
                "NaN handling strategy=impute: "
                f"imputed columns with median values: {', '.join(imputed_columns)}.",
            )

    if features.empty:
        raise DatasetError("No rows remain after NaN handling.")

    if labels.nunique() < 2:
        raise DatasetError("Need at least two label classes for training.")

    return DatasetBundle(
        raw=cleaned,
        features=features,
        labels=labels,
        dropped_columns=tuple(dropped_columns),
    )


def load_and_validate_dataset(
    path: str | Path,
    *,
    nan_strategy: Literal["drop", "impute"] = "drop",
    logger: Callable[[str], None] | None = None,
) -> DatasetBundle:
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
    except Exception as exc:  # pragma: no cover - pandas parser behavior
        raise DatasetError(f"Failed to parse CSV at {dataset_path}: {exc}") from exc

    return validate_supervised_dataframe(df, nan_strategy=nan_strategy, logger=logger)
