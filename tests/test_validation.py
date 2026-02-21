from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from music_emotion_predictor.validation import (
    DatasetError,
    load_and_validate_dataset,
    validate_supervised_dataframe,
)


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uri": ["track_1", "track_2", "track_3", "track_4"],
            "energy": [0.1, 0.2, 0.3, 0.4],
            "valence": [0.3, 0.4, 0.5, 0.6],
            "tempo": [90, 100, 110, 120],
            "danceability": [0.2, 0.3, 0.4, 0.5],
            "labels": ["Calm", "Sad", "Energetic", "Happy"],
        }
    )


def test_validation_rejects_empty_csv(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")

    with pytest.raises(DatasetError, match="empty"):
        load_and_validate_dataset(empty_csv)


def test_validation_requires_labels_column() -> None:
    df = pd.DataFrame({"energy": [0.1], "tempo": [100], "danceability": [0.2]})

    with pytest.raises(DatasetError, match="labels"):
        validate_supervised_dataframe(df)


def test_validation_rejects_unknown_labels() -> None:
    df = _valid_df()
    df["labels"] = ["Calm", "Sad", "Angry", "Happy"]

    with pytest.raises(DatasetError, match="Invalid label values"):
        validate_supervised_dataframe(df)


def test_validation_rejects_non_numeric_feature_columns() -> None:
    df = _valid_df()
    df["tempo"] = ["fast", "slow", "med", "fast"]

    with pytest.raises(DatasetError, match="non-numeric columns"):
        validate_supervised_dataframe(df)


def test_validation_drops_rows_with_nan_when_strategy_drop() -> None:
    df = _valid_df()
    df.loc[1, "energy"] = None
    logs: list[str] = []

    bundle = validate_supervised_dataframe(df, nan_strategy="drop", logger=logs.append)

    assert len(bundle.features) == 3
    assert not bundle.features.isna().any().any()
    assert any("strategy=drop" in line for line in logs)


def test_validation_imputes_nan_when_strategy_impute() -> None:
    df = _valid_df()
    df.loc[1, "energy"] = None
    logs: list[str] = []

    bundle = validate_supervised_dataframe(df, nan_strategy="impute", logger=logs.append)

    assert len(bundle.features) == 4
    assert not bundle.features.isna().any().any()
    assert any("strategy=impute" in line for line in logs)


def test_validation_reports_dropped_columns() -> None:
    df = _valid_df()
    df["Unnamed: 0"] = [1, 2, 3, 4]
    logs: list[str] = []

    bundle = validate_supervised_dataframe(df, logger=logs.append)

    assert bundle.dropped_columns == ("uri", "Unnamed: 0")
    assert any("Dropped columns: uri, Unnamed: 0" in line for line in logs)
