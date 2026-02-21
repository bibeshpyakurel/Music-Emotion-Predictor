from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validation import (
    DatasetBundle,
    load_and_validate_dataset,
    validate_supervised_dataframe,
)


def load_dataset(path: str | Path) -> pd.DataFrame:
    return load_and_validate_dataset(path).raw


def prepare_supervised_data(df: pd.DataFrame) -> DatasetBundle:
    return validate_supervised_dataframe(df)
