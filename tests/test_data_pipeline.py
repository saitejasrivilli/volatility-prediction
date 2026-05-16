import pandas as pd
import pytest

from src.config import DataConfig
from src.data_pipeline import DataCleaner, DataValidationError, DataValidator


def make_ohlcv(rows: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "Open": range(100, 100 + rows),
            "High": range(101, 101 + rows),
            "Low": range(99, 99 + rows),
            "Close": [100.5 + i for i in range(rows)],
            "Volume": [1000 + (i * 10) for i in range(rows)],
        },
        index=dates,
    )


def test_validator_rejects_duplicate_dates():
    df = make_ohlcv()
    duplicate = pd.concat([df, df.iloc[[0]]]).sort_index()
    config = DataConfig(MIN_ROWS=1)

    with pytest.raises(DataValidationError, match="duplicate dates"):
        DataValidator(duplicate, config).validate()


def test_cleaner_never_backfills_from_future_rows():
    df = make_ohlcv()
    df.loc[df.index[0], "Close"] = pd.NA

    cleaned = DataCleaner(df).clean()

    assert df.index[0] not in cleaned.index
    assert cleaned.index.min() == df.index[1]


def test_cleaner_fixes_high_low_inversions():
    df = make_ohlcv()
    df.loc[df.index[0], ["High", "Low"]] = [90, 110]

    cleaned = DataCleaner(df).clean()

    assert cleaned.iloc[0]["High"] == 110
    assert cleaned.iloc[0]["Low"] == 90
