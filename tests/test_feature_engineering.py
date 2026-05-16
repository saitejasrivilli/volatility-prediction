import numpy as np
import pandas as pd

from src.feature_engineering import FeaturePreprocessor, TargetBuilder


def test_target_builder_preserves_unknown_labels_as_nan():
    close = pd.Series(
        np.linspace(100, 130, 300),
        index=pd.date_range("2023-01-01", periods=300, freq="D"),
    )
    target = TargetBuilder.volatility_spike_target(close, close.pct_change())

    assert target.iloc[:270].isna().any()
    assert pd.isna(target.iloc[-1])


def test_preprocessor_uses_training_statistics_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [100.0]})

    preprocessor = FeaturePreprocessor().fit(train)
    transformed_test = preprocessor.transform(test)

    assert preprocessor.means["x"] == 2.0
    assert transformed_test.iloc[0]["x"] > 50


def test_transition_target_only_marks_regime_entries():
    idx = pd.date_range("2023-01-01", periods=280, freq="D")
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    returns = close.pct_change()
    target = TargetBuilder.volatility_transition_target(close, returns)

    assert set(target.dropna().unique()).issubset({0.0, 1.0})
