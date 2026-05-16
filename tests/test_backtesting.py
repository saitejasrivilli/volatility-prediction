import numpy as np
import pandas as pd

from src.backtesting import (
    Backtester,
    ClassificationEvaluator,
    ThresholdOptimizer,
    WalkForwardValidator,
)
from src.config import WalkForwardConfig


def test_backtester_only_trades_positive_spike_signals():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Close": [100.0, 102.0, 101.0, 103.0],
            "Volatility_20d": [0.02, 0.02, 0.02, 0.02],
        },
        index=dates,
    )
    signals = pd.Series([0, 1, 0, 1], index=dates)
    strengths = pd.Series([0.9, 0.9, 0.9, 0.9], index=dates)

    trades, _ = Backtester().run_backtest(df, signals, strengths, dates, kelly_pct=0.02)

    assert len(trades) == 1
    assert trades[0].date == dates[1]


def test_generate_windows_honors_step_and_test_window():
    dates = pd.date_range("2019-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"Close": np.linspace(100, 200, len(dates))}, index=dates)
    features = pd.DataFrame({"Volatility_20d": 0.02}, index=dates)
    config = WalkForwardConfig(TRAIN_WINDOW=2, TEST_WINDOW=2, STEP_SIZE=2)

    windows = WalkForwardValidator(df, features, config).generate_windows()

    assert windows[0] == ("2019-01-01", "2020-12-31", "2021-01-01", "2022-12-31")
    assert windows[1] == ("2019-01-01", "2022-12-31", "2023-01-01", "2024-12-31")


def test_classification_evaluator_reports_imbalance_sensitive_metrics():
    y_true = pd.Series([0, 0, 1, 1])
    predictions = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.4, 0.9])

    metrics = ClassificationEvaluator.evaluate(y_true, predictions, scores)

    assert metrics.accuracy == 0.75
    assert metrics.recall == 0.5
    assert metrics.confusion_matrix == (2, 0, 1, 1)


def test_threshold_optimizer_prefers_training_f1():
    y_true = pd.Series([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.45, 0.9])

    threshold = ThresholdOptimizer.optimize(y_true, scores)

    assert threshold <= 0.45


def test_walk_forward_validator_exposes_training_kelly_estimator():
    dates = pd.date_range("2019-01-01", periods=1200, freq="D")
    df = pd.DataFrame({"Close": np.linspace(100, 200, len(dates))}, index=dates)
    features = pd.DataFrame(
        {
            "Volatility_20d": 0.02,
            "MA_Ratio_5": 1.0,
        },
        index=dates,
    )

    validator = WalkForwardValidator(df, features)

    assert callable(validator._estimate_kelly_from_training)
