"""Pooled multi-ticker walk-forward experimentation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

try:
    from .backtesting import (
        AlertEvaluator,
        ClassificationEvaluator,
        ThresholdOptimizer,
        make_model,
    )
    from .config import MODEL_CONFIG
    from .feature_engineering import FeaturePreprocessor
except ImportError:  # pragma: no cover
    from backtesting import (
        AlertEvaluator,
        ClassificationEvaluator,
        ThresholdOptimizer,
        make_model,
    )
    from config import MODEL_CONFIG
    from feature_engineering import FeaturePreprocessor


@dataclass
class PooledFoldResult:
    model_kind: str
    horizon: int
    period: int
    aggregate_f1: float
    aggregate_pr_auc: float
    baseline_f1: float
    positives: int
    positive_rate: float
    alert_metrics: pd.DataFrame
    rule_baselines: pd.DataFrame
    yearly_metrics: pd.DataFrame
    drift_metrics: pd.DataFrame
    alert_rows: pd.DataFrame
    per_ticker: pd.DataFrame


class PooledExperimentRunner:
    """Train on pooled rows while reporting aggregate and ticker-level results."""

    def __init__(self, market_data, features, target, walk_config, models):
        self.market_data = market_data
        self.features = features
        self.target = target
        self.walk_config = walk_config
        self.models = models

    def run(self, horizon: int) -> list[PooledFoldResult]:
        dates = self.market_data.index.get_level_values(0)
        first_date = dates.min().normalize()
        last_date = dates.max().normalize()
        test_start = first_date + pd.DateOffset(years=self.walk_config.TRAIN_WINDOW)
        results = []
        period = 1
        while test_start <= last_date:
            test_end = (
                test_start
                + pd.DateOffset(years=self.walk_config.TEST_WINDOW)
                - pd.Timedelta(days=1)
            )
            if test_end > last_date:
                break
            train_mask = (dates >= first_date) & (
                dates <= test_start - pd.Timedelta(days=1)
            )
            test_mask = (dates >= test_start) & (dates <= test_end)
            fold_result = self._run_fold(train_mask, test_mask, period, horizon)
            results.extend(fold_result)
            period += 1
            test_start += pd.DateOffset(years=self.walk_config.STEP_SIZE)
        return results

    def _run_fold(self, train_mask, test_mask, period, horizon):
        train_raw = self.features.loc[train_mask]
        test_raw = self.features.loc[test_mask]
        y_train = self.target.loc[train_raw.index]
        y_test = self.target.loc[test_raw.index]
        train_valid = train_raw.notna().all(axis=1) & y_train.notna()
        test_valid = test_raw.notna().all(axis=1) & y_test.notna()
        train_raw, y_train = train_raw.loc[train_valid], y_train.loc[
            train_valid
        ].astype(int)
        test_raw, y_test = test_raw.loc[test_valid], y_test.loc[test_valid].astype(int)
        preprocessor = FeaturePreprocessor()
        train_features = preprocessor.fit_transform(train_raw)
        test_features = preprocessor.transform(test_raw)
        output = []
        for model_kind in self.models:
            model = CalibratedClassifierCV(
                make_model(model_kind),
                method="sigmoid",
                cv=MODEL_CONFIG.CALIBRATION_CV,
            )
            model.fit(train_features, y_train)
            train_scores = model.predict_proba(train_features)[:, 1]
            threshold = ThresholdOptimizer.optimize(y_train, train_scores)
            test_scores = model.predict_proba(test_features)[:, 1]
            predictions = test_scores >= threshold
            evaluator = ClassificationEvaluator()
            aggregate = evaluator.evaluate(y_test, predictions, test_scores)
            alert_metrics = pd.DataFrame(
                [
                    AlertEvaluator.evaluate(y_test, test_scores, fraction).__dict__
                    for fraction in MODEL_CONFIG.ALERT_FRACTIONS
                ]
            )
            alert_rows = self._alert_rows(test_raw, y_test, test_scores)
            per_ticker = self._per_ticker_metrics(y_test, predictions, test_scores)
            rule_baselines = self._rule_baselines(test_raw, y_test)
            yearly_metrics = self._yearly_metrics(y_test, predictions, test_scores)
            drift_metrics = self._drift_metrics(train_raw, test_raw)
            output.append(
                PooledFoldResult(
                    model_kind=model_kind,
                    horizon=horizon,
                    period=period,
                    aggregate_f1=aggregate.f1,
                    aggregate_pr_auc=aggregate.pr_auc,
                    baseline_f1=0.0,
                    positives=int(y_test.sum()),
                    positive_rate=float(y_test.mean()),
                    alert_metrics=alert_metrics,
                    rule_baselines=rule_baselines,
                    yearly_metrics=yearly_metrics,
                    drift_metrics=drift_metrics,
                    alert_rows=alert_rows,
                    per_ticker=per_ticker,
                )
            )
        return output

    @staticmethod
    def _per_ticker_metrics(y_test, predictions, scores):
        evaluator = ClassificationEvaluator()
        rows = []
        prediction_series = pd.Series(predictions, index=y_test.index)
        score_series = pd.Series(scores, index=y_test.index)
        for ticker in y_test.index.get_level_values("Ticker").unique():
            mask = y_test.index.get_level_values("Ticker") == ticker
            metrics = evaluator.evaluate(
                y_test.loc[mask],
                prediction_series.loc[mask].to_numpy(),
                score_series.loc[mask].to_numpy(),
            )
            rows.append({"Ticker": ticker, "F1": metrics.f1, "PR AUC": metrics.pr_auc})
        return pd.DataFrame(rows)

    @staticmethod
    def _rule_baselines(test_raw, y_test):
        evaluator = ClassificationEvaluator()
        rows = []
        rules = {
            "volatility_slope_positive": test_raw["Volatility_Slope_5d"] > 0,
            "return_shock_high": test_raw["Return_Shock_5d"]
            >= test_raw["Return_Shock_5d"].quantile(0.90),
            "bb_width_expanding": test_raw["BB_Width_Change_5d"] > 0,
        }
        for name, predictions in rules.items():
            metrics = evaluator.evaluate(
                y_test,
                predictions.astype(int).to_numpy(),
                predictions.astype(float).to_numpy(),
            )
            rows.append({"Rule": name, "F1": metrics.f1, "PR AUC": metrics.pr_auc})
        return pd.DataFrame(rows)

    @staticmethod
    def _yearly_metrics(y_test, predictions, scores):
        evaluator = ClassificationEvaluator()
        dates = y_test.index.get_level_values(0)
        rows = []
        prediction_series = pd.Series(predictions, index=y_test.index)
        score_series = pd.Series(scores, index=y_test.index)
        for year in sorted(dates.year.unique()):
            mask = dates.year == year
            metrics = evaluator.evaluate(
                y_test.loc[mask],
                prediction_series.loc[mask].to_numpy(),
                score_series.loc[mask].to_numpy(),
            )
            rows.append(
                {
                    "Year": year,
                    "F1": metrics.f1,
                    "PR AUC": metrics.pr_auc,
                    "Positive Rate": float(y_test.loc[mask].mean()),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _drift_metrics(train_raw, test_raw):
        common = train_raw.columns.intersection(test_raw.columns)
        train_mean = train_raw[common].mean()
        test_mean = test_raw[common].mean()
        train_std = train_raw[common].std().replace(0, np.nan)
        standardized_shift = ((test_mean - train_mean) / train_std).abs()
        return (
            standardized_shift.sort_values(ascending=False)
            .rename("absolute_standardized_mean_shift")
            .reset_index()
            .rename(columns={"index": "Feature"})
        )

    @staticmethod
    def _alert_rows(test_raw, y_test, scores):
        frame = test_raw.copy()
        frame["Target"] = y_test
        frame["Score"] = scores
        frame = frame.reset_index()
        return frame.sort_values("Score", ascending=False)
