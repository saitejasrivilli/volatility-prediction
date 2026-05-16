"""Walk-forward evaluation and research backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .config import MODEL_CONFIG, TRADING_CONFIG, WALK_FORWARD_CONFIG
    from .feature_engineering import FeaturePreprocessor, TargetBuilder
except ImportError:  # pragma: no cover
    from config import MODEL_CONFIG, TRADING_CONFIG, WALK_FORWARD_CONFIG
    from feature_engineering import FeaturePreprocessor, TargetBuilder

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Single simulated trade."""

    date: datetime
    entry_price: float
    exit_price: float
    signal_strength: float
    position_size: float
    pnl_dollars: float
    pnl_pct: float


@dataclass
class ClassificationMetrics:
    """Classification quality for one prediction stream."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    confusion_matrix: Tuple[int, int, int, int]


@dataclass
class BaselineMetrics:
    """Naive references for one test fold."""

    always_negative: ClassificationMetrics
    persistence: ClassificationMetrics


@dataclass
class FoldData:
    """Aligned train/test data for one validation fold."""

    train_features: pd.DataFrame
    test_features: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    current_spike_test: pd.Series


@dataclass
class BacktestPeriod:
    """Results for one train/test period."""

    period_num: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    classification: ClassificationMetrics
    baselines: BaselineMetrics
    decision_threshold: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    cumulative_returns: np.ndarray
    trades: List[TradeResult]
    coefficients: Dict[str, float]

    def summary(self) -> Dict:
        """Return concise period metrics for CSV/reporting."""
        return {
            "Period": self.period_num,
            "Train": f"{self.train_start.date()} to {self.train_end.date()}",
            "Test": f"{self.test_start.date()} to {self.test_end.date()}",
            "Accuracy": f"{self.classification.accuracy:.1%}",
            "Precision": f"{self.classification.precision:.1%}",
            "Recall": f"{self.classification.recall:.1%}",
            "F1": f"{self.classification.f1:.1%}",
            "PR AUC": f"{self.classification.pr_auc:.3f}",
            "Brier": f"{self.classification.brier_score:.3f}",
            "Threshold": f"{self.decision_threshold:.2f}",
            "Baseline F1": f"{self.baselines.persistence.f1:.1%}",
            "Trades": self.total_trades,
            "Win Rate": f"{self.win_rate:.1%}",
            "Total Return": f"{self.total_return:.2%}",
            "Sharpe": f"{self.sharpe_ratio:.2f}",
        }


class ClassificationEvaluator:
    """Evaluate predictions and simple train-free baselines."""

    @staticmethod
    def evaluate(
        y_true: pd.Series, predictions: np.ndarray, scores: np.ndarray
    ) -> ClassificationMetrics:
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        roc_auc = roc_auc_score(y_true, scores) if y_true.nunique() > 1 else 0.0
        pr_auc = (
            average_precision_score(y_true, scores) if y_true.nunique() > 1 else 0.0
        )
        return ClassificationMetrics(
            accuracy=accuracy_score(y_true, predictions),
            precision=precision_score(y_true, predictions, zero_division=0),
            recall=recall_score(y_true, predictions, zero_division=0),
            f1=f1_score(y_true, predictions, zero_division=0),
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            brier_score=brier_score_loss(y_true, scores),
            confusion_matrix=(int(tn), int(fp), int(fn), int(tp)),
        )

    def baselines(
        self, y_test: pd.Series, current_spike_test: pd.Series
    ) -> BaselineMetrics:
        always_negative = np.zeros(len(y_test), dtype=int)
        persistence = current_spike_test.astype(int).to_numpy()
        return BaselineMetrics(
            always_negative=self.evaluate(y_test, always_negative, always_negative),
            persistence=self.evaluate(y_test, persistence, persistence),
        )


class ThresholdOptimizer:
    """Choose a decision threshold using training data only."""

    @staticmethod
    def optimize(y_true: pd.Series, scores: np.ndarray) -> float:
        thresholds = np.linspace(
            MODEL_CONFIG.MIN_THRESHOLD,
            MODEL_CONFIG.MAX_THRESHOLD,
            MODEL_CONFIG.THRESHOLD_STEPS,
        )
        utility_scores = [
            fbeta_score(
                y_true,
                scores >= threshold,
                beta=MODEL_CONFIG.F_BETA,
                zero_division=0,
            )
            for threshold in thresholds
        ]
        return float(thresholds[int(np.argmax(utility_scores))])


@dataclass
class AlertMetrics:
    """Ranking quality for a top-k alert budget."""

    alert_fraction: float
    alerts: int
    precision_at_k: float
    recall_at_k: float
    lift: float


class AlertEvaluator:
    """Evaluate rare-event ranking quality under finite alert budgets."""

    @staticmethod
    def evaluate(
        y_true: pd.Series, scores: np.ndarray, alert_fraction: float
    ) -> AlertMetrics:
        alerts = max(1, int(np.ceil(len(y_true) * alert_fraction)))
        top_indices = np.argsort(scores)[-alerts:]
        positives = y_true.to_numpy()
        hits = positives[top_indices].sum()
        prevalence = positives.mean()
        precision = float(hits / alerts)
        recall = float(hits / positives.sum()) if positives.sum() else 0.0
        lift = float(precision / prevalence) if prevalence else 0.0
        return AlertMetrics(alert_fraction, alerts, precision, recall, lift)


class PositionSizer:
    """Position sizing using either fixed sizing or fractional Kelly."""

    def __init__(
        self, account_size: float, risk_pct: float = 0.01, kelly_fraction: float = 0.25
    ):
        self.account_size = account_size
        self.risk_pct = risk_pct
        self.kelly_fraction = kelly_fraction

    def kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 0.0
        odds = avg_win / avg_loss
        kelly = (win_rate * odds - (1 - win_rate)) / odds
        return max(
            0.0, min(kelly * self.kelly_fraction, TRADING_CONFIG.MAX_POSITION_SIZE_PCT)
        )

    def size_position(
        self, signal_strength: float, volatility: float, kelly_pct: float
    ) -> float:
        if TRADING_CONFIG.POSITION_SIZE_METHOD == "fixed":
            return self.account_size * TRADING_CONFIG.FIXED_POSITION_PCT
        base_risk = self.account_size * self.risk_pct
        volatility_adjusted = base_risk / max(volatility, 0.001)
        max_position = self.account_size * TRADING_CONFIG.MAX_POSITION_SIZE_PCT
        return min(volatility_adjusted * kelly_pct * signal_strength, max_position)


class Backtester:
    """Simulate a long-volatility premium-hurdle strategy."""

    def __init__(self, config=TRADING_CONFIG):
        self.config = config

    def run_backtest(
        self,
        test_df: pd.DataFrame,
        signals: pd.Series,
        signal_strengths: pd.Series,
        dates: pd.DatetimeIndex,
        *,
        kelly_pct: float,
    ) -> Tuple[List[TradeResult], np.ndarray]:
        trades: List[TradeResult] = []
        account_value = self.config.ACCOUNT_SIZE
        cumulative_returns: List[float] = []
        sizer = PositionSizer(self.config.ACCOUNT_SIZE)

        for idx in range(len(test_df) - 1):
            signal = signals.iloc[idx]
            strength = signal_strengths.iloc[idx]
            if signal != 1 or strength < self.config.MIN_SIGNAL_STRENGTH:
                continue

            drawdown = (
                account_value - self.config.ACCOUNT_SIZE
            ) / self.config.ACCOUNT_SIZE
            if drawdown < -self.config.MAX_DRAWDOWN_LIMIT:
                logger.warning("Max drawdown limit hit: %.2f%%", drawdown * 100)
                break

            current_price = float(test_df["Close"].iloc[idx])
            next_price = float(test_df["Close"].iloc[idx + 1])
            volatility = float(test_df["Volatility_20d"].iloc[idx])
            realized_move = abs((next_price - current_price) / current_price)
            premium_hurdle = volatility * self.config.VOL_PREMIUM_MULTIPLIER
            execution_cost = self.config.BID_ASK_SPREAD_PCT + (
                2 * self.config.SLIPPAGE_PCT
            )
            commission_pct = self.config.COMMISSION_PER_TRADE / max(current_price, 1.0)
            pnl_pct = realized_move - premium_hurdle - execution_cost - commission_pct

            position_size = sizer.size_position(strength, volatility, kelly_pct)
            if position_size <= 0:
                continue
            pnl_dollars = position_size * pnl_pct
            account_value += pnl_dollars
            trades.append(
                TradeResult(
                    date=dates[idx],
                    entry_price=current_price,
                    exit_price=next_price,
                    signal_strength=float(strength),
                    position_size=position_size,
                    pnl_dollars=pnl_dollars,
                    pnl_pct=pnl_pct,
                )
            )
            cumulative_returns.append((account_value / self.config.ACCOUNT_SIZE) - 1)
        return trades, np.array(cumulative_returns)

    def calculate_metrics(self, trades: List[TradeResult], returns: np.ndarray) -> Dict:
        if not trades:
            return {}
        pnl_pcts = np.array([trade.pnl_pct for trade in trades])
        wins = pnl_pcts > 0
        losses = pnl_pcts < 0
        gross_profit = pnl_pcts[wins].sum() if wins.any() else 0.0
        gross_loss = abs(pnl_pcts[losses].sum()) if losses.any() else 0.0
        equity_curve = (
            np.concatenate([[1.0], returns + 1.0]) if len(returns) else np.array([1.0])
        )
        daily_returns = (
            (equity_curve[1:] / equity_curve[:-1]) - 1 if len(returns) else np.array([])
        )
        sharpe = self._ratio(
            daily_returns, daily_returns.std() if len(daily_returns) else 0.0
        )
        downside = daily_returns[daily_returns < 0]
        sortino = self._ratio(daily_returns, downside.std() if len(downside) else 0.0)
        running_max = np.maximum.accumulate(equity_curve)
        max_drawdown = ((equity_curve / running_max) - 1).min()
        total_return = returns[-1] if len(returns) else 0.0
        annual_return = (
            total_return * (252 / len(returns))
            if 0 < len(returns) < 252
            else total_return
        )
        return {
            "total_trades": len(trades),
            "winning_trades": int(wins.sum()),
            "losing_trades": int(losses.sum()),
            "win_rate": float(wins.mean()),
            "avg_win": float(pnl_pcts[wins].mean()) if wins.any() else 0.0,
            "avg_loss": float(abs(pnl_pcts[losses].mean())) if losses.any() else 0.0,
            "profit_factor": (
                float(gross_profit / gross_loss) if gross_loss > 0 else 0.0
            ),
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
        }

    def _ratio(self, daily_returns: np.ndarray, denominator: float) -> float:
        if len(daily_returns) == 0 or denominator == 0:
            return 0.0
        excess = daily_returns.mean() - (self.config.RISK_FREE_RATE_ANNUAL / 252)
        return float((excess / denominator) * np.sqrt(252))


class WalkForwardValidator:
    """Expanding-window validation with fold-local preprocessing and calibration."""

    def __init__(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        config=WALK_FORWARD_CONFIG,
        *,
        target_kind: str = "transition",
        model_kind: str = "logistic",
    ):
        self.df = df
        self.features = features
        self.config = config
        self.target_kind = target_kind
        self.model_kind = model_kind
        self.results: List[BacktestPeriod] = []
        self.full_returns = self.df["Close"].pct_change()
        self.full_target = self._build_target()
        volatility = self.full_returns.rolling(window=20).std()
        threshold = volatility.rolling(window=252).quantile(0.75)
        self.current_spike = (volatility > threshold).fillna(False).astype(int)
        self.current_transition = (
            self.current_spike.diff().clip(lower=0).fillna(0).astype(int)
        )

    def generate_windows(self) -> List[Tuple[str, str, str, str]]:
        windows = []
        first_date = self.df.index.min().normalize()
        last_date = self.df.index.max().normalize()
        test_start = first_date + pd.DateOffset(years=self.config.TRAIN_WINDOW)
        while test_start <= last_date:
            test_end = (
                test_start
                + pd.DateOffset(years=self.config.TEST_WINDOW)
                - pd.Timedelta(days=1)
            )
            if test_end > last_date:
                break
            windows.append(
                tuple(
                    item.strftime("%Y-%m-%d")
                    for item in (
                        first_date,
                        test_start - pd.Timedelta(days=1),
                        test_start,
                        test_end,
                    )
                )
            )
            test_start += pd.DateOffset(years=self.config.STEP_SIZE)
        return windows

    def run(self) -> List[BacktestPeriod]:
        self.results = []
        for period_num, window in enumerate(self.generate_windows(), 1):
            fold = self._build_fold_data(*window)
            if fold is None:
                continue
            model, threshold, train_scores = self._fit_model(fold)
            test_scores = model.predict_proba(fold.test_features)[:, 1]
            test_predictions = test_scores >= threshold
            evaluator = ClassificationEvaluator()
            classification = evaluator.evaluate(
                fold.y_test, test_predictions, test_scores
            )
            baselines = evaluator.baselines(fold.y_test, fold.current_spike_test)
            kelly_pct = self._estimate_kelly_from_training(
                fold, train_scores, threshold
            )
            trades, returns = Backtester().run_backtest(
                fold.test_df,
                pd.Series(test_predictions.astype(int), index=fold.test_df.index),
                pd.Series(test_scores, index=fold.test_df.index),
                fold.test_df.index,
                kelly_pct=kelly_pct,
            )
            self.results.append(
                self._build_period_result(
                    period_num,
                    window,
                    classification,
                    baselines,
                    threshold,
                    trades,
                    returns,
                    model,
                )
            )
        return self.results

    def _build_fold_data(
        self, train_start, train_end, test_start, test_end
    ) -> FoldData | None:
        train_mask = (self.df.index.astype(str) >= train_start) & (
            self.df.index.astype(str) <= train_end
        )
        test_mask = (self.df.index.astype(str) >= test_start) & (
            self.df.index.astype(str) <= test_end
        )
        train_raw = self.features.loc[train_mask]
        test_raw = self.features.loc[test_mask]
        train_valid = (
            train_raw.notna().all(axis=1) & self.full_target.loc[train_mask].notna()
        )
        test_valid = (
            test_raw.notna().all(axis=1) & self.full_target.loc[test_mask].notna()
        )
        train_raw = train_raw.loc[train_valid]
        test_raw = test_raw.loc[test_valid]
        if len(train_raw) < 100 or len(test_raw) < 50:
            return None
        preprocessor = FeaturePreprocessor()
        volatility_feature = self.features[["Volatility_20d"]]
        return FoldData(
            train_features=preprocessor.fit_transform(train_raw),
            test_features=preprocessor.transform(test_raw),
            y_train=self.full_target.loc[train_raw.index].astype(int),
            y_test=self.full_target.loc[test_raw.index].astype(int),
            train_df=self.df.loc[train_raw.index].join(volatility_feature, how="left"),
            test_df=self.df.loc[test_raw.index].join(volatility_feature, how="left"),
            current_spike_test=(
                self.current_transition.loc[test_raw.index]
                if self.target_kind == "transition"
                else self.current_spike.loc[test_raw.index]
            ),
        )

    def _fit_model(self, fold: FoldData):
        base_model = self._make_model()
        model = CalibratedClassifierCV(
            base_model, method="sigmoid", cv=MODEL_CONFIG.CALIBRATION_CV
        )
        model.fit(fold.train_features, fold.y_train)
        train_scores = model.predict_proba(fold.train_features)[:, 1]
        return (
            model,
            ThresholdOptimizer.optimize(fold.y_train, train_scores),
            train_scores,
        )

    def _build_target(self) -> pd.Series:
        if self.target_kind == "transition":
            return TargetBuilder.volatility_transition_target(
                self.df["Close"], self.full_returns
            )
        if self.target_kind == "spike":
            return TargetBuilder.volatility_spike_target(
                self.df["Close"], self.full_returns
            )
        raise ValueError(f"Unknown target kind: {self.target_kind}")

    def _make_model(self):
        if self.model_kind == "logistic":
            return LogisticRegression(
                C=MODEL_CONFIG.LR_C,
                l1_ratio=0,
                solver=MODEL_CONFIG.LR_SOLVER,
                max_iter=MODEL_CONFIG.LR_MAX_ITER,
                random_state=MODEL_CONFIG.RANDOM_STATE,
                class_weight="balanced",
            )
        if self.model_kind == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=10,
                random_state=MODEL_CONFIG.RANDOM_STATE,
                class_weight="balanced",
            )
        raise ValueError(f"Unknown model kind: {self.model_kind}")

    def _estimate_kelly_from_training(
        self, fold: FoldData, train_scores: np.ndarray, threshold: float
    ) -> float:
        if TRADING_CONFIG.POSITION_SIZE_METHOD == "fixed":
            return TRADING_CONFIG.FIXED_POSITION_PCT
        train_predictions = train_scores >= threshold
        trades, _ = Backtester().run_backtest(
            fold.train_df,
            pd.Series(train_predictions.astype(int), index=fold.train_df.index),
            pd.Series(train_scores, index=fold.train_df.index),
            fold.train_df.index,
            kelly_pct=TRADING_CONFIG.FIXED_POSITION_PCT,
        )
        if len(trades) < 10:
            return TRADING_CONFIG.FIXED_POSITION_PCT
        pnl = np.array([trade.pnl_pct for trade in trades])
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        if len(wins) == 0 or len(losses) == 0:
            return TRADING_CONFIG.FIXED_POSITION_PCT
        return PositionSizer(TRADING_CONFIG.ACCOUNT_SIZE).kelly_size(
            len(wins) / len(pnl), float(wins.mean()), float(abs(losses.mean()))
        )

    def _build_period_result(
        self,
        period_num: int,
        window,
        classification: ClassificationMetrics,
        baselines: BaselineMetrics,
        threshold: float,
        trades: List[TradeResult],
        returns: np.ndarray,
        model,
    ) -> BacktestPeriod:
        metrics = Backtester().calculate_metrics(trades, returns)
        return BacktestPeriod(
            period_num=period_num,
            train_start=pd.to_datetime(window[0]),
            train_end=pd.to_datetime(window[1]),
            test_start=pd.to_datetime(window[2]),
            test_end=pd.to_datetime(window[3]),
            classification=classification,
            baselines=baselines,
            decision_threshold=threshold,
            total_trades=metrics.get("total_trades", 0),
            winning_trades=metrics.get("winning_trades", 0),
            losing_trades=metrics.get("losing_trades", 0),
            win_rate=metrics.get("win_rate", 0.0),
            avg_win=metrics.get("avg_win", 0.0),
            avg_loss=metrics.get("avg_loss", 0.0),
            profit_factor=metrics.get("profit_factor", 0.0),
            total_return=metrics.get("total_return", 0.0),
            annual_return=metrics.get("annual_return", 0.0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            sortino_ratio=metrics.get("sortino_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            cumulative_returns=returns,
            trades=trades,
            coefficients=self._extract_coefficients(model),
        )

    def coefficient_stability(self) -> pd.DataFrame:
        """Summarize signed coefficient stability across validation folds."""
        if not self.results:
            return pd.DataFrame()
        coefficient_frame = pd.DataFrame(
            [result.coefficients for result in self.results]
        )
        if coefficient_frame.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "feature": coefficient_frame.columns,
                "mean_coefficient": coefficient_frame.mean().values,
                "mean_abs_coefficient": coefficient_frame.abs().mean().values,
                "std_coefficient": coefficient_frame.std(ddof=0).values,
                "positive_fraction": (coefficient_frame > 0).mean().values,
            }
        ).sort_values("mean_abs_coefficient", ascending=False)

    @staticmethod
    def _extract_coefficients(model) -> Dict[str, float]:
        estimators = [item.estimator for item in model.calibrated_classifiers_]
        if not estimators or not hasattr(estimators[0], "coef_"):
            return {}
        coefficient_matrix = np.vstack([estimator.coef_[0] for estimator in estimators])
        mean_coefficients = coefficient_matrix.mean(axis=0)
        return dict(zip(model.feature_names_in_, mean_coefficients))

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([result.summary() for result in self.results])

    def print_summary(self) -> None:
        if self.results:
            logger.info("\n%s", self.summary().to_string(index=False))


def make_model(model_kind: str):
    """Public model factory for experiment runners."""
    if model_kind == "logistic":
        return LogisticRegression(
            C=MODEL_CONFIG.LR_C,
            l1_ratio=0,
            solver=MODEL_CONFIG.LR_SOLVER,
            max_iter=MODEL_CONFIG.LR_MAX_ITER,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            class_weight="balanced",
        )
    if model_kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            class_weight="balanced",
        )
    raise ValueError(f"Unknown model kind: {model_kind}")
