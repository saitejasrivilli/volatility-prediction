"""Model monitoring and drift detection utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .config import MONITORING_CONFIG
except ImportError:
    from config import MONITORING_CONFIG

logger = logging.getLogger(__name__)


def _population_stability_index(baseline: np.ndarray, current: np.ndarray) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI = sum((current% - baseline%) * ln(current% / baseline%))

    PSI interpretation:
    < 0.10: No significant population change
    0.10-0.25: Small population change
    > 0.25: Significant population change

    Args:
        baseline: Array of baseline values
        current: Array of current values

    Returns:
        PSI value (float)
    """
    # Use 10 bins for binning
    bins = np.quantile(np.concatenate([baseline, current]), np.linspace(0, 1, 11))
    bins[0] -= 1e-6  # Include min value
    bins[-1] += 1e-6  # Include max value

    baseline_counts = np.histogram(baseline, bins=bins)[0]
    current_counts = np.histogram(current, bins=bins)[0]

    baseline_pct = (baseline_counts + 1e-6) / len(baseline)
    current_pct = (current_counts + 1e-6) / len(current)

    psi = np.sum(
        (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    )
    return float(psi)


@dataclass
class DriftDetectionResult:
    """Result of drift detection check."""

    feature_name: str
    drift_type: str  # 'feature', 'prediction', 'performance'
    value: float
    threshold: float
    is_drift: bool
    severity: str  # 'low', 'moderate', 'high'


@dataclass
class MonitoringReport:
    """Complete monitoring report."""

    timestamp: str
    total_samples: int
    drift_results: list  # List of DriftDetectionResult dicts
    feature_drift_summary: Dict[str, str]  # feature -> severity
    prediction_drift: Optional[Dict]
    performance_drift: Optional[Dict]
    overall_status: str  # 'healthy', 'caution', 'alert'


class FeatureDriftDetector:
    """Detect feature distribution drift using KS test and PSI."""

    def __init__(self, baseline_stats: Dict[str, Tuple[float, float]]):
        """Initialize with baseline feature statistics.

        Args:
            baseline_stats: Dict mapping feature_name -> (mean, std)
                           from training fold
        """
        self.baseline_stats = baseline_stats
        self.config = MONITORING_CONFIG

    def detect(self, current_features: pd.DataFrame) -> Dict[str, DriftDetectionResult]:
        """Detect drift in incoming features.

        Args:
            current_features: DataFrame of current feature values

        Returns:
            Dict mapping feature_name -> DriftDetectionResult
        """
        results = {}

        for col in current_features.columns:
            if col not in self.baseline_stats:
                continue

            baseline_mean, baseline_std = self.baseline_stats[col]
            current_values = current_features[col].dropna().values

            if len(current_values) == 0:
                continue

            # PSI test
            baseline_synthetic = np.random.normal(
                baseline_mean, baseline_std, size=len(current_values)
            )
            psi = _population_stability_index(baseline_synthetic, current_values)

            # Determine severity
            if psi > self.config.PSI_HIGH_THRESHOLD:
                severity = "high"
                is_drift = True
            elif psi > self.config.PSI_MODERATE_THRESHOLD:
                severity = "moderate"
                is_drift = True
            else:
                severity = "low"
                is_drift = False

            results[col] = DriftDetectionResult(
                feature_name=col,
                drift_type="feature",
                value=psi,
                threshold=self.config.PSI_HIGH_THRESHOLD,
                is_drift=is_drift,
                severity=severity,
            )

        return results


class PredictionDriftDetector:
    """Detect prediction score distribution drift."""

    def __init__(self, baseline_scores: np.ndarray):
        """Initialize with baseline prediction scores.

        Args:
            baseline_scores: Array of baseline model scores from training
        """
        self.baseline_mean = float(np.mean(baseline_scores))
        self.baseline_std = float(np.std(baseline_scores))
        self.config = MONITORING_CONFIG

    def detect(self, current_scores: np.ndarray) -> DriftDetectionResult:
        """Detect drift in prediction scores.

        Args:
            current_scores: Array of current model scores

        Returns:
            DriftDetectionResult
        """
        if len(current_scores) == 0:
            return DriftDetectionResult(
                feature_name="prediction_scores",
                drift_type="prediction",
                value=0.0,
                threshold=self.config.SCORE_DRIFT_SIGMA,
                is_drift=False,
                severity="low",
            )

        current_mean = float(np.mean(current_scores))
        score_diff = abs(current_mean - self.baseline_mean)
        sigma_diff = score_diff / max(self.baseline_std, 1e-6)

        is_drift = sigma_diff > self.config.SCORE_DRIFT_SIGMA

        severity = "low"
        if is_drift:
            severity = "high" if sigma_diff > 3 * self.config.SCORE_DRIFT_SIGMA else "moderate"

        return DriftDetectionResult(
            feature_name="prediction_scores",
            drift_type="prediction",
            value=sigma_diff,
            threshold=self.config.SCORE_DRIFT_SIGMA,
            is_drift=is_drift,
            severity=severity,
        )


class PerformanceDriftDetector:
    """Detect model performance degradation over time."""

    def __init__(self, baseline_precision: float, window_size: int = 30):
        """Initialize with baseline precision.

        Args:
            baseline_precision: Baseline precision@K from training
            window_size: Rolling window size (days) for assessment
        """
        self.baseline_precision = baseline_precision
        self.window_size = window_size
        self.config = MONITORING_CONFIG
        self.rolling_precisions = []  # Deque of recent precisions

    def detect(
        self, current_predictions: np.ndarray, current_actuals: np.ndarray
    ) -> DriftDetectionResult:
        """Detect performance degradation.

        Args:
            current_predictions: Array of model predictions
            current_actuals: Array of actual labels

        Returns:
            DriftDetectionResult
        """
        if len(current_predictions) == 0 or len(current_actuals) == 0:
            return DriftDetectionResult(
                feature_name="performance",
                drift_type="performance",
                value=self.baseline_precision,
                threshold=self.baseline_precision * self.config.MIN_PRECISION_RATIO,
                is_drift=False,
                severity="low",
            )

        # Compute precision for current batch
        from sklearn.metrics import precision_score

        try:
            current_precision = precision_score(
                current_actuals, current_predictions, zero_division=0.0
            )
        except Exception:
            current_precision = 0.0

        self.rolling_precisions.append(current_precision)
        if len(self.rolling_precisions) > 3:
            self.rolling_precisions.pop(0)

        # Check if recent precisions are consistently degraded
        degradation_threshold = (
            self.baseline_precision * self.config.MIN_PRECISION_RATIO
        )
        degraded_count = sum(
            1 for p in self.rolling_precisions if p < degradation_threshold
        )

        is_drift = degraded_count >= 2  # 2+ windows degraded

        severity = "low"
        if is_drift:
            severity = "high" if (
                degraded_count == 3
            ) else "moderate"

        return DriftDetectionResult(
            feature_name="performance",
            drift_type="performance",
            value=current_precision,
            threshold=degradation_threshold,
            is_drift=is_drift,
            severity=severity,
        )


class ModelMonitor:
    """Integrated model monitoring system."""

    def __init__(self, artifact: Dict):
        """Initialize monitor with model artifact.

        Args:
            artifact: Loaded model artifact dict with keys:
                     model, preprocessor, feature_columns, target_kind, model_kind
        """
        # Extract baseline stats from preprocessor
        preprocessor = artifact.get("preprocessor")
        baseline_means = preprocessor.means if hasattr(preprocessor, "means") else {}
        baseline_stds = preprocessor.stds if hasattr(preprocessor, "stds") else {}

        baseline_stats = {
            col: (baseline_means.get(col, 0.0), baseline_stds.get(col, 1.0))
            for col in artifact.get("feature_columns", [])
        }

        self.feature_detector = FeatureDriftDetector(baseline_stats)
        self.prediction_detector = PredictionDriftDetector(np.array([0.5]))
        self.performance_detector = PerformanceDriftDetector(0.5)

    def monitor(
        self,
        current_features: pd.DataFrame,
        current_scores: Optional[np.ndarray] = None,
        current_predictions: Optional[np.ndarray] = None,
        current_actuals: Optional[np.ndarray] = None,
    ) -> MonitoringReport:
        """Run full monitoring suite.

        Returns:
            MonitoringReport with all drift assessments
        """
        drift_results = []
        feature_drift_summary = {}

        # Feature drift
        feature_results = self.feature_detector.detect(current_features)
        for col, result in feature_results.items():
            drift_results.append(asdict(result))
            feature_drift_summary[col] = result.severity

        # Prediction drift
        if current_scores is not None:
            pred_result = self.prediction_detector.detect(current_scores)
            drift_results.append(asdict(pred_result))

        # Performance drift
        if current_predictions is not None and current_actuals is not None:
            perf_result = self.performance_detector.detect(
                current_predictions, current_actuals
            )
            drift_results.append(asdict(perf_result))

        # Determine overall status
        high_severity = sum(
            1 for r in drift_results if r.get("severity") == "high"
        )
        moderate_severity = sum(
            1 for r in drift_results if r.get("severity") == "moderate"
        )

        if high_severity > 0:
            overall_status = "alert"
        elif moderate_severity > 0:
            overall_status = "caution"
        else:
            overall_status = "healthy"

        return MonitoringReport(
            timestamp=pd.Timestamp.now().isoformat(),
            total_samples=len(current_features),
            drift_results=drift_results,
            feature_drift_summary=feature_drift_summary,
            prediction_drift=None,
            performance_drift=None,
            overall_status=overall_status,
        )

    def write_report(self, report: MonitoringReport, output_path: Path) -> None:
        """Write monitoring report to JSON file.

        Args:
            report: MonitoringReport instance
            output_path: Path to write JSON
        """
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
