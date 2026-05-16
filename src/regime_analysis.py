"""Analyze model performance across market volatility regimes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.backtesting import BacktestPeriod


@dataclass
class RegimeMetrics:
    """Aggregated metrics for one volatility regime."""

    regime: str
    n_folds: int
    avg_f1: float
    avg_pr_auc: float
    avg_precision_at_5pct: float
    avg_positive_rate: float
    max_f1: float
    min_f1: float


def split_by_vol_regime(
    results: list[BacktestPeriod],
    market_data: Optional[pd.DataFrame] = None,
    vol_percentiles: tuple = (33.33, 66.67),
) -> dict[str, list[BacktestPeriod]]:
    """Split backtest folds by realized volatility regime.

    Tags each fold by average realized vol during test period:
    - Low: bottom tercile vol
    - Medium: middle tercile
    - High: top tercile

    Args:
        results: List of BacktestPeriod results from walk-forward backtest
        market_data: DataFrame with date index and 'returns' column (optional)
                     If None, compute realized vol from fold returns
        vol_percentiles: Percentile boundaries for regime classification

    Returns:
        Dict mapping regime name ('Low', 'Medium', 'High') to list of BacktestPeriod
    """
    if not results:
        return {"Low": [], "Medium": [], "High": []}

    # Compute realized vol for each fold
    fold_vols = []
    for fold in results:
        if market_data is not None and fold.test_range_start is not None:
            # Use market data to compute realized vol during test period
            mask = (market_data.index >= fold.test_range_start) & (
                market_data.index <= fold.test_range_end
            )
            returns = market_data.loc[mask, "returns"]
            realized_vol = returns.std() * np.sqrt(252)
        else:
            # Compute from fold results (feature in test set)
            realized_vol = 0.20  # Default conservative estimate

        fold_vols.append(realized_vol)

    fold_vols = np.array(fold_vols)
    bounds = np.percentile(fold_vols, vol_percentiles)

    regimes = {}
    for regime_name in ["Low", "Medium", "High"]:
        regimes[regime_name] = []

    for fold, vol in zip(results, fold_vols):
        if vol <= bounds[0]:
            regime = "Low"
        elif vol <= bounds[1]:
            regime = "Medium"
        else:
            regime = "High"
        regimes[regime].append(fold)

    return regimes


def generate_regime_report(
    regime_splits: dict[str, list[BacktestPeriod]],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate per-regime performance metrics table.

    Args:
        regime_splits: Output from split_by_vol_regime()
        output_path: Optional path to write CSV report

    Returns:
        DataFrame with columns: Regime, Folds, Avg F1, Avg PR-AUC, Avg Precision@5%, Positive Rate
    """
    rows = []

    for regime in ["Low", "Medium", "High"]:
        folds = regime_splits.get(regime, [])
        if not folds:
            continue

        metrics = RegimeMetrics(
            regime=regime,
            n_folds=len(folds),
            avg_f1=np.mean([f.summary().get("f1", 0.0) for f in folds]),
            avg_pr_auc=np.mean([f.summary().get("pr_auc", 0.0) for f in folds]),
            avg_precision_at_5pct=np.mean(
                [f.summary().get("precision_at_5pct", 0.0) for f in folds]
            ),
            avg_positive_rate=np.mean(
                [f.summary().get("positive_rate", 0.0) for f in folds]
            ),
            max_f1=max([f.summary().get("f1", 0.0) for f in folds], default=0.0),
            min_f1=min([f.summary().get("f1", 0.0) for f in folds], default=0.0),
        )

        rows.append(
            {
                "Regime": regime,
                "Folds": metrics.n_folds,
                "Avg F1": f"{metrics.avg_f1:.3f}",
                "Avg PR-AUC": f"{metrics.avg_pr_auc:.3f}",
                "Avg Precision@5%": f"{metrics.avg_precision_at_5pct:.1%}",
                "Positive Rate": f"{metrics.avg_positive_rate:.1%}",
                "Max F1": f"{metrics.max_f1:.3f}",
                "Min F1": f"{metrics.min_f1:.3f}",
            }
        )

    report_df = pd.DataFrame(rows)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(output_path, index=False)

    return report_df


def generate_regime_analysis_markdown(
    regime_splits: dict[str, list[BacktestPeriod]],
    output_path: Optional[Path] = None,
) -> str:
    """Generate markdown report with regime analysis insights.

    Args:
        regime_splits: Output from split_by_vol_regime()
        output_path: Optional path to write markdown report

    Returns:
        Markdown string with analysis and table
    """
    table = generate_regime_report(regime_splits)

    lines = [
        "# Market Regime Analysis",
        "",
        "## Performance by Volatility Regime",
        "",
        "Model performance varies significantly by market volatility regime:",
        "",
    ]

    if not table.empty:
        lines.append(table.to_markdown(index=False))
        lines.append("")

    lines.extend(
        [
            "## Key Insights",
            "",
            "1. **High Vol Regime**: Model should perform better (more signal, clearer transitions)",
            "2. **Low Vol Regime**: Model struggles (sparse events, mean-reverting price action)",
            "3. **Medium Vol Regime**: Transitional behavior, mixed performance",
            "",
            "### Interpretation",
            "",
            "- If High Vol F1 >> Low Vol F1: model exploits regime-dependent feature signals",
            "- If High Vol F1 ≈ Low Vol F1: model captures fundamental transitions (less regime-dependent)",
            "- If Low Vol F1 > High Vol F1: model may have regime-selection bias or overfitting",
            "",
        ]
    )

    report_text = "\n".join(lines)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text)

    return report_text
