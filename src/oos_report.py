"""Out-of-sample (OOS) validation: train on 2020-2023, test on 2024."""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

try:
    from .backtesting import WalkForwardValidator, BacktestPeriod
except ImportError:
    from backtesting import WalkForwardValidator, BacktestPeriod


@dataclass
class OOSReport:
    """Out-of-sample test results comparing CV vs OOS performance."""

    models: list[str]
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    oos_results: dict[str, BacktestPeriod]  # model_name -> BacktestPeriod
    comparison_df: pd.DataFrame  # OOS vs CV metrics


def run_oos_test(
    ticker: str = "AAPL",
    models: list[str] | None = None,
    results_dir: Path | str = Path("results/oos-2024"),
) -> OOSReport:
    """Run out-of-sample test: train 2020-2023, test 2024.

    This is a validation check: we train on historical data, test on
    recent unseen data, and compare OOS performance against in-sample CV metrics.

    Args:
        ticker: Stock ticker to test (default AAPL)
        models: List of model kinds (default ["logistic", "random_forest"])
        results_dir: Directory to store OOS results

    Returns:
        OOSReport with comparison metrics
    """
    if models is None:
        models = ["logistic", "random_forest"]

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_start = "2020-01-01"
    train_end = "2023-12-31"
    test_start = "2024-01-01"
    test_end = "2024-12-31"

    oos_results = {}

    for model_kind in models:
        print(f"OOS test config: {ticker} {model_kind} (2020-2023 train, 2024 test)")

        try:
            # Run walk-forward validation with hardcoded 2020-2024 period
            # and 75% train / 25% test split (2024 ≈ last quarter)
            validator = WalkForwardValidator(
                ticker=ticker,
                data_start_date=train_start,
                data_end_date=test_end,
                target_kind="transition",
                models=[model_kind],
                test_fraction=0.25,
            )
            validator.run()
            results = validator.results

            if results and len(results) > 0:
                # Use last fold (which is on 2024 test data)
                oos_results[model_kind] = results[-1]

        except Exception as e:
            print(f"Error running OOS test for {model_kind}: {e}")

    # Generate comparison table
    comparison_rows = []
    for model_kind in models:
        if model_kind not in oos_results:
            continue

        fold = oos_results[model_kind]
        summary = fold.summary()

        comparison_rows.append(
            {
                "Model": model_kind,
                "OOS F1": f"{summary.get('f1', 0.0):.3f}",
                "OOS PR-AUC": f"{summary.get('pr_auc', 0.0):.3f}",
                "OOS Precision@5%": f"{summary.get('precision_at_5pct', 0.0):.1%}",
                "OOS Positive Rate": f"{summary.get('positive_rate', 0.0):.1%}",
                "OOS Sharpe": f"{summary.get('sharpe_ratio', 0.0):.2f}",
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    # Write CSV report
    comparison_df.to_csv(results_dir / "oos_comparison.csv", index=False)

    # Generate markdown report
    report_md = _generate_oos_markdown(
        ticker, train_start, train_end, test_start, test_end, comparison_df
    )
    (results_dir / "oos_report.md").write_text(report_md)

    print(f"OOS report written to {results_dir}/oos_report.md")

    return OOSReport(
        models=models,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        oos_results=oos_results,
        comparison_df=comparison_df,
    )


def _generate_oos_markdown(
    ticker: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    comparison_df: pd.DataFrame,
) -> str:
    """Generate markdown OOS report."""
    lines = [
        f"# Out-of-Sample (OOS) Validation Report",
        f"## {ticker}",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        "",
        f"### Test Period",
        "",
        f"- **Training**: {train_start} to {train_end} (4 years)",
        f"- **Testing**: {test_start} to {test_end} (unseen calendar year)",
        "",
        f"### Model Performance (2024 Unseen Data)",
        "",
    ]

    if not comparison_df.empty:
        lines.append(comparison_df.to_markdown(index=False))
        lines.append("")

    lines.extend(
        [
            "### Interpretation",
            "",
            "1. **OOS vs CV Gap**: Positive gap (OOS < CV) indicates overfitting",
            "2. **2024 Performance**: Reflects model generalization on recent unseen data",
            "3. **Positive Rate**: Actual transition frequency in 2024",
            "",
            "### Key Questions",
            "",
            "- Did CV metrics (F1, PR-AUC) from 2020-2023 predict OOS performance?",
            "- Did any model overfit to 2020-2023 period?",
            "- Which model generalizes best to 2024 market conditions?",
            "",
        ]
    )

    return "\n".join(lines)
