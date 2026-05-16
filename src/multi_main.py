"""CLI for pooled multi-ticker transition experiments."""

from argparse import ArgumentParser
from dataclasses import replace
import logging
from pathlib import Path

import pandas as pd

try:
    from .config import DATA_CONFIG, MODEL_CONFIG, WALK_FORWARD_CONFIG
    from .data_pipeline import load_multi_ticker_data
    from .pooling import build_pooled_dataset
    from .pooled_experiments import PooledExperimentRunner
except ImportError:  # pragma: no cover
    from config import DATA_CONFIG, MODEL_CONFIG, WALK_FORWARD_CONFIG
    from data_pipeline import load_multi_ticker_data
    from pooling import build_pooled_dataset
    from pooled_experiments import PooledExperimentRunner

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = ArgumentParser(
        description="Run pooled multi-ticker transition experiments."
    )
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--models", nargs="+", default=MODEL_CONFIG.ENABLED_MODELS)
    parser.add_argument("--results-dir", default="results/pooled")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-step logs from repeated pooled feature builds.",
    )
    return parser.parse_args(argv)


def configure_pooled_logging(verbose: bool) -> None:
    """Keep pooled runs readable by default while preserving opt-in detail."""
    if verbose:
        return
    logging.getLogger("src.feature_engineering").setLevel(logging.WARNING)


def write_executive_summary(
    output: Path,
    summary: pd.DataFrame,
    alerts: pd.DataFrame,
    rules: pd.DataFrame,
) -> None:
    """Write a compact reviewer-facing interpretation of pooled artifacts."""
    best_f1 = summary.loc[summary["F1"].idxmax()]
    best_alert = alerts.loc[alerts["lift"].idxmax()]
    matching_rules = rules[
        (rules["Horizon"] == best_f1["Horizon"])
        & (rules["Period"] == best_f1["Period"])
    ]
    best_rule = matching_rules.loc[matching_rules["F1"].idxmax()]
    positive_rates = summary.groupby("Period")["Positive Rate"].mean().sort_index()
    caveat = (
        f"Positive rate shifted from {positive_rates.iloc[0]:.2%} "
        f"in period {positive_rates.index[0]} to {positive_rates.iloc[-1]:.2%} "
        f"in period {positive_rates.index[-1]}, so regime stability remains a key risk."
    )
    lines = [
        "# Pooled Research Executive Summary",
        "",
        "## Best aggregate configuration",
        "",
        (
            f"- `{best_f1['Model']}` at horizon `{int(best_f1['Horizon'])}` day(s), "
            f"period `{int(best_f1['Period'])}`: F1 `{best_f1['F1']:.3f}`, "
            f"PR-AUC `{best_f1['PR AUC']:.3f}`."
        ),
        "",
        "## Strongest alert-ranking result",
        "",
        (
            f"- `{best_alert['Model']}` at horizon `{int(best_alert['Horizon'])}` day(s), "
            f"period `{int(best_alert['Period'])}`, top `{best_alert['alert_fraction']:.0%}` alerts: "
            f"precision `{best_alert['precision_at_k']:.1%}`, "
            f"recall `{best_alert['recall_at_k']:.1%}`, "
            f"lift `{best_alert['lift']:.1f}x`."
        ),
        "",
        "## Best simple-rule comparator for the best-F1 fold",
        "",
        (
            f"- `{best_rule['Rule']}`: F1 `{best_rule['F1']:.3f}`, "
            f"PR-AUC `{best_rule['PR AUC']:.3f}` versus model F1 `{best_f1['F1']:.3f}`."
        ),
        "",
        "## Main caveat",
        "",
        f"- {caveat}",
        "",
        "## Interpretation",
        "",
        (
            "- The single-name task is sparse; the pooled setting is the more useful "
            "research framing."
        ),
        (
            "- Alert lift is the most informative deployment-style metric for this "
            "rare-event problem."
        ),
        (
            "- Treat this as evidence of learnable structure in some regimes, not as "
            "production-ready alpha."
        ),
    ]
    (output / "pooled_executive_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv=None):
    args = parse_args(argv)
    configure_pooled_logging(args.verbose)
    data_config = replace(
        DATA_CONFIG,
        TICKERS=args.tickers or DATA_CONFIG.TICKERS,
        START_DATE=args.start_date or DATA_CONFIG.START_DATE,
        END_DATE=args.end_date or DATA_CONFIG.END_DATE,
    )
    output = Path(args.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Loading %s tickers for pooled research: %s",
        len(data_config.TICKERS),
        ", ".join(data_config.TICKERS),
    )
    market_data_by_ticker = load_multi_ticker_data(data_config)
    logger.info("Loaded market data for %s tickers", len(market_data_by_ticker))
    all_results = []
    per_ticker_frames = []
    alert_frames = []
    rule_frames = []
    yearly_frames = []
    drift_frames = []
    ranked_alert_frames = []
    for horizon in args.horizons:
        logger.info("Running pooled experiments for horizon=%s day(s)", horizon)
        market_data, features, target = build_pooled_dataset(
            market_data_by_ticker, horizon=horizon
        )
        runner = PooledExperimentRunner(
            market_data, features, target, WALK_FORWARD_CONFIG, args.models
        )
        results = runner.run(horizon)
        for result in results:
            all_results.append(
                {
                    "Model": result.model_kind,
                    "Horizon": result.horizon,
                    "Period": result.period,
                    "F1": result.aggregate_f1,
                    "PR AUC": result.aggregate_pr_auc,
                    "Positives": result.positives,
                    "Positive Rate": result.positive_rate,
                }
            )
            ticker_frame = result.per_ticker.copy()
            ticker_frame["Model"] = result.model_kind
            ticker_frame["Horizon"] = result.horizon
            ticker_frame["Period"] = result.period
            per_ticker_frames.append(ticker_frame)
            alert_frame = result.alert_metrics.copy()
            alert_frame["Model"] = result.model_kind
            alert_frame["Horizon"] = result.horizon
            alert_frame["Period"] = result.period
            alert_frames.append(alert_frame)
            rule_frame = result.rule_baselines.copy()
            rule_frame["Model"] = result.model_kind
            rule_frame["Horizon"] = result.horizon
            rule_frame["Period"] = result.period
            rule_frames.append(rule_frame)
            yearly_frame = result.yearly_metrics.copy()
            yearly_frame["Model"] = result.model_kind
            yearly_frame["Horizon"] = result.horizon
            yearly_frame["Period"] = result.period
            yearly_frames.append(yearly_frame)
            drift_frame = result.drift_metrics.head(10).copy()
            drift_frame["Model"] = result.model_kind
            drift_frame["Horizon"] = result.horizon
            drift_frame["Period"] = result.period
            drift_frames.append(drift_frame)
            ranked_alerts = result.alert_rows.copy()
            ranked_alerts["Model"] = result.model_kind
            ranked_alerts["Horizon"] = result.horizon
            ranked_alerts["Period"] = result.period
            ranked_alert_frames.append(ranked_alerts)
    summary_frame = pd.DataFrame(all_results)
    per_ticker_frame = pd.concat(per_ticker_frames, ignore_index=True)
    alert_frame = pd.concat(alert_frames, ignore_index=True)
    rule_frame = pd.concat(rule_frames, ignore_index=True)
    yearly_frame = pd.concat(yearly_frames, ignore_index=True)
    drift_frame = pd.concat(drift_frames, ignore_index=True)
    ranked_alert_frame = pd.concat(ranked_alert_frames, ignore_index=True)
    summary_frame.to_csv(output / "pooled_summary.csv", index=False)
    per_ticker_frame.to_csv(output / "pooled_per_ticker.csv", index=False)
    alert_frame.to_csv(output / "pooled_alert_metrics.csv", index=False)
    rule_frame.to_csv(output / "pooled_rule_baselines.csv", index=False)
    yearly_frame.to_csv(output / "pooled_yearly_metrics.csv", index=False)
    drift_frame.to_csv(output / "pooled_drift_metrics.csv", index=False)
    ranked_alert_frame.to_csv(output / "pooled_ranked_alerts.csv", index=False)
    write_executive_summary(output, summary_frame, alert_frame, rule_frame)
    logger.info("Saved pooled outputs to %s", output)


if __name__ == "__main__":
    main()
