"""CLI entrypoint for the volatility prediction pipeline."""

from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import sys

try:
    from .config import (
        DATA_CONFIG,
        FEATURE_CONFIG,
        LOGGING_CONFIG,
        MODEL_CONFIG,
        TRADING_CONFIG,
        WALK_FORWARD_CONFIG,
    )
    from .data_pipeline import load_data
    from .experiments import ExperimentRunner
    from .feature_engineering import TargetBuilder, build_features
except ImportError:  # pragma: no cover
    from config import (
        DATA_CONFIG,
        FEATURE_CONFIG,
        LOGGING_CONFIG,
        MODEL_CONFIG,
        TRADING_CONFIG,
        WALK_FORWARD_CONFIG,
    )
    from data_pipeline import load_data
    from experiments import ExperimentRunner
    from feature_engineering import TargetBuilder, build_features

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = ArgumentParser(
        description="Run walk-forward volatility research pipeline."
    )
    parser.add_argument("--ticker")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--train-window-years", type=int)
    parser.add_argument("--test-window-years", type=int)
    parser.add_argument("--step-size-years", type=int)
    parser.add_argument("--results-dir")
    parser.add_argument("--target-kind", choices=["transition", "spike"])
    parser.add_argument("--models", nargs="+")
    return parser.parse_args(argv)


def configure_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(results_dir / LOGGING_CONFIG.LOG_FILE),
            logging.StreamHandler(),
        ],
        force=True,
    )


def apply_overrides(args):
    data_config = replace(
        DATA_CONFIG,
        TICKER=args.ticker or DATA_CONFIG.TICKER,
        START_DATE=args.start_date or DATA_CONFIG.START_DATE,
        END_DATE=args.end_date or DATA_CONFIG.END_DATE,
    )
    walk_config = replace(
        WALK_FORWARD_CONFIG,
        TRAIN_WINDOW=args.train_window_years or WALK_FORWARD_CONFIG.TRAIN_WINDOW,
        TEST_WINDOW=args.test_window_years or WALK_FORWARD_CONFIG.TEST_WINDOW,
        STEP_SIZE=args.step_size_years or WALK_FORWARD_CONFIG.STEP_SIZE,
    )
    results_dir = Path(args.results_dir or LOGGING_CONFIG.RESULTS_DIR)
    target_kind = args.target_kind or "transition"
    models = args.models or MODEL_CONFIG.ENABLED_MODELS
    return data_config, walk_config, results_dir, target_kind, models


def write_metadata(results_dir: Path, data_config, walk_config) -> None:
    metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "data_config": asdict(data_config),
        "walk_forward_config": asdict(walk_config),
        "feature_config": asdict(FEATURE_CONFIG),
        "model_config": asdict(MODEL_CONFIG),
        "trading_config": asdict(TRADING_CONFIG),
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main(argv=None):
    """Run the pipeline and return fold results."""
    args = parse_args(argv)
    data_config, walk_config, results_dir, target_kind, models = apply_overrides(args)
    configure_logging(results_dir)
    logger.info("Starting pipeline for %s", data_config.TICKER)

    market_data = load_data(data_config)
    features = build_features(market_data, FEATURE_CONFIG)
    features.to_csv(results_dir / "features_raw.csv")

    target_builder = (
        TargetBuilder.volatility_transition_target
        if target_kind == "transition"
        else TargetBuilder.volatility_spike_target
    )
    target = target_builder(market_data["Close"], market_data["Close"].pct_change())
    known_target = target.dropna()
    logger.info(
        "Known target distribution: positives=%s (%.1f%%), negatives=%s (%.1f%%)",
        int((known_target == 1).sum()),
        (known_target == 1).mean() * 100,
        int((known_target == 0).sum()),
        (known_target == 0).mean() * 100,
    )

    runner = ExperimentRunner(market_data, features, walk_config, target_kind, models)
    results = runner.run()
    registry = runner.registry(results)
    registry.to_csv(results_dir / "experiment_registry.csv", index=False)
    for result in results:
        result.summary().to_csv(
            results_dir / f"validation_summary_{result.model_kind}.csv", index=False
        )
        stability = result.validator.coefficient_stability()
        if not stability.empty:
            stability.to_csv(
                results_dir / f"coefficient_stability_{result.model_kind}.csv",
                index=False,
            )
    runner.write_research_memo(results, results_dir / "research_memo.md")
    if results:
        runner.save_final_model(results[0], results_dir / "model_artifact.joblib")
    write_metadata(results_dir, data_config, walk_config)
    logger.info("Saved outputs to %s", results_dir)
    return results


if __name__ == "__main__":
    main()
