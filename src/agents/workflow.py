"""Multi-step experiment workflow with checkpointing and resumption."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    from ..config import DATA_CONFIG
    from ..data_pipeline import load_data
    from ..feature_engineering import FeatureEngineer, TargetBuilder
    from ..pooled_experiments import PooledExperimentRunner
    from .portfolio_agent import PortfolioDecisionAgent
    from .research_agent import ResearchOrchestrationAgent
except ImportError:
    from config import DATA_CONFIG
    from data_pipeline import load_data
    from feature_engineering import FeatureEngineer, TargetBuilder
    from pooled_experiments import PooledExperimentRunner
    from portfolio_agent import PortfolioDecisionAgent
    from research_agent import ResearchOrchestrationAgent

logger = logging.getLogger(__name__)


class WorkflowStep(Enum):
    """Workflow execution steps."""

    INIT = "init"
    FETCH = "fetch"
    FEATURES = "features"
    EXPERIMENTS = "experiments"
    ALERTS = "alerts"
    DECISIONS = "decisions"
    OPTIONS = "options"
    DONE = "done"


@dataclass
class WorkflowCheckpoint:
    """Workflow checkpoint for resumption."""

    step: WorkflowStep
    timestamp: str
    tickers: List[str]
    results_dir: Path


class ExperimentWorkflow:
    """Checkpointed multi-step experiment workflow.

    Steps:
    1. INIT - initialize paths and config
    2. FETCH - load data for all tickers
    3. FEATURES - engineer features
    4. EXPERIMENTS - run walk-forward experiments
    5. ALERTS - generate ranked alerts
    6. DECISIONS - run portfolio agent
    7. OPTIONS - optional: options backtest
    8. DONE - finalize and report
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        results_dir: Path | str,
        target_kind: str = "transition",
        models: Optional[List[str]] = None,
    ):
        """Initialize workflow.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            results_dir: Directory for results/checkpoints
            target_kind: 'transition' or 'spike'
            models: List of model families to train
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.results_dir = Path(results_dir)
        self.target_kind = target_kind
        self.models = models or ["logistic", "random_forest"]

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.results_dir / "workflow_state.json"

        self.current_step = WorkflowStep.INIT
        self.artifacts = {}  # Store intermediate results

    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists. Return True if resumed."""
        if not self.checkpoint_path.exists():
            return False

        with open(self.checkpoint_path) as f:
            data = json.load(f)

        step = WorkflowStep(data["step"])
        self.current_step = step
        logger.info(f"Resumed from step: {step.value}")
        return True

    def save_checkpoint(self) -> None:
        """Save current step as checkpoint."""
        checkpoint = WorkflowCheckpoint(
            step=self.current_step,
            timestamp=datetime.now().isoformat(),
            tickers=self.tickers,
            results_dir=str(self.results_dir),
        )
        with open(self.checkpoint_path, "w") as f:
            json.dump(
                {
                    "step": checkpoint.step.value,
                    "timestamp": checkpoint.timestamp,
                    "tickers": checkpoint.tickers,
                    "results_dir": str(checkpoint.results_dir),
                },
                f,
            )
        logger.info(f"Checkpoint saved: {self.current_step.value}")

    def run(self) -> bool:
        """Run workflow with automatic checkpointing.

        Returns:
            True if completed successfully
        """
        # Resume from checkpoint if available
        self.load_checkpoint()

        steps_to_run = list(WorkflowStep)
        current_idx = steps_to_run.index(self.current_step)

        for step in steps_to_run[current_idx:]:
            try:
                logger.info(f"Running step: {step.value}")
                self.current_step = step

                if step == WorkflowStep.INIT:
                    self._step_init()
                elif step == WorkflowStep.FETCH:
                    self._step_fetch()
                elif step == WorkflowStep.FEATURES:
                    self._step_features()
                elif step == WorkflowStep.EXPERIMENTS:
                    self._step_experiments()
                elif step == WorkflowStep.ALERTS:
                    self._step_alerts()
                elif step == WorkflowStep.DECISIONS:
                    self._step_decisions()
                elif step == WorkflowStep.OPTIONS:
                    self._step_options()
                elif step == WorkflowStep.DONE:
                    self._step_done()

                self.save_checkpoint()
                logger.info(f"Step completed: {step.value}")

            except Exception as e:
                logger.error(f"Step {step.value} failed: {e}", exc_info=True)
                self.save_checkpoint()
                return False

        return True

    def _step_init(self) -> None:
        """Initialize paths and config."""
        logger.info(f"Initializing workflow for tickers: {self.tickers}")
        self.artifacts["init_config"] = {
            "tickers": self.tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "target_kind": self.target_kind,
            "models": self.models,
        }

    def _step_fetch(self) -> None:
        """Load market data for all tickers."""
        logger.info(f"Fetching data for {len(self.tickers)} tickers")
        market_data_dict = {}

        for ticker in self.tickers:
            try:
                config = DATA_CONFIG
                config.TICKER = ticker
                config.START_DATE = self.start_date
                config.END_DATE = self.end_date
                data = load_data(config)
                market_data_dict[ticker] = data
                logger.info(f"Loaded {len(data)} rows for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {e}")

        self.artifacts["market_data"] = market_data_dict

    def _step_features(self) -> None:
        """Engineer features for all tickers."""
        market_data_dict = self.artifacts.get("market_data", {})
        features_dict = {}

        engineer = FeatureEngineer()

        for ticker, market_data in market_data_dict.items():
            try:
                features = engineer.engineer(market_data)
                features_dict[ticker] = features
                logger.info(f"Engineered {features.shape[1]} features for {ticker}")
            except Exception as e:
                logger.warning(f"Feature engineering failed for {ticker}: {e}")

        self.artifacts["features"] = features_dict

    def _step_experiments(self) -> None:
        """Run experiments across all tickers."""
        market_data_dict = self.artifacts.get("market_data", {})
        features_dict = self.artifacts.get("features", {})

        experiment_results = {}

        for ticker in self.tickers:
            if ticker not in market_data_dict or ticker not in features_dict:
                logger.warning(f"Skipping {ticker}: missing data or features")
                continue

            try:
                market_data = market_data_dict[ticker]
                features = features_dict[ticker]

                # Run single-ticker pooled experiment
                target_builder = TargetBuilder()
                if self.target_kind == "transition":
                    target = target_builder.volatility_transition_target(
                        market_data
                    )
                else:
                    target = target_builder.volatility_spike_target(market_data)

                # Store for later pooling
                experiment_results[ticker] = {
                    "market_data": market_data,
                    "features": features,
                    "target": target,
                }
                logger.info(f"Prepared experiment data for {ticker}")

            except Exception as e:
                logger.warning(f"Experiment prep failed for {ticker}: {e}")

        self.artifacts["experiment_results"] = experiment_results

    def _step_alerts(self) -> None:
        """Generate ranked alerts (placeholder for pooled runner)."""
        logger.info("Generating alerts from experiments")
        # In production, would run PooledExperimentRunner here
        # For now, store placeholder
        self.artifacts["alerts_df"] = pd.DataFrame()

    def _step_decisions(self) -> None:
        """Run portfolio decision agent."""
        logger.info("Running portfolio decision agent")
        alerts_df = self.artifacts.get("alerts_df", pd.DataFrame())

        if alerts_df.empty:
            logger.warning("No alerts to process")
            return

        agent = PortfolioDecisionAgent()
        approved_df = agent.evaluate_alerts(alerts_df)

        self.artifacts["approved_alerts"] = approved_df
        logger.info(
            f"Approved {approved_df['approved'].sum()} of "
            f"{len(approved_df)} alerts"
        )

    def _step_options(self) -> None:
        """Optional: Run options backtest on approved alerts."""
        logger.info("Options step (placeholder)")

    def _step_done(self) -> None:
        """Finalize and write summary reports."""
        logger.info("Workflow complete")

        # Write final report
        summary_path = self.results_dir / "workflow_summary.json"
        summary = {
            "completed_at": datetime.now().isoformat(),
            "tickers": self.tickers,
            "steps_completed": [s.value for s in WorkflowStep],
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary written to {summary_path}")
