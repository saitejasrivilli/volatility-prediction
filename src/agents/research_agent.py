"""Research orchestration agent for multi-ticker analysis and recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class ResearchRanking:
    """Ranking of ticker quality based on signal metrics."""

    ticker: str
    pr_auc: float
    f1: float
    lift_1pct: float
    rank_score: float  # Composite score


class ResearchOrchestrationAgent:
    """Orchestrates research across multiple tickers.

    Ranks tickers by signal quality and generates allocation recommendations.
    """

    def __init__(self):
        """Initialize research agent."""
        self.rankings: List[ResearchRanking] = []

    def rank_tickers_from_registry(
        self, registry_csv: Path
    ) -> List[ResearchRanking]:
        """Rank tickers from experiment registry.

        Args:
            registry_csv: Path to experiment_registry.csv

        Returns:
            List of ResearchRanking sorted by rank_score (descending)
        """
        df = pd.read_csv(registry_csv)

        rankings = []
        for _, row in df.iterrows():
            # Extract metrics
            ticker = row.get("Ticker", "UNKNOWN")
            pr_auc = row.get("PR_AUC", 0.0)
            f1 = row.get("F1", 0.0)
            lift = row.get("Lift@1%", 0.0)

            # Composite rank: weighted average of normalized metrics
            # Assuming 0-1 range for PR_AUC and F1, lift could be higher
            lift_norm = min(lift / 20.0, 1.0)  # Normalize lift assuming max ~20x
            rank_score = 0.4 * pr_auc + 0.4 * f1 + 0.2 * lift_norm

            ranking = ResearchRanking(
                ticker=ticker,
                pr_auc=pr_auc,
                f1=f1,
                lift_1pct=lift,
                rank_score=rank_score,
            )
            rankings.append(ranking)

        # Sort by rank score descending
        rankings.sort(key=lambda x: x.rank_score, reverse=True)
        self.rankings = rankings
        return rankings

    def generate_allocation_weights(self) -> Dict[str, float]:
        """Generate allocation weights proportional to rank scores.

        Returns:
            Dict mapping ticker -> allocation fraction (sums to 1.0)
        """
        if not self.rankings:
            return {}

        total_score = sum(r.rank_score for r in self.rankings)
        if total_score == 0:
            # Equal weight fallback
            n = len(self.rankings)
            return {r.ticker: 1.0 / n for r in self.rankings}

        return {
            r.ticker: r.rank_score / total_score for r in self.rankings
        }

    def write_research_report(
        self, output_path: Path, registry_csv: Path
    ) -> None:
        """Write research report ranking tickers and allocation.

        Args:
            output_path: Path to write research_report.md
            registry_csv: Path to experiment_registry.csv for source data
        """
        self.rank_tickers_from_registry(registry_csv)
        weights = self.generate_allocation_weights()

        report_lines = [
            "# Research Orchestration Report\n",
            "## Ticker Rankings by Signal Quality\n",
            "| Rank | Ticker | PR-AUC | F1 | Lift@1% | Rank Score |",
            "|------|--------|--------|----|---------|----|",
        ]

        for i, ranking in enumerate(self.rankings, 1):
            report_lines.append(
                f"| {i} | {ranking.ticker} | {ranking.pr_auc:.4f} | "
                f"{ranking.f1:.4f} | {ranking.lift_1pct:.2f}x | "
                f"{ranking.rank_score:.4f} |"
            )

        report_lines.extend([
            "\n## Recommended Allocation Weights\n",
            "| Ticker | Allocation % |",
            "|--------|--------------|",
        ])

        for ticker, weight in weights.items():
            pct = weight * 100
            report_lines.append(f"| {ticker} | {pct:.1f}% |")

        report_lines.extend([
            "\n## Methodology\n",
            "Allocation weights are proportional to composite rank score, "
            "computed as: 40% PR-AUC + 40% F1 + 20% Normalized Lift@1%.\n",
            "\nThis ensures portfolio concentration on tickers with strongest "
            "signal quality across precision, recall, and ranking metrics.\n",
        ])

        report_text = "\n".join(report_lines)
        Path(output_path).write_text(report_text)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert rankings to DataFrame."""
        if not self.rankings:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "Ticker": r.ticker,
                "PR_AUC": r.pr_auc,
                "F1": r.f1,
                "Lift@1%": r.lift_1pct,
                "Rank_Score": r.rank_score,
            }
            for r in self.rankings
        ])
