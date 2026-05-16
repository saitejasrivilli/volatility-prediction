"""A/B testing framework for champion vs challenger model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ABTestResult:
    """Result of A/B test comparing two models."""

    champion_name: str
    challenger_name: str
    f1_champion: float
    f1_challenger: float
    f1_delta: float
    pr_auc_champion: float
    pr_auc_challenger: float
    pr_auc_delta: float
    brier_champion: float
    brier_challenger: float
    brier_delta: float
    p_value: float
    is_significant: bool
    winner: str  # 'champion', 'challenger', 'tie'
    effect_size: float
    recommendation: str

    def to_markdown(self) -> str:
        """Format result as markdown."""
        lines = [
            "# A/B Test Results\n",
            f"## Comparison: {self.champion_name} (Champion) vs {self.challenger_name} (Challenger)\n",
            "### Metrics\n",
            "| Metric | Champion | Challenger | Delta |",
            "|--------|----------|-----------|-------|",
            f"| F1 | {self.f1_champion:.4f} | {self.f1_challenger:.4f} | {self.f1_delta:+.4f} |",
            f"| PR-AUC | {self.pr_auc_champion:.4f} | {self.pr_auc_challenger:.4f} | {self.pr_auc_delta:+.4f} |",
            f"| Brier | {self.brier_champion:.4f} | {self.brier_challenger:.4f} | {self.brier_delta:+.4f} |",
            "\n### Statistical Significance\n",
            f"- p-value: {self.p_value:.4f}",
            f"- Significant at α=0.05: {'Yes' if self.is_significant else 'No'}",
            f"- Effect size (Cohen's d): {self.effect_size:.4f}\n",
            "### Decision\n",
            f"- **Winner**: {self.winner.title()}",
            f"- **Recommendation**: {self.recommendation}\n",
        ]
        return "\n".join(lines)


class ABTestFramework:
    """Compare two model results using statistical hypothesis testing."""

    def __init__(self, alpha: float = 0.05):
        """Initialize A/B test framework.

        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha

    def compare_fold_results(
        self,
        champion_f1_scores: List[float],
        challenger_f1_scores: List[float],
        champion_name: str = "Champion",
        challenger_name: str = "Challenger",
        champion_pr_auc: Optional[float] = None,
        challenger_pr_auc: Optional[float] = None,
        champion_brier: Optional[float] = None,
        challenger_brier: Optional[float] = None,
    ) -> ABTestResult:
        """Compare two models using fold-level results.

        Performs Mann-Whitney U test (non-parametric) on fold-level F1 scores.

        Args:
            champion_f1_scores: List of F1 scores per fold for champion
            challenger_f1_scores: List of F1 scores per fold for challenger
            champion_name: Name for champion model
            challenger_name: Name for challenger model
            champion_pr_auc: Overall PR-AUC for champion (optional)
            challenger_pr_auc: Overall PR-AUC for challenger (optional)
            champion_brier: Overall Brier score for champion (optional)
            challenger_brier: Overall Brier score for challenger (optional)

        Returns:
            ABTestResult with statistical comparison
        """
        champion_f1 = float(np.mean(champion_f1_scores))
        challenger_f1 = float(np.mean(challenger_f1_scores))
        f1_delta = challenger_f1 - champion_f1

        # Mann-Whitney U test (non-parametric, appropriate for small N)
        statistic, p_value = stats.mannwhitneyu(
            challenger_f1_scores, champion_f1_scores, alternative="two-sided"
        )

        is_significant = p_value < self.alpha

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(champion_f1_scores, ddof=1) ** 2
             + np.std(challenger_f1_scores, ddof=1) ** 2)
            / 2
        )
        effect_size = f1_delta / max(pooled_std, 1e-6)

        # Overall metrics
        champion_pr_auc = champion_pr_auc or champion_f1
        challenger_pr_auc = challenger_pr_auc or challenger_f1
        pr_auc_delta = challenger_pr_auc - champion_pr_auc

        champion_brier = champion_brier or 0.0
        challenger_brier = challenger_brier or 0.0
        brier_delta = champion_brier - challenger_brier  # Lower is better

        # Determine winner
        if f1_delta > 0.01 and is_significant:
            winner = "challenger"
            recommendation = (
                f"{challenger_name} is statistically significantly better "
                f"(F1 +{f1_delta:.4f}, p={p_value:.4f}). "
                f"Recommend promotion."
            )
        elif f1_delta < -0.01 and is_significant:
            winner = "champion"
            recommendation = (
                f"{champion_name} remains better (F1 {f1_delta:+.4f}, p={p_value:.4f}). "
                f"Keep champion."
            )
        else:
            winner = "tie"
            recommendation = (
                f"No significant difference (p={p_value:.4f}). "
                f"Performance comparable; decide on other factors."
            )

        return ABTestResult(
            champion_name=champion_name,
            challenger_name=challenger_name,
            f1_champion=champion_f1,
            f1_challenger=challenger_f1,
            f1_delta=f1_delta,
            pr_auc_champion=champion_pr_auc,
            pr_auc_challenger=challenger_pr_auc,
            pr_auc_delta=pr_auc_delta,
            brier_champion=champion_brier,
            brier_challenger=challenger_brier,
            brier_delta=brier_delta,
            p_value=p_value,
            is_significant=is_significant,
            winner=winner,
            effect_size=effect_size,
            recommendation=recommendation,
        )

    def compare_result_dirs(
        self,
        champion_dir: Path | str,
        challenger_dir: Path | str,
    ) -> ABTestResult:
        """Compare two experiment result directories.

        Args:
            champion_dir: Path to champion results directory
            challenger_dir: Path to challenger results directory

        Returns:
            ABTestResult
        """
        champion_dir = Path(champion_dir)
        challenger_dir = Path(challenger_dir)

        # Load validation summaries
        champion_summary_path = champion_dir / "validation_summary_logistic.csv"
        challenger_summary_path = challenger_dir / "validation_summary_logistic.csv"

        if not champion_summary_path.exists():
            champion_summary_path = champion_dir / "validation_summary_random_forest.csv"

        if not challenger_summary_path.exists():
            challenger_summary_path = challenger_dir / "validation_summary_random_forest.csv"

        champion_df = pd.read_csv(champion_summary_path)
        challenger_df = pd.read_csv(challenger_summary_path)

        champion_f1 = champion_df["F1"].tolist()
        challenger_f1 = challenger_df["F1"].tolist()

        champion_pr_auc = champion_df["PR_AUC"].mean()
        challenger_pr_auc = challenger_df["PR_AUC"].mean()

        champion_brier = champion_df.get("Brier_Score", [0.0]).iloc[0]
        challenger_brier = challenger_df.get("Brier_Score", [0.0]).iloc[0]

        return self.compare_fold_results(
            champion_f1,
            challenger_f1,
            champion_name=str(champion_dir.name),
            challenger_name=str(challenger_dir.name),
            champion_pr_auc=float(champion_pr_auc),
            challenger_pr_auc=float(challenger_pr_auc),
            champion_brier=float(champion_brier),
            challenger_brier=float(challenger_brier),
        )

    def write_report(self, result: ABTestResult, output_path: Path) -> None:
        """Write A/B test report to markdown file.

        Args:
            result: ABTestResult
            output_path: Path to write markdown
        """
        Path(output_path).write_text(result.to_markdown())
