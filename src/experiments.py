"""Experiment orchestration and research memo generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

try:
    from .backtesting import WalkForwardValidator
    from .config import MODEL_CONFIG
    from .feature_engineering import FeaturePreprocessor
except ImportError:  # pragma: no cover
    from backtesting import WalkForwardValidator
    from config import MODEL_CONFIG
    from feature_engineering import FeaturePreprocessor


@dataclass
class ExperimentResult:
    """One model experiment plus its fold-level validator."""

    model_kind: str
    target_kind: str
    validator: WalkForwardValidator

    def summary(self) -> pd.DataFrame:
        frame = self.validator.summary().copy()
        frame.insert(0, "Model", self.model_kind)
        frame.insert(1, "Target", self.target_kind)
        return frame


class ExperimentRunner:
    """Run comparable walk-forward experiments across model families."""

    def __init__(
        self, market_data, features, walk_config, target_kind: str, models: List[str]
    ):
        self.market_data = market_data
        self.features = features
        self.walk_config = walk_config
        self.target_kind = target_kind
        self.models = models

    def run(self) -> List[ExperimentResult]:
        results = []
        for model_kind in self.models:
            validator = WalkForwardValidator(
                self.market_data,
                self.features,
                self.walk_config,
                target_kind=self.target_kind,
                model_kind=model_kind,
            )
            validator.run()
            results.append(ExperimentResult(model_kind, self.target_kind, validator))
        return results

    @staticmethod
    def registry(results: List[ExperimentResult]) -> pd.DataFrame:
        frames = [result.summary() for result in results]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def write_research_memo(results: List[ExperimentResult], path: Path) -> None:
        registry = ExperimentRunner.registry(results)
        if registry.empty:
            path.write_text(
                "# Research Memo\n\nNo experiments completed.\n", encoding="utf-8"
            )
            return
        numeric = registry.copy()
        for column in ("F1", "PR AUC", "Brier", "Baseline F1"):
            numeric[column] = numeric[column].astype(str).str.rstrip("%").astype(
                float
            ) / (100 if column in {"F1", "Baseline F1"} else 1)
        aggregate = numeric.groupby("Model")[
            ["F1", "PR AUC", "Brier", "Baseline F1"]
        ].mean()
        best_model = aggregate["F1"].idxmax()
        aggregate_table = aggregate.to_csv()
        lines = [
            "# Research Memo",
            "",
            f"## Best average F1 model: `{best_model}`",
            "",
            "## Aggregate comparison",
            "",
            "```csv",
            aggregate_table.strip(),
            "```",
            "",
            "## Interpretation",
            "",
            "- Compare model F1 against the baseline before making economic claims.",
            "- Prefer lower Brier scores when calibrated probabilities drive sizing.",
            "- Review coefficient stability and fold dispersion before deployment.",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def save_final_model(result: ExperimentResult, path: Path) -> None:
        """Fit and persist a final model artifact on all valid available rows."""
        validator = result.validator
        valid = validator.features.notna().all(axis=1) & validator.full_target.notna()
        raw_features = validator.features.loc[valid]
        labels = validator.full_target.loc[valid].astype(int)
        preprocessor = FeaturePreprocessor()
        transformed = preprocessor.fit_transform(raw_features)
        model = CalibratedClassifierCV(
            validator._make_model(), method="sigmoid", cv=MODEL_CONFIG.CALIBRATION_CV
        )
        model.fit(transformed, labels)
        joblib.dump(
            {
                "model": model,
                "preprocessor": preprocessor,
                "feature_columns": list(raw_features.columns),
                "target_kind": result.target_kind,
                "model_kind": result.model_kind,
            },
            path,
        )
