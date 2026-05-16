"""Model governance and regulatory audit logging."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DecisionLogEntry:
    """Single audit log entry."""

    timestamp: str
    entry_type: str  # 'prediction', 'alert_approved', 'alert_rejected', 'model_promoted', 'risk_check'
    model_version: str
    ticker: str
    signal_date: str
    score: float
    decision: str  # 'approved' | 'rejected' | 'prediction_served' | 'promoted'
    reason: str
    probability: float
    threshold: float
    position_size: float
    operator_id: str
    run_id: str
    audit_id: str = ""


class AuditLogger:
    """Append-only audit log for regulatory compliance."""

    def __init__(self, log_path: Path | str):
        """Initialize audit logger.

        Args:
            log_path: Path to audit_log.jsonl file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.run_id = str(uuid.uuid4())
        self._entry_count = 0

    def log_prediction(
        self,
        request: Dict[str, Any],
        response: Dict[str, Any],
        model_version: str,
        ticker: str = "unknown",
        signal_date: str = "",
    ) -> str:
        """Log a prediction request/response.

        Args:
            request: Request dict with 'features' key
            response: Response dict with 'probability' key
            model_version: Model version string
            ticker: Ticker symbol
            signal_date: Signal date (ISO format)

        Returns:
            audit_id for this prediction
        """
        audit_id = str(uuid.uuid4())
        probability = response.get("probability", 0.0)
        features = request.get("features", {})

        entry = DecisionLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            entry_type="prediction",
            model_version=model_version,
            ticker=ticker,
            signal_date=signal_date or datetime.utcnow().isoformat(),
            score=probability,
            decision="prediction_served",
            reason="model inference",
            probability=probability,
            threshold=0.55,  # From TRADING_CONFIG
            position_size=0.0,
            operator_id="system",
            run_id=self.run_id,
            audit_id=audit_id,
        )

        self._write_entry(entry)
        return audit_id

    def log_alert_decision(
        self,
        alert: Dict[str, Any],
        approved: bool,
        reason: str,
        model_version: str = "unknown",
    ) -> None:
        """Log alert approval/rejection decision.

        Args:
            alert: Alert dict with Date, Ticker, Score columns
            approved: Whether alert was approved
            reason: Reason for decision
            model_version: Model version
        """
        entry = DecisionLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            entry_type="alert_approved" if approved else "alert_rejected",
            model_version=model_version,
            ticker=alert.get("Ticker", "unknown"),
            signal_date=str(alert.get("Date", datetime.utcnow().isoformat())),
            score=alert.get("Score", 0.0),
            decision="approved" if approved else "rejected",
            reason=reason,
            probability=alert.get("Score", 0.0),
            threshold=0.55,
            position_size=alert.get("position_size", 0.0),
            operator_id="system",
            run_id=self.run_id,
            audit_id=str(uuid.uuid4()),
        )

        self._write_entry(entry)

    def log_model_promotion(
        self,
        version_tag: str,
        promoted_by: str = "system",
        review_rationale: str = "",
    ) -> None:
        """Log model promotion to champion.

        Args:
            version_tag: Version being promoted
            promoted_by: User/system that triggered promotion
            review_rationale: Why this version was promoted
        """
        entry = DecisionLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            entry_type="model_promoted",
            model_version=version_tag,
            ticker="all",
            signal_date=datetime.utcnow().isoformat(),
            score=0.0,
            decision="promoted",
            reason=review_rationale or "model promotion",
            probability=0.0,
            threshold=0.0,
            position_size=0.0,
            operator_id=promoted_by,
            run_id=self.run_id,
            audit_id=str(uuid.uuid4()),
        )

        self._write_entry(entry)

    def log_risk_check(
        self,
        ticker: str,
        check_name: str,
        passed: bool,
        details: str = "",
    ) -> None:
        """Log risk limit check result.

        Args:
            ticker: Ticker checked
            check_name: Risk check name (e.g., 'portfolio_capacity')
            passed: Whether check passed
            details: Additional details
        """
        entry = DecisionLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            entry_type="risk_check",
            model_version="N/A",
            ticker=ticker,
            signal_date=datetime.utcnow().isoformat(),
            score=1.0 if passed else 0.0,
            decision="passed" if passed else "failed",
            reason=f"{check_name}: {details}",
            probability=1.0 if passed else 0.0,
            threshold=1.0,
            position_size=0.0,
            operator_id="system",
            run_id=self.run_id,
            audit_id=str(uuid.uuid4()),
        )

        self._write_entry(entry)

    def _write_entry(self, entry: DecisionLogEntry) -> None:
        """Write entry to log file (thread-safe, append-only).

        Args:
            entry: DecisionLogEntry to write
        """
        with self._lock:
            try:
                entry_dict = asdict(entry)
                entry_dict["checksum"] = self._compute_checksum(entry_dict)

                with open(self.log_path, "a") as f:
                    f.write(json.dumps(entry_dict) + "\n")

                self._entry_count += 1
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}", exc_info=True)

    @staticmethod
    def _compute_checksum(entry_dict: Dict) -> str:
        """Compute SHA-256 checksum of entry (for tamper detection)."""
        # Exclude checksum field itself
        entry_copy = {k: v for k, v in entry_dict.items() if k != "checksum"}
        entry_json = json.dumps(entry_copy, sort_keys=True)
        return hashlib.sha256(entry_json.encode()).hexdigest()

    def to_dataframe(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert audit log to DataFrame for analysis.

        Args:
            start_date: ISO format start date (inclusive)
            end_date: ISO format end date (inclusive)

        Returns:
            DataFrame with audit entries
        """
        if not self.log_path.exists():
            return pd.DataFrame()

        entries = []
        with open(self.log_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

        if not entries:
            return pd.DataFrame()

        df = pd.DataFrame(entries)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]

        return df.sort_values("timestamp")

    def write_governance_report(self, output_path: Path | str) -> None:
        """Write daily governance summary report.

        Args:
            output_path: Path to write markdown report
        """
        df = self.to_dataframe()

        if df.empty:
            Path(output_path).write_text("# Governance Report\n\nNo audit entries found.\n")
            return

        # Summary statistics
        total_predictions = len(df[df["entry_type"] == "prediction"])
        total_approvals = len(df[df["entry_type"] == "alert_approved"])
        total_rejections = len(df[df["entry_type"] == "alert_rejected"])
        approval_rate = (
            total_approvals / (total_approvals + total_rejections)
            if (total_approvals + total_rejections) > 0
            else 0
        )

        models_used = df[df["entry_type"] == "prediction"]["model_version"].unique()
        avg_probability = df[df["entry_type"] == "prediction"]["probability"].mean()

        report_lines = [
            "# Model Governance Report\n",
            f"**Generated**: {datetime.utcnow().isoformat()}\n",
            f"**Audit Log**: {self.log_path}\n",
            "\n## Summary\n",
            f"- Total Predictions Served: {total_predictions}",
            f"- Alert Approvals: {total_approvals}",
            f"- Alert Rejections: {total_rejections}",
            f"- Approval Rate: {approval_rate:.1%}\n",
            "## Models in Use\n",
        ]

        for model in models_used:
            count = len(df[(df["entry_type"] == "prediction") & (df["model_version"] == model)])
            report_lines.append(f"- `{model}`: {count} predictions")

        report_lines.extend([
            f"\n## Prediction Quality\n",
            f"- Average Probability: {avg_probability:.4f}\n",
            "## Risk Checks\n",
        ])

        risk_checks = df[df["entry_type"] == "risk_check"]
        if not risk_checks.empty:
            risk_summary = risk_checks.groupby("reason").size()
            for reason, count in risk_summary.items():
                report_lines.append(f"- {reason}: {count}")
        else:
            report_lines.append("- No risk checks logged")

        report_text = "\n".join(report_lines)
        Path(output_path).write_text(report_text)
        logger.info(f"Governance report written to {output_path}")

    def is_writable(self) -> bool:
        """Check if audit log is writable.

        Returns:
            True if log file can be written to
        """
        try:
            with open(self.log_path, "a") as f:
                f.write("")
            return True
        except Exception:
            return False

    def get_entry_count(self) -> int:
        """Get total number of entries logged."""
        return self._entry_count


def validate_model_artifact(artifact: Dict[str, Any]) -> bool:
    """Validate model artifact has required governance fields.

    Args:
        artifact: Model artifact dict

    Returns:
        True if all required fields present
    """
    required_keys = ["model", "preprocessor", "feature_columns", "target_kind", "model_kind"]
    return all(key in artifact for key in required_keys)
