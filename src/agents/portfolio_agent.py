"""Portfolio-aware decision agent for alert filtering and position management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import pandas as pd

try:
    from ..config import AGENT_CONFIG, TRADING_CONFIG
except ImportError:
    from config import AGENT_CONFIG, TRADING_CONFIG


@dataclass
class ApprovedAlert:
    """Alert approved for trading with justification."""

    date: datetime
    ticker: str
    score: float
    adjusted_score: float
    reason: str
    position_size: float


@dataclass
class PortfolioState:
    """Current portfolio state for constraint checking."""

    open_alerts: List[ApprovedAlert] = field(default_factory=list)
    ticker_positions: Dict[str, List[datetime]] = field(default_factory=dict)
    last_alert_per_ticker: Dict[str, datetime] = field(default_factory=dict)

    def add_alert(self, alert: ApprovedAlert) -> None:
        """Add alert to open positions."""
        self.open_alerts.append(alert)
        if alert.ticker not in self.ticker_positions:
            self.ticker_positions[alert.ticker] = []
        self.ticker_positions[alert.ticker].append(alert.date)
        self.last_alert_per_ticker[alert.ticker] = alert.date

    def remove_expired(self, cutoff_date: datetime, hold_days: int = 5) -> None:
        """Remove alerts older than hold_days."""
        self.open_alerts = [
            a for a in self.open_alerts if (cutoff_date - a.date).days < hold_days
        ]

    def count_approved_in_window(
        self, ticker: str, current_date: datetime, window_days: int = 5
    ) -> int:
        """Count approved alerts for ticker in rolling window."""
        if ticker not in self.ticker_positions:
            return 0
        dates = self.ticker_positions[ticker]
        cutoff = current_date - timedelta(days=window_days)
        return sum(1 for d in dates if d > cutoff)


class PortfolioDecisionAgent:
    """Autonomous agent for portfolio-aware alert filtering.

    Applies risk constraints:
    - Max alerts per ticker per rolling window
    - Max total concurrent approved alerts (portfolio capacity)
    - Min score threshold
    - Sector concentration cap (if provided)
    - Correlation penalty between same-sector alerts
    """

    def __init__(
        self,
        sector_map: Optional[Dict[str, str]] = None,
        hold_days: int = 5,
    ):
        """Initialize portfolio decision agent.

        Args:
            sector_map: Optional dict mapping ticker -> sector
            hold_days: Days to hold each position before expiry
        """
        self.state = PortfolioState()
        self.sector_map = sector_map or {}
        self.hold_days = hold_days
        self.config = AGENT_CONFIG
        self.trading_config = TRADING_CONFIG

    def evaluate_alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate ranked alerts and return approved subset.

        Args:
            df: DataFrame with columns [Date, Ticker, Score, ...]
                sorted descending by Score

        Returns:
            Same DataFrame with added columns:
            - approved: bool
            - reason: str
            - adjusted_score: float
        """
        df = df.copy()

        # Handle empty DataFrame
        if len(df) == 0:
            df["approved"] = False
            df["reason"] = ""
            df["adjusted_score"] = 0.0
            return df

        # Initialize new columns
        df["approved"] = False
        df["reason"] = ""
        df["adjusted_score"] = df["Score"].values

        approved_alerts = []
        approved_count = 0

        # Clean expired positions
        first_date = pd.to_datetime(df["Date"].iloc[0])
        self.state.remove_expired(first_date, self.hold_days)

        for idx, row in df.iterrows():
            current_date = pd.to_datetime(row["Date"])
            ticker = row["Ticker"]
            score = row["Score"]

            # Check min score threshold
            if score < self.trading_config.MIN_SIGNAL_STRENGTH:
                df.at[idx, "reason"] = (
                    f"Score {score:.3f} < min {self.trading_config.MIN_SIGNAL_STRENGTH:.3f}"
                )
                continue

            # Check portfolio capacity
            if approved_count >= self.config.PORTFOLIO_CAPACITY:
                df.at[idx, "reason"] = (
                    f"Portfolio at capacity ({self.config.PORTFOLIO_CAPACITY})"
                )
                continue

            # Check max alerts per ticker in window
            alerts_in_window = self.state.count_approved_in_window(
                ticker, current_date, window_days=5
            )
            if alerts_in_window >= self.config.MAX_ALERTS_PER_TICKER_WINDOW:
                df.at[idx, "reason"] = (
                    f"{alerts_in_window} alerts in 5d window "
                    f"(max {self.config.MAX_ALERTS_PER_TICKER_WINDOW})"
                )
                continue

            # Calculate adjusted score with penalties
            adjusted_score = score
            penalty_reason = []

            # Correlation penalty for same-sector alerts
            if self.sector_map:
                ticker_sector = self.sector_map.get(ticker)
                if ticker_sector:
                    same_sector_count = sum(
                        1
                        for a in self.state.open_alerts
                        if self.sector_map.get(a.ticker) == ticker_sector
                    )
                    if same_sector_count > 0:
                        penalty = (
                            self.config.CORRELATION_PENALTY_SAME_SECTOR
                            * same_sector_count
                        )
                        adjusted_score -= penalty
                        penalty_reason.append(
                            f"sector_overlap_{same_sector_count}"
                        )

            # Approve if adjusted score still above threshold
            if adjusted_score < self.trading_config.MIN_SIGNAL_STRENGTH:
                df.at[idx, "reason"] = (
                    f"Adjusted score {adjusted_score:.3f} "
                    f"< min {self.trading_config.MIN_SIGNAL_STRENGTH:.3f} "
                    f"({'+'.join(penalty_reason)})"
                )
                continue

            # All checks passed
            position_size = self._calculate_position_size(score)
            alert = ApprovedAlert(
                date=current_date,
                ticker=ticker,
                score=score,
                adjusted_score=adjusted_score,
                reason="approved",
                position_size=position_size,
            )
            self.state.add_alert(alert)
            approved_alerts.append(alert)
            approved_count += 1

            df.at[idx, "approved"] = True
            df.at[idx, "reason"] = "approved"
            df.at[idx, "adjusted_score"] = adjusted_score

        return df

    def _calculate_position_size(self, score: float) -> float:
        """Calculate position size as fraction of account.

        Score-weighted Kelly-like sizing, capped at max position pct.
        """
        # Normalize score to [0, 1] for sizing
        min_score = self.trading_config.MIN_SIGNAL_STRENGTH
        max_score = 1.0
        normalized = (score - min_score) / (max_score - min_score)
        normalized = max(0.0, min(1.0, normalized))

        # Apply Kelly fraction (simplified: proportional to score)
        kelly_fraction = normalized * self.trading_config.MAX_POSITION_SIZE_PCT
        return min(kelly_fraction, self.trading_config.MAX_POSITION_SIZE_PCT)

    def get_open_positions(self) -> pd.DataFrame:
        """Return DataFrame of currently open approved alerts."""
        if not self.state.open_alerts:
            return pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "score",
                    "adjusted_score",
                    "position_size",
                ]
            )

        data = {
            "date": [a.date for a in self.state.open_alerts],
            "ticker": [a.ticker for a in self.state.open_alerts],
            "score": [a.score for a in self.state.open_alerts],
            "adjusted_score": [a.adjusted_score for a in self.state.open_alerts],
            "position_size": [a.position_size for a in self.state.open_alerts],
        }
        return pd.DataFrame(data)

    def reset_state(self) -> None:
        """Reset portfolio state (for testing or new runs)."""
        self.state = PortfolioState()
