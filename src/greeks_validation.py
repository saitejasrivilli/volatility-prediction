"""Validate Black-Scholes Greeks against market option chain data."""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from src.greeks import (
    black_scholes_price,
    implied_volatility,
)


@dataclass
class GreeksValidationReport:
    """Summary of Greeks/pricing validation against a market option chain."""

    n_contracts: int
    avg_pricing_error_pct: float
    avg_iv_error_pct: float
    avg_delta_error: float
    within_tolerance_pct: float
    report_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_markdown(self) -> str:
        """Format report as markdown table."""
        lines = [
            "## Greeks Validation Report",
            "",
            f"- **Contracts Analyzed**: {self.n_contracts}",
            f"- **Avg Pricing Error**: {self.avg_pricing_error_pct:.2%}",
            f"- **Avg IV Error**: {self.avg_iv_error_pct:.2%}",
            f"- **Avg Delta Error**: {self.avg_delta_error:.4f}",
            f"- **Within Tolerance (±10%)**: {self.within_tolerance_pct:.1%}",
            "",
        ]

        if not self.report_df.empty:
            lines.append("### Per-Contract Details")
            lines.append("")
            lines.append(self.report_df.to_markdown(index=False))

        return "\n".join(lines)


def validate_greeks(
    chain_df: pd.DataFrame,
    spot: float,
    as_of: Optional[str] = None,
    T_override: Optional[float] = None,
    r: float = 0.05,
    tolerance_pct: float = 0.10,
) -> GreeksValidationReport:
    """Validate Black-Scholes Greeks against market option chain data.

    Compares BS-computed prices/Greeks against market mid-prices and
    implied volatilities from chain.

    Args:
        chain_df: Options chain from market (requires: strike, bid, ask,
                  impliedVolatility, option_type, expiration_date)
        spot: Current spot price
        as_of: Reference date for T calculation (defaults to today)
        T_override: Override time-to-expiry in years (use if chain missing expiration_date)
        r: Risk-free rate (default 5%)
        tolerance_pct: Tolerance for pricing error (default 10%)

    Returns:
        GreeksValidationReport with metrics and per-contract details
    """
    if chain_df.empty:
        return GreeksValidationReport(
            n_contracts=0,
            avg_pricing_error_pct=0.0,
            avg_iv_error_pct=0.0,
            avg_delta_error=0.0,
            within_tolerance_pct=0.0,
        )

    chain = chain_df.copy()
    required_cols = {"strike", "bid", "ask", "option_type"}
    missing = required_cols - set(chain.columns)
    if missing:
        raise ValueError(f"Chain missing required columns: {missing}")

    if as_of is None:
        as_of = pd.Timestamp.now()
    else:
        as_of = pd.Timestamp(as_of)

    # Compute T if not provided
    if T_override is not None:
        chain["T"] = T_override
    elif "expiration_date" in chain.columns:
        exp_dates = pd.to_datetime(chain["expiration_date"])
        chain["T"] = (exp_dates - as_of).dt.days / 365.0
        chain["T"] = chain["T"].clip(lower=0.001)  # Avoid T=0
    else:
        raise ValueError("Must provide T_override or expiration_date column")

    # Market mid-price
    chain["market_mid"] = (chain["bid"] + chain["ask"]) / 2.0

    # BS price
    chain["bs_price"] = chain.apply(
        lambda row: black_scholes_price(
            S=spot,
            K=row["strike"],
            T=row["T"],
            r=r,
            sigma=row.get("implied_volatility", row.get("impliedVolatility", 0.20)),
            option_type=row["option_type"].lower()[0],  # 'c' or 'p'
        ),
        axis=1,
    )

    # Pricing error
    chain["pricing_error_pct"] = np.abs(chain["bs_price"] - chain["market_mid"]) / (
        chain["market_mid"] + 1e-6
    )
    chain["within_pricing_tolerance"] = (
        chain["pricing_error_pct"] <= tolerance_pct
    )

    # IV error: extract market IV and compare to BS-back-computed IV
    chain["market_iv"] = chain.get("implied_volatility", chain.get("impliedVolatility", 0.20))

    def backcompute_iv(row):
        try:
            return implied_volatility(
                market_price=row["market_mid"],
                S=spot,
                K=row["strike"],
                T=row["T"],
                r=r,
                option_type=row["option_type"].lower()[0],
            )
        except Exception:
            return np.nan

    chain["bs_iv"] = chain.apply(backcompute_iv, axis=1)
    chain["iv_error_pct"] = np.abs(chain["bs_iv"] - chain["market_iv"]) / (
        chain["market_iv"] + 1e-6
    )

    # Delta comparison (if available in chain)
    if "delta" in chain.columns:
        from src.greeks import delta as compute_delta

        chain["bs_delta"] = chain.apply(
            lambda row: compute_delta(
                S=spot,
                K=row["strike"],
                T=row["T"],
                r=r,
                sigma=row["market_iv"],
                option_type=row["option_type"].lower()[0],
            ),
            axis=1,
        )
        chain["delta_error"] = np.abs(chain["bs_delta"] - chain["delta"])
    else:
        chain["delta_error"] = np.nan

    # Summary metrics
    n_contracts = len(chain)
    avg_pricing_error = chain["pricing_error_pct"].mean()
    avg_iv_error = chain["iv_error_pct"].mean()
    avg_delta_error = chain["delta_error"].mean() if "delta" in chain.columns else 0.0
    within_tolerance = chain["within_pricing_tolerance"].mean()

    # Per-contract detail report
    report_cols = [
        "strike",
        "option_type",
        "T",
        "market_mid",
        "bs_price",
        "pricing_error_pct",
        "market_iv",
        "bs_iv",
        "iv_error_pct",
    ]
    if "delta" in chain.columns:
        report_cols.append("delta_error")

    report_df = chain[report_cols].copy()
    report_df["pricing_error_pct"] = (report_df["pricing_error_pct"] * 100).round(2)
    report_df["iv_error_pct"] = (report_df["iv_error_pct"] * 100).round(2)
    report_df["T"] = report_df["T"].round(4)
    report_df["market_mid"] = report_df["market_mid"].round(2)
    report_df["bs_price"] = report_df["bs_price"].round(2)
    report_df["market_iv"] = (report_df["market_iv"] * 100).round(2)
    report_df["bs_iv"] = (report_df["bs_iv"] * 100).round(2)

    return GreeksValidationReport(
        n_contracts=n_contracts,
        avg_pricing_error_pct=avg_pricing_error,
        avg_iv_error_pct=avg_iv_error,
        avg_delta_error=avg_delta_error,
        within_tolerance_pct=within_tolerance,
        report_df=report_df,
    )
