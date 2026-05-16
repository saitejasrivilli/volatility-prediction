"""Hedging strategies for options positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog

try:
    from .greeks import delta, gamma, vega, GreeksCalculator, GreeksSnapshot
except ImportError:
    from greeks import delta, gamma, vega, GreeksCalculator, GreeksSnapshot


@dataclass
class HedgeResult:
    """Result of hedging simulation."""

    entry_date: str
    exit_date: str
    option_pnl: float
    hedge_pnl: float
    total_pnl: float
    rebalance_count: int
    avg_delta_exposure: float
    hedge_type: str  # 'delta_neutral', 'vega_neutral'


class DeltaHedger:
    """Delta hedging simulator for neutral rebalancing."""

    def __init__(self, rebalance_frequency: str = "daily"):
        """Initialize delta hedger.

        Args:
            rebalance_frequency: 'daily', 'weekly', or 'on_threshold'
        """
        self.rebalance_frequency = rebalance_frequency

    def simulate(
        self,
        S_path: np.ndarray,
        K: float,
        T0: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> HedgeResult:
        """Simulate delta hedging along a price path.

        Args:
            S_path: Array of spot prices over time
            K: Strike price
            T0: Initial time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            HedgeResult with P&L breakdown
        """
        n = len(S_path)
        times = np.linspace(T0, 0, n)
        deltas = []
        hedge_positions = []
        rebalance_count = 0
        hedge_cost_cumulative = 0.0

        # Initial option position
        initial_delta = delta(S_path[0], K, times[0], r, sigma, option_type)
        hedge_positions.append(-initial_delta)  # Short stock to delta-hedge

        for i in range(1, n):
            S = S_path[i]
            T = times[i]

            # Compute current delta
            current_delta = delta(S, K, T, r, sigma, option_type)
            deltas.append(current_delta)

            # Determine if rehedge needed
            target_hedge = -current_delta
            current_hedge = hedge_positions[-1]

            rehedge_threshold = 0.05  # Rehedge if delta drifts > 5%
            if abs(target_hedge - current_hedge) > rehedge_threshold:
                # Rehedge: adjust stock position
                trade_size = target_hedge - current_hedge
                hedge_cost_cumulative += trade_size * S  # Cost of trade
                hedge_positions.append(target_hedge)
                rebalance_count += 1
            else:
                hedge_positions.append(current_hedge)

        # Compute P&L at maturity
        S_final = S_path[-1]
        option_payoff = max(S_final - K, 0.0) if option_type == "call" else max(K - S_final, 0.0)

        # Stock hedge P&L
        hedge_pnl = sum(
            hedge_positions[i] * (S_path[i] - S_path[i - 1])
            for i in range(1, n)
        ) - hedge_cost_cumulative

        # Total P&L
        total_pnl = option_payoff + hedge_pnl

        avg_delta = float(np.mean(np.abs(deltas))) if deltas else 0.0

        return HedgeResult(
            entry_date="",
            exit_date="",
            option_pnl=option_payoff,
            hedge_pnl=hedge_pnl,
            total_pnl=total_pnl,
            rebalance_count=rebalance_count,
            avg_delta_exposure=avg_delta,
            hedge_type="delta_neutral",
        )


class VegaNeutralizer:
    """Construct vega-neutral options portfolio."""

    @staticmethod
    def find_hedge_contracts(
        primary_vega: float,
        hedge_contracts: pd.DataFrame,
        max_contracts: Optional[int] = None,
    ) -> pd.DataFrame:
        """Find minimum-cost vega hedge.

        Args:
            primary_vega: Vega exposure of primary position (positive for long)
            hedge_contracts: DataFrame with columns: Strike, Vega, Price, ...
            max_contracts: Max number of hedge contracts to use

        Returns:
            DataFrame of selected hedge contracts with quantities
        """
        if hedge_contracts.empty:
            return pd.DataFrame()

        hedge_contracts = hedge_contracts.copy()

        # Single-contract matching (simple case)
        if len(hedge_contracts) == 1 or max_contracts == 1:
            contract = hedge_contracts.iloc[0]
            quantity = primary_vega / max(contract["Vega"], 1e-6)
            return pd.DataFrame([{
                **contract.to_dict(),
                "Quantity": quantity,
                "NetVega": quantity * contract["Vega"],
            }])

        # Multi-contract LP (more complex, minimize cost)
        n_contracts = len(hedge_contracts)
        if max_contracts:
            n_contracts = min(n_contracts, max_contracts)

        # Objective: minimize cost
        costs = hedge_contracts.head(n_contracts)["Price"].values

        # Constraint: vega exposure = primary_vega
        vegas = hedge_contracts.head(n_contracts)["Vega"].values
        A_eq = [vegas]
        b_eq = [primary_vega]

        # Bounds: quantities >= 0
        bounds = [(0, None) for _ in range(n_contracts)]

        try:
            result = linprog(
                costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
            )

            if result.success:
                selected = hedge_contracts.head(n_contracts).copy()
                selected["Quantity"] = result.x
                selected["NetVega"] = (
                    selected["Quantity"] * selected["Vega"]
                )
                return selected[selected["Quantity"] > 1e-6]
        except Exception:
            pass

        # Fallback: single largest vega contract
        largest_vega_idx = hedge_contracts["Vega"].abs().idxmax()
        contract = hedge_contracts.loc[largest_vega_idx]
        quantity = primary_vega / max(contract["Vega"], 1e-6)

        return pd.DataFrame([{
            **contract.to_dict(),
            "Quantity": quantity,
            "NetVega": quantity * contract["Vega"],
        }])


class PortfolioGreeksAggregator:
    """Aggregate Greeks across positions."""

    @staticmethod
    def aggregate(positions: List[GreeksSnapshot]) -> dict:
        """Aggregate Greeks across all positions.

        Args:
            positions: List of GreeksSnapshot objects

        Returns:
            Dict with aggregated: delta, gamma, vega, theta, rho
        """
        if not positions:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            }

        return {
            "delta": float(sum(p.delta for p in positions)),
            "gamma": float(sum(p.gamma for p in positions)),
            "vega": float(sum(p.vega for p in positions)),
            "theta": float(sum(p.theta for p in positions)),
            "rho": float(sum(p.rho for p in positions)),
        }
