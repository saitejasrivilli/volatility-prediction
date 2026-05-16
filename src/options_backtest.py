"""Minimal historical straddle backtest primitives."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    from .options_data import AtTheMoneyStraddleSelector, OptionsDataError
    from .greeks import GreeksSnapshot
except ImportError:
    from options_data import AtTheMoneyStraddleSelector, OptionsDataError
    from greeks import GreeksSnapshot


@dataclass
class StraddleTrade:
    symbol: str
    signal_date: pd.Timestamp
    expiration_date: pd.Timestamp
    strike: float
    entry_debit: float
    exit_value: float
    pnl: float
    entry_greeks: Optional[GreeksSnapshot] = None
    exit_greeks: Optional[GreeksSnapshot] = None
    hedge_pnl: float = 0.0


class HistoricalStraddleBacktester:
    """Evaluate alert dates using historical option chains from a provider client."""

    def __init__(
        self,
        options_client,
        min_days_to_expiry: int = 7,
        compute_greeks: bool = False,
        hedge_delta: bool = False,
        risk_free_rate: float = 0.05,
    ):
        self.options_client = options_client
        self.min_days_to_expiry = min_days_to_expiry
        self.compute_greeks = compute_greeks
        self.hedge_delta = hedge_delta
        self.risk_free_rate = risk_free_rate

    def evaluate_alert(
        self,
        symbol: str,
        signal_date,
        exit_date,
        spot_entry: float,
        spot_exit: float,
        volatility: float = 0.20,
    ) -> StraddleTrade:
        chain = self.options_client.fetch_chain(symbol, signal_date)
        call, put = AtTheMoneyStraddleSelector.select(
            chain, spot_entry, signal_date, self.min_days_to_expiry
        )
        if any(pd.isna(value) for value in (call.get("ask"), put.get("ask"))):
            raise OptionsDataError("Missing ask prices for entry legs")
        entry_debit = float(call["ask"] + put["ask"])
        strike = float(call["strike"])
        expiration = pd.Timestamp(call["expiration_date"])
        exit_timestamp = pd.Timestamp(exit_date)

        # Compute Greeks at entry if requested
        entry_greeks = None
        if self.compute_greeks:
            from .greeks import GreeksCalculator
            T_entry = (expiration - pd.Timestamp(signal_date)).days / 365.0
            entry_greeks = GreeksCalculator.compute_snapshot(
                spot_entry, strike, T_entry, self.risk_free_rate, volatility, "call"
            )

        if exit_timestamp < expiration:
            exit_value = self._mark_to_market_exit(
                symbol,
                exit_timestamp,
                expiration,
                strike,
            )
        else:
            exit_value = self._intrinsic_value(strike, spot_exit)

        # Compute Greeks at exit if requested
        exit_greeks = None
        if self.compute_greeks:
            from .greeks import GreeksCalculator
            T_exit = (expiration - exit_timestamp).days / 365.0
            T_exit = max(T_exit, 0.0)
            exit_greeks = GreeksCalculator.compute_snapshot(
                spot_exit, strike, T_exit, self.risk_free_rate, volatility, "call"
            )

        pnl = exit_value - entry_debit

        return StraddleTrade(
            symbol=symbol,
            signal_date=pd.Timestamp(signal_date),
            expiration_date=expiration,
            strike=strike,
            entry_debit=entry_debit,
            exit_value=exit_value,
            pnl=pnl,
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
        )

    def _mark_to_market_exit(
        self,
        symbol: str,
        exit_date: pd.Timestamp,
        expiration: pd.Timestamp,
        strike: float,
    ) -> float:
        exit_chain = self.options_client.fetch_chain(symbol, exit_date)
        same_contract = exit_chain[
            (exit_chain["expiration_date"] == expiration)
            & (exit_chain["strike"] == strike)
        ]
        call = same_contract[
            same_contract["option_type"].str.lower().str.startswith("c")
        ]
        put = same_contract[
            same_contract["option_type"].str.lower().str.startswith("p")
        ]
        if call.empty or put.empty:
            raise OptionsDataError("Could not find exit quotes for selected straddle")
        if any(
            pd.isna(value)
            for value in (call.iloc[0].get("bid"), put.iloc[0].get("bid"))
        ):
            raise OptionsDataError("Missing bid prices for exit legs")
        return float(call.iloc[0]["bid"] + put.iloc[0]["bid"])

    @staticmethod
    def _intrinsic_value(strike: float, spot_price: float) -> float:
        return max(spot_price - strike, 0) + max(strike - spot_price, 0)
