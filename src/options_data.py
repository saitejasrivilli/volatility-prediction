"""Historical options-chain access for true options backtesting workflows."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
import os
from typing import Literal

import pandas as pd
import requests


class OptionsDataError(RuntimeError):
    """Raised when historical options data cannot be retrieved or parsed."""


class HistoricalOptionsClient(ABC):
    """Provider-agnostic contract consumed by downstream options workflows."""

    @abstractmethod
    def fetch_chain(self, symbol: str, as_of: date | str) -> pd.DataFrame:
        """Return a normalized historical end-of-day option chain."""


class OptionsChainNormalizer:
    """Normalize vendor payloads into the schema required by the backtester."""

    REQUIRED_COLUMNS = {"expiration_date", "strike", "option_type"}

    @staticmethod
    def normalize(frame: pd.DataFrame, rename_map: dict[str, str] | None = None):
        normalized = frame.rename(columns=rename_map or {}).copy()
        for column in ("strike", "bid", "ask", "last", "implied_volatility", "delta"):
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if "expiration_date" in normalized.columns:
            normalized["expiration_date"] = pd.to_datetime(
                normalized["expiration_date"], errors="coerce"
            )
        missing = OptionsChainNormalizer.REQUIRED_COLUMNS - set(normalized.columns)
        if missing:
            raise OptionsDataError(
                f"Normalized chain missing required columns: {sorted(missing)}"
            )
        return normalized


@dataclass
class AlphaVantageHistoricalOptionsClient(HistoricalOptionsClient):
    """Fetch historical US options chains from Alpha Vantage."""

    api_key: str | None = None
    base_url: str = "https://www.alphavantage.co/query"
    timeout_seconds: int = 30

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise OptionsDataError(
                "Missing Alpha Vantage API key. Set ALPHAVANTAGE_API_KEY."
            )

    def fetch_chain(self, symbol: str, as_of: date | str) -> pd.DataFrame:
        """Return one historical end-of-day option chain."""
        response = requests.get(
            self.base_url,
            params={
                "function": "HISTORICAL_OPTIONS",
                "symbol": symbol,
                "date": str(as_of),
                "apikey": self.api_key,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if "data" not in payload:
            raise OptionsDataError(f"Unexpected options response: {payload}")
        frame = pd.DataFrame(payload["data"])
        if frame.empty:
            raise OptionsDataError(f"No options returned for {symbol} on {as_of}")
        return OptionsChainNormalizer.normalize(
            frame,
            rename_map={
                "expiration": "expiration_date",
                "type": "option_type",
            },
        )


@dataclass
class PolygonHistoricalOptionsClient(HistoricalOptionsClient):
    """Extension point for Polygon-backed historical option chains."""

    api_key: str | None = None

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise OptionsDataError("Missing Polygon API key. Set POLYGON_API_KEY.")

    def fetch_chain(self, symbol: str, as_of: date | str) -> pd.DataFrame:
        raise NotImplementedError(
            "Polygon adapter is configured but not implemented yet. "
            "Polygon's available historical products need to be mapped to the "
            "normalized chain schema before live use."
        )


@dataclass
class OratsHistoricalOptionsClient(HistoricalOptionsClient):
    """Extension point for ORATS-backed historical option chains."""

    api_key: str | None = None

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("ORATS_API_KEY")
        if not self.api_key:
            raise OptionsDataError("Missing ORATS API key. Set ORATS_API_KEY.")

    def fetch_chain(self, symbol: str, as_of: date | str) -> pd.DataFrame:
        raise NotImplementedError(
            "ORATS adapter is configured but not implemented yet. "
            "Vendor response fields need to be mapped to the normalized chain schema."
        )


@dataclass
class DemoHistoricalOptionsClient(HistoricalOptionsClient):
    """Credential-free synthetic chains for demos; not valid research data."""

    min_strike: int = 25
    max_strike: int = 500
    strike_step: int = 5

    def fetch_chain(self, symbol: str, as_of: date | str) -> pd.DataFrame:
        as_of_date = pd.Timestamp(as_of).normalize()
        expirations = self._next_weekly_expirations(as_of_date)
        rows = []
        for expiration in expirations:
            days_to_expiry = max((expiration - as_of_date).days, 1)
            for strike in range(self.min_strike, self.max_strike + 1, self.strike_step):
                time_value = max(days_to_expiry / 10, 0.5)
                spread = 0.20
                for option_type in ("call", "put"):
                    rows.append(
                        {
                            "expiration_date": expiration,
                            "strike": float(strike),
                            "option_type": option_type,
                            "bid": round(time_value, 2),
                            "ask": round(time_value + spread, 2),
                            "data_source": "demo_synthetic",
                        }
                    )
        return OptionsChainNormalizer.normalize(pd.DataFrame(rows))

    @staticmethod
    def _next_weekly_expirations(as_of_date: pd.Timestamp) -> list[pd.Timestamp]:
        days_until_friday = (4 - as_of_date.weekday()) % 7
        first_friday = as_of_date + pd.Timedelta(days=days_until_friday)
        if first_friday <= as_of_date:
            first_friday += pd.Timedelta(days=7)
        return [first_friday + pd.Timedelta(days=7 * offset) for offset in range(3)]


@dataclass
class YFinanceHistoricalOptionsClient(HistoricalOptionsClient):
    """Fetch US options chains from yfinance (live/delayed, not historical).

    yfinance provides current option chains with Greeks and IV from market data.
    For backtesting use-cases, treats latest chain as proxy for validation workflows.
    """

    def fetch_chain(self, symbol: str, as_of: date | str) -> pd.DataFrame:
        """Return normalized options chain from yfinance.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            as_of: Date to query (nearest available expiration used)

        Returns:
            Normalized DataFrame with columns: expiration_date, strike, option_type,
            bid, ask, implied_volatility, delta, gamma, vega, theta
        """
        try:
            import yfinance as yf
        except ImportError:
            raise OptionsDataError(
                "yfinance not installed. Install via: pip install yfinance"
            )

        ticker = yf.Ticker(symbol)
        try:
            expirations = ticker.options
        except Exception as e:
            raise OptionsDataError(f"Failed to fetch expirations for {symbol}: {e}")

        if not expirations:
            raise OptionsDataError(f"No option expirations available for {symbol}")

        # Find nearest available expiration
        as_of_date = pd.Timestamp(as_of).normalize()
        nearest_exp = self._find_nearest_expiration(expirations, as_of_date)

        try:
            option_chain = ticker.option_chain(nearest_exp)
        except Exception as e:
            raise OptionsDataError(
                f"Failed to fetch option chain for {symbol} exp {nearest_exp}: {e}"
            )

        calls = option_chain.calls.copy()
        puts = option_chain.puts.copy()

        if calls.empty or puts.empty:
            raise OptionsDataError(
                f"Empty option chain for {symbol} expiration {nearest_exp}"
            )

        calls["option_type"] = "call"
        puts["option_type"] = "put"
        calls["expiration_date"] = nearest_exp
        puts["expiration_date"] = nearest_exp
        calls["data_source"] = "yfinance_live"
        puts["data_source"] = "yfinance_live"

        combined = pd.concat([calls, puts], ignore_index=True)

        rename_map = {
            "impliedVolatility": "implied_volatility",
        }

        return OptionsChainNormalizer.normalize(combined, rename_map=rename_map)

    @staticmethod
    def _find_nearest_expiration(expirations: list[str], as_of: pd.Timestamp) -> str:
        """Find nearest available expiration to as_of date."""
        exp_dates = pd.to_datetime(expirations)
        future_exps = exp_dates[exp_dates >= as_of]
        if len(future_exps) > 0:
            return future_exps.min().strftime("%Y-%m-%d")
        return exp_dates.max().strftime("%Y-%m-%d")


OptionsProviderName = Literal["alpha_vantage", "polygon", "orats", "demo", "yfinance"]


def build_historical_options_client(
    provider: OptionsProviderName | str | None = None,
) -> HistoricalOptionsClient:
    """Construct the configured historical-options provider client."""
    provider_name = (provider or os.getenv("OPTIONS_PROVIDER", "alpha_vantage")).lower()
    clients = {
        "alpha_vantage": AlphaVantageHistoricalOptionsClient,
        "polygon": PolygonHistoricalOptionsClient,
        "orats": OratsHistoricalOptionsClient,
        "demo": DemoHistoricalOptionsClient,
        "yfinance": YFinanceHistoricalOptionsClient,
    }
    try:
        return clients[provider_name]()
    except KeyError as exc:
        raise OptionsDataError(
            f"Unsupported options provider: {provider_name}. "
            f"Choose one of {sorted(clients)}."
        ) from exc


class AtTheMoneyStraddleSelector:
    """Select a same-expiration ATM call/put pair from a historical chain."""

    @staticmethod
    def select(
        chain: pd.DataFrame,
        spot_price: float,
        as_of: date | str | pd.Timestamp,
        min_days_to_expiry: int = 7,
    ):
        required = {"expiration_date", "strike", "option_type"}
        missing = required - set(chain.columns)
        if missing:
            raise OptionsDataError(
                f"Missing required option columns: {sorted(missing)}"
            )
        anchor_date = pd.Timestamp(as_of).normalize()
        eligible = chain[
            chain["expiration_date"]
            >= anchor_date + pd.Timedelta(days=min_days_to_expiry)
        ].copy()
        if eligible.empty:
            raise OptionsDataError("No eligible expirations in options chain")
        expiration = eligible["expiration_date"].min()
        same_expiry = eligible[eligible["expiration_date"] == expiration].copy()
        strike = same_expiry.iloc[
            (same_expiry["strike"] - spot_price).abs().argsort()[:1]
        ]["strike"].iloc[0]
        pair = same_expiry[same_expiry["strike"] == strike]
        call = pair[pair["option_type"].str.lower().str.startswith("c")]
        put = pair[pair["option_type"].str.lower().str.startswith("p")]
        if call.empty or put.empty:
            raise OptionsDataError("Could not find matching ATM call/put pair")
        return call.iloc[0], put.iloc[0]
