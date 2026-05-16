"""Intraday tick data pipeline and realized volatility estimators."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from .config import DATA_CONFIG
except ImportError:
    from config import DATA_CONFIG


class TickDataPipeline:
    """Load and process intraday tick data."""

    @staticmethod
    def load_intraday(
        ticker: str,
        interval: str = "5m",
        period: str = "60d",
    ) -> pd.DataFrame:
        """Load intraday data from yfinance.

        Args:
            ticker: Ticker symbol
            interval: '1m', '5m', '15m', '30m', '60m', '1h'
            period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'

        Returns:
            DataFrame with OHLCV data, DatetimeIndex
        """
        data = yf.download(ticker, interval=interval, period=period, progress=False)

        if data.empty:
            raise ValueError(f"No data found for {ticker}")

        # Ensure proper column names and index
        data.index.name = "Date"
        return data.sort_index()

    @staticmethod
    def close_to_close_vol(df: pd.DataFrame) -> float:
        """Close-to-close realized volatility (standard).

        Args:
            df: DataFrame with 'Close' column

        Returns:
            Annualized volatility
        """
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        if len(returns) < 2:
            return 0.0

        # Intraday sampling frequency (assume 5m bars = 78 per trading day)
        periods_per_year = 252 * 78  # Adjust based on interval
        return float(returns.std() * np.sqrt(periods_per_year))

    @staticmethod
    def parkinson_vol(df: pd.DataFrame) -> float:
        """Parkinson volatility estimator (uses High/Low).

        More efficient than close-to-close, assumes zero drift.

        Args:
            df: DataFrame with 'High' and 'Low' columns

        Returns:
            Annualized volatility
        """
        hl_ratio = np.log(df["High"] / df["Low"])
        n = len(hl_ratio)

        if n < 2:
            return 0.0

        c = 4 * np.log(2)
        vol_daily = np.sqrt((hl_ratio**2).sum() / (n * c))

        return float(vol_daily * np.sqrt(252))

    @staticmethod
    def rogers_satchell_vol(df: pd.DataFrame) -> float:
        """Rogers-Satchell volatility estimator (OHLC-based).

        Zero-drift assumption, good for trending markets.

        Args:
            df: DataFrame with 'Open', 'High', 'Low', 'Close' columns

        Returns:
            Annualized volatility
        """
        co = np.log(df["Close"] / df["Open"])
        ch = np.log(df["High"] / df["Close"])
        cl = np.log(df["Close"] / df["Low"])

        rs = co * (co - ch) + ch * (ch - cl) + cl * (cl - co)
        rs = rs.dropna()

        if len(rs) < 2:
            return 0.0

        vol_daily = np.sqrt(rs.sum() / len(rs))
        return float(vol_daily * np.sqrt(252))

    @staticmethod
    def yang_zhang_vol(df: pd.DataFrame) -> float:
        """Yang-Zhang volatility estimator (combines open jump + RS + close).

        Best estimator for assets with gaps.

        Args:
            df: DataFrame with 'Open', 'High', 'Low', 'Close' columns

        Returns:
            Annualized volatility
        """
        # Overnight jump component
        open_close_prev = np.log(df["Open"] / df["Close"].shift(1))
        overnight_vol = open_close_prev.dropna().std()

        # Intraday (Rogers-Satchell)
        co = np.log(df["Close"] / df["Open"])
        ch = np.log(df["High"] / df["Close"])
        cl = np.log(df["Close"] / df["Low"])
        rs = co * (co - ch) + ch * (ch - cl) + cl * (cl - co)
        intraday_vol = np.sqrt((rs.dropna()).sum() / len(rs))

        # Close component (close-to-close)
        close_vol = np.log(df["Close"] / df["Close"].shift(1)).dropna().std()

        # Weights (equal weighting simplified)
        combined_vol = np.sqrt(
            0.34 * overnight_vol**2 + 0.33 * intraday_vol**2 + 0.33 * close_vol**2
        )
        return float(combined_vol * np.sqrt(252))

    @staticmethod
    def compute_vwap(df: pd.DataFrame) -> np.ndarray:
        """Compute Volume-Weighted Average Price.

        Args:
            df: DataFrame with 'Close' and 'Volume' columns

        Returns:
            Array of VWAP values
        """
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap = (
            (typical_price * df["Volume"]).rolling(window=len(df)).sum()
            / df["Volume"].rolling(window=len(df)).sum()
        )
        return vwap.values

    @staticmethod
    def intraday_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute intraday volatility and other features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional columns:
            - VWAP
            - Intraday_Vol_1h, 4h, 1d (rolling realized vol)
            - Volume_Weighted_Return
        """
        features = df.copy()

        # VWAP
        features["VWAP"] = TickDataPipeline.compute_vwap(df)

        # Volume-weighted return
        features["Volume_Weighted_Return"] = (
            (df["Close"] - df["Open"]) / df["Open"]
        ) * df["Volume"] / df["Volume"].mean()

        # Rolling realized volatility at different horizons
        for hours, window in [(1, 12), (4, 48), (24, 288)]:  # Assuming 5m bars
            label = f"Intraday_Vol_{hours}h" if hours < 24 else "Intraday_Vol_1d"
            log_returns = np.log(df["Close"] / df["Close"].shift(1))
            features[label] = (
                log_returns.rolling(window=window).std() * np.sqrt(252 * 78)
            )

        return features


class TickDataFeatureEngineer:
    """Add tick-based features to existing feature set."""

    @staticmethod
    def add_tick_features(
        daily_df: pd.DataFrame,
        ticker: str,
        interval: str = "5m",
        period: str = "60d",
    ) -> pd.DataFrame:
        """Add intraday-derived features to daily dataframe.

        Args:
            daily_df: Daily OHLCV DataFrame
            ticker: Ticker symbol for data fetch
            interval: Intraday interval
            period: Look-back period for intraday

        Returns:
            Daily DataFrame with added tick features
        """
        # Load intraday for most recent period
        try:
            intraday = TickDataPipeline.load_intraday(ticker, interval, period)
        except Exception as e:
            print(f"Warning: Could not load intraday data for {ticker}: {e}")
            return daily_df

        # Compute realized vols for each day
        intraday_features = TickDataPipeline.intraday_features(intraday)

        # Resample to daily (take last value of day)
        daily_agg = intraday_features.resample("D").agg({
            "VWAP": "last",
            "Intraday_Vol_1h": "last",
            "Intraday_Vol_4h": "last",
            "Intraday_Vol_1d": "last",
            "Volume_Weighted_Return": "sum",
        })

        # Merge into daily DataFrame
        result = daily_df.copy()
        for col in daily_agg.columns:
            result[f"Tick_{col}"] = daily_agg[col]

        # Forward-fill NaN from early days
        result = result.fillna(method="bfill")

        return result
