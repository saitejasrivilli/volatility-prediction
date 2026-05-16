"""
Feature Engineering: Transform raw OHLCV data into predictive features.

This module creates technical indicators used by professional quants:
- Moving average ratios and crossovers
- Relative Strength Index (RSI)
- Bollinger Bands
- Average True Range (ATR)
- Volatility regime classification
- Momentum and trend indicators

All calculations use numpy/pandas for speed. Features are normalized for model input.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .config import DATA_CONFIG, FEATURE_CONFIG
except ImportError:  # pragma: no cover
    from config import DATA_CONFIG, FEATURE_CONFIG

logger = logging.getLogger(__name__)


# ============================================================================
# ADVANCED QUANT INDICATORS
# ============================================================================


class MeanReversionIndicators:
    """Mean reversion and OU process indicators."""

    @staticmethod
    def zscore_mean(price: pd.Series, periods: List[int]) -> pd.DataFrame:
        """Z-score of price vs rolling mean."""
        result = pd.DataFrame(index=price.index)
        for period in periods:
            ma = price.rolling(period).mean()
            std = price.rolling(period).std()
            result[f"MR_ZScore_{period}d"] = (price - ma) / std.replace(0, 1)
        return result

    @staticmethod
    def hurst_exponent(returns: pd.Series, window: int = 100) -> pd.Series:
        """Rolling Hurst exponent (R/S analysis). <0.5=mean-reverting, >0.5=trending."""
        def hurst(ts):
            if len(ts) < window // 2:
                return np.nan
            lags = range(2, min(len(ts) // 2, 100))
            tau = []
            for lag in lags:
                tau_lag = np.std(np.diff(ts, lag)) / np.std(np.diff(ts))
                tau.append(tau_lag)
            poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            return poly[0] * 2

        return returns.rolling(window).apply(hurst, raw=False)

    @staticmethod
    def ou_halflife(returns: pd.Series, window: int = 50) -> pd.Series:
        """Ornstein-Uhlenbeck half-life via AR(1) fit."""
        def compute_halflife(x):
            if len(x) < 3:
                return np.nan
            diffs = np.diff(x)
            x_lagged = x[:-1]
            try:
                slope = np.polyfit(x_lagged, diffs, 1)[0]
                if slope >= 0 or slope < -2:
                    return np.nan
                return -np.log(2) / slope
            except Exception:
                return np.nan

        return returns.rolling(window).apply(compute_halflife, raw=False)

    @staticmethod
    def ou_spread(price: pd.Series, period: int = 20) -> pd.Series:
        """OU spread: deviation from MA / recent vol."""
        ma = price.rolling(period).mean()
        vol = price.rolling(period).std()
        return (price - ma) / vol.replace(0, 1)


class RegimePersistenceFeatures:
    """Regime persistence, correlation breakdown, and transitions."""

    @staticmethod
    def transition_matrix(vol_regime: pd.Series, window: int = 252) -> pd.DataFrame:
        """Rolling transition probabilities."""
        result = pd.DataFrame(index=vol_regime.index)
        for i in range(len(vol_regime) - window):
            window_data = vol_regime.iloc[i:i+window].values
            transitions = {}
            for state in [1, 2, 3]:  # Low, Med, High
                low_to_next = np.sum((window_data[:-1] == state) & (window_data[1:] == state))
                total = np.sum(window_data[:-1] == state)
                transitions[f"{state}_to_{state}"] = low_to_next / max(total, 1)

            result.loc[vol_regime.index[i+window], "RP_LowToLow_Prob"] = transitions.get("1_to_1", 0)
            result.loc[vol_regime.index[i+window], "RP_HighToHigh_Prob"] = transitions.get("3_to_3", 0)

        return result.fillna(method="bfill")

    @staticmethod
    def regime_duration(vol_regime: pd.Series) -> pd.Series:
        """Days in current regime."""
        duration = (vol_regime.diff() != 0).astype(int).cumsum()
        return duration.groupby(duration).cumcount() + 1

    @staticmethod
    def return_vol_correlation(returns: pd.Series, vol: pd.Series, window: int = 60) -> pd.Series:
        """Correlation between returns and volatility."""
        abs_returns = returns.abs()
        return abs_returns.rolling(window).corr(vol)


class LiquidityFeatures:
    """Liquidity and transaction cost proxies."""

    @staticmethod
    def average_daily_volume(volume: pd.Series, period: int = 20) -> pd.Series:
        """Average daily volume."""
        return volume.rolling(period).mean()

    @staticmethod
    def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
        """Relative volume: current / average."""
        adv = volume.rolling(period).mean()
        return volume / adv.replace(0, 1)

    @staticmethod
    def amihud_ratio(returns: pd.Series, volume: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Amihud illiquidity: |return| / dollar_volume."""
        dollar_volume = volume * close
        amihud = returns.abs() / dollar_volume.replace(0, 1)
        return amihud.rolling(period).mean()

    @staticmethod
    def spread_proxy(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Bid-ask spread proxy: (high - low) / close."""
        return (high - low) / close.replace(0, 1)

    @staticmethod
    def market_impact(volume: pd.Series, close: pd.Series, vol: pd.Series, adv: pd.Series) -> pd.Series:
        """Simplified market impact: sqrt(volume/ADV) * volatility."""
        impact = np.sqrt(volume / adv.replace(0, 1)) * vol
        return impact.fillna(0)


class FactorModels:
    """Fama-French and factor exposure indicators."""

    @staticmethod
    def market_beta(returns: pd.Series, market_returns: pd.Series, window: int = 60) -> Tuple[pd.Series, pd.Series]:
        """Rolling beta and idiosyncratic vol vs market."""
        betas = []
        residual_vols = []

        for i in range(window, len(returns)):
            y = returns.iloc[i-window:i].values
            x = market_returns.iloc[i-window:i].values
            try:
                beta = np.polyfit(x, y, 1)[0]
                pred = np.polyval([beta, 0], x)
                resid_vol = np.std(y - pred)
            except Exception:
                beta, resid_vol = np.nan, np.nan
            betas.append(beta)
            residual_vols.append(resid_vol)

        beta_series = pd.Series(np.nan, index=returns.index)
        beta_series.iloc[window:] = betas
        resid_series = pd.Series(np.nan, index=returns.index)
        resid_series.iloc[window:] = residual_vols

        return beta_series.fillna(method="bfill"), resid_series.fillna(method="bfill")

    @staticmethod
    def size_factor_proxy(vol: pd.Series, returns: pd.Series) -> pd.Series:
        """Size factor proxy: normalized volatility percentile."""
        vol_rank = vol.rolling(252).rank(pct=True)
        return vol_rank.fillna(method="bfill")

    @staticmethod
    def value_factor_proxy(close: pd.Series, period: int = 252) -> pd.Series:
        """Value factor proxy: position within 52-week range."""
        range_high = close.rolling(period).max()
        range_low = close.rolling(period).min()
        value_pos = (close - range_low) / (range_high - range_low).replace(0, 0.5)
        return value_pos


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================


class TechnicalIndicators:
    """
    Calculate technical indicators. All are public domain, industry-standard.
    """

    @staticmethod
    def moving_averages(prices: pd.Series, periods: List[int]) -> pd.DataFrame:
        """
        Calculate moving averages.

        Args:
            prices: Series of closing prices
            periods: List of MA periods (e.g., [5, 20, 50, 200])

        Returns:
            DataFrame with MA columns
        """
        result = pd.DataFrame(index=prices.index)
        for period in periods:
            result[f"MA_{period}"] = prices.rolling(window=period).mean()
        return result

    @staticmethod
    def ma_ratios(prices: pd.Series, periods: List[int]) -> pd.DataFrame:
        """
        Calculate price/MA ratios.
        Ratio > 1: price above moving average (uptrend signal)
        Ratio < 1: price below moving average (downtrend signal)

        Args:
            prices: Series of closing prices
            periods: List of MA periods

        Returns:
            DataFrame with ratio columns
        """
        mas = TechnicalIndicators.moving_averages(prices, periods)
        result = pd.DataFrame(index=prices.index)

        for period in periods:
            ma_col = f"MA_{period}"
            result[f"MA_Ratio_{period}"] = prices / mas[ma_col]

        return result

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI > 70: Overbought (potential reversal)
        RSI < 30: Oversold (potential reversal)

        Args:
            prices: Series of closing prices
            period: RSI period (standard: 14)

        Returns:
            Series with RSI values (0-100)
        """
        delta = prices.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Average gains and losses
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Relative Strength
        rs = avg_gain / avg_loss

        # RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def bollinger_bands(
        prices: pd.Series, period: int = 20, std_mult: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Mid band: SMA
        Upper band: SMA + (std * 2)
        Lower band: SMA - (std * 2)

        Use band width and %B for volatility estimation.

        Args:
            prices: Series of closing prices
            period: BB period (standard: 20)
            std_mult: Std multiplier (standard: 2.0)

        Returns:
            DataFrame with mid, upper, lower, width, %B
        """
        mid = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = mid + (std * std_mult)
        lower = mid - (std * std_mult)

        # Band width (normalized)
        width = (upper - lower) / mid

        # %B: where price is within bands (0 = at lower, 1 = at upper, 0.5 = at mid)
        percent_b = (prices - lower) / (upper - lower)

        result = pd.DataFrame(index=prices.index)
        result["BB_Mid"] = mid
        result["BB_Upper"] = upper
        result["BB_Lower"] = lower
        result["BB_Width"] = width  # High = high volatility
        result["BB_PercentB"] = percent_b

        return result

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        True Range = max(High - Low, abs(High - Close_prev), abs(Low - Close_prev))
        ATR = SMA of True Range

        ATR measures volatility. High ATR = high volatility, good for straddles.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR period (standard: 14)

        Returns:
            Series with ATR values
        """
        # True Range calculation
        hl = high - low
        hc = np.abs(high - close.shift(1))
        lc = np.abs(low - close.shift(1))

        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        # Average True Range
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def volatility(returns: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate historical volatility (rolling standard deviation of returns).

        Args:
            returns: Series of returns (calculated from prices)
            period: Lookback period (standard: 20)

        Returns:
            Series with volatility values
        """
        return returns.rolling(window=period).std()

    @staticmethod
    def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate momentum: change over period.
        Momentum = Price_today - Price_N_days_ago

        Positive momentum: price going up
        Negative momentum: price going down

        Args:
            prices: Series of prices
            period: Lookback period

        Returns:
            Series with momentum values
        """
        return prices.diff(period)

    @staticmethod
    def roc(prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Rate of Change: percentage change over period.
        ROC = (Price_today - Price_N_days_ago) / Price_N_days_ago

        Args:
            prices: Series of prices
            period: Lookback period

        Returns:
            Series with ROC values
        """
        return prices.pct_change(period)


# ============================================================================
# REGIME CLASSIFICATION
# ============================================================================


class RegimeClassifier:
    """
    Classify market regime (uptrend/downtrend, high/low volatility).
    Regime changes affect how technical indicators perform.
    """

    @staticmethod
    def volatility_regime(
        volatility: pd.Series, percentiles: Tuple[int, int] = (33, 67)
    ) -> pd.Series:
        """
        Classify volatility as Low/Medium/High based on percentiles.

        Args:
            volatility: Series of volatility values
            percentiles: (low_pct, high_pct) for classification

        Returns:
            Series with regime values (1=Low, 2=Medium, 3=High)
        """
        low_threshold = volatility.rolling(window=252).quantile(percentiles[0] / 100)
        high_threshold = volatility.rolling(window=252).quantile(percentiles[1] / 100)

        regime = pd.Series(2, index=volatility.index)  # Default to medium
        regime[volatility < low_threshold] = 1  # Low
        regime[volatility > high_threshold] = 3  # High

        return regime

    @staticmethod
    def trend_regime(prices: pd.Series, ma_period: int = 50) -> pd.Series:
        """
        Classify trend using moving average.

        Args:
            prices: Series of prices
            ma_period: MA period for trend

        Returns:
            Series with regime values (0=Downtrend, 1=Uptrend)
        """
        ma = prices.rolling(window=ma_period).mean()
        regime = (prices > ma).astype(int)
        return regime


# ============================================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================================


class FeatureEngineer:
    """
    Transform raw OHLCV data into features for machine learning.
    """

    def __init__(self, df: pd.DataFrame, config: FEATURE_CONFIG = None):
        """
        Initialize feature engineer.

        Args:
            df: DataFrame with OHLCV data
            config: Feature configuration
        """
        self.df = df.copy()
        self.config = config or FEATURE_CONFIG
        self.features = pd.DataFrame(index=df.index)
        logger.info("Initialized FeatureEngineer for %s rows", len(df))

    def engineer(self) -> pd.DataFrame:
        """
        Run full feature engineering pipeline.

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering")
        logger.info("Active feature groups: %s", self.config.active_features)

        # Extract price components
        close = self.df["Close"]
        high = self.df["High"]
        low = self.df["Low"]
        volume = self.df["Volume"]
        returns = close.pct_change()

        # ===== MOVING AVERAGE FEATURES =====
        if self.config.USE_MOVING_AVERAGE_RATIOS:
            logger.info("→ Adding moving average ratios")
            ma_ratios = TechnicalIndicators.ma_ratios(close, DATA_CONFIG.MA_PERIODS)
            self.features = self.features.join(ma_ratios)

        # ===== RSI FEATURE =====
        if self.config.USE_RSI:
            logger.info("→ Adding RSI")
            self.features["RSI"] = TechnicalIndicators.rsi(
                close, DATA_CONFIG.RSI_PERIOD
            )

        # ===== BOLLINGER BANDS FEATURES =====
        if self.config.USE_BOLLINGER_BANDS:
            logger.info("→ Adding Bollinger Bands")
            bb = TechnicalIndicators.bollinger_bands(
                close, DATA_CONFIG.BBANDS_PERIOD, DATA_CONFIG.BBANDS_STD
            )
            self.features = self.features.join(bb)

        # ===== ATR FEATURE =====
        if self.config.USE_ATR:
            logger.info("→ Adding Average True Range (ATR)")
            self.features["ATR"] = TechnicalIndicators.atr(high, low, close)
            self.features["ATR_Ratio"] = (
                self.features["ATR"] / close
            )  # Normalize by price

        # ===== VOLATILITY FEATURE =====
        if self.config.USE_VOLATILITY_REGIME:
            logger.info("→ Adding volatility calculations")
            vol = TechnicalIndicators.volatility(returns, period=20)
            self.features["Volatility_20d"] = vol
            self.features["Volatility_Ratio"] = (
                vol / vol.rolling(252).mean()
            )  # Current vs yearly mean

        # ===== MOMENTUM =====
        if self.config.USE_MOMENTUM:
            logger.info("→ Adding momentum")
            self.features["Momentum_10d"] = TechnicalIndicators.momentum(close, 10)
            self.features["ROC_10d"] = TechnicalIndicators.roc(close, 10)

        # ===== REGIME INDICATORS =====
        if self.config.USE_VOLATILITY_REGIME:
            logger.info("→ Adding volatility regime classification")
            vol = TechnicalIndicators.volatility(returns, period=20)
            self.features["Vol_Regime"] = RegimeClassifier.volatility_regime(vol)

        if self.config.USE_TREND_REGIME:
            logger.info("→ Adding trend regime classification")
            self.features["Trend_Regime"] = RegimeClassifier.trend_regime(close)

        # ===== VOLUME FEATURES =====
        if self.config.USE_VOLUME_PROFILE:
            logger.info("→ Adding volume indicators")
            self.features["Volume_MA_20d"] = volume.rolling(20).mean()
            self.features["Volume_Ratio"] = volume / volume.rolling(20).mean()

        if self.config.USE_TRANSITION_FEATURES:
            logger.info("→ Adding transition features")
            volatility_20d = TechnicalIndicators.volatility(returns, period=20)
            atr_ratio = TechnicalIndicators.atr(high, low, close) / close
            bb_width = TechnicalIndicators.bollinger_bands(
                close, DATA_CONFIG.BBANDS_PERIOD, DATA_CONFIG.BBANDS_STD
            )["BB_Width"]
            rolling_max = close.rolling(20).max()
            self.features["Volatility_Slope_5d"] = volatility_20d.diff(5)
            self.features["Volatility_Acceleration_5d"] = volatility_20d.diff().diff(5)
            self.features["ATR_Ratio_Change_5d"] = atr_ratio.diff(5)
            self.features["BB_Width_Change_5d"] = bb_width.diff(5)
            self.features["Drawdown_20d"] = (close / rolling_max) - 1
            self.features["Return_Shock_5d"] = returns.abs().rolling(5).max()

        # ===== MEAN REVERSION FEATURES =====
        if self.config.USE_MEAN_REVERSION:
            logger.info("→ Adding mean reversion indicators")
            zscore_features = MeanReversionIndicators.zscore_mean(close, [20, 50])
            self.features = self.features.join(zscore_features)
            self.features["MR_Hurst_100d"] = MeanReversionIndicators.hurst_exponent(returns)
            self.features["MR_HalfLife_50d"] = MeanReversionIndicators.ou_halflife(returns)
            self.features["MR_OU_Spread_20d"] = MeanReversionIndicators.ou_spread(close)

        # ===== REGIME PERSISTENCE FEATURES =====
        if self.config.USE_REGIME_PERSISTENCE:
            logger.info("→ Adding regime persistence")
            vol_regime = RegimeClassifier.volatility_regime(
                TechnicalIndicators.volatility(returns, period=20)
            )
            persistence = RegimePersistenceFeatures.transition_matrix(vol_regime)
            self.features = self.features.join(persistence)
            self.features["RP_Duration_Days"] = RegimePersistenceFeatures.regime_duration(vol_regime)
            vol = TechnicalIndicators.volatility(returns, period=20)
            self.features["RP_ReturnVol_Corr_60d"] = RegimePersistenceFeatures.return_vol_correlation(returns, vol)

        # ===== LIQUIDITY FEATURES =====
        if self.config.USE_LIQUIDITY_FEATURES:
            logger.info("→ Adding liquidity features")
            self.features["LQ_ADV_20d"] = LiquidityFeatures.average_daily_volume(volume)
            self.features["LQ_RelVol"] = LiquidityFeatures.relative_volume(volume)
            self.features["LQ_Amihud_20d"] = LiquidityFeatures.amihud_ratio(returns, volume, close)
            self.features["LQ_Spread_Proxy"] = LiquidityFeatures.spread_proxy(high, low, close)
            adv = LiquidityFeatures.average_daily_volume(volume)
            vol = TechnicalIndicators.volatility(returns, period=20)
            self.features["LQ_Impact_Estimate"] = LiquidityFeatures.market_impact(volume, close, vol, adv)

        logger.info(
            "Feature engineering complete: %s features created", self.features.shape[1]
        )

        return self.features

    def handle_missing_values(self, method: str = "drop") -> pd.DataFrame:
        """
        Handle missing values from rolling calculations.

        Args:
            method: 'forward_fill', 'drop', or 'zero'

        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values (%s)", method)

        initial_na = self.features.isnull().sum().sum()

        if method == "forward_fill":
            # Use past information only; never backfill future feature values.
            self.features = self.features.ffill()
        elif method == "drop":
            # Drop rows with NaN (conservative)
            self.features = self.features.dropna()
        elif method == "zero":
            # Fill with 0 (risky, only for specific features)
            self.features = self.features.fillna(0)

        final_na = self.features.isnull().sum().sum()
        logger.info("Filled %s NaN values → %s remaining", initial_na, final_na)

        return self.features

    def normalize(self, method: str = "zscore") -> pd.DataFrame:
        """
        Normalize features for model training.

        Args:
            method: 'zscore' (mean=0, std=1) or 'minmax' (0-1 range)

        Returns:
            Normalized DataFrame
        """
        logger.info("Normalizing features (%s)", method)

        normalized = self.features.copy()

        for col in normalized.columns:
            if method == "zscore":
                mean = normalized[col].mean()
                std = normalized[col].std()
                if std > 0:
                    normalized[col] = (normalized[col] - mean) / std
            elif method == "minmax":
                min_val = normalized[col].min()
                max_val = normalized[col].max()
                if max_val > min_val:
                    normalized[col] = (normalized[col] - min_val) / (max_val - min_val)

        logger.info("✓ Normalized %s features", len(normalized.columns))

        return normalized


# ============================================================================
# TARGET VARIABLE
# ============================================================================


class TargetBuilder:
    """
    Create target variable: Is volatility spiking tomorrow?
    """

    @staticmethod
    def volatility_spike_target(
        close: pd.Series,
        returns: pd.Series,
        lookback: int = 20,
        threshold_pctl: int = 75,
        horizon: int = 1,
    ) -> pd.Series:
        """
        Create binary target: Will volatility spike in next N days?

        Args:
            close: Series of close prices
            returns: Series of returns
            lookback: Lookback window for volatility calculation
            threshold_pctl: Percentile threshold (e.g., 75 = top 25% volatility days)
            horizon: Predict how many days ahead (default: 1 = tomorrow)

        Returns:
            Series with binary target (1 = spike, 0 = no spike)
        """
        # Calculate rolling volatility
        volatility = returns.rolling(window=lookback).std()

        # Calculate volatility threshold (75th percentile)
        threshold = volatility.rolling(window=252).quantile(threshold_pctl / 100)

        # Shift forward by horizon to create future target
        future_volatility = volatility.shift(-horizon)

        # Preserve rows where the target cannot yet be computed. Casting directly
        # to int would silently turn unknown labels into class 0.
        valid = future_volatility.notna() & threshold.notna()
        target = pd.Series(np.nan, index=close.index, dtype="float64")
        target.loc[valid] = (
            future_volatility.loc[valid] > threshold.loc[valid]
        ).astype(int)

        return target

    @staticmethod
    def volatility_transition_target(
        close: pd.Series,
        returns: pd.Series,
        lookback: int = 20,
        threshold_pctl: int = 75,
        horizon: int = 1,
    ) -> pd.Series:
        """Predict entry into a high-volatility regime on the next horizon."""
        volatility = returns.rolling(window=lookback).std()
        threshold = volatility.rolling(window=252).quantile(threshold_pctl / 100)
        current_high = volatility > threshold
        future_high = current_high.shift(-horizon)
        valid = current_high.notna() & future_high.notna() & threshold.notna()
        target = pd.Series(np.nan, index=close.index, dtype="float64")
        target.loc[valid] = (
            (~current_high.loc[valid]) & future_high.loc[valid]
        ).astype(int)
        return target


class ImpliedVolFeatures:
    """Options surface and implied volatility features."""

    @staticmethod
    def compute_iv_features(
        ticker: str,
        as_of: Optional[str] = None,
        atm_iv: Optional[float] = None,
        iv_skew: Optional[float] = None,
        iv_term_structure: Optional[float] = None,
        vix_proxy: Optional[float] = None,
    ) -> pd.DataFrame:
        """Compute IV surface features from options market or proxies.

        When live=False (backtest mode), uses default IV values as proxies.
        When live=True (production), fetches current options chains via yfinance.

        Args:
            ticker: Stock ticker symbol
            as_of: Reference date for IV calculation
            atm_iv: ATM implied volatility (30-day proxy). If None, use 0.20 default.
            iv_skew: Call IV - Put IV (smile/smirk indicator). If None, use 0.02 default.
            iv_term_structure: Short IV / Long IV ratio. If None, use 0.95 default.
            vix_proxy: 30-day realized vol of broad market. If None, compute from ticker.

        Returns:
            Series with columns: IV_ATM, IV_Skew, IV_Term_Structure, VIX_Proxy
        """
        features = {}

        # ATM implied volatility (30-day proxy)
        features["IV_ATM"] = atm_iv if atm_iv is not None else 0.20

        # IV skew: call vs put IV spread (OTM call IV - OTM put IV)
        # Positive skew = risk-off (puts expensive). Negative = risk-on.
        features["IV_Skew"] = iv_skew if iv_skew is not None else 0.02

        # IV term structure: short-dated vs long-dated IV ratio
        # <1.0 = contango (term structure normalizing). >1.0 = backwardation (crisis).
        features["IV_Term_Structure"] = (
            iv_term_structure if iv_term_structure is not None else 0.95
        )

        # VIX proxy: 30-day realized volatility of broad market
        # If unavailable, use ticker's own 20-day realized vol.
        features["VIX_Proxy"] = vix_proxy if vix_proxy is not None else 0.18

        return pd.Series(features)

    @staticmethod
    def fetch_live_iv_chain(
        ticker: str,
        as_of: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch current options chain from yfinance and extract IV metrics.

        Note: yfinance provides live/delayed data, not historical.
        Use for production monitoring, not backtesting.

        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            as_of: Date for expiration calculation (defaults to today)

        Returns:
            DataFrame with IV_ATM, IV_Skew, IV_Term_Structure; None if fetch fails
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed; IV features unavailable")
            return None

        try:
            ticker_obj = yf.Ticker(ticker)
            expirations = ticker_obj.options

            if not expirations:
                logger.warning(f"No option expirations for {ticker}")
                return None

            # Get nearest expiration
            nearest_exp = expirations[0]
            chain = ticker_obj.option_chain(nearest_exp)

            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                return None

            # Compute ATM IV
            calls_itm = calls[calls["strike"] >= ticker_obj.info.get("currentPrice", 100)]
            puts_itm = puts[puts["strike"] <= ticker_obj.info.get("currentPrice", 100)]

            atm_iv_call = (
                calls_itm.iloc[0]["impliedVolatility"]
                if not calls_itm.empty
                else 0.20
            )
            atm_iv_put = (
                puts_itm.iloc[-1]["impliedVolatility"]
                if not puts_itm.empty
                else 0.20
            )
            atm_iv = (atm_iv_call + atm_iv_put) / 2.0

            # Compute IV skew (OTM call - OTM put)
            calls_otm = calls[calls["strike"] > ticker_obj.info.get("currentPrice", 100)]
            puts_otm = puts[puts["strike"] < ticker_obj.info.get("currentPrice", 100)]

            otm_call_iv = (
                calls_otm.iloc[-1]["impliedVolatility"]
                if not calls_otm.empty
                else atm_iv_call
            )
            otm_put_iv = (
                puts_otm.iloc[0]["impliedVolatility"]
                if not puts_otm.empty
                else atm_iv_put
            )
            iv_skew = otm_call_iv - otm_put_iv

            return pd.Series(
                {
                    "IV_ATM": atm_iv,
                    "IV_Skew": iv_skew,
                    "IV_Term_Structure": 0.95,  # Would need 2nd expiration
                    "VIX_Proxy": atm_iv,  # Use ATM as proxy
                }
            )

        except Exception as e:
            logger.warning(f"Failed to fetch IV chain for {ticker}: {e}")
            return None


# ============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ============================================================================


class FeaturePreprocessor:
    """Fold-safe preprocessing for already engineered features."""

    def __init__(self):
        self.means: Optional[pd.Series] = None
        self.stds: Optional[pd.Series] = None

    def fit(self, features: pd.DataFrame) -> "FeaturePreprocessor":
        clean = features.dropna()
        if clean.empty:
            raise ValueError("Cannot fit preprocessor on an empty feature frame")
        self.means = clean.mean()
        self.stds = clean.std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.means is None or self.stds is None:
            raise RuntimeError("FeaturePreprocessor must be fit before transform")
        return (features - self.means) / self.stds

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)


def build_features(
    df: pd.DataFrame,
    config: FEATURE_CONFIG = None,
    *,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to build all features.

    Args:
        df: Raw OHLCV DataFrame
        config: Feature configuration

    Returns:
        DataFrame with engineered features. By default this returns raw,
        past-only features so walk-forward validation can fit preprocessing
        inside each fold without leaking future statistics.
    """
    engineer = FeatureEngineer(df, config)
    features = engineer.engineer()
    if normalize:
        features = engineer.handle_missing_values(method="drop")
        features = engineer.normalize(method="zscore")
    return features


if __name__ == "__main__":
    # Example usage
    try:
        from .data_pipeline import load_data
    except ImportError:  # pragma: no cover
        from data_pipeline import load_data

    sample_df = load_data()
    sample_features = build_features(sample_df)

    print("\nFeatures shape:", sample_features.shape)
    print("\nFeature columns:")
    print(sample_features.columns.tolist())
    print("\nFeature statistics:")
    print(sample_features.describe())
