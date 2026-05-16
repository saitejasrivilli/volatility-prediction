"""
Configuration and constants for volatility prediction system.
Centralized to avoid magic numbers and enable easy experimentation.
"""

from dataclasses import dataclass
from typing import List
import os


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# ============================================================================
# DATA CONFIGURATION
# ============================================================================


@dataclass
class DataConfig:
    """Configuration for data fetching and processing."""

    # Data source
    TICKER: str = _env_str("TICKER", "AAPL")
    TICKERS: List[str] = None
    START_DATE: str = _env_str("START_DATE", "2019-01-01")
    END_DATE: str = _env_str("END_DATE", "2024-12-31")

    # Data validation
    MIN_ROWS: int = 1000  # Minimum rows to consider data valid
    MAX_NA_PERCENTAGE: float = 0.05  # Max 5% missing values allowed
    DUPLICATE_CHECK: bool = True

    # Technical indicators periods (industry standard)
    RSI_PERIOD: int = 14
    BBANDS_PERIOD: int = 20
    BBANDS_STD: float = 2.0
    MA_PERIODS: List[int] = None  # Will be set in __post_init__

    def __post_init__(self):
        if self.MA_PERIODS is None:
            self.MA_PERIODS = [5, 20, 50, 200]
        if self.TICKERS is None:
            raw_tickers = _env_str("TICKERS", "AAPL,MSFT,NVDA,AMZN,META,GOOGL,SPY,QQQ")
            self.TICKERS = [
                ticker.strip() for ticker in raw_tickers.split(",") if ticker.strip()
            ]


# ============================================================================
# VOLATILITY PREDICTION TARGET
# ============================================================================


@dataclass
class VolatilityTarget:
    """Define what we're predicting: volatility spike."""

    # Target definition
    LOOKBACK_WINDOW: int = 20  # Calculate volatility over last 20 days
    SPIKE_THRESHOLD_PERCENTILE: int = 75  # 75th percentile = "spike"
    PREDICTION_HORIZON: int = 1  # Predict next 1 day
    TARGET_KIND: str = _env_str("TARGET_KIND", "transition")

    # Alternative threshold (for testing)
    SPIKE_THRESHOLD_STD_MULTIPLIER: float = 1.5  # 1.5x median volatility


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================


@dataclass
class WalkForwardConfig:
    """Configuration for rigorous out-of-sample testing."""

    # Time periods (in years)
    TRAIN_WINDOW: int = _env_int("TRAIN_WINDOW_YEARS", 3)
    TEST_WINDOW: int = _env_int("TEST_WINDOW_YEARS", 1)
    STEP_SIZE: int = _env_int("STEP_SIZE_YEARS", 1)

    # This creates:
    # Period 1: Train 2019-2021, Test 2022
    # Period 2: Train 2019-2022, Test 2023
    # Period 3: Train 2019-2023, Test 2024


# ============================================================================
# BACKTESTING & TRADING MECHANICS
# ============================================================================


@dataclass
class TradingConfig:
    """Real-world trading constraints and costs."""

    # Risk management
    ACCOUNT_SIZE: float = _env_float("ACCOUNT_SIZE", 100_000.0)
    MAX_POSITION_SIZE_PCT: float = 0.10  # Max 10% of account per position
    MAX_DRAWDOWN_LIMIT: float = _env_float("MAX_DRAWDOWN_LIMIT", 0.20)

    # Transaction costs (bid-ask spread + commission)
    BID_ASK_SPREAD_PCT: float = _env_float("BID_ASK_SPREAD_PCT", 0.001)
    COMMISSION_PER_TRADE: float = _env_float("COMMISSION_PER_TRADE", 0.0)
    SLIPPAGE_PCT: float = _env_float("SLIPPAGE_PCT", 0.0005)

    # Position sizing
    POSITION_SIZE_METHOD: str = _env_str("POSITION_SIZE_METHOD", "kelly")
    FIXED_POSITION_PCT: float = _env_float("FIXED_POSITION_PCT", 0.02)

    # Strategy parameters
    MIN_SIGNAL_STRENGTH: float = _env_float("MIN_SIGNAL_STRENGTH", 0.55)
    VOL_PREMIUM_MULTIPLIER: float = _env_float("VOL_PREMIUM_MULTIPLIER", 1.0)

    # Risk-free rate for Sharpe calculation
    RISK_FREE_RATE_ANNUAL: float = 0.05  # 5% (current rate approx)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================


@dataclass
class ModelConfig:
    """Machine learning model hyperparameters."""

    # Logistic Regression
    LR_C: float = 1.0
    LR_SOLVER: str = "lbfgs"
    LR_MAX_ITER: int = 1000

    RANDOM_STATE: int = 42
    CALIBRATION_CV: int = 3
    MIN_THRESHOLD: float = 0.01
    MAX_THRESHOLD: float = 0.50
    THRESHOLD_STEPS: int = 100
    F_BETA: float = 2.0
    ALERT_FRACTIONS: List[float] = None
    ENABLED_MODELS: List[str] = None

    def __post_init__(self):
        if self.ENABLED_MODELS is None:
            self.ENABLED_MODELS = ["logistic", "random_forest"]
        if self.ALERT_FRACTIONS is None:
            self.ALERT_FRACTIONS = [0.01, 0.05, 0.10]


# ============================================================================
# FEATURE SELECTION
# ============================================================================


@dataclass
class FeatureConfig:
    """Which features to use for model training."""

    # Technical indicators (all free to calculate)
    USE_MOVING_AVERAGE_RATIOS: bool = True
    USE_RSI: bool = True
    USE_BOLLINGER_BANDS: bool = True
    USE_ATR: bool = True  # Average True Range
    USE_MOMENTUM: bool = True
    USE_VOLUME_PROFILE: bool = True
    USE_TRANSITION_FEATURES: bool = True

    # Regime indicators
    USE_VOLATILITY_REGIME: bool = True  # Classify as low/med/high vol
    USE_TREND_REGIME: bool = True  # Uptrend/downtrend

    # High-frequency features
    USE_TICK_FEATURES: bool = False

    # Advanced quant features
    USE_MEAN_REVERSION: bool = True
    USE_REGIME_PERSISTENCE: bool = True
    USE_LIQUIDITY_FEATURES: bool = True
    USE_FACTOR_MODELS: bool = False  # Requires market returns reference

    @property
    def active_features(self) -> int:
        """Count how many feature groups are enabled."""
        count = sum(
            [
                self.USE_MOVING_AVERAGE_RATIOS,
                self.USE_RSI,
                self.USE_BOLLINGER_BANDS,
                self.USE_ATR,
                self.USE_MOMENTUM,
                self.USE_VOLUME_PROFILE,
                self.USE_TRANSITION_FEATURES,
                self.USE_VOLATILITY_REGIME,
                self.USE_TREND_REGIME,
                self.USE_TICK_FEATURES,
                self.USE_MEAN_REVERSION,
                self.USE_REGIME_PERSISTENCE,
                self.USE_LIQUIDITY_FEATURES,
                self.USE_FACTOR_MODELS,
            ]
        )
        return count


# ============================================================================
# LOGGING & REPORTING
# ============================================================================


@dataclass
class LoggingConfig:
    """Configuration for logging and output."""

    LOG_LEVEL: str = _env_str("LOG_LEVEL", "INFO")
    LOG_TO_FILE: bool = True
    LOG_FILE: str = "volatility_predictor.log"

    # Results output
    RESULTS_DIR: str = _env_str("RESULTS_DIR", "./results")
    PRINT_DETAILED_TRADES: bool = False  # Very verbose


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================


@dataclass
class AgentConfig:
    """Configuration for autonomous agent behavior."""

    MAX_ALERTS_PER_TICKER_WINDOW: int = _env_int("MAX_ALERTS_PER_TICKER_WINDOW", 3)
    PORTFOLIO_CAPACITY: int = _env_int("PORTFOLIO_CAPACITY", 10)
    CORRELATION_PENALTY_SAME_SECTOR: float = _env_float("CORRELATION_PENALTY_SAME_SECTOR", 0.15)
    USE_LLM_SYNTHESIS: bool = False  # Only if ANTHROPIC_API_KEY set


# ============================================================================
# MONITORING CONFIGURATION
# ============================================================================


@dataclass
class MonitoringConfig:
    """Configuration for drift detection and performance monitoring."""

    PSI_HIGH_THRESHOLD: float = _env_float("PSI_HIGH_THRESHOLD", 0.25)
    PSI_MODERATE_THRESHOLD: float = _env_float("PSI_MODERATE_THRESHOLD", 0.10)
    SCORE_DRIFT_SIGMA: float = _env_float("SCORE_DRIFT_SIGMA", 2.0)
    PERFORMANCE_WINDOW: int = _env_int("PERFORMANCE_WINDOW", 30)
    MIN_PRECISION_RATIO: float = _env_float("MIN_PRECISION_RATIO", 0.5)


# ============================================================================
# MODEL REGISTRY CONFIGURATION
# ============================================================================


@dataclass
class RegistryConfig:
    """Configuration for model versioning and registry."""

    REGISTRY_DIR: str = _env_str("REGISTRY_DIR", "results/registry")


# ============================================================================
# CONVENIENCE INSTANTIATION
# ============================================================================

# Create default instances
DATA_CONFIG = DataConfig()
VOLATILITY_TARGET = VolatilityTarget()
WALK_FORWARD_CONFIG = WalkForwardConfig()
TRADING_CONFIG = TradingConfig()
MODEL_CONFIG = ModelConfig()
FEATURE_CONFIG = FeatureConfig()
LOGGING_CONFIG = LoggingConfig()
AGENT_CONFIG = AgentConfig()
MONITORING_CONFIG = MonitoringConfig()
REGISTRY_CONFIG = RegistryConfig()


def print_config() -> None:
    """Print all active configuration for transparency."""
    print("=" * 80)
    print("SYSTEM CONFIGURATION")
    print("=" * 80)
    print(f"\nDATA:")
    print(f"  Ticker: {DATA_CONFIG.TICKER}")
    print(f"  Period: {DATA_CONFIG.START_DATE} to {DATA_CONFIG.END_DATE}")
    print(f"\nVOLATILITY TARGET:")
    print(f"  Lookback: {VOLATILITY_TARGET.LOOKBACK_WINDOW} days")
    print(
        f"  Spike threshold: {VOLATILITY_TARGET.SPIKE_THRESHOLD_PERCENTILE}th percentile"
    )
    print(f"\nWALK-FORWARD VALIDATION:")
    print(f"  Train window: {WALK_FORWARD_CONFIG.TRAIN_WINDOW} years")
    print(f"  Test window: {WALK_FORWARD_CONFIG.TEST_WINDOW} year")
    print(f"\nTRADING MECHANICS:")
    print(f"  Account size: ${TRADING_CONFIG.ACCOUNT_SIZE:,.0f}")
    print(f"  Bid-ask spread: {TRADING_CONFIG.BID_ASK_SPREAD_PCT * 100:.3f}%")
    print(
        f"  Max position: {TRADING_CONFIG.MAX_POSITION_SIZE_PCT * 100:.1f}% of account"
    )
    print(f"\nFEATURES:")
    print(f"  Active feature groups: {FEATURE_CONFIG.active_features}")
    print("=" * 80)
