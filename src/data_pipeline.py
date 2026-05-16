"""
Data pipeline: fetch, validate, and clean financial data.

This module handles data acquisition from yfinance with:
- Retry logic for network failures
- Comprehensive validation checks
- Logging for debugging
- Error handling that doesn't silently fail
"""

import logging
import pandas as pd
import yfinance as yf

# Import config
try:  # Support both `python -m src...` and direct script execution.
    from .config import DATA_CONFIG, LOGGING_CONFIG
except ImportError:  # pragma: no cover - exercised only for direct script execution
    from config import DATA_CONFIG, LOGGING_CONFIG

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class DataFetchError(Exception):
    """Raised when data cannot be fetched."""


class DataValidationError(Exception):
    """Raised when data fails validation checks."""


class DataCleaningError(Exception):
    """Raised when data cannot be cleaned."""


# ============================================================================
# DATA FETCHER
# ============================================================================


class DataFetcher:
    """
    Fetch stock data from yfinance with error handling and logging.

    Example:
        fetcher = DataFetcher(ticker="AAPL")
        df = fetcher.fetch()
    """

    def __init__(
        self, ticker: str, start_date: str, end_date: str, max_retries: int = 3
    ):
        """
        Initialize fetcher.

        Args:
            ticker: Stock ticker (e.g., "AAPL")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_retries: Number of retries on network failure
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.max_retries = max_retries
        logger.info(
            "Initialized fetcher for %s: %s to %s", ticker, start_date, end_date
        )

    def fetch(self) -> pd.DataFrame:
        """
        Fetch data with retry logic.

        Returns:
            DataFrame with OHLCV data

        Raises:
            DataFetchError: If data cannot be fetched after retries
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Fetching %s (attempt %s/%s)",
                    self.ticker,
                    attempt,
                    self.max_retries,
                )
                df = yf.download(
                    self.ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                )

                if df is None or df.empty:
                    raise DataFetchError(f"No data returned for {self.ticker}")

                df = self._normalize_columns(df)

                logger.info("Successfully fetched %s rows for %s", len(df), self.ticker)
                return df

            except Exception as e:
                logger.warning("Attempt %s failed: %s", attempt, e)
                if attempt == self.max_retries:
                    raise DataFetchError(
                        f"Could not fetch {self.ticker} after {self.max_retries} attempts: {e}"
                    ) from e

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize yfinance output to the single-level OHLCV shape expected downstream.

        Recent yfinance versions return MultiIndex columns such as
        ('Close', 'AAPL') even when only one ticker is requested. The rest of this
        pipeline intentionally works with plain column names like 'Close'.
        """
        if not isinstance(df.columns, pd.MultiIndex):
            return df

        ticker_level = df.columns.get_level_values(-1)
        tickers = pd.Index(ticker_level).dropna().unique()

        if len(tickers) != 1:
            raise DataFetchError(
                f"Expected one ticker in downloaded data, found {list(tickers)}"
            )

        normalized = df.copy()
        normalized.columns = normalized.columns.get_level_values(0)
        logger.info(
            "Normalized multi-level yfinance columns to single-level OHLCV columns"
        )
        return normalized

    def save(self, df: pd.DataFrame, path: str) -> None:
        """Save data to CSV."""
        df.to_csv(path)
        logger.info("Saved data to %s", path)


# ============================================================================
# DATA VALIDATOR
# ============================================================================


class DataValidator:
    """
    Validate data quality using configurable rules.

    Checks:
    - Non-null values
    - Correct column names
    - Date ordering
    - No duplicates
    - Price reasonableness
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, df: pd.DataFrame, config: DATA_CONFIG = None):
        """
        Initialize validator.

        Args:
            df: DataFrame to validate
            config: DataConfig object with validation rules
        """
        self.df = df
        self.config = config or DATA_CONFIG
        self.errors = []
        self.warnings = []

    def validate(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if data passes all checks

        Raises:
            DataValidationError: If critical checks fail
        """
        logger.info("Validating %s rows", len(self.df))

        self._check_shape()
        self._check_columns()
        self._check_date_index()
        self._check_duplicates()
        self._check_missing_values()
        self._check_price_sanity()

        # Log warnings but don't fail on them
        for warning in self.warnings:
            logger.warning("⚠️  %s", warning)

        # Fail on critical errors
        if self.errors:
            error_msg = "Data validation failed:\n" + "\n".join(self.errors)
            logger.error(error_msg)
            raise DataValidationError(error_msg)

        logger.info("✓ All validation checks passed")
        return True

    def _check_shape(self) -> None:
        """Check minimum number of rows."""
        if len(self.df) < self.config.MIN_ROWS:
            self.errors.append(
                f"Insufficient data: {len(self.df)} rows < {self.config.MIN_ROWS} minimum"
            )

    def _check_columns(self) -> None:
        """Check required columns exist."""
        missing = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing:
            self.errors.append(f"Missing columns: {missing}")

    def _check_date_index(self) -> None:
        """Check index is datetime and sorted."""
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.errors.append("Index is not DatetimeIndex")
        elif not self.df.index.is_monotonic_increasing:
            self.errors.append("Index is not sorted chronologically")

    def _check_duplicates(self) -> None:
        """Check no duplicate dates."""
        if not self.config.DUPLICATE_CHECK:
            return

        duplicates = self.df.index.duplicated().sum()
        if duplicates > 0:
            self.errors.append(f"Found {duplicates} duplicate dates")

    def _check_missing_values(self) -> None:
        """Check missing value percentage."""
        total_cells = len(self.df) * len(self.REQUIRED_COLUMNS)
        missing_cells = self.df[self.REQUIRED_COLUMNS].isnull().sum().sum()
        missing_pct = missing_cells / total_cells

        if missing_pct > self.config.MAX_NA_PERCENTAGE:
            self.errors.append(
                f"Too many missing values: {missing_pct:.2%} > {self.config.MAX_NA_PERCENTAGE:.2%}"
            )
        elif missing_pct > 0:
            self.warnings.append(
                f"Missing values: {missing_pct:.2%} (will be filled forward)"
            )

    def _check_price_sanity(self) -> None:
        """Check prices are positive and ordered (High > Low, etc)."""
        # Prices should be positive
        if (self.df["Close"] <= 0).any():
            self.errors.append("Found non-positive close prices")

        # High should be >= Low
        if (self.df["High"] < self.df["Low"]).any():
            bad_count = (self.df["High"] < self.df["Low"]).sum()
            self.warnings.append(
                f"{bad_count} rows where High < Low (will be corrected)"
            )

        # Check for extreme price jumps (gap limiter)
        daily_returns = self.df["Close"].pct_change().abs()
        extreme_moves = (daily_returns > 0.30).sum()  # > 30% in one day
        if extreme_moves > 0:
            self.warnings.append(
                f"Found {extreme_moves} days with > 30% moves (check for splits/dividends)"
            )


# ============================================================================
# DATA CLEANER
# ============================================================================


class DataCleaner:
    """
    Clean data: fill missing values, handle outliers, fix inconsistencies.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize cleaner."""
        self.df = df.copy()
        self.original_shape = df.shape
        logger.info("Initialized cleaner for %s rows", len(df))

    def clean(self) -> pd.DataFrame:
        """
        Run full cleaning pipeline.

        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning")

        # 1. Fix High < Low issues
        original_high_low = self.df[["High", "Low"]].copy()
        self.df["High"] = original_high_low.max(axis=1)
        self.df["Low"] = original_high_low.min(axis=1)

        # 2. Fill missing values using only information available at that time.
        # Backfilling would leak future prices into earlier rows.
        self.df = self.df.ffill()

        # 3. Remove any leading rows that still have NaN after forward filling.
        initial_len = len(self.df)
        self.df = self.df.dropna()
        if len(self.df) < initial_len:
            logger.warning("Dropped %s rows with NaN", initial_len - len(self.df))

        # 4. Ensure numeric types
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # 5. Remove zero volume days (trading halts, etc)
        zero_volume = (self.df["Volume"] == 0).sum()
        self.df = self.df[self.df["Volume"] > 0]
        if zero_volume > 0:
            logger.info("Removed %s zero-volume days", zero_volume)

        logger.info("Cleaning complete: %s → %s", self.original_shape, self.df.shape)
        return self.df


# ============================================================================
# MAIN DATA PIPELINE
# ============================================================================


class DataPipeline:
    """
    Full data pipeline: fetch → validate → clean

    Example:
        pipeline = DataPipeline()
        df = pipeline.run()
    """

    def __init__(self, config: DATA_CONFIG = None):
        """Initialize pipeline."""
        self.config = config or DATA_CONFIG
        logger.info("Initialized DataPipeline")

    def run(self) -> pd.DataFrame:
        """
        Execute full pipeline.

        Returns:
            Clean, validated DataFrame ready for feature engineering
        """
        try:
            # Step 1: Fetch
            logger.info("=" * 80)
            logger.info("STEP 1: FETCHING DATA")
            logger.info("=" * 80)
            fetcher = DataFetcher(
                ticker=self.config.TICKER,
                start_date=self.config.START_DATE,
                end_date=self.config.END_DATE,
            )
            df = fetcher.fetch()

            # Step 2: Validate
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: VALIDATING DATA")
            logger.info("=" * 80)
            validator = DataValidator(df, self.config)
            validator.validate()

            # Step 3: Clean
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: CLEANING DATA")
            logger.info("=" * 80)
            cleaner = DataCleaner(df)
            df = cleaner.clean()

            logger.info("\n" + "=" * 80)
            logger.info("✓ DATA PIPELINE COMPLETE")
            logger.info("=" * 80)
            logger.info("Final shape: %s", df.shape)
            logger.info(
                "Date range: %s to %s", df.index.min().date(), df.index.max().date()
            )
            logger.info(
                "Close price range: $%.2f - $%.2f",
                df["Close"].min(),
                df["Close"].max(),
            )
            logger.info("=" * 80 + "\n")

            return df

        except (DataFetchError, DataValidationError, DataCleaningError) as e:
            logger.error("Pipeline failed: %s", e)
            raise


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def load_data(config: DATA_CONFIG = None) -> pd.DataFrame:
    """
    Quick function to fetch and clean data.

    Args:
        config: Optional DataConfig (uses default if not provided)

    Returns:
        Clean DataFrame ready for feature engineering
    """
    pipeline = DataPipeline(config)
    return pipeline.run()


def load_multi_ticker_data(config: DATA_CONFIG = None) -> dict[str, pd.DataFrame]:
    """Fetch and clean one DataFrame per configured ticker."""
    resolved = config or DATA_CONFIG
    frames = {}
    for ticker in resolved.TICKERS:
        ticker_config = type(resolved)(
            **{
                **resolved.__dict__,
                "TICKER": ticker,
                "TICKERS": resolved.TICKERS,
            }
        )
        frames[ticker] = load_data(ticker_config)
    return frames


if __name__ == "__main__":
    # Example usage
    df = load_data()
    print("\nData summary:")
    print(df.describe())
    print("\nFirst few rows:")
    print(df.head())
