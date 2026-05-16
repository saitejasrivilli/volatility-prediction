"""Build pooled multi-ticker research datasets."""

import pandas as pd

try:
    from .feature_engineering import TargetBuilder, build_features
except ImportError:  # pragma: no cover
    from feature_engineering import TargetBuilder, build_features


def build_pooled_dataset(
    market_data_by_ticker: dict[str, pd.DataFrame],
    *,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return pooled OHLCV rows, features, and transition targets."""
    pooled_frames = []
    pooled_features = []
    pooled_targets = []

    for ticker, frame in market_data_by_ticker.items():
        features = build_features(frame)
        target = TargetBuilder.volatility_transition_target(
            frame["Close"], frame["Close"].pct_change(), horizon=horizon
        )
        indexed_frame = frame.copy()
        indexed_frame["Ticker"] = ticker
        indexed_frame = indexed_frame.set_index("Ticker", append=True)
        indexed_features = features.copy()
        indexed_features["Ticker"] = ticker
        indexed_features = indexed_features.set_index("Ticker", append=True)
        indexed_target = target.copy()
        indexed_target.index = indexed_features.index

        pooled_frames.append(indexed_frame)
        pooled_features.append(indexed_features)
        pooled_targets.append(indexed_target)

    market_data = pd.concat(pooled_frames).sort_index()
    feature_frame = pd.concat(pooled_features).sort_index()
    target = pd.concat(pooled_targets).sort_index()
    return market_data, feature_frame, target
