import pandas as pd

from src.options_backtest import HistoricalStraddleBacktester


class FakeOptionsClient:
    def fetch_chain(self, symbol, as_of):
        as_of = pd.Timestamp(as_of)
        if as_of == pd.Timestamp("2024-01-02"):
            return pd.DataFrame(
                {
                    "expiration_date": pd.to_datetime(
                        ["2024-01-05", "2024-01-12", "2024-01-12"]
                    ),
                    "strike": [100, 100, 100],
                    "option_type": ["call", "call", "put"],
                    "ask": [1.0, 3.0, 2.0],
                    "bid": [0.5, 2.5, 1.5],
                }
            )
        return pd.DataFrame(
            {
                "expiration_date": pd.to_datetime(["2024-01-12", "2024-01-12"]),
                "strike": [100, 100],
                "option_type": ["call", "put"],
                "ask": [8.5, 0.5],
                "bid": [8.0, 0.25],
            }
        )


def test_historical_straddle_backtester_uses_chain_prices():
    trade = HistoricalStraddleBacktester(FakeOptionsClient()).evaluate_alert(
        "AAA", "2024-01-02", "2024-01-07", 99.0, 108.0
    )

    assert trade.entry_debit == 5.0
    assert trade.exit_value == 8.25
    assert trade.pnl == 3.25


def test_historical_straddle_backtester_uses_intrinsic_at_expiry():
    trade = HistoricalStraddleBacktester(FakeOptionsClient()).evaluate_alert(
        "AAA", "2024-01-02", "2024-01-12", 99.0, 108.0
    )

    assert trade.exit_value == 8.0
