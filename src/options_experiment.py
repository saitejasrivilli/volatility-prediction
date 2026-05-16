"""Signal-to-options pilot experiment orchestration."""

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import DATA_CONFIG
    from .data_pipeline import load_data
    from .options_backtest import HistoricalStraddleBacktester
    from .options_data import build_historical_options_client
except ImportError:  # pragma: no cover
    from config import DATA_CONFIG
    from data_pipeline import load_data
    from options_backtest import HistoricalStraddleBacktester
    from options_data import build_historical_options_client


def parse_args(argv=None):
    parser = ArgumentParser(description="Run a signal-to-options pilot experiment.")
    parser.add_argument("--alerts-file", required=True)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument(
        "--options-provider",
        choices=["alpha_vantage", "polygon", "orats", "demo"],
        default=None,
        help="Historical options provider; defaults to OPTIONS_PROVIDER env var.",
    )
    parser.add_argument("--results-dir", default="results/options-pilot")
    return parser.parse_args(argv)


def select_alerts(frame: pd.DataFrame, args) -> pd.DataFrame:
    subset = frame[
        (frame["Ticker"] == args.symbol)
        & (frame["Model"] == args.model)
        & (frame["Horizon"] == args.horizon)
        & (frame["Period"] == args.period)
    ].copy()
    count = max(1, int(len(subset) * args.top_fraction))
    return subset.nlargest(count, "Score")


def load_alerts(path: str | Path) -> pd.DataFrame:
    """Load ranked-alert exports across the supported historical schemas."""
    alerts = pd.read_csv(path)
    if "Date" in alerts.columns:
        alerts["Date"] = pd.to_datetime(alerts["Date"])
        return alerts
    if "level_0" in alerts.columns:
        alerts = alerts.rename(columns={"level_0": "Date"})
        alerts["Date"] = pd.to_datetime(alerts["Date"])
        return alerts
    raise ValueError("Alerts file must contain a Date or level_0 column.")


def market_data_config_for_symbol(symbol: str):
    """Return a data config aligned with the requested options symbol."""
    return replace(DATA_CONFIG, TICKER=symbol)


def build_demo_market_data(
    selected_alerts: pd.DataFrame, hold_days: int
) -> pd.DataFrame:
    """Build deterministic synthetic spot data for credential-free demos."""
    if selected_alerts.empty:
        return pd.DataFrame(columns=["Close"])
    start = pd.Timestamp(selected_alerts["Date"].min()).normalize()
    end = pd.Timestamp(selected_alerts["Date"].max()).normalize() + pd.offsets.BDay(
        hold_days
    )
    index = pd.bdate_range(start, end)
    close = 150 + np.linspace(0, len(index) - 1, len(index)) * 0.75
    return pd.DataFrame({"Close": close}, index=index)


def summarize_trades(trades: pd.DataFrame, provider: str | None = None) -> pd.DataFrame:
    """Return one-row pilot diagnostics for the straddle trades."""
    data_mode = "demo_synthetic" if provider == "demo" else "historical_provider"
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "provider": provider,
                    "data_mode": data_mode,
                    "trade_count": 0,
                    "win_rate": float("nan"),
                    "average_pnl": float("nan"),
                    "median_pnl": float("nan"),
                    "total_pnl": 0.0,
                    "best_trade": float("nan"),
                    "worst_trade": float("nan"),
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "provider": provider,
                "data_mode": data_mode,
                "trade_count": len(trades),
                "win_rate": float((trades["pnl"] > 0).mean()),
                "average_pnl": float(trades["pnl"].mean()),
                "median_pnl": float(trades["pnl"].median()),
                "total_pnl": float(trades["pnl"].sum()),
                "best_trade": float(trades["pnl"].max()),
                "worst_trade": float(trades["pnl"].min()),
            }
        ]
    )


def write_run_note(output: Path, provider: str | None) -> None:
    """Write a human-readable note distinguishing demo from research runs."""
    if provider == "demo":
        note = (
            "Demo run only: this output uses synthetic spot and options data "
            "(`data_mode=demo_synthetic`) and must not be interpreted as "
            "historical research evidence.\n"
        )
    else:
        note = (
            "Historical-provider run: interpret results only after confirming "
            "vendor entitlements, quote completeness, and data-license scope.\n"
        )
    (output / "RUN_NOTE.txt").write_text(note)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    alerts = load_alerts(args.alerts_file)
    selected = select_alerts(alerts, args)
    selected.to_csv(output / "selected_alerts.csv", index=False)

    if args.options_provider == "demo":
        market_data = build_demo_market_data(selected, args.hold_days)
    else:
        market_data = load_data(market_data_config_for_symbol(args.symbol))
    client = build_historical_options_client(args.options_provider)
    backtester = HistoricalStraddleBacktester(client)
    trades = []
    for _, alert in selected.iterrows():
        signal_date = pd.Timestamp(alert["Date"])
        if signal_date not in market_data.index:
            continue
        loc = market_data.index.get_loc(signal_date)
        exit_loc = min(loc + args.hold_days, len(market_data) - 1)
        trade = backtester.evaluate_alert(
            args.symbol,
            signal_date,
            market_data.index[exit_loc],
            float(market_data["Close"].iloc[loc]),
            float(market_data["Close"].iloc[exit_loc]),
        )
        trades.append(trade.__dict__)
    trades_frame = pd.DataFrame(trades)
    trades_frame.to_csv(output / "straddle_trades.csv", index=False)
    summarize_trades(trades_frame, args.options_provider).to_csv(
        output / "straddle_summary.csv", index=False
    )
    write_run_note(output, args.options_provider)


if __name__ == "__main__":
    main()
