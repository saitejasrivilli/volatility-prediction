# Quick Start

## Install

```bash
python -m pip install -r requirements-dev.txt
```

## Run

```bash
python -m src.main --target-kind transition --models logistic random_forest
```

For pooled multi-ticker research, the default output is concise:

```bash
python -m src.multi_main \
  --tickers AAPL MSFT NVDA AMZN META GOOGL SPY QQQ \
  --horizons 1 5 \
  --models logistic random_forest
```

Add `--verbose` if you want the repeated feature-engineering step logs during pooled runs.

## Test

```bash
python -m pytest
```

## What to expect

The pipeline will:

1. download OHLCV data,
2. build raw features,
3. create a next-day volatility-spike target,
4. run expanding-window walk-forward validation,
5. compare model families with naive baselines,
6. write reproducible experiment artifacts to `results/`.

The generated metrics are the source of truth for the current run. This project intentionally does not advertise fixed Sharpe ratios or win rates in advance because those depend on data, dates, configuration, and the current implementation.

## Common adjustments

Use environment variables when you want a quick experiment:

```bash
TICKER=MSFT START_DATE=2018-01-01 END_DATE=2025-12-31 python -m src.main
```

You can also use CLI flags:

```bash
python -m src.main --ticker MSFT --start-date 2018-01-01 --end-date 2025-12-31
```

For a deployable surface:

```bash
uvicorn src.service:app --reload
```

For a credential-free options architecture demo:

```bash
python -m src.options_experiment \
  --alerts-file data/demo_ranked_alerts.csv \
  --symbol AAPL \
  --model random_forest \
  --horizon 1 \
  --period 1 \
  --top-fraction 1.0 \
  --hold-days 5 \
  --options-provider demo \
  --results-dir results/options-demo-aapl
```

That options command uses synthetic demo data. It proves the workflow runs end-to-end; it is not historical research evidence.
