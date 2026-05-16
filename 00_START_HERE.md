# Start Here

This repository is a **zero-cost equity-volatility research platform** with a separate options-engineering demo path.

## What is real research here

The core project uses free daily equity data and implements:

- market-data ingestion and validation,
- feature engineering,
- walk-forward experimentation,
- baseline comparison,
- probability calibration,
- coefficient-stability and drift reporting,
- a lightweight inference service.

The generated result files are the source of truth for each run. The project intentionally does **not** hardcode flattering Sharpe ratios, win rates, or fixed performance claims.

## What is a demo

The options layer is intentionally separated:

- the architecture is real and provider-pluggable,
- the included `demo` mode uses synthetic spot and options data,
- demo outputs are labeled `data_mode=demo_synthetic`,
- no options PnL from demo mode should be treated as research evidence.

This keeps the project runnable for reviewers while remaining honest about what requires licensed historical options data.

## Recommended first commands

```bash
python -m pip install -r requirements-dev.txt
python -m src.main --target-kind transition --models logistic random_forest
python -m pytest
```

For the credential-free options walkthrough:

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

## Read next

1. `README.md`
2. `QUICKSTART.md`
3. `STRUCTURE.md`
4. `docs/IMPLEMENTATION_GUIDE.md`
