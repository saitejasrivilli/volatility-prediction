# Implementation Guide

## 1. Install

```bash
python -m pip install -r requirements-dev.txt
```

## 2. Run the pipeline

```bash
python -m src.main
```

## 3. Understand the flow

```text
OHLCV data
  -> validation and past-only cleaning
  -> raw feature engineering
  -> target creation
  -> fold-specific preprocessing
  -> calibration + training-only threshold selection
  -> walk-forward validation against naive baselines
  -> long-volatility premium-hurdle backtest
```

## 4. Inspect outputs

After a run, inspect:

- `results/features_raw.csv`
- `results/validation_summary.csv`
- `results/coefficient_stability.csv`
- `results/run_metadata.json`
- `volatility_predictor.log`

`features_raw.csv` is intentionally unscaled. Scaling happens inside each validation fold so training data never learns from future test-period statistics.

## 5. Change the experiment

Use environment variables for quick changes:

```bash
TICKER=MSFT START_DATE=2018-01-01 END_DATE=2025-12-31 python -m src.main
```

Supported values are documented in `.env.example`.

## 6. Run tests

```bash
python -m pytest
```

The test suite is designed to catch the failure modes that matter most here:

- future-value leakage during cleaning,
- target rows being silently mislabeled,
- preprocessing using test-fold statistics,
- incorrect walk-forward windows,
- unintended trades on non-spike predictions.

## 7. Interpret results carefully

This project is a research pipeline, not a production trading system. The backtest approximates a long-volatility signal using realized absolute next-day movement less an expected-move hurdle and execution costs. It does **not** price options, model implied volatility, or simulate a real listed-options book.
