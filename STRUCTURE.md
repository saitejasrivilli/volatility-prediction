# Project Structure

```text
volatility-prediction/
├── src/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── backtesting.py
│   ├── experiments.py
│   ├── pooling.py
│   ├── pooled_experiments.py
│   ├── options_data.py
│   ├── options_backtest.py
│   ├── options_experiment.py
│   └── main.py
├── data/
│   └── demo_ranked_alerts.csv
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_feature_engineering.py
│   └── test_backtesting.py
├── docs/
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── Makefile
└── README.md
```

Generated local artifacts such as `results/`, virtual environments, logs, caches, and `.DS_Store` files are ignored by `.gitignore` and should not be committed.

The default research workflow is zero-cost and uses free equity data. The options modules are kept separate as a provider-pluggable extension; the built-in `demo` path uses synthetic data and is explicitly not a source of research claims.
