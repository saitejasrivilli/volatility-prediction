# Equity Volatility Intelligence Platform

Production-grade AI/quant research system for equity volatility prediction and trading signal generation. Demonstrates end-to-end ML workflow from research to deployment.

## What It Does

Predicts equity volatility regime transitions (entry into high-volatility periods) using walk-forward validation, rigorous baseline comparison, and production-safe inference.

- **Signal Generation**: ML models (logistic, random forest, XGBoost, LightGBM) predict volatility transitions
- **Portfolio Management**: Autonomous agents filter alerts by risk constraints, portfolio capacity, correlation
- **Compliance**: Immutable audit logs, model governance, decision tracking
- **Deployment**: Docker, Kubernetes, GitHub Actions CI/CD, Prometheus monitoring
- **Options Integration**: Live options data (yfinance), Greeks validation, straddle backtesting with hedging
- **Market Regime Analysis**: Performance splits by volatility regime, out-of-sample validation
- **Validation & Intuition**: Greeks validation against market, financial intuition documentation

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run experiment (single ticker)
python -m src.main \
  --ticker AAPL \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --target-kind transition \
  --models logistic random_forest xgboost lightgbm \
  --results-dir results/aapl

# Or with just gradient boosting (faster F1 improvement)
python -m src.main \
  --ticker AAPL \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --target-kind transition \
  --models xgboost lightgbm \
  --results-dir results/aapl-boost

# Start API
uvicorn src.service:app --reload

# Try prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"MA_Ratio_20": 1.05, "RSI": 65, ...}}'
```

## Architecture

### Core Pipeline
```
raw OHLCV
  ↓ data_pipeline.py (validation, cleaning)
  ↓ feature_engineering.py (12+ feature groups)
  ↓ backtesting.py (walk-forward validation)
  ↓ experiments.py (model comparison, baselines)
  ↓ model_artifact.joblib (exported for serving)
  ↓ service.py (FastAPI inference)
```

### Key Components

| Module | Purpose |
|--------|---------|
| `src/agents/` | Autonomous portfolio & research agents |
| `src/audit.py` | Regulatory compliance, decision logging |
| `src/monitoring.py` | Drift detection (feature, prediction, performance) |
| `src/ab_testing.py` | Champion vs challenger comparison |
| `src/model_registry.py` | Artifact versioning & governance |
| `src/greeks.py` | Black-Scholes Greeks, options pricing |
| `src/greeks_validation.py` | Validate Greeks/pricing against market chains |
| `src/hedging.py` | Delta neutralization, vega hedging |
| `src/tick_data.py` | Intraday data, realized volatility |
| `src/options_data.py` | Historical & live options chains (Alpha Vantage, yfinance, Polygon, ORATS) |
| `src/regime_analysis.py` | Performance splits by volatility regime (low/medium/high) |
| `src/oos_report.py` | Out-of-sample validation (2020-2023 train, 2024 test) |

### Feature Groups (15+ total, configurable)

**Technical** (standard)
- Moving average ratios, RSI, Bollinger Bands, ATR, momentum

**Regime** (existing)
- Volatility regime, trend regime, volume profile

**Advanced** (existing)
- Mean reversion (Z-score, Hurst, OU process)
- Regime persistence (transition matrix, correlation breakdown)
- Liquidity (ADV, Amihud, bid-ask impact)
- Factor models (beta, SMB, HML proxies)

**Options & Implied Vol** (new)
- IV_ATM: 30-day at-the-money implied volatility
- IV_Skew: Call vs put IV spread (risk-off detector)
- IV_Term_Structure: Short-dated vs long-dated IV ratio
- VIX_Proxy: 30-day realized vol of broad market

## Deployment

### Local (Docker Compose)
```bash
docker-compose up
curl http://localhost:8000/health
curl http://localhost:9090  # Prometheus
```

### Kubernetes
```bash
kubectl apply -f k8s/
kubectl get pods
kubectl port-forward svc/volatility-api 8080:80
```

### CI/CD (GitHub Actions)
- Test suite on every PR (Python 3.10/3.11/3.12)
- Docker build & push on merge to main
- Automatic k8s deployment

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + monitoring status |
| POST | `/predict` | Single prediction + drift check |
| GET | `/model-info` | Model metadata & experiment results |
| GET | `/metrics` | Prometheus metrics (predictions, cache, drift events) |
| GET | `/model-version` | Current champion version |
| POST | `/register-model` | Register artifact in registry |
| GET | `/ab-test` | Compare two model versions |

## Testing

37 essential tests covering:
- **Backtesting**: walk-forward logic, threshold selection, Kelly sizing
- **Data Integrity**: duplicate detection, leakage prevention, malformed data
- **ML Correctness**: preprocessing, target encoding, feature leakage
- **Greeks**: pricing, delta/gamma/vega/theta bounds, expiry edges
- **Governance**: artifact storage, version promotion, audit logging

```bash
pytest tests/ -v           # Run all
pytest tests/ -x -q        # Fail fast
```

## Configuration

All via `src/config.py`. Environment override support:

```bash
TICKER=MSFT LOG_LEVEL=DEBUG python -m src.main --end-date 2024-12-31
```

Key settings:
- `MIN_SIGNAL_STRENGTH`: Score threshold for alerts (default 0.55)
- `PORTFOLIO_CAPACITY`: Max concurrent approved alerts (default 10)
- `PSI_HIGH_THRESHOLD`: Feature drift alert level (default 0.25)
- `USE_MEAN_REVERSION`, `USE_LIQUIDITY_FEATURES`: Toggle feature groups

## Evidence

| Test Case | Result | Interpretation |
|-----------|--------|-----------------|
| Single-ticker AAPL transition | F1 0.000 | Too sparse for standalone classifier |
| Pooled 8-ticker model | F1 0.225, PR-AUC 0.144 | Rare event learnable via aggregation |
| Top 1% alert precision | 38.1%, lift 13.9x | Ranking > raw accuracy for alert prioritization |
| vs naive rule | F1 0.065 | Learned model adds value |

This reflects disciplined research: reports failures honestly, baselines always compared, no overfitting claims.

## Advanced Features (2025)

### 1. Real Options Data Integration
- **Source**: yfinance (free, no API key) + Alpha Vantage + Polygon + ORATS extensible framework
- **`YFinanceHistoricalOptionsClient`**: Fetches current option chains, finds nearest expirations, normalizes to unified schema
- **Live data**: Supports both backtest (demo chains) and production (live yfinance) modes

### 2. Greeks Validation
- **Module**: `src/greeks_validation.py`
- **validates**: Black-Scholes pricing vs market mid, IV surface, delta errors
- **Output**: Per-contract validation report with tolerance checks (±10% default)

### 3. Market Regime Analysis
- **Module**: `src/regime_analysis.py`
- **Splits**: Backtest folds by volatility regime (low/medium/high terciles)
- **Reports**: Per-regime metrics (Avg F1, PR-AUC, Precision@5%, positive rate)
- **Insight**: Model should perform better in high-vol regimes (more signal), worse in low-vol (sparse events)

### 4. Out-of-Sample Validation
- **Module**: `src/oos_report.py`
- **Hardcoded**: Train 2020-2023, test 2024 (calendar year holdout)
- **Compares**: CV metrics vs unseen 2024 data (detects overfitting)
- **Output**: `oos_comparison.csv`, `oos_report.md`

### 5. Financial Intuition Documentation
- **File**: `docs/FINANCIAL_INTUITION.md`
- **Covers**: Why volatility transitions are hard, class imbalance math, regime persistence, feature lag
- **Honest assessment**: F1 fundamental ceiling ~0.25-0.35 with daily bars; 13.9x ranking lift is the real win
- **Prescriptions**: What data would break the ceiling (intraday, options surface, alternative data)

### 6. Gradient Boosting + Implied Vol Features
- **Models**: XGBoost (scale_pos_weight=20), LightGBM (is_unbalance=True) handle rare events better
- **IV Features**: `ImpliedVolFeatures.compute_iv_features()` provides IV_ATM, IV_Skew, IV_Term_Structure, VIX_Proxy
- **Expected F1 lift**: +0.05-0.10 vs random forest via explicit imbalance handling
- **Config**: Feature groups configurable; IV features off by default (need live data)

## Production Readiness

✅ Walk-forward validation (no lookahead bias)  
✅ Fold-local preprocessing (leakage-safe)  
✅ Baseline comparison (naive + persistence)  
✅ Coefficient stability reports (logistic only)  
✅ Immutable audit trail (decision logging)  
✅ Model versioning & promotion (governance)  
✅ Drift monitoring (feature, prediction, performance)  
✅ A/B testing framework (champion/challenger)  
✅ Containerized inference (Docker, k8s)  
✅ Comprehensive test suite (37 critical tests)  

⚠️ Research-grade: backtesting only, no live trading  
⚠️ Demo options data: licensed data integration needed for real trading

## Quicklinks

- `00_START_HERE.md` — full documentation
- `docs/INTERVIEW_TALK_TRACK.md` — how to explain this project
- `docs/IMPLEMENTATION_GUIDE.md` — adding new components
- `docs/FINANCIAL_INTUITION.md` — why F1 is low, why pooling helps, fundamental ceiling with daily data
- `.github/workflows/` — CI/CD setup

## Tech Stack

- **ML**: scikit-learn (logistic, random forest, calibration), XGBoost, LightGBM
- **Data**: pandas, numpy, scipy, yfinance
- **API**: FastAPI, uvicorn
- **Testing**: pytest
- **Deployment**: Docker, Kubernetes, GitHub Actions
- **Monitoring**: Prometheus
- **Python**: 3.10+
