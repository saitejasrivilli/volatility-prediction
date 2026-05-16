# Equity Volatility Intelligence Platform

Production-grade AI/quant research system for equity volatility prediction and trading signal generation. Demonstrates end-to-end ML workflow from research to deployment.

## What It Does

Predicts equity volatility regime transitions (entry into high-volatility periods) using walk-forward validation, rigorous baseline comparison, and production-safe inference.

- **Signal Generation**: ML models (logistic regression, random forest) predict volatility transitions
- **Portfolio Management**: Autonomous agents filter alerts by risk constraints, portfolio capacity, correlation
- **Compliance**: Immutable audit logs, model governance, decision tracking
- **Deployment**: Docker, Kubernetes, GitHub Actions CI/CD, Prometheus monitoring
- **Options Integration**: Straddle backtesting with Greeks-based hedging

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
  --models logistic random_forest \
  --results-dir results/aapl

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
| `src/hedging.py` | Delta neutralization, vega hedging |
| `src/tick_data.py` | Intraday data, realized volatility |

### Feature Groups (12 total, configurable)

**Technical** (standard)
- Moving average ratios, RSI, Bollinger Bands, ATR, momentum

**Regime** (existing)
- Volatility regime, trend regime, volume profile

**Advanced** (new)
- Mean reversion (Z-score, Hurst, OU process)
- Regime persistence (transition matrix, correlation breakdown)
- Liquidity (ADV, Amihud, bid-ask impact)
- Factor models (beta, SMB, HML proxies)

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
- `.github/workflows/` — CI/CD setup

## Tech Stack

- **ML**: scikit-learn (logistic, random forest, calibration)
- **Data**: pandas, numpy, scipy
- **API**: FastAPI, uvicorn
- **Testing**: pytest
- **Deployment**: Docker, Kubernetes, GitHub Actions
- **Monitoring**: Prometheus
- **Python**: 3.10+
