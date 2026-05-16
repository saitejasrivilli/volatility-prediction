"""Lightweight inference service for research artifacts."""

from pathlib import Path
import json
from collections import OrderedDict
from typing import Optional

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

try:
    from .model_registry import ModelRegistry
    from .monitoring import ModelMonitor
    from .ab_testing import ABTestFramework
except ImportError:
    from model_registry import ModelRegistry
    from monitoring import ModelMonitor
    from ab_testing import ABTestFramework

app = FastAPI(title="Equity Volatility Intelligence")

# In-process model cache (LRU, max 5 models)
_model_cache = OrderedDict()
_cache_max_size = 5
_prediction_count = 0
_cache_hits = 0


class PredictionRequest(BaseModel):
    features: dict[str, float]


def _get_cached_artifact(results_dir: str) -> dict:
    """Get artifact from cache or load from disk.

    Args:
        results_dir: Results directory path

    Returns:
        Loaded artifact dict
    """
    global _model_cache, _cache_hits

    if results_dir in _model_cache:
        # Move to end (most recent)
        _model_cache.move_to_end(results_dir)
        _cache_hits += 1
        return _model_cache[results_dir]

    # Load from disk
    artifact_path = Path(results_dir) / "model_artifact.joblib"
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="No model artifact found")

    artifact = joblib.load(artifact_path)

    # Add to cache
    _model_cache[results_dir] = artifact
    if len(_model_cache) > _cache_max_size:
        _model_cache.popitem(last=False)  # Remove oldest

    return artifact


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info(results_dir: str = "results/latest"):
    metadata_path = Path(results_dir) / "run_metadata.json"
    registry_path = Path(results_dir) / "experiment_registry.csv"
    if not metadata_path.exists() or not registry_path.exists():
        raise HTTPException(status_code=404, detail="No experiment artifacts found")
    return {
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
        "experiments": registry_path.read_text(encoding="utf-8"),
    }


@app.post("/predict")
def predict(request: PredictionRequest, results_dir: str = "results/latest"):
    global _prediction_count
    _prediction_count += 1

    artifact = _get_cached_artifact(results_dir)

    columns = artifact["feature_columns"]
    missing = [column for column in columns if column not in request.features]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_features": missing})

    frame = pd.DataFrame([{column: request.features[column] for column in columns}])
    transformed = artifact["preprocessor"].transform(frame)
    probability = float(artifact["model"].predict_proba(transformed)[0, 1])

    return {
        "probability": probability,
        "target_kind": artifact["target_kind"],
        "model_kind": artifact["model_kind"],
    }


@app.get("/metrics")
def metrics():
    """Prometheus-format metrics endpoint."""
    cache_ratio = _cache_hits / max(_prediction_count, 1)
    cache_size = len(_model_cache)

    metrics_text = f"""# HELP prediction_total Total predictions served
# TYPE prediction_total counter
prediction_total {_prediction_count}

# HELP cache_hits Total cache hits
# TYPE cache_hits counter
cache_hits {_cache_hits}

# HELP cache_hit_ratio Cache hit ratio (0-1)
# TYPE cache_hit_ratio gauge
cache_hit_ratio {cache_ratio:.2f}

# HELP model_cache_size Current model cache size
# TYPE model_cache_size gauge
model_cache_size {cache_size}
"""
    return metrics_text


@app.get("/model-version")
def model_version(registry_dir: Optional[str] = None):
    """Get current champion model version."""
    registry = ModelRegistry(registry_dir)
    champion = registry.load_champion()

    if not champion:
        raise HTTPException(status_code=404, detail="No champion model registered")

    return champion


@app.post("/register-model")
def register_model(
    source_dir: str,
    version_tag: str,
    ticker: Optional[str] = None,
    model_kind: Optional[str] = None,
    pr_auc: float = 0.0,
    f1: float = 0.0,
    registry_dir: Optional[str] = None,
):
    """Register a model artifact in the registry."""
    registry = ModelRegistry(registry_dir)

    metrics = {
        "ticker": ticker or "unknown",
        "model_kind": model_kind or "unknown",
        "pr_auc": pr_auc,
        "f1": f1,
    }

    version_path = registry.register(source_dir, version_tag, metrics)
    return {"version": version_tag, "path": str(version_path)}


@app.get("/ab-test")
def ab_test(champion: str, challenger: str):
    """Compare champion and challenger models."""
    framework = ABTestFramework()

    try:
        result = framework.compare_result_dirs(
            champion_dir=champion, challenger_dir=challenger
        )
        return {
            "winner": result.winner,
            "p_value": result.p_value,
            "is_significant": result.is_significant,
            "f1_delta": result.f1_delta,
            "pr_auc_delta": result.pr_auc_delta,
            "recommendation": result.recommendation,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
