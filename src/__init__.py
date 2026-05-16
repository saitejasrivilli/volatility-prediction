"""
Volatility Prediction System

A production-grade ML system for predicting equity volatility spikes.
Uses walk-forward validation, realistic transaction costs, and risk management.

Main modules:
- config: Configuration and parameters
- data_pipeline: Fetch, validate, clean financial data
- feature_engineering: Technical indicators and features
- backtesting: Walk-forward validation and performance metrics
- main: Orchestrate entire pipeline
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Volatility Prediction System with Walk-Forward Validation"

from .config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    TRADING_CONFIG,
    WALK_FORWARD_CONFIG,
    print_config,
)

__all__ = [
    "DATA_CONFIG",
    "MODEL_CONFIG",
    "TRADING_CONFIG",
    "WALK_FORWARD_CONFIG",
    "print_config",
]
