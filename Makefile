.PHONY: help install test run clean docs lint format

help:
	@echo "Volatility Prediction System - Available Commands"
	@echo ""
	@echo "  make install          Install dependencies"
	@echo "  make run              Run complete pipeline"
	@echo "  make test             Run unit tests"
	@echo "  make lint             Check code quality"
	@echo "  make format           Format code with black"
	@echo "  make clean            Remove generated files"
	@echo "  make docs             Generate documentation"
	@echo ""

install:
	pip install -r requirements.txt

run:
	python -m src.main

test:
	python -m pytest tests/ -v

lint:
	python -m pylint src/ --disable=C0111,C0103

format:
	python -m black src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache .coverage
	rm -rf results/*.csv results/*.log
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	@echo "Documentation is in docs/ folder"
	@echo "Start with: README.md, QUICKSTART.md"

# Development targets
install-dev:
	pip install -r requirements.txt
	pip install pytest black pylint sphinx

run-quick:
	python -c "from src.data_pipeline import load_data; from src.feature_engineering import build_features; df = load_data(); features = build_features(df); print(f'✓ Data shape: {df.shape}, Features shape: {features.shape}')"

check:
	@echo "Running quick validation..."
	python -c "import pandas, numpy, sklearn, yfinance; print('✓ All dependencies installed')"
	python -m src.config
