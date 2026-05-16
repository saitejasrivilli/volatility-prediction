# Contributing to Volatility Prediction System

## Development Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/volatility-prediction.git
cd volatility-prediction
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies
```bash
pip install -r requirements.txt
pip install pytest black pylint sphinx
```

## Code Standards

### Style Guide
- Follow PEP 8
- Use type hints on all functions
- Maximum line length: 100 characters
- Use meaningful variable names

### Formatting
```bash
# Format code with black
black src/ tests/

# Check code quality
pylint src/ --disable=C0111,C0103
```

### Documentation
- Every function needs a docstring
- Use Google-style docstrings
- Include examples in docstrings
- Update README if adding features

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Workflow

1. Create a branch for your feature
```bash
git checkout -b feature/your-feature-name
```

2. Make changes and test locally
```bash
make test
make lint
```

3. Commit with clear messages
```bash
git commit -m "Add feature: description of what you did"
```

4. Push and create pull request
```bash
git push origin feature/your-feature-name
```

## Commit Message Style

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 50 characters
- Reference issues when relevant: "Fixes #123"

## Areas for Contribution

- [ ] Additional technical indicators
- [ ] Machine learning models (XGBoost, LightGBM)
- [ ] Extended testing suite
- [ ] Documentation improvements
- [ ] Performance optimizations
- [ ] Additional asset classes
- [ ] Visualization tools

## Questions?

Open an issue on GitHub with the "question" label.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
