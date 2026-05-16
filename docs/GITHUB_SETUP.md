# GitHub Setup Notes

## What to commit

Commit source, tests, documentation, and configuration templates:

- `src/`
- `tests/`
- `docs/`
- `README.md`
- `QUICKSTART.md`
- `requirements*.txt`
- `setup.py`
- `.env.example`

Do **not** commit generated local artifacts such as:

- `venv/`
- `results/`
- `*.log`
- `__pycache__/`
- `.DS_Store`

The repository already includes ignore rules for these files.

## Suggested first commands

```bash
git init
git add .
git commit -m "Initial cleaned volatility prediction pipeline"
```

## CI expectations

The GitHub Actions workflow should fail when formatting, linting, or tests fail. Avoid masking failures with `|| true`; they make green CI less meaningful.

## Repository description

Prefer a truthful description such as:

> Research-oriented volatility classification pipeline with fold-safe preprocessing and walk-forward validation.

Avoid fixed performance claims unless they are tied to a reproducible tagged experiment and the exact configuration is documented.
