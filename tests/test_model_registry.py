"""Unit tests for model registry."""

import json
import tempfile
from pathlib import Path

import pytest

try:
    from src.model_registry import ModelRegistry
except ImportError:
    from model_registry import ModelRegistry


class TestModelRegistry:
    """Test model registry functionality."""

    def test_init_creates_directory(self):
        """Test that init creates registry directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            assert Path(tmpdir).exists()

    def test_versions_csv_initialized(self):
        """Test that versions.csv is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            versions_file = Path(tmpdir) / "versions.csv"
            assert versions_file.exists()

    def test_register_model(self):
        """Test registering a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source directory with dummy artifacts
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()
            (source_dir / "model_artifact.joblib").touch()
            (source_dir / "run_metadata.json").write_text('{"key": "value"}')

            registry = ModelRegistry(Path(tmpdir) / "registry")
            metrics = {"ticker": "AAPL", "model_kind": "logistic", "pr_auc": 0.75}
            version_path = registry.register(source_dir, "v1", metrics)

            assert version_path.exists()
            assert (version_path / "model_artifact.joblib").exists()

    def test_list_versions(self):
        """Test listing versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            versions = registry.list_versions()
            assert isinstance(versions, object)  # DataFrame

    def test_promote_and_load_champion(self):
        """Test promoting and loading champion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source directory
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()
            (source_dir / "model_artifact.joblib").touch()
            (source_dir / "run_metadata.json").write_text("{}")

            registry = ModelRegistry(Path(tmpdir) / "registry")
            registry.register(source_dir, "v1")
            registry.promote("v1")

            champion = registry.load_champion()
            assert champion["version"] == "v1"

    def test_get_champion_artifact_path(self):
        """Test getting champion artifact path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()
            (source_dir / "model_artifact.joblib").touch()
            (source_dir / "run_metadata.json").write_text("{}")

            registry = ModelRegistry(Path(tmpdir) / "registry")
            registry.register(source_dir, "v1")
            registry.promote("v1")

            path = registry.get_champion_artifact_path()
            assert path is not None
            assert path.name == "model_artifact.joblib"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
