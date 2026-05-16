"""Model registry for versioning and champion/challenger management."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    from .config import REGISTRY_CONFIG
except ImportError:
    from config import REGISTRY_CONFIG


class ModelRegistry:
    """Directory-based model versioning system.

    Manages model artifacts, metadata, and promotion of champion models.
    """

    def __init__(self, registry_dir: Path | str | None = None):
        """Initialize model registry.

        Args:
            registry_dir: Root directory for versioned models.
                         Defaults to REGISTRY_CONFIG.REGISTRY_DIR
        """
        if registry_dir is None:
            registry_dir = REGISTRY_CONFIG.REGISTRY_DIR

        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.versions_csv = self.registry_dir / "versions.csv"
        self.champion_json = self.registry_dir / "champion.json"

        # Initialize versions.csv if not exists
        if not self.versions_csv.exists():
            self._init_versions_csv()

    def _init_versions_csv(self) -> None:
        """Initialize empty versions.csv."""
        df = pd.DataFrame(
            columns=[
                "version",
                "ticker",
                "model_kind",
                "target_kind",
                "pr_auc",
                "f1",
                "brier_score",
                "registered_at",
                "promoted",
            ]
        )
        df.to_csv(self.versions_csv, index=False)

    def register(
        self,
        source_dir: Path | str,
        version_tag: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Register model artifact from source directory.

        Args:
            source_dir: Directory containing model_artifact.joblib and metadata
            version_tag: Version identifier (e.g., 'v1', 'v2')
            metrics: Dict with keys like 'pr_auc', 'f1', 'brier_score', 'ticker', 'model_kind'

        Returns:
            Path to registered version directory
        """
        source_dir = Path(source_dir)
        version_dir = self.registry_dir / version_tag
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifact
        artifact_src = source_dir / "model_artifact.joblib"
        artifact_dst = version_dir / "model_artifact.joblib"
        if artifact_src.exists():
            shutil.copy2(artifact_src, artifact_dst)

        # Copy metadata
        metadata_src = source_dir / "run_metadata.json"
        metadata_dst = version_dir / "run_metadata.json"
        if metadata_src.exists():
            shutil.copy2(metadata_src, metadata_dst)

        # Record in versions.csv
        metrics = metrics or {}
        new_row = {
            "version": version_tag,
            "ticker": metrics.get("ticker", "unknown"),
            "model_kind": metrics.get("model_kind", "unknown"),
            "target_kind": metrics.get("target_kind", "unknown"),
            "pr_auc": metrics.get("pr_auc", 0.0),
            "f1": metrics.get("f1", 0.0),
            "brier_score": metrics.get("brier_score", 0.0),
            "registered_at": datetime.now().isoformat(),
            "promoted": False,
        }

        df = pd.read_csv(self.versions_csv)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.versions_csv, index=False)

        return version_dir

    def promote(self, version_tag: str) -> None:
        """Promote version to champion status.

        Args:
            version_tag: Version to promote
        """
        version_dir = self.registry_dir / version_tag
        if not version_dir.exists():
            raise ValueError(f"Version {version_tag} not found")

        # Update champion.json
        champion_info = {
            "version": version_tag,
            "promoted_at": datetime.now().isoformat(),
            "artifact_path": str(version_dir / "model_artifact.joblib"),
        }

        with open(self.champion_json, "w") as f:
            json.dump(champion_info, f, indent=2)

        # Update versions.csv to mark as promoted
        df = pd.read_csv(self.versions_csv)
        df.loc[df["version"] == version_tag, "promoted"] = True
        df.loc[df["version"] != version_tag, "promoted"] = False
        df.to_csv(self.versions_csv, index=False)

    def rollback(self, version_tag: str) -> None:
        """Rollback champion to a previous version.

        Args:
            version_tag: Version to rollback to
        """
        self.promote(version_tag)

    def load_champion(self) -> Dict[str, Any]:
        """Load current champion model artifact.

        Returns:
            Dict with keys: version, artifact_path, promoted_at
        """
        if not self.champion_json.exists():
            return {}

        with open(self.champion_json) as f:
            return json.load(f)

    def get_champion_artifact_path(self) -> Optional[Path]:
        """Get path to champion model artifact.

        Returns:
            Path to model_artifact.joblib or None if no champion
        """
        champion = self.load_champion()
        if champion and "artifact_path" in champion:
            return Path(champion["artifact_path"])
        return None

    def list_versions(self) -> pd.DataFrame:
        """List all registered versions.

        Returns:
            DataFrame with version information
        """
        return pd.read_csv(self.versions_csv)

    def get_version(self, version_tag: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific version.

        Args:
            version_tag: Version identifier

        Returns:
            Dict with version info or None if not found
        """
        df = pd.read_csv(self.versions_csv)
        rows = df[df["version"] == version_tag]

        if rows.empty:
            return None

        row = rows.iloc[0]
        return {
            "version": row["version"],
            "ticker": row["ticker"],
            "model_kind": row["model_kind"],
            "pr_auc": row["pr_auc"],
            "f1": row["f1"],
            "registered_at": row["registered_at"],
            "promoted": bool(row["promoted"]),
        }

    def delete_version(self, version_tag: str) -> None:
        """Delete a registered version (cannot delete promoted).

        Args:
            version_tag: Version to delete

        Raises:
            ValueError if version is promoted
        """
        df = pd.read_csv(self.versions_csv)
        rows = df[df["version"] == version_tag]

        if not rows.empty and rows.iloc[0]["promoted"]:
            raise ValueError(f"Cannot delete promoted version {version_tag}")

        version_dir = self.registry_dir / version_tag
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # Remove from CSV
        df = df[df["version"] != version_tag]
        df.to_csv(self.versions_csv, index=False)
