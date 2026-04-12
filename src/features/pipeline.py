"""Feature engineering pipeline with fit/transform semantics and persistence.

The :class:`FeaturePipeline` wraps a :class:`FeatureRegistry`, computes
requested features, and optionally normalises them using statistics learned
during :meth:`fit`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
import structlog

from src.features.registry import FeatureRegistry

logger = structlog.get_logger(__name__)


class FeaturePipeline:
    """Compute features from a registry, with optional z-score normalization.

    Parameters
    ----------
    registry:
        The :class:`FeatureRegistry` containing feature definitions.
    feature_names:
        Subset of features to compute.  ``None`` means *all* registered
        features.
    normalize:
        If ``True``, :meth:`transform` will z-score normalize features using
        statistics from :meth:`fit`.
    """

    def __init__(
        self,
        registry: FeatureRegistry,
        feature_names: list[str] | None = None,
        normalize: bool = False,
    ) -> None:
        self._registry = registry
        self._feature_names = feature_names
        self._normalize = normalize

        # Normalization statistics populated by fit()
        self._stats: dict[str, dict[str, float]] = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> FeaturePipeline:
        """Compute features and learn per-feature mean/std for normalization.

        Returns ``self`` for method-chaining.
        """
        logger.info("pipeline_fit_start", rows=len(data))
        df = self._compute(data)
        df = self._handle_missing(df)

        feature_cols = self.get_feature_names()
        self._stats = {}
        for col in feature_cols:
            if col in df.columns:
                self._stats[col] = {
                    "mean": float(df[col].mean()) if not df[col].isna().all() else 0.0,
                    "std": float(df[col].std()) if not df[col].isna().all() else 1.0,
                }
                # Guard against zero std
                if self._stats[col]["std"] == 0.0 or np.isnan(self._stats[col]["std"]):
                    self._stats[col]["std"] = 1.0

        self._is_fitted = True
        logger.info("pipeline_fit_complete", features_fitted=len(self._stats))
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute features and optionally normalize using learned stats.

        If :attr:`normalize` is ``True``, :meth:`fit` must have been called
        first (or stats loaded via :meth:`load`).
        """
        logger.info("pipeline_transform_start", rows=len(data))
        df = self._compute(data)
        df = self._handle_missing(df)

        if self._normalize:
            if not self._is_fitted:
                raise RuntimeError(
                    "Pipeline has not been fitted. Call fit() or load() before "
                    "transform() when normalize=True."
                )
            feature_cols = [c for c in self.get_feature_names() if c in df.columns]
            for col in feature_cols:
                if col in self._stats:
                    mean = self._stats[col]["mean"]
                    std = self._stats[col]["std"]
                    df[col] = (df[col] - mean) / std

        logger.info("pipeline_transform_complete", output_cols=len(df.columns), rows=len(df))
        return df

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convenience: :meth:`fit` then :meth:`transform` in one call."""
        self.fit(data)
        return self.transform(data)

    def get_feature_names(self) -> list[str]:
        """Return the list of feature names this pipeline will produce."""
        if self._feature_names is not None:
            return list(self._feature_names)
        return self._registry.feature_names

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist normalization parameters to *path* as JSON (via orjson)."""
        path = Path(path)
        payload: dict[str, Any] = {
            "feature_names": self._feature_names,
            "normalize": self._normalize,
            "is_fitted": self._is_fitted,
            "stats": self._stats,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        logger.info("pipeline_saved", path=str(path))

    def load(self, path: str | Path) -> FeaturePipeline:
        """Restore normalization parameters from *path*.

        Returns ``self`` for method-chaining.
        """
        path = Path(path)
        raw = orjson.loads(path.read_bytes())
        self._feature_names = raw.get("feature_names")
        self._normalize = raw.get("normalize", False)
        self._is_fitted = raw.get("is_fitted", False)
        self._stats = raw.get("stats", {})
        logger.info("pipeline_loaded", path=str(path), features=len(self._stats))
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Delegate feature computation to the registry."""
        if data.empty:
            logger.warning("pipeline_empty_input")
            return data
        if self._feature_names is not None:
            return self._registry.compute_subset(data, self._feature_names)
        return self._registry.compute_all(data)

    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill then drop leading NaN rows."""
        df = df.ffill()
        # Drop rows where *any* column is still NaN (typically the head
        # where look-back windows haven't filled yet)
        first_valid = df.apply(lambda s: s.first_valid_index()).max()
        if first_valid is not None and first_valid in df.index:
            idx = df.index.get_loc(first_valid)
            if isinstance(idx, slice):
                idx = idx.start or 0
            elif isinstance(idx, np.ndarray):
                idx = int(np.argmax(idx))
            df = df.iloc[int(idx):]
        return df
