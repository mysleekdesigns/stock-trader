"""Ridge regression stacking ensemble for combining base model predictions.

The :class:`EnsemblePredictor` collects predictions from heterogeneous base
models (tree-based, LSTM, transformer) and learns a Ridge meta-learner per
forecast horizon that optimally blends them.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
import structlog
from sklearn.linear_model import Ridge

from src.core.exceptions import ModelError, ModelPredictionError, ModelTrainingError
from src.models.base import BaseModelPredictor

logger = structlog.get_logger(__name__)


class EnsemblePredictor(BaseModelPredictor):
    """Stacking ensemble that blends base model predictions via Ridge regression.

    Parameters
    ----------
    models:
        List of fitted :class:`BaseModelPredictor` instances whose predictions
        will be stacked.
    horizons:
        Forecast horizons (in bars/days) for which to learn separate
        meta-learners.
    min_weight_floor:
        Minimum weight any single model can have after weight adjustment.
    ridge_alpha:
        L2 regularization strength for the Ridge meta-learners.
    """

    def __init__(
        self,
        models: list[BaseModelPredictor],
        horizons: list[int] | None = None,
        min_weight_floor: float = 0.05,
        ridge_alpha: float = 1.0,
    ) -> None:
        super().__init__(name="ensemble")
        self.models = models
        self.horizons: list[int] = horizons or [1, 2, 5, 10, 21]
        self.min_weight_floor = min_weight_floor
        self.ridge_alpha = ridge_alpha

        # One Ridge meta-learner per horizon
        self._meta_learners: dict[int, Ridge] = {}

        # Model weights (soft blend coefficients, sum to 1)
        self._model_weights: dict[str, float] = {
            m.name: 1.0 / len(models) for m in models
        }

        logger.info(
            "ensemble_init",
            n_models=len(models),
            model_names=[m.name for m in models],
            horizons=self.horizons,
        )

    # ------------------------------------------------------------------
    # fit / predict / predict_proba
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any, **kwargs: Any) -> None:
        """Fit per-horizon Ridge meta-learners on stacked base-model predictions.

        Parameters
        ----------
        X:
            Feature matrix used to generate base-model predictions.
        y:
            Target DataFrame or Series.  If a DataFrame it should contain
            columns named ``returns_h{horizon}`` for each horizon, or a single
            ``returns`` column that will be used for all horizons.  A Series is
            broadcast to every horizon.
        """
        logger.info("ensemble_fit_start", n_samples=len(X), n_models=len(self.models))

        meta_features = self._collect_meta_features(X)

        if isinstance(y, pd.Series):
            targets: dict[int, np.ndarray] = {h: y.values for h in self.horizons}
        elif isinstance(y, pd.DataFrame):
            targets = {}
            for h in self.horizons:
                col = f"returns_h{h}"
                if col in y.columns:
                    targets[h] = y[col].values
                elif "returns" in y.columns:
                    targets[h] = y["returns"].values
                else:
                    raise ModelTrainingError(
                        f"Target DataFrame must contain '{col}' or 'returns' column",
                        details={"available_columns": list(y.columns)},
                    )
        else:
            targets = {h: np.asarray(y) for h in self.horizons}

        # Align lengths (meta_features may be shorter if models drop rows)
        n = meta_features.shape[0]
        for h in self.horizons:
            t = targets[h]
            if len(t) > n:
                targets[h] = t[-n:]
            elif len(t) < n:
                meta_features = meta_features[-len(t):]
                n = meta_features.shape[0]

        for h in self.horizons:
            ridge = Ridge(alpha=self.ridge_alpha)
            mask = ~np.isnan(targets[h]) & ~np.isnan(meta_features).any(axis=1)
            if mask.sum() < 10:
                logger.warning(
                    "ensemble_insufficient_samples",
                    horizon=h,
                    valid_samples=int(mask.sum()),
                )
                continue

            ridge.fit(meta_features[mask], targets[h][mask])
            self._meta_learners[h] = ridge
            logger.info(
                "ensemble_meta_learner_fitted",
                horizon=h,
                n_samples=int(mask.sum()),
                coefs=ridge.coef_.tolist(),
            )

        self.is_fitted = True
        self.feature_names = list(X.columns)
        logger.info("ensemble_fit_complete", horizons_fitted=list(self._meta_learners.keys()))

    def predict(self, X: pd.DataFrame) -> dict[str, Any]:
        """Return combined predictions per horizon.

        Returns
        -------
        dict
            Keys are ``predictions_h{horizon}`` with arrays of predicted
            returns, plus ``direction`` with the sign-based direction array
            for the shortest horizon.
        """
        self._check_fitted()
        meta_features = self._collect_meta_features(X)

        result: dict[str, Any] = {}
        for h in self.horizons:
            if h not in self._meta_learners:
                continue
            preds = self._meta_learners[h].predict(meta_features)
            result[f"predictions_h{h}"] = preds

        # Direction based on the shortest fitted horizon
        shortest = min(self._meta_learners.keys()) if self._meta_learners else self.horizons[0]
        if f"predictions_h{shortest}" in result:
            preds_short = result[f"predictions_h{shortest}"]
            result["direction"] = np.sign(preds_short)
            result["returns"] = preds_short

        return result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return combined direction probabilities (probability of positive return).

        Aggregates base-model probabilities using the current model weights.
        """
        self._check_fitted()

        weighted_proba = np.zeros(len(X), dtype=np.float64)
        total_weight = 0.0

        for model in self.models:
            w = self._model_weights.get(model.name, 0.0)
            if w <= 0:
                continue
            try:
                proba = model.predict_proba(X)
                # Handle multi-column output: take probability of positive class
                if proba.ndim == 2:
                    proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                # Align length
                n = min(len(weighted_proba), len(proba))
                weighted_proba[:n] += w * proba[:n]
                total_weight += w
            except Exception:
                logger.warning("ensemble_proba_model_failed", model=model.name, exc_info=True)

        if total_weight > 0:
            weighted_proba /= total_weight

        return weighted_proba

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def update_weights(
        self,
        rolling_sharpes: dict[str, float],
        decay_factor: float = 0.95,
    ) -> None:
        """Adjust model weights using exponential decay on rolling Sharpe ratios.

        Parameters
        ----------
        rolling_sharpes:
            Mapping of model name to its recent rolling Sharpe ratio.
        decay_factor:
            Exponential decay applied to current weights before blending with
            the new Sharpe-based weights.
        """
        logger.info(
            "ensemble_weight_update_start",
            rolling_sharpes=rolling_sharpes,
            decay_factor=decay_factor,
        )

        # Compute Sharpe-based raw weights (shift to non-negative)
        sharpe_vals = np.array(
            [rolling_sharpes.get(m.name, 0.0) for m in self.models], dtype=np.float64
        )
        shifted = sharpe_vals - sharpe_vals.min() + 1e-8
        sharpe_weights = shifted / shifted.sum()

        # Blend with decayed current weights
        for i, model in enumerate(self.models):
            old_w = self._model_weights.get(model.name, 1.0 / len(self.models))
            new_w = decay_factor * old_w + (1.0 - decay_factor) * sharpe_weights[i]
            self._model_weights[model.name] = new_w

        # Enforce floor
        for name in self._model_weights:
            self._model_weights[name] = max(self._model_weights[name], self.min_weight_floor)

        # Normalize to sum to 1
        total = sum(self._model_weights.values())
        if total > 0:
            for name in self._model_weights:
                self._model_weights[name] /= total

        logger.info("ensemble_weight_update_complete", weights=self._model_weights)

    def get_model_weights(self) -> dict[str, float]:
        """Return current model weights."""
        return dict(self._model_weights)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        """Return weighted average of feature importance across base models."""
        combined: dict[str, float] = {}
        total_weight = 0.0

        for model in self.models:
            w = self._model_weights.get(model.name, 0.0)
            if w <= 0:
                continue
            try:
                importance = model.get_feature_importance()
                for feat, imp in importance.items():
                    combined[feat] = combined.get(feat, 0.0) + w * imp
                total_weight += w
            except Exception:
                logger.warning(
                    "ensemble_feature_importance_failed", model=model.name, exc_info=True
                )

        if total_weight > 0:
            combined = {k: v / total_weight for k, v in combined.items()}

        return dict(sorted(combined.items(), key=lambda kv: kv[1], reverse=True))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist meta-learners, weights, and model list metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Meta-learners
        meta_path = path / "meta_learners.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(self._meta_learners, f)

        # Metadata — ensure all values are JSON-native types
        metadata = {
            "model_names": [m.name for m in self.models],
            "model_weights": {k: float(v) for k, v in self._model_weights.items()},
            "horizons": self.horizons,
            "min_weight_floor": float(self.min_weight_floor),
            "ridge_alpha": float(self.ridge_alpha),
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        meta_json = path / "metadata.json"
        meta_json.write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

        logger.info("ensemble_saved", path=str(path))

    def load(self, path: Path) -> None:
        """Restore meta-learners, weights, and metadata from *path*."""
        path = Path(path)

        meta_path = path / "meta_learners.pkl"
        if not meta_path.exists():
            raise ModelError(
                f"Meta-learner file not found: {meta_path}",
                details={"path": str(meta_path)},
            )
        with open(meta_path, "rb") as f:
            self._meta_learners = pickle.load(f)  # noqa: S301

        meta_json = path / "metadata.json"
        if meta_json.exists():
            data = orjson.loads(meta_json.read_bytes())
            self._model_weights = data.get("model_weights", self._model_weights)
            self.horizons = data.get("horizons", self.horizons)
            self.min_weight_floor = data.get("min_weight_floor", self.min_weight_floor)
            self.ridge_alpha = data.get("ridge_alpha", self.ridge_alpha)
            self.feature_names = data.get("feature_names", self.feature_names)
            self.is_fitted = data.get("is_fitted", False)

        logger.info("ensemble_loaded", path=str(path), horizons=list(self._meta_learners.keys()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Stack base-model predictions into a meta-feature matrix.

        Each model contributes one column (its predicted return).  If a model
        fails, its column is filled with zeros.
        """
        columns: list[np.ndarray] = []

        for model in self.models:
            try:
                preds = model.predict(X)
                if isinstance(preds, dict):
                    # Use the 'returns' key if available, else first numeric array
                    if "returns" in preds:
                        col = np.asarray(preds["returns"], dtype=np.float64)
                    else:
                        for v in preds.values():
                            arr = np.asarray(v, dtype=np.float64)
                            if arr.ndim <= 1:
                                col = arr
                                break
                        else:
                            col = np.zeros(len(X), dtype=np.float64)
                elif isinstance(preds, np.ndarray):
                    col = preds.astype(np.float64).ravel()
                else:
                    col = np.asarray(preds, dtype=np.float64).ravel()

                # Ensure correct length
                if len(col) != len(X):
                    padded = np.zeros(len(X), dtype=np.float64)
                    n = min(len(col), len(X))
                    padded[-n:] = col[-n:]
                    col = padded

                columns.append(col)
            except Exception:
                logger.warning("ensemble_base_predict_failed", model=model.name, exc_info=True)
                columns.append(np.zeros(len(X), dtype=np.float64))

        if not columns:
            raise ModelPredictionError(
                "No base models produced predictions",
                details={"n_models": len(self.models)},
            )

        return np.column_stack(columns)
