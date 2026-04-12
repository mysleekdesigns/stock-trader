"""LightGBM predictor with direction / return / volatility sub-models.

Wraps three LightGBM estimators behind a single :class:`BaseModelPredictor`
interface so the rest of the system interacts with one object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
from lightgbm import LGBMClassifier, LGBMRegressor

from src.core.exceptions import (
    ModelLoadError,
    ModelPredictionError,
    ModelTrainingError,
)
from src.models.base import BaseModelPredictor

logger = structlog.get_logger(__name__)

# Sub-model file stems persisted alongside each other.
_DIRECTION_FILE = "direction_classifier.joblib"
_RETURN_FILE = "return_regressor.joblib"
_VOLATILITY_FILE = "volatility_regressor.joblib"
_META_FILE = "meta.joblib"


class LightGBMPredictor(BaseModelPredictor):
    """Gradient-boosted tree predictor backed by LightGBM.

    Internally maintains three sub-models:

    * **direction_classifier** – binary classification (1 = up, 0 = down)
    * **return_regressor** – predicted log-return magnitude
    * **volatility_regressor** – predicted realised volatility
    """

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    _DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    _EARLY_STOPPING_ROUNDS = 50

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(name="lightgbm")

        user_params = params or {}
        merged = {**self._DEFAULT_PARAMS, **user_params}

        # Build each sub-model with objective-specific overrides.
        clf_params = {**merged, "objective": "binary", "metric": "binary_logloss"}
        ret_params = {**merged, "objective": "regression", "metric": "mse"}
        vol_params = {**merged, "objective": "regression", "metric": "mae"}

        self.direction_classifier = LGBMClassifier(**clf_params)
        self.return_regressor = LGBMRegressor(**ret_params)
        self.volatility_regressor = LGBMRegressor(**vol_params)

        self._early_stopping_rounds = user_params.get(
            "early_stopping_rounds", self._EARLY_STOPPING_ROUNDS
        )

        logger.info("lightgbm_predictor_init", params=merged)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs: Any) -> None:
        """Train all three sub-models.

        Parameters
        ----------
        X:
            Feature matrix (n_samples, n_features).
        y:
            Target DataFrame **must** contain columns
            ``["direction", "returns", "volatility"]``.
        kwargs:
            Optional ``val_X`` and ``val_y`` for early-stopping evaluation.
        """
        try:
            X = self.validate_input(X)
            self._validate_targets(y)

            self.feature_names = list(X.columns)

            val_X: pd.DataFrame | None = kwargs.get("val_X")
            val_y: pd.DataFrame | None = kwargs.get("val_y")
            if val_X is not None:
                val_X = self.validate_input(val_X)

            fit_params = self._build_fit_params(X, y, val_X, val_y)

            logger.info(
                "lightgbm_training_start",
                n_samples=X.shape[0],
                n_features=X.shape[1],
                has_validation=val_X is not None,
            )

            # --- direction classifier ---
            self.direction_classifier.fit(
                X, y["direction"], **fit_params["direction"]
            )
            logger.info("lightgbm_direction_trained")

            # --- return regressor ---
            self.return_regressor.fit(
                X, y["returns"], **fit_params["returns"]
            )
            logger.info("lightgbm_return_trained")

            # --- volatility regressor ---
            self.volatility_regressor.fit(
                X, y["volatility"], **fit_params["volatility"]
            )
            logger.info("lightgbm_volatility_trained")

            self.is_fitted = True
            logger.info("lightgbm_training_complete")

        except (ModelTrainingError, ModelPredictionError):
            raise
        except Exception as exc:
            raise ModelTrainingError(
                f"LightGBM training failed: {exc}",
                details={"original_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # predict / predict_proba
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return predictions from all three sub-models.

        Returns a dict with keys ``direction`` (int array), ``returns``
        (float array), and ``volatility`` (float array).
        """
        self._check_fitted()
        try:
            X = self.validate_input(X)
            return {
                "direction": self.direction_classifier.predict(X).astype(int),
                "returns": self.return_regressor.predict(X).astype(np.float64),
                "volatility": self.volatility_regressor.predict(X).astype(np.float64),
            }
        except ModelPredictionError:
            raise
        except Exception as exc:
            raise ModelPredictionError(
                f"LightGBM prediction failed: {exc}",
                details={"original_error": str(exc)},
            ) from exc

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return direction-class probabilities (n_samples, 2)."""
        self._check_fitted()
        try:
            X = self.validate_input(X)
            return self.direction_classifier.predict_proba(X)
        except ModelPredictionError:
            raise
        except Exception as exc:
            raise ModelPredictionError(
                f"LightGBM predict_proba failed: {exc}",
                details={"original_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist all sub-models and metadata to *path* directory."""
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.direction_classifier, path / _DIRECTION_FILE)
        joblib.dump(self.return_regressor, path / _RETURN_FILE)
        joblib.dump(self.volatility_regressor, path / _VOLATILITY_FILE)
        joblib.dump(
            {"feature_names": self.feature_names, "name": self.name},
            path / _META_FILE,
        )
        logger.info("lightgbm_model_saved", path=str(path))

    def load(self, path: Path) -> None:
        """Restore sub-models and metadata from *path* directory."""
        path = Path(path)
        try:
            self.direction_classifier = joblib.load(path / _DIRECTION_FILE)
            self.return_regressor = joblib.load(path / _RETURN_FILE)
            self.volatility_regressor = joblib.load(path / _VOLATILITY_FILE)

            meta: dict[str, Any] = joblib.load(path / _META_FILE)
            self.feature_names = meta.get("feature_names", [])
            self.name = meta.get("name", self.name)

            self.is_fitted = True
            logger.info(
                "lightgbm_model_loaded",
                path=str(path),
                n_features=len(self.feature_names),
            )
        except FileNotFoundError as exc:
            raise ModelLoadError(
                f"LightGBM model files not found at {path}",
                details={"path": str(path), "original_error": str(exc)},
            ) from exc
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load LightGBM model: {exc}",
                details={"path": str(path), "original_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances averaged across all three sub-models."""
        self._check_fitted()

        importances: dict[str, list[float]] = {f: [] for f in self.feature_names}

        for model in (
            self.direction_classifier,
            self.return_regressor,
            self.volatility_regressor,
        ):
            raw = model.feature_importances_
            # Normalise to [0, 1] per sub-model so each contributes equally.
            total = float(raw.sum()) or 1.0
            for fname, score in zip(self.feature_names, raw):
                importances[fname].append(score / total)

        return {
            fname: float(np.mean(scores))
            for fname, scores in importances.items()
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_targets(y: pd.DataFrame) -> None:
        required = {"direction", "returns", "volatility"}
        if not isinstance(y, pd.DataFrame):
            raise ModelTrainingError(
                "Target y must be a DataFrame with columns: direction, returns, volatility",
                details={"received_type": type(y).__name__},
            )
        missing = required - set(y.columns)
        if missing:
            raise ModelTrainingError(
                f"Target DataFrame missing columns: {missing}",
                details={"missing": sorted(missing), "present": sorted(y.columns)},
            )

    def _build_fit_params(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        val_X: pd.DataFrame | None,
        val_y: pd.DataFrame | None,
    ) -> dict[str, dict[str, Any]]:
        """Build per-sub-model ``fit()`` keyword arguments."""
        params: dict[str, dict[str, Any]] = {
            "direction": {},
            "returns": {},
            "volatility": {},
        }

        if val_X is not None and val_y is not None:
            callbacks = [
                _early_stopping_callback(self._early_stopping_rounds),
            ]
            params["direction"]["eval_set"] = [(val_X, val_y["direction"])]
            params["direction"]["callbacks"] = callbacks

            params["returns"]["eval_set"] = [(val_X, val_y["returns"])]
            params["returns"]["callbacks"] = list(callbacks)

            params["volatility"]["eval_set"] = [(val_X, val_y["volatility"])]
            params["volatility"]["callbacks"] = list(callbacks)

        return params


def _early_stopping_callback(stopping_rounds: int) -> Any:
    """Return a LightGBM early-stopping callback."""
    from lightgbm import early_stopping

    return early_stopping(stopping_rounds=stopping_rounds, verbose=False)
