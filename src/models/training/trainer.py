"""Model training orchestrator with walk-forward validation.

The :class:`ModelTrainer` coordinates feature engineering, model training,
validation, and ensemble fitting across walk-forward splits, ensuring strict
temporal separation at every stage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from src.core.exceptions import ModelTrainingError
from src.features.pipeline import FeaturePipeline
from src.models.base import BaseModelPredictor
from src.models.ensemble import EnsemblePredictor
from src.models.training.walk_forward import WalkForwardSplitter

logger = structlog.get_logger(__name__)


class ModelTrainer:
    """Orchestrates walk-forward model training, evaluation, and ensemble fitting.

    Parameters
    ----------
    models:
        List of base predictor instances to train.
    feature_pipeline:
        Feature engineering pipeline with fit/transform semantics.
    splitter:
        Walk-forward splitter instance.
    """

    def __init__(
        self,
        models: list[BaseModelPredictor],
        feature_pipeline: FeaturePipeline,
        splitter: WalkForwardSplitter,
    ) -> None:
        self.models = models
        self.feature_pipeline = feature_pipeline
        self.splitter = splitter

        logger.info(
            "trainer_init",
            n_models=len(models),
            model_names=[m.name for m in models],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all(
        self,
        data: pd.DataFrame,
        target_col: str = "returns",
    ) -> dict[str, Any]:
        """Run full walk-forward training pipeline.

        Parameters
        ----------
        data:
            Raw OHLCV+ DataFrame with a DatetimeIndex.
        target_col:
            Column name for the primary return target.

        Returns
        -------
        dict
            ``models_metrics`` per model per split, ``ensemble_metrics``,
            ``aggregate_metrics``, and the trained ``ensemble`` instance.
        """
        logger.info("train_all_start", data_rows=len(data), target=target_col)

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ModelTrainingError(
                "Data must have a DatetimeIndex",
                details={"index_type": type(data.index).__name__},
            )

        timestamps = data.index
        splits = list(self.splitter.split(timestamps))

        if not splits:
            raise ModelTrainingError(
                "No valid walk-forward splits found",
                details={
                    "data_span_days": (timestamps.max() - timestamps.min()).days,
                    "required_days": (
                        self.splitter.train_days
                        + self.splitter.val_days
                        + self.splitter.test_days
                    ),
                },
            )

        all_results: dict[str, list[dict[str, Any]]] = {m.name: [] for m in self.models}
        oos_predictions: dict[str, list[np.ndarray]] = {m.name: [] for m in self.models}
        oos_targets: list[np.ndarray] = []
        oos_features: list[pd.DataFrame] = []

        for split_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
            logger.info("train_all_split", split=split_idx, n_splits=len(splits))

            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            test_data = data.iloc[test_idx]

            # Fit feature pipeline on training data only
            self.feature_pipeline.fit(train_data)
            X_train = self.feature_pipeline.transform(train_data)
            X_val = self.feature_pipeline.transform(val_data)
            X_test = self.feature_pipeline.transform(test_data)

            # Create targets
            y_train = self._create_targets(X_train, train_data, target_col)
            y_val = self._create_targets(X_val, val_data, target_col)
            y_test = self._create_targets(X_test, test_data, target_col)

            # Drop any non-feature columns from X matrices
            feature_cols = self.feature_pipeline.get_feature_names()
            X_train = self._select_features(X_train, feature_cols)
            X_val = self._select_features(X_val, feature_cols)
            X_test = self._select_features(X_test, feature_cols)

            for model in self.models:
                try:
                    # Train and validate
                    train_metrics = self.train_single(model, X_train, y_train, X_val, y_val)

                    # Evaluate on test
                    test_metrics = self.evaluate(model, X_test, y_test)

                    result = {
                        "split": split_idx,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                    }
                    all_results[model.name].append(result)

                    # Collect OOS predictions for ensemble training
                    preds = model.predict(X_test)
                    if isinstance(preds, dict) and "returns" in preds:
                        oos_predictions[model.name].append(preds["returns"])
                    elif isinstance(preds, np.ndarray):
                        oos_predictions[model.name].append(preds.ravel())
                    else:
                        oos_predictions[model.name].append(
                            np.zeros(len(X_test), dtype=np.float64)
                        )

                except Exception:
                    logger.error(
                        "train_all_model_failed",
                        model=model.name,
                        split=split_idx,
                        exc_info=True,
                    )
                    all_results[model.name].append(
                        {"split": split_idx, "error": True}
                    )

            oos_targets.append(y_test["returns"].values if "returns" in y_test else y_test.values)
            oos_features.append(X_test)

        # Train ensemble on all OOS predictions
        ensemble = self._train_ensemble(oos_features, oos_targets)

        # Aggregate metrics
        aggregate = self._aggregate_metrics(all_results)

        logger.info("train_all_complete", n_splits=len(splits), aggregate=aggregate)

        return {
            "models_metrics": all_results,
            "aggregate_metrics": aggregate,
            "ensemble": ensemble,
            "n_splits": len(splits),
        }

    def train_single(
        self,
        model: BaseModelPredictor,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame | pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame | pd.Series,
    ) -> dict[str, Any]:
        """Train a single model and return validation metrics.

        Parameters
        ----------
        model:
            Predictor instance to train.
        X_train, y_train:
            Training features and targets.
        X_val, y_val:
            Validation features and targets.

        Returns
        -------
        dict
            Validation metrics including accuracy, AUC, MSE, directional accuracy.
        """
        logger.info(
            "train_single_start",
            model=model.name,
            train_samples=len(X_train),
            val_samples=len(X_val),
        )

        # Extract returns target for tree models
        if isinstance(y_train, pd.DataFrame):
            y_fit = y_train["direction"] if "direction" in y_train.columns else y_train.iloc[:, 0]
        else:
            y_fit = y_train

        model.fit(X_train, y_fit, X_val=X_val, y_val=y_val)

        metrics = self.evaluate(model, X_val, y_val)
        logger.info("train_single_complete", model=model.name, val_metrics=metrics)
        return metrics

    def evaluate(
        self,
        model: BaseModelPredictor,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame | pd.Series,
    ) -> dict[str, Any]:
        """Evaluate model on held-out data.

        Returns
        -------
        dict
            Metrics: accuracy, auc, mse, directional_accuracy, n_samples.
        """
        if isinstance(y_test, pd.DataFrame):
            y_direction = y_test["direction"].values if "direction" in y_test.columns else None
            y_returns = y_test["returns"].values if "returns" in y_test.columns else None
        else:
            y_returns = y_test.values
            y_direction = np.sign(y_returns)

        metrics: dict[str, Any] = {"n_samples": len(X_test)}

        try:
            preds = model.predict(X_test)
            if isinstance(preds, dict):
                pred_returns = preds.get("returns", np.zeros(len(X_test)))
            else:
                pred_returns = np.asarray(preds).ravel()

            pred_direction = np.sign(pred_returns)

            if y_returns is not None:
                metrics["mse"] = float(mean_squared_error(y_returns, pred_returns[:len(y_returns)]))

            if y_direction is not None:
                dir_pred = pred_direction[:len(y_direction)]
                # Directional accuracy: fraction where signs match
                mask = y_direction != 0
                if mask.sum() > 0:
                    metrics["directional_accuracy"] = float(
                        np.mean(dir_pred[mask] == y_direction[mask])
                    )
        except Exception:
            logger.warning("evaluate_predict_failed", model=model.name, exc_info=True)

        try:
            proba = model.predict_proba(X_test)
            if proba.ndim == 2:
                proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

            if y_direction is not None:
                # For AUC, convert direction to binary (1 = positive, 0 = negative/zero)
                y_binary = (y_direction > 0).astype(int)
                p = proba[:len(y_binary)]
                if len(np.unique(y_binary)) > 1:
                    metrics["auc"] = float(roc_auc_score(y_binary, p))

                # Accuracy using 0.5 threshold
                pred_binary = (p > 0.5).astype(int)
                metrics["accuracy"] = float(accuracy_score(y_binary, pred_binary))
        except Exception:
            logger.warning("evaluate_proba_failed", model=model.name, exc_info=True)

        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_targets(
        self,
        X: pd.DataFrame,
        raw_data: pd.DataFrame,
        target_col: str,
    ) -> pd.DataFrame:
        """Create target variables aligned with feature matrix X.

        Creates:
        - ``returns``: raw return values
        - ``direction``: sign of returns (1, -1, 0)
        - ``volatility``: rolling 21-day standard deviation of returns
        """
        # Align raw data index to X's index
        common_idx = X.index.intersection(raw_data.index)

        if target_col in raw_data.columns:
            returns = raw_data.loc[common_idx, target_col].values
        elif "close" in raw_data.columns:
            close = raw_data.loc[common_idx, "close"]
            returns = close.pct_change().fillna(0).values
        else:
            raise ModelTrainingError(
                f"Cannot find target column '{target_col}' or 'close' in data",
                details={"available_columns": list(raw_data.columns)},
            )

        direction = np.sign(returns).astype(np.float64)

        # Rolling volatility
        ret_series = pd.Series(returns, index=common_idx)
        volatility = ret_series.rolling(window=21, min_periods=5).std().fillna(0).values

        targets = pd.DataFrame(
            {"returns": returns, "direction": direction, "volatility": volatility},
            index=common_idx,
        )
        return targets

    @staticmethod
    def _select_features(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Select only feature columns present in X."""
        available = [c for c in feature_cols if c in X.columns]
        if available:
            return X[available]
        return X

    def _train_ensemble(
        self,
        oos_features: list[pd.DataFrame],
        oos_targets: list[np.ndarray],
    ) -> EnsemblePredictor:
        """Fit ensemble on collected out-of-sample predictions."""
        if not oos_features or not oos_targets:
            logger.warning("train_ensemble_no_oos_data")
            return EnsemblePredictor(models=self.models)

        X_oos = pd.concat(oos_features, axis=0)
        y_oos = np.concatenate(oos_targets, axis=0)

        # Align lengths
        n = min(len(X_oos), len(y_oos))
        X_oos = X_oos.iloc[:n]
        y_oos = y_oos[:n]

        ensemble = EnsemblePredictor(models=self.models)
        try:
            ensemble.fit(X_oos, y_oos)
            logger.info("train_ensemble_complete", n_oos_samples=n)
        except Exception:
            logger.error("train_ensemble_failed", exc_info=True)

        return ensemble

    @staticmethod
    def _aggregate_metrics(
        all_results: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, float]]:
        """Compute mean metrics across splits for each model."""
        aggregate: dict[str, dict[str, float]] = {}

        for model_name, split_results in all_results.items():
            metric_sums: dict[str, list[float]] = {}

            for result in split_results:
                test_m = result.get("test_metrics", {})
                for key, val in test_m.items():
                    if isinstance(val, (int, float)):
                        metric_sums.setdefault(key, []).append(float(val))

            aggregate[model_name] = {
                k: float(np.mean(v)) for k, v in metric_sums.items()
            }

        return aggregate
