"""Optuna-based hyperparameter optimization with financial pruning.

The :class:`HyperoptTuner` defines per-model-type search spaces and optimizes
for risk-adjusted performance (Sharpe ratio or AUC) using walk-forward
validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
import structlog

from src.core.exceptions import ModelTrainingError
from src.features.pipeline import FeaturePipeline
from src.models.base import BaseModelPredictor
from src.models.training.walk_forward import WalkForwardSplitter

logger = structlog.get_logger(__name__)

# Suppress Optuna's internal logging to avoid noise
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Search space definitions per model type
# ---------------------------------------------------------------------------

_SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "lightgbm": {
        "n_estimators": {"type": "int", "low": 100, "high": 2000, "log": True},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "num_leaves": {"type": "int", "low": 15, "high": 127},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
        "min_child_samples": {"type": "int", "low": 5, "high": 100},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    },
    "xgboost": {
        "n_estimators": {"type": "int", "low": 100, "high": 2000, "log": True},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 20},
        "gamma": {"type": "float", "low": 1e-8, "high": 5.0, "log": True},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    },
    "lstm": {
        "hidden_size": {"type": "int", "low": 32, "high": 256, "step": 32},
        "num_layers": {"type": "int", "low": 1, "high": 4},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "sequence_length": {"type": "int", "low": 10, "high": 120, "step": 10},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
    },
    "tft": {
        "hidden_size": {"type": "int", "low": 16, "high": 256, "step": 16},
        "attention_head_size": {"type": "int", "low": 1, "high": 8},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "hidden_continuous_size": {"type": "int", "low": 8, "high": 64, "step": 8},
    },
}


class _FinancialPruner(optuna.pruners.BasePruner):
    """Prune trials whose intermediate rolling Sharpe drops below a threshold."""

    def __init__(self, sharpe_threshold: float = -0.5, warmup_steps: int = 5) -> None:
        self.sharpe_threshold = sharpe_threshold
        self.warmup_steps = warmup_steps

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        step = trial.last_step
        if step is None or step < self.warmup_steps:
            return False

        values = [
            trial.intermediate_values[s]
            for s in sorted(trial.intermediate_values.keys())
        ]
        if not values:
            return False

        arr = np.array(values, dtype=np.float64)
        if arr.std() == 0:
            return False

        sharpe = float(arr.mean() / arr.std()) * np.sqrt(252)
        return sharpe < self.sharpe_threshold


class HyperoptTuner:
    """Optuna-based hyperparameter optimizer for trading ML models.

    Parameters
    ----------
    model_class:
        The class (not instance) of the model to tune.  Must accept
        keyword arguments matching the search space.
    feature_pipeline:
        Feature pipeline for data preparation.
    splitter:
        Walk-forward splitter for temporal cross-validation.
    n_trials:
        Number of Optuna trials to run.
    metric:
        Optimization metric: ``"sharpe"`` or ``"auc"``.
    sharpe_threshold:
        Pruning threshold for intermediate rolling Sharpe.
    """

    def __init__(
        self,
        model_class: type[BaseModelPredictor],
        feature_pipeline: FeaturePipeline,
        splitter: WalkForwardSplitter,
        n_trials: int = 100,
        metric: str = "sharpe",
        sharpe_threshold: float = -0.5,
    ) -> None:
        self.model_class = model_class
        self.feature_pipeline = feature_pipeline
        self.splitter = splitter
        self.n_trials = n_trials
        self.metric = metric
        self.sharpe_threshold = sharpe_threshold

        # Detect model type from class name
        self._model_type = self._detect_model_type(model_class)

        logger.info(
            "hyperopt_init",
            model_class=model_class.__name__,
            model_type=self._model_type,
            n_trials=n_trials,
            metric=metric,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(self, data: pd.DataFrame, target_col: str = "returns") -> dict[str, Any]:
        """Run hyperparameter optimization.

        Parameters
        ----------
        data:
            Raw OHLCV+ DataFrame with DatetimeIndex.
        target_col:
            Column name for the return target.

        Returns
        -------
        dict
            ``best_params``, ``best_value``, ``n_trials``, ``n_pruned``.
        """
        logger.info("hyperopt_tune_start", n_trials=self.n_trials, data_rows=len(data))

        self._data = data
        self._target_col = target_col

        pruner = _FinancialPruner(
            sharpe_threshold=self.sharpe_threshold,
            warmup_steps=3,
        )

        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        best = study.best_trial
        result = {
            "best_params": best.params,
            "best_value": best.value,
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }

        logger.info("hyperopt_tune_complete", **result)

        # Clean up reference
        self._data = None
        self._target_col = None

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Single trial objective: returns negative Sharpe or AUC on validation."""
        params = self._sample_params(trial)

        try:
            model = self.model_class(**params)
        except TypeError:
            # Some models need a name kwarg
            model = self.model_class(name=f"trial_{trial.number}", **params)

        data = self._data
        target_col = self._target_col
        timestamps = data.index

        split_scores: list[float] = []

        for step, (train_idx, val_idx, _test_idx) in enumerate(self.splitter.split(timestamps)):
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]

            self.feature_pipeline.fit(train_data)
            X_train = self.feature_pipeline.transform(train_data)
            X_val = self.feature_pipeline.transform(val_data)

            # Create targets
            y_train = self._extract_target(X_train, train_data, target_col)
            y_val = self._extract_target(X_val, val_data, target_col)

            feature_cols = self.feature_pipeline.get_feature_names()
            avail_train = [c for c in feature_cols if c in X_train.columns]
            avail_val = [c for c in feature_cols if c in X_val.columns]
            X_train = X_train[avail_train] if avail_train else X_train
            X_val = X_val[avail_val] if avail_val else X_val

            try:
                model.fit(X_train, np.sign(y_train))
                preds = model.predict(X_val)

                if isinstance(preds, dict):
                    pred_returns = preds.get("returns", np.zeros(len(X_val)))
                else:
                    pred_returns = np.asarray(preds).ravel()

                score = self._compute_score(pred_returns, y_val)
                split_scores.append(score)

                # Report intermediate value for pruning
                trial.report(score, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception:
                logger.warning(
                    "hyperopt_trial_split_failed",
                    trial=trial.number,
                    step=step,
                    exc_info=True,
                )
                split_scores.append(-1.0)

        if not split_scores:
            return -999.0

        return float(np.mean(split_scores))

    def _sample_params(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Sample hyperparameters from the model-specific search space."""
        space = _SEARCH_SPACES.get(self._model_type, {})
        params: dict[str, Any] = {}

        for name, spec in space.items():
            ptype = spec["type"]
            if ptype == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    step=spec.get("step", 1),
                    log=spec.get("log", False),
                )
            elif ptype == "float":
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])

        return params

    def _compute_score(self, pred_returns: np.ndarray, actual_returns: np.ndarray) -> float:
        """Compute objective score from predictions and actuals."""
        n = min(len(pred_returns), len(actual_returns))
        pred = pred_returns[:n]
        actual = actual_returns[:n]

        if self.metric == "sharpe":
            # Strategy returns: go long when predicted positive, short when negative
            strategy_returns = np.sign(pred) * actual
            if strategy_returns.std() == 0:
                return 0.0
            sharpe = float(strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
            return sharpe

        elif self.metric == "auc":
            from sklearn.metrics import roc_auc_score

            y_binary = (actual > 0).astype(int)
            # Use predicted return magnitude as score
            if len(np.unique(y_binary)) < 2:
                return 0.5
            return float(roc_auc_score(y_binary, pred))

        return 0.0

    @staticmethod
    def _extract_target(
        X: pd.DataFrame,
        raw_data: pd.DataFrame,
        target_col: str,
    ) -> np.ndarray:
        """Extract return target aligned with feature matrix."""
        common_idx = X.index.intersection(raw_data.index)
        if target_col in raw_data.columns:
            return raw_data.loc[common_idx, target_col].values.astype(np.float64)
        elif "close" in raw_data.columns:
            return raw_data.loc[common_idx, "close"].pct_change().fillna(0).values.astype(
                np.float64
            )
        raise ModelTrainingError(
            f"Cannot find '{target_col}' or 'close' in data",
            details={"available_columns": list(raw_data.columns)},
        )

    @staticmethod
    def _detect_model_type(model_class: type) -> str:
        """Detect model type from class name for search space selection."""
        name = model_class.__name__.lower()
        if "lightgbm" in name or "lgbm" in name:
            return "lightgbm"
        elif "xgboost" in name or "xgb" in name:
            return "xgboost"
        elif "lstm" in name:
            return "lstm"
        elif "tft" in name or "temporal" in name or "transformer" in name:
            return "tft"
        return "lightgbm"  # default fallback
