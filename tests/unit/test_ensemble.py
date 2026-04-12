"""Unit tests for the Ensemble meta-learner.

Tests verify weighted prediction combination, dynamic weight updates based on
rolling Sharpe ratios, minimum weight floors, normalization, fit/predict round-
trips, and save/load persistence.

The tests use simple mock predictors so they can run even before the real model
implementations exist.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Attempt to import the real EnsemblePredictor; skip gracefully if absent.
# ---------------------------------------------------------------------------

EnsemblePredictor = pytest.importorskip(
    "src.models.ensemble", reason="src.models.ensemble not yet available"
).EnsemblePredictor


# ---------------------------------------------------------------------------
# Mock / stub predictors
# ---------------------------------------------------------------------------

class _MockPredictor:
    """Minimal predictor that satisfies the BasePredictor protocol."""

    def __init__(
        self,
        name: str,
        direction_value: int = 1,
        returns_value: float = 0.01,
        volatility_value: float = 0.02,
        proba_value: float = 0.7,
    ) -> None:
        self.name = name
        self.is_fitted = True
        self.feature_names: list[str] = []
        self._direction = direction_value
        self._returns = returns_value
        self._volatility = volatility_value
        self._proba = proba_value

    # -- protocol methods --------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any, **kwargs: Any) -> None:
        self.is_fitted = True
        self.feature_names = list(X.columns)

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(X)
        return {
            "direction": np.full(n, self._direction, dtype=int),
            "returns": np.full(n, self._returns, dtype=np.float64),
            "volatility": np.full(n, self._volatility, dtype=np.float64),
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.column_stack(
            [np.full(n, 1 - self._proba), np.full(n, self._proba)]
        )

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass

    def get_feature_importance(self) -> dict[str, float]:
        return {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_df() -> pd.DataFrame:
    """Small synthetic feature DataFrame."""
    rng = np.random.RandomState(42)
    n = 50
    return pd.DataFrame(
        {f"feat_{i}": rng.randn(n) for i in range(5)},
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )


@pytest.fixture
def target_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic targets aligned with ``feature_df``."""
    rng = np.random.RandomState(42)
    n = len(feature_df)
    return pd.DataFrame(
        {
            "direction": rng.randint(0, 2, size=n),
            "returns": rng.randn(n) * 0.01,
            "volatility": np.abs(rng.randn(n) * 0.02),
        },
        index=feature_df.index,
    )


@pytest.fixture
def two_predictors() -> list[_MockPredictor]:
    """Two mock predictors with differing return forecasts."""
    return [
        _MockPredictor("model_a", returns_value=0.01, proba_value=0.6),
        _MockPredictor("model_b", returns_value=0.03, proba_value=0.9),
    ]


@pytest.fixture
def ensemble(two_predictors: list[_MockPredictor], feature_df: pd.DataFrame, target_df: pd.DataFrame) -> EnsemblePredictor:
    """EnsemblePredictor wrapping two mock predictors, pre-fitted."""
    ens = EnsemblePredictor(models=two_predictors)
    ens.fit(feature_df, target_df)
    return ens


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnsembleCombinesPredictions:
    """Verify the ensemble output is a weighted combination of sub-models."""

    def test_ensemble_combines_predictions(
        self, ensemble: EnsemblePredictor, feature_df: pd.DataFrame
    ) -> None:
        preds = ensemble.predict(feature_df)

        # Must return a dict with at least direction and returns keys.
        assert isinstance(preds, dict)
        for key in ("direction", "returns"):
            assert key in preds, f"Missing key: {key}"
            assert len(preds[key]) == len(feature_df)

        # Returns should be finite and non-NaN after ensemble combination.
        mean_ret = float(np.mean(preds["returns"]))
        assert np.isfinite(mean_ret), f"Combined return {mean_ret} is not finite"


class TestEnsembleWeightUpdate:
    """Verify weight adjustment via rolling Sharpe ratios."""

    def test_ensemble_weight_update(
        self, ensemble: EnsemblePredictor
    ) -> None:
        # Simulate rolling Sharpe ratios: model_b is much better.
        sharpes = {"model_a": 0.5, "model_b": 2.0}

        ensemble.update_weights(sharpes)

        weights = ensemble.get_model_weights()
        assert weights["model_b"] > weights["model_a"], (
            "Higher-Sharpe model should receive higher weight"
        )


class TestEnsembleMinWeightFloor:
    """No model weight should drop below the configured floor."""

    def test_ensemble_min_weight_floor(
        self, ensemble: EnsemblePredictor
    ) -> None:
        # Give one model a very poor Sharpe.
        sharpes = {"model_a": -1.0, "model_b": 5.0}
        ensemble.update_weights(sharpes)

        weights = ensemble.get_model_weights()
        floor = getattr(ensemble, "min_weight", 0.05)

        for name, w in weights.items():
            assert w >= floor, (
                f"Weight for {name} ({w:.4f}) fell below floor ({floor})"
            )


class TestEnsembleWeightNormalization:
    """Weights must always sum to 1.0."""

    def test_ensemble_weight_normalization(
        self, ensemble: EnsemblePredictor
    ) -> None:
        # Default weights should be normalized.
        weights = ensemble.get_model_weights()
        assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0

        # After an update they should still be normalized.
        ensemble.update_weights({"model_a": 1.0, "model_b": 3.0})
        weights = ensemble.get_model_weights()
        assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0

    def test_normalization_with_many_models(self, feature_df, target_df) -> None:
        """Property check: random Sharpe values still yield sum-to-one."""
        rng = np.random.RandomState(99)
        predictors = [_MockPredictor(f"m_{i}") for i in range(10)]
        ens = EnsemblePredictor(models=predictors)
        ens.fit(feature_df, target_df)

        for _ in range(20):
            sharpes = {f"m_{i}": rng.randn() for i in range(10)}
            ens.update_weights(sharpes)
            weights = ens.get_model_weights()
            assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0


class TestEnsembleFitPredict:
    """Fit the ensemble on synthetic data, verify predict output."""

    def test_ensemble_fit_predict(
        self,
        feature_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        predictors = [
            _MockPredictor("mock_a"),
            _MockPredictor("mock_b"),
        ]
        ens = EnsemblePredictor(models=predictors)
        ens.fit(feature_df, target_df)

        preds = ens.predict(feature_df)

        assert isinstance(preds, dict)
        # Must have at least direction and returns
        for key in ("direction", "returns"):
            assert key in preds
            arr = preds[key]
            assert isinstance(arr, np.ndarray)
            assert arr.shape[0] == len(feature_df)


class TestEnsembleSaveLoad:
    """Save then load: predictions should match."""

    def test_ensemble_save_load(
        self,
        feature_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        predictors = [
            _MockPredictor("mock_a"),
            _MockPredictor("mock_b"),
        ]
        ens = EnsemblePredictor(models=predictors)
        ens.fit(feature_df, target_df)

        preds_before = ens.predict(feature_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "ensemble_model"
            ens.save(save_path)

            # Create a fresh ensemble and load.
            ens2 = EnsemblePredictor(models=predictors)
            ens2.load(save_path)

            preds_after = ens2.predict(feature_df)

        for key in ("returns", "direction"):
            np.testing.assert_allclose(
                preds_before[key],
                preds_after[key],
                atol=1e-6,
                err_msg=f"Predictions diverge for '{key}' after save/load",
            )
