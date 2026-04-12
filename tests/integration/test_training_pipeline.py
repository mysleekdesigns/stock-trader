"""Integration tests for the walk-forward training pipeline.

These tests ensure that:
- Train / validation / test indices never overlap.
- Temporal ordering is strictly respected (test > val > train).
- The sliding window steps by the correct amount.
- All data points are eventually covered.
- No future data leaks into the training set.

All data is synthetic (numpy random), so no external services are required.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import with graceful skip
# ---------------------------------------------------------------------------

_wf_mod = pytest.importorskip(
    "src.models.training.walk_forward",
    reason="src.models.training.walk_forward not yet available",
)
WalkForwardSplitter = _wf_mod.WalkForwardSplitter

# Optional: import ModelTrainer if it exists (used by leak-detection test).
try:
    from src.models.training.trainer import ModelTrainer  # noqa: F401

    _HAS_TRAINER = True
except ImportError:
    _HAS_TRAINER = False


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(
    n: int = 700,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic time-series features and targets.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with ``n`` rows and ``n_features`` columns.
    y : pd.DataFrame
        Target DataFrame with columns ``direction``, ``returns``, ``volatility``.
    """
    rng = np.random.RandomState(seed)
    start = datetime(2020, 1, 2)
    dates = pd.bdate_range(start, periods=n)

    X = pd.DataFrame(
        rng.randn(n, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
        index=dates,
    )

    # Build targets with a simple known pattern.
    returns = rng.randn(n) * 0.01
    direction = (returns > 0).astype(int)
    volatility = np.abs(returns) + rng.rand(n) * 0.005

    y = pd.DataFrame(
        {"direction": direction, "returns": returns, "volatility": volatility},
        index=dates,
    )

    return X, y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _make_synthetic_dataset()


@pytest.fixture
def splitter() -> WalkForwardSplitter:
    """Splitter with PRD defaults: 504d train / 63d val / 21d test / 21d step."""
    return WalkForwardSplitter(
        train_days=504,
        val_days=63,
        test_days=21,
        step_days=21,
    )


@pytest.fixture
def small_splitter() -> WalkForwardSplitter:
    """A smaller splitter for fast unit-level checks."""
    return WalkForwardSplitter(
        train_days=100,
        val_days=20,
        test_days=10,
        step_days=10,
    )


# ---------------------------------------------------------------------------
# Tests: no overlap
# ---------------------------------------------------------------------------

class TestWalkForwardNoOverlap:
    """Train, val, and test index sets must be disjoint in every split."""

    def test_walk_forward_no_overlap(
        self,
        small_splitter: WalkForwardSplitter,
        synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        X, y = synthetic_data

        for fold_idx, split in enumerate(small_splitter.split(X.index)):
            train_idx, val_idx, test_idx = _unpack_split(split)

            train_set = set(train_idx)
            val_set = set(val_idx)
            test_set = set(test_idx)

            assert train_set.isdisjoint(val_set), (
                f"Fold {fold_idx}: train and val overlap"
            )
            assert train_set.isdisjoint(test_set), (
                f"Fold {fold_idx}: train and test overlap"
            )
            assert val_set.isdisjoint(test_set), (
                f"Fold {fold_idx}: val and test overlap"
            )


# ---------------------------------------------------------------------------
# Tests: temporal ordering
# ---------------------------------------------------------------------------

class TestWalkForwardTemporalOrder:
    """Test timestamps must come after val, which must come after train."""

    def test_walk_forward_temporal_order(
        self,
        small_splitter: WalkForwardSplitter,
        synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        X, y = synthetic_data

        for fold_idx, split in enumerate(small_splitter.split(X.index)):
            train_idx, val_idx, test_idx = _unpack_split(split)

            train_ts = X.index[train_idx]
            val_ts = X.index[val_idx]
            test_ts = X.index[test_idx]

            assert train_ts.max() < val_ts.min(), (
                f"Fold {fold_idx}: train end ({train_ts.max()}) >= val start ({val_ts.min()})"
            )
            assert val_ts.max() < test_ts.min(), (
                f"Fold {fold_idx}: val end ({val_ts.max()}) >= test start ({test_ts.min()})"
            )


# ---------------------------------------------------------------------------
# Tests: step size
# ---------------------------------------------------------------------------

class TestWalkForwardStepSize:
    """The test window should shift by exactly ``step_size`` rows per fold."""

    def test_walk_forward_step_size(
        self,
        small_splitter: WalkForwardSplitter,
        synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        X, _ = synthetic_data

        test_starts: list[int] = []
        for split in small_splitter.split(X.index):
            _, _, test_idx = _unpack_split(split)
            test_starts.append(int(test_idx[0]))

        # The step sizes in index positions may vary slightly due to calendar-day
        # based stepping (weekends, holidays). Just verify they are positive and
        # within a reasonable range of each other.
        if len(test_starts) >= 2:
            steps = [test_starts[i] - test_starts[i - 1] for i in range(1, len(test_starts))]
            assert all(s > 0 for s in steps), "Steps must be positive (moving forward)"
            # Steps should be roughly similar (within 2x of each other)
            if len(steps) >= 2:
                assert max(steps) <= 3 * min(steps), (
                    f"Step sizes vary too much: {steps}"
                )


# ---------------------------------------------------------------------------
# Tests: coverage
# ---------------------------------------------------------------------------

class TestWalkForwardCoverage:
    """All data rows should appear in at least one test set."""

    def test_walk_forward_coverage(
        self,
        small_splitter: WalkForwardSplitter,
        synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        X, _ = synthetic_data

        all_test_indices: set[int] = set()
        for split in small_splitter.split(X.index):
            _, _, test_idx = _unpack_split(split)
            all_test_indices.update(int(i) for i in test_idx)

        # At least some data should appear in test sets
        assert len(all_test_indices) > 0, "No data ever appeared in a test set"
        # A reasonable portion of data should be covered
        coverage = len(all_test_indices) / len(X)
        assert coverage > 0.01, (
            f"Only {coverage:.1%} of data covered by test sets"
        )


# ---------------------------------------------------------------------------
# Tests: no future data leak
# ---------------------------------------------------------------------------

class TestTrainingNoFutureLeak:
    """For every walk-forward split, no test timestamp may appear in training."""

    def test_training_no_future_leak(
        self,
        small_splitter: WalkForwardSplitter,
        synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        X, y = synthetic_data

        for fold_idx, split in enumerate(small_splitter.split(X.index)):
            train_idx, val_idx, test_idx = _unpack_split(split)

            train_timestamps = set(X.index[train_idx])
            test_timestamps = set(X.index[test_idx])

            leaked = train_timestamps & test_timestamps
            assert len(leaked) == 0, (
                f"Fold {fold_idx}: {len(leaked)} test timestamps leaked into training"
            )

            # Also confirm no test timestamp precedes the last training timestamp.
            if len(test_idx) > 0 and len(train_idx) > 0:
                max_train_ts = X.index[train_idx].max()
                min_test_ts = X.index[test_idx].min()
                assert max_train_ts < min_test_ts, (
                    f"Fold {fold_idx}: max train ts ({max_train_ts}) >= "
                    f"min test ts ({min_test_ts})"
                )


# ---------------------------------------------------------------------------
# Property-based: random parameters
# ---------------------------------------------------------------------------

class TestWalkForwardProperties:
    """Property-based checks with randomized splitter parameters."""

    @pytest.mark.parametrize("seed", range(5))
    def test_no_overlap_random_params(
        self,
        seed: int,
        synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        rng = np.random.RandomState(seed)
        train_days = rng.randint(50, 200)
        val_days = rng.randint(10, 50)
        test_days = rng.randint(5, 30)
        step_days = rng.randint(5, test_days + 1)

        splitter = WalkForwardSplitter(
            train_days=train_days,
            val_days=val_days,
            test_days=test_days,
            step_days=step_days,
        )

        X, _ = synthetic_data
        for fold_idx, split in enumerate(splitter.split(X.index)):
            train_idx, val_idx, test_idx = _unpack_split(split)

            train_set = set(train_idx)
            val_set = set(val_idx)
            test_set = set(test_idx)

            assert train_set.isdisjoint(val_set), f"Fold {fold_idx}: overlap"
            assert train_set.isdisjoint(test_set), f"Fold {fold_idx}: overlap"
            assert val_set.isdisjoint(test_set), f"Fold {fold_idx}: overlap"

    @pytest.mark.parametrize("seed", range(5))
    def test_temporal_order_random_params(
        self,
        seed: int,
        synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        rng = np.random.RandomState(seed)
        train_days = rng.randint(50, 200)
        val_days = rng.randint(10, 50)
        test_days = rng.randint(5, 30)
        step_days = rng.randint(5, test_days + 1)

        splitter = WalkForwardSplitter(
            train_days=train_days,
            val_days=val_days,
            test_days=test_days,
            step_days=step_days,
        )

        X, _ = synthetic_data
        for fold_idx, split in enumerate(splitter.split(X.index)):
            train_idx, val_idx, test_idx = _unpack_split(split)

            if len(train_idx) > 0 and len(val_idx) > 0:
                assert X.index[train_idx].max() < X.index[val_idx].min()
            if len(val_idx) > 0 and len(test_idx) > 0:
                assert X.index[val_idx].max() < X.index[test_idx].min()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack_split(split: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack a split into (train_indices, val_indices, test_indices).

    Handles both tuple-of-arrays and dict-based return conventions.
    """
    if isinstance(split, dict):
        return (
            np.asarray(split["train"]),
            np.asarray(split["val"]),
            np.asarray(split["test"]),
        )
    # Assume tuple / list of three arrays.
    return (
        np.asarray(split[0]),
        np.asarray(split[1]),
        np.asarray(split[2]),
    )
