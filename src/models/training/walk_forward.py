"""Walk-forward validation splitter with strict temporal ordering.

Generates non-overlapping (train, validation, test) windows that advance
through time by a configurable step size, guaranteeing no data leakage.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class WalkForwardSplitter:
    """Walk-forward cross-validation splitter for time-series data.

    Parameters
    ----------
    train_days:
        Number of calendar days in each training window.
    val_days:
        Number of calendar days in the validation window immediately
        following training.
    test_days:
        Number of calendar days in the out-of-sample test window
        following validation.
    step_days:
        Number of calendar days to advance between successive splits.
    """

    def __init__(
        self,
        train_days: int = 504,
        val_days: int = 63,
        test_days: int = 21,
        step_days: int = 21,
    ) -> None:
        if train_days <= 0 or val_days <= 0 or test_days <= 0 or step_days <= 0:
            raise ValueError("All day parameters must be positive integers")

        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days
        self.step_days = step_days

        logger.info(
            "walk_forward_splitter_init",
            train_days=train_days,
            val_days=val_days,
            test_days=test_days,
            step_days=step_days,
        )

    def split(
        self, timestamps: pd.DatetimeIndex
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Yield (train_idx, val_idx, test_idx) index arrays.

        Each split satisfies:
            max(train_timestamps) < min(val_timestamps) < max(val_timestamps) < min(test_timestamps)

        Parameters
        ----------
        timestamps:
            Sorted DatetimeIndex of the full dataset.

        Yields
        ------
        tuple of (train_indices, val_indices, test_indices)
            Integer index arrays into the original data.
        """
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)

        if not timestamps.is_monotonic_increasing:
            sort_order = timestamps.argsort()
            timestamps = timestamps[sort_order]
        else:
            sort_order = None

        ts_min = timestamps.min()
        ts_max = timestamps.max()
        total_window = pd.Timedelta(days=self.train_days + self.val_days + self.test_days)

        if (ts_max - ts_min) < total_window:
            logger.warning(
                "walk_forward_insufficient_data",
                data_span_days=(ts_max - ts_min).days,
                required_days=total_window.days,
            )
            return

        split_idx = 0
        train_start = ts_min

        while True:
            train_end = train_start + pd.Timedelta(days=self.train_days)
            val_start = train_end
            val_end = val_start + pd.Timedelta(days=self.val_days)
            test_start = val_end
            test_end = test_start + pd.Timedelta(days=self.test_days)

            if test_end > ts_max + pd.Timedelta(days=1):
                break

            train_mask = (timestamps >= train_start) & (timestamps < train_end)
            val_mask = (timestamps >= val_start) & (timestamps < val_end)
            test_mask = (timestamps >= test_start) & (timestamps < test_end)

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            test_indices = np.where(test_mask)[0]

            # Skip splits with insufficient data in any partition
            if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
                train_start += pd.Timedelta(days=self.step_days)
                continue

            # Map back to original ordering if timestamps were unsorted
            if sort_order is not None:
                train_indices = sort_order[train_indices]
                val_indices = sort_order[val_indices]
                test_indices = sort_order[test_indices]

            # Validate no temporal leakage
            assert timestamps[train_indices].max() <= timestamps[val_indices].min(), (
                "Leakage: train overlaps validation"
            )
            assert timestamps[val_indices].max() <= timestamps[test_indices].min(), (
                "Leakage: validation overlaps test"
            )

            logger.info(
                "walk_forward_split",
                split=split_idx,
                train_start=str(train_start.date()),
                train_end=str(train_end.date()),
                val_start=str(val_start.date()),
                val_end=str(val_end.date()),
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                train_n=len(train_indices),
                val_n=len(val_indices),
                test_n=len(test_indices),
            )

            yield train_indices, val_indices, test_indices

            split_idx += 1
            train_start += pd.Timedelta(days=self.step_days)

    def get_n_splits(self, timestamps: pd.DatetimeIndex) -> int:
        """Return the number of splits that will be generated."""
        return sum(1 for _ in self.split(timestamps))

    def __repr__(self) -> str:
        return (
            f"WalkForwardSplitter(train={self.train_days}d, val={self.val_days}d, "
            f"test={self.test_days}d, step={self.step_days}d)"
        )
