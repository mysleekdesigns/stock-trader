"""A/B testing framework for champion vs. challenger model evaluation.

Provides :class:`ABTestFramework` which collects paired predictions from two
models, computes risk-adjusted performance metrics, and uses a Welch t-test to
decide whether the challenger model is a statistically significant improvement.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import structlog
from scipy.stats import ttest_ind

logger = structlog.get_logger(__name__)


class ABTestFramework:
    """Run a statistical A/B test between a champion and a challenger model.

    Parameters
    ----------
    challenger_model
        The candidate model under evaluation.
    champion_model
        The current production model (baseline).
    min_samples
        Minimum number of recorded predictions per model before evaluation
        is considered valid.
    significance_level
        Two-tailed significance level for the t-test (default 0.05).
    """

    CHAMPION = "champion"
    CHALLENGER = "challenger"

    def __init__(
        self,
        challenger_model: Any,
        champion_model: Any,
        min_samples: int = 100,
        significance_level: float = 0.05,
    ) -> None:
        self._challenger_model = challenger_model
        self._champion_model = champion_model
        self._min_samples = min_samples
        self._significance_level = significance_level

        # Keyed by model name -> list of (prediction, actual) tuples.
        self._records: dict[str, list[tuple[float, float]]] = defaultdict(list)

        logger.info(
            "ab_test.init",
            challenger=str(challenger_model),
            champion=str(champion_model),
            min_samples=min_samples,
            significance_level=significance_level,
        )

    # ── recording ──────────────────────────────────────────────────────

    def record_prediction(self, model_name: str, prediction: float, actual: float) -> None:
        """Store a single prediction/actual pair for the given model.

        Parameters
        ----------
        model_name
            Either ``"champion"`` or ``"challenger"``.
        prediction
            Model's predicted value (e.g. expected return).
        actual
            Realised value.
        """
        if model_name not in (self.CHAMPION, self.CHALLENGER):
            raise ValueError(
                f"model_name must be '{self.CHAMPION}' or '{self.CHALLENGER}', "
                f"got '{model_name}'"
            )

        self._records[model_name].append((prediction, actual))

        logger.debug(
            "ab_test.record",
            model=model_name,
            prediction=round(prediction, 6),
            actual=round(actual, 6),
            total=len(self._records[model_name]),
        )

    # ── evaluation ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_returns(records: list[tuple[float, float]]) -> np.ndarray:
        """Derive per-prediction returns: ``actual * sign(prediction)``.

        This models a simple directional strategy: if the model predicts a
        positive move we go long, otherwise short.  The return is the actual
        move multiplied by the predicted direction.
        """
        returns = []
        for prediction, actual in records:
            direction = 1.0 if prediction >= 0 else -1.0
            returns.append(direction * actual)
        return np.array(returns)

    @staticmethod
    def _sharpe_ratio(returns: np.ndarray, annualization: float = 252.0) -> float:
        """Annualised Sharpe ratio (assume zero risk-free rate)."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(annualization))

    def evaluate(self) -> dict[str, Any]:
        """Evaluate champion vs. challenger using recorded predictions.

        Returns
        -------
        dict
            Keys: ``champion_sharpe``, ``challenger_sharpe``, ``p_value``,
            ``significant``, ``recommendation``, ``champion_n``,
            ``challenger_n``, ``enough_samples``.
        """
        champ_records = self._records.get(self.CHAMPION, [])
        chall_records = self._records.get(self.CHALLENGER, [])

        enough_samples = (
            len(champ_records) >= self._min_samples
            and len(chall_records) >= self._min_samples
        )

        champ_returns = self._compute_returns(champ_records) if champ_records else np.array([])
        chall_returns = self._compute_returns(chall_records) if chall_records else np.array([])

        champ_sharpe = self._sharpe_ratio(champ_returns) if len(champ_returns) >= 2 else 0.0
        chall_sharpe = self._sharpe_ratio(chall_returns) if len(chall_returns) >= 2 else 0.0

        # Statistical test on return distributions (Welch's t-test).
        if len(champ_returns) >= 2 and len(chall_returns) >= 2:
            t_stat, p_value = ttest_ind(chall_returns, champ_returns, equal_var=False)
            p_value = float(p_value)
        else:
            p_value = 1.0

        significant = p_value < self._significance_level and enough_samples

        # Recommendation logic
        if not enough_samples:
            recommendation = "collect_more_data"
        elif significant and chall_sharpe > champ_sharpe:
            recommendation = "promote_challenger"
        elif significant and chall_sharpe < champ_sharpe:
            recommendation = "keep_champion"
        else:
            recommendation = "no_significant_difference"

        result = {
            "champion_sharpe": round(champ_sharpe, 4),
            "challenger_sharpe": round(chall_sharpe, 4),
            "p_value": round(p_value, 6),
            "significant": significant,
            "recommendation": recommendation,
            "champion_n": len(champ_records),
            "challenger_n": len(chall_records),
            "enough_samples": enough_samples,
        }

        logger.info("ab_test.evaluate", **result)
        return result

    # ── promotion ──────────────────────────────────────────────────────

    def promote_challenger(self) -> bool:
        """Promote the challenger to champion if significantly better.

        Returns
        -------
        bool
            ``True`` if the challenger was promoted, ``False`` otherwise.
        """
        evaluation = self.evaluate()

        if evaluation["recommendation"] == "promote_challenger":
            logger.info(
                "ab_test.promote_challenger",
                challenger_sharpe=evaluation["challenger_sharpe"],
                champion_sharpe=evaluation["champion_sharpe"],
                p_value=evaluation["p_value"],
            )
            # Swap models
            self._champion_model, self._challenger_model = (
                self._challenger_model,
                self._champion_model,
            )
            # Reset records for a fresh evaluation cycle
            self._records.clear()
            return True

        logger.info(
            "ab_test.promotion_declined",
            recommendation=evaluation["recommendation"],
            p_value=evaluation["p_value"],
        )
        return False

    # ── accessors ──────────────────────────────────────────────────────

    @property
    def champion_model(self) -> Any:
        """Return the current champion model."""
        return self._champion_model

    @property
    def challenger_model(self) -> Any:
        """Return the current challenger model."""
        return self._challenger_model
