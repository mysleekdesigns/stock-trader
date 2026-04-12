"""Model monitoring: drift detection, performance decay, and regime changes.

Provides :class:`DriftMonitor` which runs statistical tests to detect when
production data diverges from the training distribution, when model accuracy
degrades, or when macro-regime shifts invalidate current model assumptions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy.stats import ks_2samp

logger = structlog.get_logger(__name__)


class DriftMonitor:
    """Monitors data and model drift against a reference dataset.

    Parameters
    ----------
    reference_data
        Historical feature DataFrame that represents the "expected" distribution
        (typically the training set or a recent stable window).
    config
        Optional configuration overrides.  Supported keys:

        - ``p_value_threshold`` (float): KS-test significance level.  Default 0.05.
        - ``min_sharpe`` (float): Minimum acceptable Sharpe ratio.  Default 0.5.
        - ``vix_threshold`` (float): VIX level above which a regime change is
          flagged.  Default 25.0.
    """

    def __init__(self, reference_data: pd.DataFrame, config: dict[str, Any] | None = None) -> None:
        self._reference = reference_data.copy()
        self._config = config or {}
        self._p_value_threshold: float = self._config.get("p_value_threshold", 0.05)
        self._min_sharpe: float = self._config.get("min_sharpe", 0.5)
        self._vix_threshold: float = self._config.get("vix_threshold", 25.0)

        # Cache reference predictions for later comparison (set via generate_report).
        self._reference_predictions: np.ndarray | None = None
        self._last_report: dict[str, Any] | None = None

        logger.info(
            "drift_monitor.init",
            reference_rows=len(self._reference),
            reference_features=list(self._reference.columns),
            p_value_threshold=self._p_value_threshold,
        )

    # ── feature drift ──────────────────────────────────────────────────

    def check_feature_drift(
        self,
        current_data: pd.DataFrame,
    ) -> dict[str, dict[str, float | bool]]:
        """Run a two-sample Kolmogorov-Smirnov test per feature.

        Parameters
        ----------
        current_data
            Recent production feature DataFrame with the same columns as
            ``reference_data``.

        Returns
        -------
        dict
            ``{feature_name: {"drifted": bool, "p_value": float, "statistic": float}}``
        """
        results: dict[str, dict[str, float | bool]] = {}
        common_features = [c for c in self._reference.columns if c in current_data.columns]

        if not common_features:
            logger.warning("drift_monitor.no_common_features")
            return results

        for feature in common_features:
            ref_values = self._reference[feature].dropna().values
            cur_values = current_data[feature].dropna().values

            if len(ref_values) == 0 or len(cur_values) == 0:
                logger.warning("drift_monitor.empty_feature", feature=feature)
                continue

            statistic, p_value = ks_2samp(ref_values, cur_values)
            drifted = bool(p_value < self._p_value_threshold)

            results[feature] = {
                "drifted": drifted,
                "p_value": float(p_value),
                "statistic": float(statistic),
            }

            if drifted:
                logger.warning(
                    "drift_monitor.feature_drift_detected",
                    feature=feature,
                    p_value=round(p_value, 6),
                    statistic=round(statistic, 4),
                )

        drifted_count = sum(1 for v in results.values() if v["drifted"])
        logger.info(
            "drift_monitor.feature_drift_check_complete",
            total_features=len(results),
            drifted_features=drifted_count,
        )
        return results

    # ── prediction drift ───────────────────────────────────────────────

    def check_prediction_drift(
        self,
        predictions: np.ndarray,
        reference_predictions: np.ndarray,
    ) -> dict[str, Any]:
        """Compare current model predictions against reference predictions.

        Uses the KS test to determine if the prediction distribution has shifted.

        Returns
        -------
        dict
            ``{"drifted": bool, "p_value": float, "statistic": float,
              "mean_shift": float, "std_shift": float}``
        """
        predictions = np.asarray(predictions).ravel()
        reference_predictions = np.asarray(reference_predictions).ravel()

        if len(predictions) == 0 or len(reference_predictions) == 0:
            logger.warning("drift_monitor.empty_predictions")
            return {
                "drifted": False,
                "p_value": 1.0,
                "statistic": 0.0,
                "mean_shift": 0.0,
                "std_shift": 0.0,
            }

        statistic, p_value = ks_2samp(predictions, reference_predictions)
        drifted = bool(p_value < self._p_value_threshold)

        mean_shift = float(np.mean(predictions) - np.mean(reference_predictions))
        ref_std = float(np.std(reference_predictions))
        cur_std = float(np.std(predictions))
        std_shift = cur_std - ref_std

        result = {
            "drifted": drifted,
            "p_value": float(p_value),
            "statistic": float(statistic),
            "mean_shift": mean_shift,
            "std_shift": std_shift,
        }

        if drifted:
            logger.warning("drift_monitor.prediction_drift_detected", **result)
        else:
            logger.info("drift_monitor.prediction_drift_ok", p_value=round(p_value, 6))

        return result

    # ── performance decay ──────────────────────────────────────────────

    def check_performance_decay(
        self,
        current_sharpe: float,
        min_sharpe: float | None = None,
    ) -> bool:
        """Return ``True`` if the current Sharpe ratio is below the minimum.

        Parameters
        ----------
        current_sharpe
            The live or recent-window Sharpe ratio.
        min_sharpe
            Override the configured minimum.  Defaults to ``self._min_sharpe``.
        """
        threshold = min_sharpe if min_sharpe is not None else self._min_sharpe
        decayed = current_sharpe < threshold

        if decayed:
            logger.warning(
                "drift_monitor.performance_decay",
                current_sharpe=round(current_sharpe, 4),
                min_sharpe=threshold,
            )
        else:
            logger.info(
                "drift_monitor.performance_ok",
                current_sharpe=round(current_sharpe, 4),
                min_sharpe=threshold,
            )

        return decayed

    # ── regime change ──────────────────────────────────────────────────

    def check_regime_change(
        self,
        vix: float,
        threshold: float | None = None,
    ) -> bool:
        """Return ``True`` if the VIX exceeds the configured threshold.

        Parameters
        ----------
        vix
            Current CBOE Volatility Index level.
        threshold
            Override the configured VIX threshold.
        """
        limit = threshold if threshold is not None else self._vix_threshold
        regime_change = vix > limit

        if regime_change:
            logger.warning(
                "drift_monitor.regime_change",
                vix=round(vix, 2),
                threshold=limit,
            )
        else:
            logger.info(
                "drift_monitor.regime_stable",
                vix=round(vix, 2),
                threshold=limit,
            )

        return regime_change

    # ── aggregate report ───────────────────────────────────────────────

    def generate_report(
        self,
        current_data: pd.DataFrame | None = None,
        predictions: np.ndarray | None = None,
        reference_predictions: np.ndarray | None = None,
        current_sharpe: float | None = None,
        vix: float | None = None,
    ) -> dict[str, Any]:
        """Run all available checks and return a consolidated monitoring report.

        Parameters that are ``None`` are skipped in the report.

        Returns
        -------
        dict
            Keys: ``timestamp``, ``feature_drift``, ``prediction_drift``,
            ``performance_decay``, ``regime_change``, ``alert``.
        """
        report: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        alert = False

        # Feature drift
        if current_data is not None:
            feature_drift = self.check_feature_drift(current_data)
            report["feature_drift"] = feature_drift
            if any(v["drifted"] for v in feature_drift.values()):
                alert = True

        # Prediction drift
        if predictions is not None and reference_predictions is not None:
            pred_drift = self.check_prediction_drift(predictions, reference_predictions)
            report["prediction_drift"] = pred_drift
            if pred_drift["drifted"]:
                alert = True

        # Performance decay
        if current_sharpe is not None:
            decayed = self.check_performance_decay(current_sharpe)
            report["performance_decay"] = {
                "decayed": decayed,
                "current_sharpe": current_sharpe,
                "min_sharpe": self._min_sharpe,
            }
            if decayed:
                alert = True

        # Regime change
        if vix is not None:
            regime = self.check_regime_change(vix)
            report["regime_change"] = {
                "changed": regime,
                "vix": vix,
                "threshold": self._vix_threshold,
            }
            if regime:
                alert = True

        report["alert"] = alert
        self._last_report = report

        logger.info("drift_monitor.report_generated", alert=alert)
        return report
