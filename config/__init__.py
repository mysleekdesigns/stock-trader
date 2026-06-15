"""Application configuration package.

This package is the PRD-canonical home for configuration:

* :mod:`config.settings` – Pydantic ``Settings`` loaded from the environment.
* :mod:`config.logging_config` – structlog setup.
* ``assets.yaml`` / ``strategies.yaml`` / ``risk_limits.yaml`` – the asset
  universe, strategy parameters, and risk thresholds consumed by the backtest
  runner and training pipeline (``--config-dir config`` resolves here).

The Pydantic settings model itself lives in :mod:`src.core.config` so that the
rest of the codebase can import it without a dependency on this package; it is
re-exported here for convenience and to match the PRD module layout.
"""

from src.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
