"""Pydantic settings (PRD-canonical re-export).

The single source of truth for the ``Settings`` model is :mod:`src.core.config`.
This module re-exports it so code and tooling can import from ``config.settings``
exactly as the PRD specifies, without duplicating the model definition.
"""

from src.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
