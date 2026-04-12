"""Feature engineering module for the trading system.

Provides a registry-based feature computation framework with dependency resolution,
technical analysis indicators, price-derived features, and a normalization pipeline.
"""

from src.features.registry import FeatureRegistry
from src.features.pipeline import FeaturePipeline

__all__ = ["FeatureRegistry", "FeaturePipeline"]
