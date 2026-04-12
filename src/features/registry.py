"""Feature registry with dependency DAG and topological execution ordering.

All feature definitions are registered here. The registry resolves computation
order via topological sort so that each feature's dependencies are guaranteed
to be available when it runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeatureDefinition:
    """Metadata and compute function for a single feature."""

    name: str
    compute_fn: Callable[[pd.DataFrame], pd.Series | pd.DataFrame]
    dependencies: list[str] = field(default_factory=list)
    description: str = ""
    group: str = ""


class FeatureRegistry:
    """Central registry of feature definitions with dependency-aware execution.

    Features are registered with optional dependencies on other features.
    The registry uses Kahn's algorithm to resolve a valid execution order
    so that every feature's inputs are computed before it runs.
    """

    def __init__(self) -> None:
        self._features: dict[str, FeatureDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        compute_fn: Callable[[pd.DataFrame], pd.Series | pd.DataFrame],
        dependencies: list[str] | None = None,
        description: str = "",
        group: str = "",
    ) -> None:
        """Register a feature definition."""
        if name in self._features:
            logger.warning("feature_overwritten", feature=name)
        defn = FeatureDefinition(
            name=name,
            compute_fn=compute_fn,
            dependencies=list(dependencies) if dependencies else [],
            description=description,
            group=group,
        )
        self._features[name] = defn
        logger.debug("feature_registered", feature=name, group=group)

    def feature(
        self,
        name: str,
        dependencies: list[str] | None = None,
        group: str = "",
        description: str = "",
    ) -> Callable:
        """Decorator form of :meth:`register`."""

        def decorator(fn: Callable) -> Callable:
            self.register(
                name=name,
                compute_fn=fn,
                dependencies=dependencies,
                description=description or fn.__doc__ or "",
                group=group,
            )
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Dependency resolution (Kahn's algorithm)
    # ------------------------------------------------------------------

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """Return adjacency list of feature -> its dependencies."""
        return {name: list(defn.dependencies) for name, defn in self._features.items()}

    def validate(self) -> None:
        """Check for missing dependencies and circular references.

        Raises ``ValueError`` on any structural problem.
        """
        graph = self.get_dependency_graph()
        all_names = set(graph.keys())

        # Missing dependencies
        for name, deps in graph.items():
            missing = [d for d in deps if d not in all_names]
            if missing:
                raise ValueError(
                    f"Feature '{name}' depends on unregistered features: {missing}"
                )

        # Circular dependency detection via topological sort attempt
        try:
            self._topological_sort(graph)
        except ValueError:
            raise  # re-raise the cycle error from _topological_sort

        logger.info("registry_validated", feature_count=len(self._features))

    def resolve_order(self) -> list[str]:
        """Return feature names in a valid topological execution order."""
        graph = self.get_dependency_graph()
        return self._topological_sort(graph)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute every registered feature in dependency order."""
        order = self.resolve_order()
        return self._compute_ordered(data, order)

    def compute_subset(
        self,
        data: pd.DataFrame,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Compute only *feature_names* and their transitive dependencies."""
        needed = self._collect_dependencies(feature_names)
        full_order = self.resolve_order()
        subset_order = [f for f in full_order if f in needed]
        return self._compute_ordered(data, subset_order)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_ordered(
        self,
        data: pd.DataFrame,
        order: list[str],
    ) -> pd.DataFrame:
        """Run compute functions in the given order, attaching results to *data*."""
        if data.empty:
            logger.warning("compute_skipped_empty_dataframe")
            return data

        df = data.copy()
        for name in order:
            defn = self._features[name]
            try:
                result = defn.compute_fn(df)
                if isinstance(result, pd.Series):
                    df[name] = result
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        df[col] = result[col]
                else:
                    df[name] = result
                logger.debug("feature_computed", feature=name)
            except Exception:
                logger.exception("feature_computation_failed", feature=name)
                raise
        return df

    def _collect_dependencies(self, names: list[str]) -> set[str]:
        """Gather *names* and all their transitive dependencies."""
        visited: set[str] = set()
        stack = list(names)
        while stack:
            name = stack.pop()
            if name in visited:
                continue
            if name not in self._features:
                raise ValueError(f"Unknown feature requested: '{name}'")
            visited.add(name)
            stack.extend(self._features[name].dependencies)
        return visited

    @staticmethod
    def _topological_sort(graph: dict[str, list[str]]) -> list[str]:
        """Kahn's algorithm — returns topological order or raises on cycle."""
        in_degree: dict[str, int] = {node: 0 for node in graph}
        for deps in graph.values():
            for dep in deps:
                in_degree[dep] = in_degree.get(dep, 0) + 1

        # Seed with zero-in-degree nodes (sorted for determinism)
        queue: list[str] = sorted(
            [node for node, deg in in_degree.items() if deg == 0]
        )
        order: list[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for deps in graph.values():
                pass  # adjacency is inverted below

        # Re-do properly: build *reverse* adjacency (dep -> dependents)
        order.clear()
        in_degree = {node: 0 for node in graph}
        reverse_adj: dict[str, list[str]] = {node: [] for node in graph}
        for node, deps in graph.items():
            in_degree[node] = len(deps)
            for dep in deps:
                reverse_adj.setdefault(dep, []).append(node)

        queue = sorted([n for n, d in in_degree.items() if d == 0])
        while queue:
            node = queue.pop(0)
            order.append(node)
            for dependent in sorted(reverse_adj.get(node, [])):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(graph):
            remaining = set(graph.keys()) - set(order)
            raise ValueError(f"Circular dependency detected among: {remaining}")

        return order

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        return list(self._features.keys())

    def __len__(self) -> int:
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        return name in self._features

    def __getitem__(self, name: str) -> FeatureDefinition:
        return self._features[name]
