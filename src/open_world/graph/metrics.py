"""Quantitative diagnostics for the district graph.

These exist so that iterating on the graph-generation algorithm is a
question of comparing numbers across runs, not eyeballing a picture and
guessing whether a change helped. Two things are specifically operationalized:

* How much of the graph crosses province/state lines at all
  (``cross_province_fraction``, ``cross_state_fraction``), and whether that
  looks like real border traffic or accidental noise.
* Whether high-elevation districts form the kind of cross-border "mountain
  range" clusters the spatial constraints call for
  (``high_elevation_cross_state_cluster_count``).
"""

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx
import polars as pl
from loguru import logger

from open_world.data.schema import ELEVATION, PROVINCE, STATE
from open_world.graph.edge_types import EDGE_KIND

DEFAULT_HIGH_ELEVATION_PERCENTILE = 0.9


@dataclass
class GraphMetrics:
    """Quantitative summary of a district graph.

    Attributes:
        node_count: Total number of districts.
        edge_count: Total number of edges, across all strategies.
        avg_degree: Mean node degree.
        degree_percentiles: p50/p90/p99/max of the degree distribution.
        edge_kind_counts: Number of edges contributed by each strategy.
        elevation_gap_by_kind: Per strategy, the mean/median/max absolute
            elevation difference across its edges.
        cross_province_fraction: Fraction of all edges linking districts in
            different provinces.
        cross_state_fraction: Fraction of all edges linking districts in
            different states.
        high_elevation_threshold: Elevation value at
            ``high_elevation_percentile``, used to select "mountain" nodes.
        high_elevation_cluster_count: Number of connected components formed
            by mountain nodes alone.
        high_elevation_cross_state_cluster_count: Of those clusters, how
            many contain districts from more than one state -- evidence
            that mountain ranges are actually crossing borders.
    """

    node_count: int
    edge_count: int
    avg_degree: float
    degree_percentiles: dict[str, float] = field(default_factory=dict)
    edge_kind_counts: dict[str, int] = field(default_factory=dict)
    elevation_gap_by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    cross_province_fraction: float = 0.0
    cross_state_fraction: float = 0.0
    high_elevation_threshold: float = 0.0
    high_elevation_cluster_count: int = 0
    high_elevation_cross_state_cluster_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain, JSON-serializable dict.

        Returns:
            The metrics as nested built-in types.
        """
        return asdict(self)


def compute_metrics(
    graph: nx.Graph,
    frame: pl.DataFrame,
    high_elevation_percentile: float = DEFAULT_HIGH_ELEVATION_PERCENTILE,
) -> GraphMetrics:
    """Compute diagnostic metrics for a district graph.

    Args:
        graph: Graph produced by :func:`open_world.graph.builder.build_graph`.
        frame: District table used to build the graph.
        high_elevation_percentile: Elevation quantile (0-1) above which a
            district counts as "high elevation" for the mountain-range
            cluster metrics.

    Returns:
        A populated :class:`GraphMetrics`.
    """
    degrees = [degree for _, degree in graph.degree]
    edge_kind_counts, elevation_gap_by_kind = _edge_stats(graph)
    cross_province, cross_state = _cross_boundary_fractions(graph)
    high_threshold, cluster_count, cross_state_cluster_count = _high_elevation_clusters(
        graph, frame, high_elevation_percentile
    )

    metrics = GraphMetrics(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        avg_degree=statistics.mean(degrees) if degrees else 0.0,
        degree_percentiles=_percentiles(degrees),
        edge_kind_counts=edge_kind_counts,
        elevation_gap_by_kind=elevation_gap_by_kind,
        cross_province_fraction=cross_province,
        cross_state_fraction=cross_state,
        high_elevation_threshold=high_threshold,
        high_elevation_cluster_count=cluster_count,
        high_elevation_cross_state_cluster_count=cross_state_cluster_count,
    )
    logger.info(
        "metrics: avg_degree={:.2f}, cross_state_fraction={:.2f}, "
        "high_elevation_cross_state_clusters={}/{}",
        metrics.avg_degree,
        metrics.cross_state_fraction,
        metrics.high_elevation_cross_state_cluster_count,
        metrics.high_elevation_cluster_count,
    )
    return metrics


def _percentiles(values: list[int]) -> dict[str, float]:
    """Compute p50/p90/p99/max of a list of values.

    Args:
        values: Values to summarize.

    Returns:
        A dict with ``p50``, ``p90``, ``p99`` and ``max`` keys. All zero if
        ``values`` is empty.
    """
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    series = pl.Series(values)
    return {
        "p50": float(series.quantile(0.5, interpolation="linear")),
        "p90": float(series.quantile(0.9, interpolation="linear")),
        "p99": float(series.quantile(0.99, interpolation="linear")),
        "max": float(series.max()),
    }


def _edge_stats(graph: nx.Graph) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
    """Count edges and summarize elevation gaps, grouped by strategy kind.

    Args:
        graph: Graph whose edges carry a ``kind`` attribute and whose nodes
            carry an ``elevation`` attribute.

    Returns:
        A (kind -> count) dict and a (kind -> {mean, median, max}) dict of
        absolute elevation differences.
    """
    counts: dict[str, int] = {}
    gaps_by_kind: dict[str, list[int]] = {}
    for source, target, attrs in graph.edges(data=True):
        kind = attrs[EDGE_KIND]
        counts[kind] = counts.get(kind, 0) + 1
        gap = abs(graph.nodes[source][ELEVATION] - graph.nodes[target][ELEVATION])
        gaps_by_kind.setdefault(kind, []).append(gap)

    gap_stats = {
        kind: {
            "mean": statistics.mean(gaps),
            "median": statistics.median(gaps),
            "max": max(gaps),
        }
        for kind, gaps in gaps_by_kind.items()
    }
    return counts, gap_stats


def _cross_boundary_fractions(graph: nx.Graph) -> tuple[float, float]:
    """Measure how often edges cross province/state lines, over the whole graph.

    Args:
        graph: Graph whose nodes carry ``state``/``province`` attributes.

    Returns:
        (cross_province_fraction, cross_state_fraction) over every edge.
        Both are 0.0 if the graph has no edges.
    """
    edges = list(graph.edges())
    if not edges:
        return 0.0, 0.0

    cross_province = sum(
        graph.nodes[source][PROVINCE] != graph.nodes[target][PROVINCE] for source, target in edges
    )
    cross_state = sum(
        graph.nodes[source][STATE] != graph.nodes[target][STATE] for source, target in edges
    )
    return cross_province / len(edges), cross_state / len(edges)


def _high_elevation_clusters(
    graph: nx.Graph, frame: pl.DataFrame, percentile: float
) -> tuple[float, int, int]:
    """Find connected clusters of high-elevation districts.

    Args:
        graph: Graph whose nodes carry ``state``/``elevation`` attributes.
        frame: District table, used to compute the elevation threshold.
        percentile: Elevation quantile (0-1) defining "high elevation".

    Returns:
        (threshold, cluster_count, cross_state_cluster_count).
    """
    threshold = float(frame[ELEVATION].quantile(percentile, interpolation="linear"))
    high_nodes = [node for node, attrs in graph.nodes(data=True) if attrs[ELEVATION] >= threshold]
    subgraph = graph.subgraph(high_nodes)

    cross_state_clusters = 0
    cluster_count = 0
    for component in nx.connected_components(subgraph):
        cluster_count += 1
        states = {graph.nodes[node][STATE] for node in component}
        if len(states) > 1:
            cross_state_clusters += 1

    return threshold, cluster_count, cross_state_clusters
