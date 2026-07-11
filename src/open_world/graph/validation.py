"""Structural validation of the assembled district graph.

Checks the spatial constraints that must hold regardless of which edge
strategy produced the graph: every province and state must be a single
connected region, every district must have at least one neighbour, and
node degree should stay low (a proxy for "the map should look planar,
not a hairball").
"""

from dataclasses import dataclass, field

import networkx as nx
import polars as pl
from loguru import logger

from open_world.data.schema import DISTRICT_ID, PROVINCE, STATE

DEFAULT_MAX_DEGREE = 5


@dataclass
class ValidationReport:
    """Result of validating a district graph against the spatial constraints.

    Attributes:
        disconnected_provinces: Province names whose induced subgraph is not
            a single connected component.
        disconnected_states: State names whose induced subgraph is not a
            single connected component.
        orphan_districts: DistrictIds with zero neighbours.
        high_degree_districts: DistrictId to degree, for nodes exceeding the
            configured max-degree threshold.
    """

    disconnected_provinces: list[str] = field(default_factory=list)
    disconnected_states: list[str] = field(default_factory=list)
    orphan_districts: list[str] = field(default_factory=list)
    high_degree_districts: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Whether the graph satisfies every hard constraint.

        High-degree districts are reported but do not affect validity, since
        the degree cap is a soft design target rather than a hard rule.

        Returns:
            True if there are no disconnected regions or orphan districts.
        """
        return not (
            self.disconnected_provinces or self.disconnected_states or self.orphan_districts
        )

    def to_summary_dict(self) -> dict[str, bool | int | list[str]]:
        """Condense the report into a small JSON-serializable summary.

        Returns:
            A dict with ``is_valid`` plus the disconnected/orphan lists and
            a ``high_degree_count`` (the full degree mapping is omitted to
            keep the payload small).
        """
        return {
            "is_valid": self.is_valid,
            "disconnected_provinces": self.disconnected_provinces,
            "disconnected_states": self.disconnected_states,
            "orphan_districts": self.orphan_districts,
            "high_degree_count": len(self.high_degree_districts),
        }


def validate_graph(
    graph: nx.Graph, frame: pl.DataFrame, max_degree: int = DEFAULT_MAX_DEGREE
) -> ValidationReport:
    """Validate a district graph against the spatial constraints.

    Args:
        graph: Graph produced by :func:`open_world.graph.builder.build_graph`.
        frame: District table used to build the graph, providing the
            state/province grouping to check.
        max_degree: Degree above which a district is flagged as high-degree.

    Returns:
        A populated :class:`ValidationReport`.
    """
    report = ValidationReport(
        disconnected_provinces=_disconnected_groups(graph, frame, PROVINCE),
        disconnected_states=_disconnected_groups(graph, frame, STATE),
        orphan_districts=[node for node, degree in graph.degree if degree == 0],
        high_degree_districts={
            node: degree for node, degree in graph.degree if degree > max_degree
        },
    )

    if report.is_valid:
        logger.info("graph is valid: all provinces and states are connected")
    else:
        logger.warning(
            "graph is invalid: {} disconnected provinces, {} disconnected states, "
            "{} orphan districts",
            len(report.disconnected_provinces),
            len(report.disconnected_states),
            len(report.orphan_districts),
        )
    if report.high_degree_districts:
        logger.warning(
            "{} districts exceed the max-degree target of {}",
            len(report.high_degree_districts),
            max_degree,
        )
    return report


def _disconnected_groups(graph: nx.Graph, frame: pl.DataFrame, group_column: str) -> list[str]:
    """Find hierarchy groups whose induced subgraph is not fully connected.

    Args:
        graph: Graph to check.
        frame: District table providing the ``districtId`` -> group mapping.
        group_column: Column to group by, e.g. ``state`` or ``province``.

    Returns:
        Names of groups whose members do not form a single connected
        component within ``graph``.
    """
    disconnected: list[str] = []
    for name, group in frame.group_by(group_column):
        node_ids = group[DISTRICT_ID].to_list()
        subgraph = graph.subgraph(node_ids)
        if subgraph.number_of_nodes() > 0 and not nx.is_connected(subgraph):
            disconnected.append(name[0])
    return disconnected
