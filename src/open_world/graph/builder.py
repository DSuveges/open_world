"""Assemble the district neighbour graph from the randomized assignment + repair."""

import networkx as nx
import polars as pl
from loguru import logger

from open_world.data.schema import DISTRICT, DISTRICT_ID, ELEVATION, PROVINCE, STATE
from open_world.graph.edge_types import EDGE_KIND
from open_world.graph.neighbours import (
    DEFAULT_CANDIDATE_POOL,
    DEFAULT_MAX_DEGREE,
    DEFAULT_MIN_BOUNDARY_FRACTION,
    DEFAULT_MIN_DEGREE,
    DEFAULT_SEED,
    assign_edges,
    repair_connectivity,
)


def build_graph(  # noqa: PLR0913 -- each strategy knob is meant to be tunable from here
    frame: pl.DataFrame,
    min_degree: int = DEFAULT_MIN_DEGREE,
    max_degree: int = DEFAULT_MAX_DEGREE,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    min_boundary_fraction: float = DEFAULT_MIN_BOUNDARY_FRACTION,
    seed: int = DEFAULT_SEED,
) -> nx.Graph:
    """Build the full district neighbour graph.

    Each district is randomly assigned up to `max_degree` neighbours (see
    :mod:`open_world.graph.neighbours` for the tiered same-province /
    same-state / cross-state search, the boundary quota, and the randomness
    involved), then a repair pass guarantees every province and state
    induces a connected subgraph.

    Args:
        frame: District table as returned by
            :func:`open_world.data.loader.load_districts`.
        min_degree: Passed through to
            :func:`open_world.graph.neighbours.assign_edges`.
        max_degree: Passed through to
            :func:`open_world.graph.neighbours.assign_edges` and
            :func:`open_world.graph.neighbours.repair_connectivity`.
        candidate_pool: Passed through to
            :func:`open_world.graph.neighbours.assign_edges`.
        min_boundary_fraction: Passed through to
            :func:`open_world.graph.neighbours.assign_edges`.
        seed: Passed through to
            :func:`open_world.graph.neighbours.assign_edges`.

    Returns:
        An undirected graph with one node per district. Node attributes are
        ``district``, ``state``, ``province`` and ``elevation``. Each edge
        has a ``kind`` attribute identifying which tier produced it.
    """
    graph = nx.Graph()
    _add_nodes(graph, frame)

    edges = assign_edges(
        frame,
        min_degree=min_degree,
        max_degree=max_degree,
        candidate_pool=candidate_pool,
        min_boundary_fraction=min_boundary_fraction,
        seed=seed,
    )
    for source, target, kind in edges:
        graph.add_edge(source, target, **{EDGE_KIND: kind})

    repair_connectivity(graph, frame, max_degree)

    logger.info(
        "assembled graph: {} nodes, {} edges (avg degree {:.2f})",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        2 * graph.number_of_edges() / graph.number_of_nodes(),
    )
    return graph


def _add_nodes(graph: nx.Graph, frame: pl.DataFrame) -> None:
    """Add one node per district, carrying hierarchy and elevation attributes.

    Args:
        graph: Graph to mutate in place.
        frame: District table.
    """
    for row in frame.select([DISTRICT_ID, DISTRICT, STATE, PROVINCE, ELEVATION]).iter_rows(
        named=True
    ):
        graph.add_node(
            row[DISTRICT_ID],
            district=row[DISTRICT],
            state=row[STATE],
            province=row[PROVINCE],
            elevation=row[ELEVATION],
        )
