"""A cheap, deterministic placeholder layout for evaluating the graph.

This is not the final map. It ignores the state/coastline/hex-grid
constraints entirely and just spreads nodes out based on graph structure, so
that connected clusters (provinces, states, cross-border mountain ranges)
are visually separable while iterating on the graph-generation algorithms.

Spectral layout (eigenvectors of the graph Laplacian) was chosen over a
force-directed layout for one practical reason: on this dataset's ~12k
nodes, ``networkx.spring_layout`` takes on the order of a minute, while
``networkx.spectral_layout`` takes well under a second. Iteration speed is
the whole point of this stage, so cost matters more here than the visual
polish force-directed layouts give.

Once the graph algorithm is settled, this module is the piece that gets
swapped for a real spatial embedding (hex region-growing or Voronoi
relaxation) -- both would fit the same ``compute_*_layout(graph) -> {node:
(x, y)}`` shape.
"""

import networkx as nx
from loguru import logger

Position = tuple[float, float]


def compute_placeholder_layout(graph: nx.Graph) -> dict[str, Position]:
    """Compute 2D positions for every node using a spectral layout.

    Args:
        graph: Graph to lay out.

    Returns:
        Mapping of node id to an (x, y) position.
    """
    logger.info("computing placeholder spectral layout ({} nodes)", graph.number_of_nodes())
    positions = nx.spectral_layout(graph)
    return {node: (float(x), float(y)) for node, (x, y) in positions.items()}


def apply_positions(graph: nx.Graph, positions: dict[str, Position]) -> None:
    """Store computed positions as ``x``/``y`` node attributes, in place.

    Args:
        graph: Graph to mutate.
        positions: Mapping of node id to (x, y), e.g. from
            :func:`compute_placeholder_layout`.
    """
    for node, (x, y) in positions.items():
        graph.nodes[node]["x"] = x
        graph.nodes[node]["y"] = y
