"""Two-level placeholder layout: cluster provinces coarsely, place districts locally.

Plain spectral layout on the *whole* graph degenerates badly once the graph
is dominated by within-province edges (measured at ~99% same-province on
this dataset): there's too little inter-cluster structure left for a
2-eigenvector embedding to spread into anything but a near-1D line, with a
few boundary-crossing districts stretched out to extreme coordinates. This
instead:

1. Builds a small meta-graph with one node per province (an edge between two
   provinces if any real edge connects their districts, weighted by how
   many) and lays it out with a spring simulation -- cheap at ~100 nodes,
   and produces an organic, non-degenerate spread reflecting how strongly
   provinces are connected to each other.
2. Lays out each province's own subgraph independently with spectral layout
   (cheap even for a province with several thousand districts, since it's
   still much smaller than the whole graph).
3. Composites: each district's position is its province's meta-layout
   center, plus its local position scaled to a per-province "bubble" radius
   (so bigger provinces get visually bigger footprints) and the meta-layout
   scaled so bubbles don't collide as badly.
"""

import math
import statistics

import networkx as nx
from loguru import logger

from open_world.data.schema import PROVINCE

Position = tuple[float, float]

DEFAULT_META_ITERATIONS = 200
DEFAULT_BUBBLE_SCALE = 4.0
DEFAULT_SEED = 42


def compute_clustered_layout(
    graph: nx.Graph,
    meta_iterations: int = DEFAULT_META_ITERATIONS,
    bubble_scale: float = DEFAULT_BUBBLE_SCALE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Position]:
    """Compute 2D positions using a two-level province-cluster layout.

    Args:
        graph: Graph to lay out; nodes must carry a ``province`` attribute.
        meta_iterations: Spring-layout iterations for the province-level
            meta-graph. Cheap, since it has one node per province rather
            than one per district.
        bubble_scale: How far apart to spread province bubbles, relative to
            their radii. Larger values reduce overlap between provinces.
        seed: Random seed for the meta-layout, for reproducibility.

    Returns:
        Mapping of node id to an (x, y) position.
    """
    members_by_province = _group_by_province(graph)
    meta_graph = _build_meta_graph(graph, members_by_province)
    meta_positions = nx.spring_layout(meta_graph, iterations=meta_iterations, seed=seed)

    avg_radius = statistics.mean(
        math.sqrt(len(members)) for members in members_by_province.values()
    )
    meta_scale = avg_radius * bubble_scale

    positions: dict[str, Position] = {}
    for province, members in members_by_province.items():
        local = nx.spectral_layout(graph.subgraph(members))
        radius = math.sqrt(len(members))
        center_x, center_y = meta_positions[province]
        for node, (local_x, local_y) in local.items():
            positions[node] = (
                float(center_x * meta_scale + local_x * radius),
                float(center_y * meta_scale + local_y * radius),
            )

    logger.info(
        "computed clustered layout ({} provinces, {} nodes)",
        len(members_by_province),
        graph.number_of_nodes(),
    )
    return positions


def _group_by_province(graph: nx.Graph) -> dict[str, list[str]]:
    """Bucket node ids by their `province` attribute.

    Args:
        graph: Graph whose nodes carry a ``province`` attribute.

    Returns:
        Mapping of province name to its member node ids.
    """
    members: dict[str, list[str]] = {}
    for node, attrs in graph.nodes(data=True):
        members.setdefault(attrs[PROVINCE], []).append(node)
    return members


def _build_meta_graph(graph: nx.Graph, members_by_province: dict[str, list[str]]) -> nx.Graph:
    """Build a province-level meta-graph, weighted by inter-province edge count.

    Args:
        graph: District-level graph.
        members_by_province: Province name to member node ids, as returned
            by :func:`_group_by_province`.

    Returns:
        A graph with one node per province and a weighted edge between any
        two provinces connected by at least one real edge.
    """
    province_of = {node: attrs[PROVINCE] for node, attrs in graph.nodes(data=True)}
    meta = nx.Graph()
    meta.add_nodes_from(members_by_province)
    for source, target in graph.edges():
        province_a, province_b = province_of[source], province_of[target]
        if province_a == province_b:
            continue
        if meta.has_edge(province_a, province_b):
            meta[province_a][province_b]["weight"] += 1
        else:
            meta.add_edge(province_a, province_b, weight=1)
    return meta
