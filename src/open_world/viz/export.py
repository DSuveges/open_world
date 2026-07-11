"""Serialize the district graph into a plain node-link structure."""

from typing import Any

import networkx as nx

from open_world.graph.edge_types import EDGE_KIND


def graph_to_json(graph: nx.Graph) -> dict[str, list[dict[str, Any]]]:
    """Convert a district graph into a JSON-serializable node-link dict.

    Args:
        graph: Graph produced by :func:`open_world.graph.builder.build_graph`,
            with ``state``, ``province`` and ``elevation`` node attributes
            and a ``kind`` edge attribute.

    Returns:
        A dict with a ``nodes`` list (``id`` plus every node attribute) and
        an ``edges`` list (``source``, ``target``, ``kind``).
    """
    nodes = [{"id": node, **attrs} for node, attrs in graph.nodes(data=True)]
    edges = [
        {"source": source, "target": target, "kind": attrs[EDGE_KIND]}
        for source, target, attrs in graph.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}
