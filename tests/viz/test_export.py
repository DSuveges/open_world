import networkx as nx

from open_world.graph.edge_types import EDGE_KIND
from open_world.graph.neighbours import KIND_SAME_PROVINCE
from open_world.viz.export import graph_to_json


def test_graph_to_json_serializes_nodes_and_edges():
    graph = nx.Graph()
    graph.add_node("a", state="s1", province="p1", elevation=10)
    graph.add_node("b", state="s1", province="p1", elevation=20)
    graph.add_edge("a", "b", **{EDGE_KIND: KIND_SAME_PROVINCE})

    payload = graph_to_json(graph)

    assert payload["nodes"] == [
        {"id": "a", "state": "s1", "province": "p1", "elevation": 10},
        {"id": "b", "state": "s1", "province": "p1", "elevation": 20},
    ]
    assert payload["edges"] == [{"source": "a", "target": "b", "kind": KIND_SAME_PROVINCE}]
