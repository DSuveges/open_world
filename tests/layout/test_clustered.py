import networkx as nx

from open_world.layout.clustered import compute_clustered_layout


def _sample_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("p1-a", province="p1")
    graph.add_node("p1-b", province="p1")
    graph.add_node("p1-c", province="p1")
    graph.add_node("p2-a", province="p2")
    graph.add_node("p2-b", province="p2")
    graph.add_edges_from([("p1-a", "p1-b"), ("p1-b", "p1-c"), ("p2-a", "p2-b"), ("p1-a", "p2-a")])
    return graph


def test_compute_clustered_layout_returns_a_position_per_node():
    graph = _sample_graph()

    positions = compute_clustered_layout(graph)

    assert set(positions) == set(graph.nodes)
    for x, y in positions.values():
        assert isinstance(x, float)
        assert isinstance(y, float)


def test_compute_clustered_layout_is_deterministic():
    graph = _sample_graph()

    first = compute_clustered_layout(graph, seed=7)
    second = compute_clustered_layout(graph, seed=7)

    assert first == second


def test_compute_clustered_layout_handles_a_province_with_no_boundary_edges():
    graph = nx.Graph()
    graph.add_node("a", province="p1")
    graph.add_node("b", province="p1")
    graph.add_edge("a", "b")
    graph.add_node("c", province="p2")
    graph.add_node("d", province="p2")
    graph.add_edge("c", "d")
    # p1 and p2 share no edge at all -- the meta-graph has an isolated node.

    positions = compute_clustered_layout(graph)

    assert set(positions) == {"a", "b", "c", "d"}
