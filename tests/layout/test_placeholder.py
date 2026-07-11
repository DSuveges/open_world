import networkx as nx

from open_world.layout.placeholder import apply_positions, compute_placeholder_layout


def test_compute_placeholder_layout_returns_a_position_per_node():
    graph = nx.path_graph(5)

    positions = compute_placeholder_layout(graph)

    assert set(positions) == set(graph.nodes)
    for x, y in positions.values():
        assert isinstance(x, float)
        assert isinstance(y, float)


def test_compute_placeholder_layout_is_deterministic():
    graph = nx.erdos_renyi_graph(30, 0.2, seed=7)

    first = compute_placeholder_layout(graph)
    second = compute_placeholder_layout(graph)

    assert first == second


def test_apply_positions_sets_node_attributes():
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b"])

    apply_positions(graph, {"a": (1.0, 2.0), "b": (3.0, 4.0)})

    assert graph.nodes["a"]["x"] == 1.0
    assert graph.nodes["a"]["y"] == 2.0
    assert graph.nodes["b"]["x"] == 3.0
    assert graph.nodes["b"]["y"] == 4.0
