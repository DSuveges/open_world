import networkx as nx

from open_world.graph.validation import validate_graph


def _frame_with_two_provinces(make_frame):
    return make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "p1-a", "elevation": 10},
            {"state": "s1", "province": "p1", "districtId": "p1-b", "elevation": 20},
            {"state": "s1", "province": "p2", "districtId": "p2-a", "elevation": 30},
            {"state": "s1", "province": "p2", "districtId": "p2-b", "elevation": 40},
        ]
    )


def test_fully_connected_graph_is_valid(make_frame):
    frame = _frame_with_two_provinces(make_frame)
    graph = nx.Graph()
    graph.add_nodes_from(["p1-a", "p1-b", "p2-a", "p2-b"])
    graph.add_edges_from([("p1-a", "p1-b"), ("p2-a", "p2-b"), ("p1-b", "p2-a")])

    report = validate_graph(graph, frame)

    assert report.is_valid
    assert report.disconnected_provinces == []
    assert report.disconnected_states == []
    assert report.orphan_districts == []


def test_detects_a_disconnected_province(make_frame):
    frame = _frame_with_two_provinces(make_frame)
    graph = nx.Graph()
    graph.add_nodes_from(["p1-a", "p1-b", "p2-a", "p2-b"])
    # p1-a and p1-b are never linked to each other -> p1 is split in two,
    # while p2 stays connected via its own edge.
    graph.add_edges_from([("p1-a", "p2-a"), ("p1-b", "p2-b"), ("p2-a", "p2-b")])

    report = validate_graph(graph, frame)

    assert not report.is_valid
    assert report.disconnected_provinces == ["p1"]


def test_detects_a_disconnected_state(make_frame):
    frame = _frame_with_two_provinces(make_frame)
    graph = nx.Graph()
    graph.add_nodes_from(["p1-a", "p1-b", "p2-a", "p2-b"])
    graph.add_edges_from([("p1-a", "p1-b"), ("p2-a", "p2-b")])
    # p1 and p2 are each internally connected but never linked to each other.

    report = validate_graph(graph, frame)

    assert not report.is_valid
    assert report.disconnected_states == ["s1"]


def test_detects_an_orphan_district(make_frame):
    frame = _frame_with_two_provinces(make_frame)
    graph = nx.Graph()
    graph.add_nodes_from(["p1-a", "p1-b", "p2-a", "p2-b"])
    # p2-b has no edges at all.
    graph.add_edges_from([("p1-a", "p1-b"), ("p1-b", "p2-a")])

    report = validate_graph(graph, frame)

    assert not report.is_valid
    assert report.orphan_districts == ["p2-b"]


def test_high_degree_is_flagged_but_does_not_invalidate(make_frame):
    frame = _frame_with_two_provinces(make_frame)
    graph = nx.Graph()
    graph.add_nodes_from(["p1-a", "p1-b", "p2-a", "p2-b"])
    graph.add_edges_from([("p1-a", "p1-b"), ("p1-a", "p2-a"), ("p1-a", "p2-b"), ("p2-a", "p2-b")])

    report = validate_graph(graph, frame, max_degree=2)

    assert report.is_valid
    assert report.high_degree_districts == {"p1-a": 3}
