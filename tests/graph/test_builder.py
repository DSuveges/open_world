from open_world.graph.builder import build_graph
from open_world.graph.edge_types import EDGE_KIND
from open_world.graph.neighbours import KIND_CROSS_STATE, KIND_SAME_PROVINCE, KIND_SAME_STATE


def test_build_graph_creates_a_node_per_district_with_attributes(make_frame):
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "a", "elevation": 10},
            {"state": "s1", "province": "p1", "districtId": "b", "elevation": 20},
        ]
    )

    graph = build_graph(frame)

    assert set(graph.nodes) == {"a", "b"}
    assert graph.nodes["a"] == {
        "district": "a",
        "state": "s1",
        "province": "p1",
        "elevation": 10,
    }


def test_build_graph_every_district_has_a_neighbour(make_frame):
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "a", "elevation": 10},
            {"state": "s2", "province": "p2", "districtId": "b", "elevation": 5000},
            {"state": "s3", "province": "p3", "districtId": "c", "elevation": 9000},
        ]
    )

    graph = build_graph(frame)

    assert all(degree > 0 for _, degree in graph.degree)


def test_build_graph_connects_isolated_districts_across_states(make_frame):
    # Each district is the sole member of its own province and state, so the
    # only way either gets a neighbour is the cross-state fallback tier.
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "a", "elevation": 10},
            {"state": "s2", "province": "p2", "districtId": "b", "elevation": 20},
        ]
    )

    graph = build_graph(frame)

    assert graph.edges["a", "b"][EDGE_KIND] == KIND_CROSS_STATE


def test_build_graph_labels_edges_with_originating_tier(make_frame):
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "p1-a", "elevation": 10},
            {"state": "s1", "province": "p1", "districtId": "p1-b", "elevation": 20},
            {"state": "s1", "province": "p2", "districtId": "p2-a", "elevation": 5000},
        ]
    )

    graph = build_graph(frame)

    assert graph.edges["p1-a", "p1-b"][EDGE_KIND] == KIND_SAME_PROVINCE
    assert graph.edges["p1-a", "p2-a"][EDGE_KIND] == KIND_SAME_STATE


def test_build_graph_respects_max_degree(make_frame):
    rows = [
        {"state": "s1", "province": "p1", "districtId": f"d{i}", "elevation": i} for i in range(30)
    ]
    frame = make_frame(rows)

    graph = build_graph(frame, max_degree=3)

    assert all(degree <= 3 for _, degree in graph.degree)


def test_build_graph_produces_no_duplicate_or_self_edges(make_frame):
    rows = [
        {"state": "s1", "province": "p1", "districtId": f"d{i}", "elevation": i * 7 % 50}
        for i in range(40)
    ]
    frame = make_frame(rows)

    graph = build_graph(frame)

    for node in graph.nodes:
        assert not graph.has_edge(node, node)
    assert graph.number_of_edges() == len({frozenset(e) for e in graph.edges})
