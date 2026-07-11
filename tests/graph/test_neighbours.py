import networkx as nx
import pytest

from open_world.graph.edge_types import EDGE_KIND
from open_world.graph.neighbours import (
    KIND_CONNECTIVITY_REPAIR,
    KIND_CROSS_STATE,
    KIND_SAME_PROVINCE,
    KIND_SAME_STATE,
    _closest_pair,
    assign_edges,
    repair_connectivity,
)


def test_assign_edges_rejects_non_positive_min_degree(make_frame):
    frame = make_frame([{"state": "s1", "province": "p1", "districtId": "a", "elevation": 10}])

    with pytest.raises(ValueError, match="min_degree must be a positive integer"):
        assign_edges(frame, min_degree=0)


def test_assign_edges_rejects_min_degree_above_max_degree(make_frame):
    frame = make_frame([{"state": "s1", "province": "p1", "districtId": "a", "elevation": 10}])

    with pytest.raises(ValueError, match="must not exceed max_degree"):
        assign_edges(frame, min_degree=6, max_degree=5)


def test_assign_edges_rejects_out_of_range_boundary_fraction(make_frame):
    frame = make_frame([{"state": "s1", "province": "p1", "districtId": "a", "elevation": 10}])

    with pytest.raises(ValueError, match="min_boundary_fraction must be in"):
        assign_edges(frame, min_boundary_fraction=1.5)


def test_assign_edges_prefers_same_province_when_enough_candidates(make_frame):
    # p1 has plenty of same-province candidates near x's elevation; p2 (same
    # state) also has close candidates, but tier 1 alone should satisfy the
    # fixed target degree, so with no forced boundary quota, no edge should
    # ever reach tier 2 or 3.
    rows = [{"state": "s1", "province": "p1", "districtId": "x", "elevation": 50}]
    rows += [
        {"state": "s1", "province": "p1", "districtId": f"p1-{i}", "elevation": 40 + i}
        for i in range(6)
    ]
    rows += [
        {"state": "s1", "province": "p2", "districtId": f"p2-{i}", "elevation": 45 + i}
        for i in range(6)
    ]
    frame = make_frame(rows)

    edges = assign_edges(
        frame, min_degree=3, max_degree=3, candidate_pool=8, min_boundary_fraction=0, seed=1
    )

    x_edges = [(u, v, kind) for u, v, kind in edges if "x" in (u, v)]
    assert len(x_edges) == 3
    assert all(kind == KIND_SAME_PROVINCE for _, _, kind in x_edges)


def test_assign_edges_forces_a_boundary_quota_even_with_local_capacity(make_frame):
    # Same setup as above -- p1 alone could satisfy x's whole target degree --
    # but a full boundary quota should force x to seek boundary edges first.
    rows = [{"state": "s1", "province": "p1", "districtId": "x", "elevation": 50}]
    rows += [
        {"state": "s1", "province": "p1", "districtId": f"p1-{i}", "elevation": 40 + i}
        for i in range(6)
    ]
    rows += [
        {"state": "s1", "province": "p2", "districtId": f"p2-{i}", "elevation": 45 + i}
        for i in range(6)
    ]
    frame = make_frame(rows)

    edges = assign_edges(
        frame, min_degree=3, max_degree=3, candidate_pool=8, min_boundary_fraction=1.0, seed=1
    )

    # With every district in both provinces chasing a full boundary quota,
    # contention can leave x with fewer than its target -- what matters is
    # that none of its edges are same-province, given a full quota.
    x_edges = [(u, v, kind) for u, v, kind in edges if "x" in (u, v)]
    assert len(x_edges) >= 1
    assert all(kind == KIND_SAME_STATE for _, _, kind in x_edges)


def test_assign_edges_falls_back_to_same_state_when_province_exhausted(make_frame):
    # p1 has only one other district (a), so tier 1 caps at 1 candidate for x;
    # the remaining budget must come from p2, in the same state.
    rows = [
        {"state": "s1", "province": "p1", "districtId": "x", "elevation": 50},
        {"state": "s1", "province": "p1", "districtId": "a", "elevation": 51},
    ]
    rows += [
        {"state": "s1", "province": "p2", "districtId": f"p2-{i}", "elevation": 45 + i}
        for i in range(6)
    ]
    frame = make_frame(rows)

    edges = assign_edges(frame, min_degree=3, max_degree=3, candidate_pool=8, seed=1)

    # p2's own 6 members can fully satisfy each other's target degree without
    # ever touching x, so how much spare capacity is left for x's tier-2
    # fallback depends on processing order -- assert the tier composition
    # (province exhausted -> state used, state never exhausted -> no
    # cross-state leak), not an exact count.
    x_edges = [(u, v, kind) for u, v, kind in edges if "x" in (u, v)]
    kinds = [kind for _, _, kind in x_edges]
    assert kinds.count(KIND_SAME_PROVINCE) == 1
    assert KIND_CROSS_STATE not in kinds
    assert len(x_edges) >= 2  # the guaranteed same-province edge plus at least one fallback


def test_assign_edges_falls_back_to_cross_state_when_state_exhausted(make_frame):
    # x is the sole member of both its province and its state.
    rows = [{"state": "s1", "province": "p1", "districtId": "x", "elevation": 50}]
    rows += [
        {"state": "s2", "province": "p2", "districtId": f"other-{i}", "elevation": 45 + i}
        for i in range(6)
    ]
    frame = make_frame(rows)

    edges = assign_edges(frame, min_degree=3, max_degree=3, candidate_pool=8, seed=1)

    x_edges = [(u, v, kind) for u, v, kind in edges if "x" in (u, v)]
    assert len(x_edges) >= 1
    assert all(kind == KIND_CROSS_STATE for _, _, kind in x_edges)


def test_assign_edges_respects_max_degree(make_frame):
    rows = [
        {"state": "s1", "province": "p1", "districtId": f"d{i}", "elevation": i} for i in range(30)
    ]
    frame = make_frame(rows)

    edges = assign_edges(frame, min_degree=2, max_degree=3, seed=7)

    degree: dict[str, int] = {}
    for source, target, _ in edges:
        degree[source] = degree.get(source, 0) + 1
        degree[target] = degree.get(target, 0) + 1
    assert all(d <= 3 for d in degree.values())


def test_assign_edges_produces_no_self_edges(make_frame):
    rows = [
        {"state": "s1", "province": "p1", "districtId": f"d{i}", "elevation": i} for i in range(15)
    ]
    frame = make_frame(rows)

    edges = assign_edges(frame, seed=3)

    assert all(source != target for source, target, _ in edges)


def test_assign_edges_is_deterministic_for_a_fixed_seed(make_frame):
    rows = [
        {"state": "s1", "province": "p1", "districtId": f"d{i}", "elevation": i * 3 % 20}
        for i in range(20)
    ]
    frame = make_frame(rows)

    first = assign_edges(frame, seed=99)
    second = assign_edges(frame, seed=99)

    assert first == second


def test_assign_edges_varies_with_a_different_seed(make_frame):
    rows = [
        {"state": "s1", "province": "p1", "districtId": f"d{i}", "elevation": i * 3 % 20}
        for i in range(20)
    ]
    frame = make_frame(rows)

    first = assign_edges(frame, seed=1)
    second = assign_edges(frame, seed=2)

    assert first != second


def test_repair_connectivity_merges_disconnected_province_components(make_frame):
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "a", "elevation": 10},
            {"state": "s1", "province": "p1", "districtId": "b", "elevation": 20},
            {"state": "s1", "province": "p1", "districtId": "c", "elevation": 30},
            {"state": "s1", "province": "p1", "districtId": "d", "elevation": 40},
        ]
    )
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    graph.add_edge("a", "b")  # {a, b} and {c, d} are disconnected components
    graph.add_edge("c", "d")

    repair_connectivity(graph, frame, max_degree=5)

    assert nx.is_connected(graph.subgraph(["a", "b", "c", "d"]))


def test_repair_connectivity_merges_disconnected_state_components(make_frame):
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "a", "elevation": 10},
            {"state": "s1", "province": "p1", "districtId": "b", "elevation": 20},
            {"state": "s1", "province": "p2", "districtId": "c", "elevation": 30},
            {"state": "s1", "province": "p2", "districtId": "d", "elevation": 40},
        ]
    )
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    graph.add_edge("a", "b")
    graph.add_edge("c", "d")
    # provinces are each internally connected but never linked to each other.

    repair_connectivity(graph, frame, max_degree=5)

    assert nx.is_connected(graph.subgraph(["a", "b", "c", "d"]))


def test_repair_connectivity_tags_new_edges(make_frame):
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "a", "elevation": 10},
            {"state": "s1", "province": "p1", "districtId": "b", "elevation": 20},
        ]
    )
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b"])

    repair_connectivity(graph, frame, max_degree=5)

    assert graph.edges["a", "b"][EDGE_KIND] == KIND_CONNECTIVITY_REPAIR


def test_repair_connectivity_can_exceed_max_degree_as_a_last_resort(make_frame):
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "a", "elevation": 10},
            {"state": "s1", "province": "p1", "districtId": "b", "elevation": 20},
        ]
    )
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b"])

    repair_connectivity(graph, frame, max_degree=0)

    assert graph.has_edge("a", "b")


def test_closest_pair_finds_the_smallest_elevation_gap():
    elevation_of = {"a1": 10, "a2": 100, "b1": 95, "b2": 200}

    pair = _closest_pair({"a1", "a2"}, {"b1", "b2"}, elevation_of)

    assert pair == ("a2", "b1")
