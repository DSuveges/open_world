import networkx as nx
import pytest

from open_world.graph.edge_types import EDGE_KIND
from open_world.graph.metrics import compute_metrics
from open_world.graph.neighbours import KIND_CROSS_STATE, KIND_SAME_PROVINCE, KIND_SAME_STATE


@pytest.fixture
def sample_graph_and_frame(make_frame):
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "p1a", "elevation": 10},
            {"state": "s1", "province": "p1", "districtId": "p1b", "elevation": 20},
            {"state": "s1", "province": "p2", "districtId": "p2a", "elevation": 30},
            {"state": "s1", "province": "p2", "districtId": "p2b", "elevation": 9000},
            {"state": "s2", "province": "p3", "districtId": "p3a", "elevation": 9010},
        ]
    )

    graph = nx.Graph()
    for row in frame.iter_rows(named=True):
        graph.add_node(
            row["districtId"],
            state=row["state"],
            province=row["province"],
            elevation=row["elevation"],
        )

    graph.add_edge("p1a", "p1b", **{EDGE_KIND: KIND_SAME_PROVINCE})
    graph.add_edge("p2a", "p2b", **{EDGE_KIND: KIND_SAME_PROVINCE})
    graph.add_edge("p1b", "p2a", **{EDGE_KIND: KIND_SAME_STATE})
    graph.add_edge("p2b", "p3a", **{EDGE_KIND: KIND_CROSS_STATE})  # cross-state, cross-province
    graph.add_edge("p1a", "p2a", **{EDGE_KIND: KIND_SAME_STATE})  # same-state, cross-province

    return graph, frame


def test_basic_counts(sample_graph_and_frame):
    graph, frame = sample_graph_and_frame

    metrics = compute_metrics(graph, frame, high_elevation_percentile=0.6)

    assert metrics.node_count == 5
    assert metrics.edge_count == 5
    assert metrics.avg_degree == pytest.approx(2.0)
    assert metrics.edge_kind_counts == {
        KIND_SAME_PROVINCE: 2,
        KIND_SAME_STATE: 2,
        KIND_CROSS_STATE: 1,
    }


def test_elevation_gap_by_kind(sample_graph_and_frame):
    graph, frame = sample_graph_and_frame

    metrics = compute_metrics(graph, frame, high_elevation_percentile=0.6)

    assert metrics.elevation_gap_by_kind[KIND_SAME_PROVINCE] == {
        "mean": pytest.approx(4490.0),
        "median": pytest.approx(4490.0),
        "max": 8970,
    }
    assert metrics.elevation_gap_by_kind[KIND_SAME_STATE] == {
        "mean": pytest.approx(15.0),
        "median": pytest.approx(15.0),
        "max": 20,
    }
    assert metrics.elevation_gap_by_kind[KIND_CROSS_STATE] == {
        "mean": pytest.approx(10.0),
        "median": pytest.approx(10.0),
        "max": 10,
    }


def test_cross_boundary_fractions(sample_graph_and_frame):
    graph, frame = sample_graph_and_frame

    metrics = compute_metrics(graph, frame, high_elevation_percentile=0.6)

    # cross-province: (p1b,p2a), (p2b,p3a), (p1a,p2a) = 3/5
    assert metrics.cross_province_fraction == pytest.approx(0.6)
    # cross-state: only (p2b,p3a) = 1/5
    assert metrics.cross_state_fraction == pytest.approx(0.2)


def test_fractions_are_zero_for_an_edgeless_graph(make_frame):
    frame = make_frame([{"state": "s1", "province": "p1", "districtId": "a", "elevation": 10}])
    graph = nx.Graph()
    graph.add_node("a", state="s1", province="p1", elevation=10)

    metrics = compute_metrics(graph, frame)

    assert metrics.cross_province_fraction == 0.0
    assert metrics.cross_state_fraction == 0.0


def test_high_elevation_cross_state_cluster(sample_graph_and_frame):
    graph, frame = sample_graph_and_frame

    metrics = compute_metrics(graph, frame, high_elevation_percentile=0.6)

    # p2b (9000) and p3a (9010) are the two high-elevation districts, and they
    # are linked by an edge -> one cluster spanning two states.
    assert metrics.high_elevation_cluster_count == 1
    assert metrics.high_elevation_cross_state_cluster_count == 1
