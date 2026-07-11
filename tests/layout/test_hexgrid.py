import networkx as nx

from open_world.layout.hexgrid import (
    HEX_NEIGHBOR_OFFSETS,
    _axial_distance,
    _axial_neighbors,
    compute_hex_layout,
)


def _sample_graph() -> nx.Graph:
    graph = nx.Graph()
    # s1: a chain of 4, elevations descending a->d
    graph.add_node("s1-a", state="s1", elevation=100)
    graph.add_node("s1-b", state="s1", elevation=80)
    graph.add_node("s1-c", state="s1", elevation=60)
    graph.add_node("s1-d", state="s1", elevation=40)
    graph.add_edges_from([("s1-a", "s1-b"), ("s1-b", "s1-c"), ("s1-c", "s1-d")])
    # s2: a separate small state
    graph.add_node("s2-a", state="s2", elevation=50)
    graph.add_node("s2-b", state="s2", elevation=30)
    graph.add_edge("s2-a", "s2-b")
    # a deliberate cross-state edge, which placement must ignore spatially
    graph.add_edge("s1-d", "s2-a")
    return graph


def test_compute_hex_layout_places_every_node():
    graph = _sample_graph()

    positions = compute_hex_layout(graph)

    assert set(positions) == set(graph.nodes)


def test_compute_hex_layout_gives_every_node_a_unique_hex():
    graph = _sample_graph()

    positions = compute_hex_layout(graph)

    assert len(set(positions.values())) == len(positions)


def test_compute_hex_layout_keeps_each_state_hex_contiguous():
    graph = _sample_graph()

    positions = compute_hex_layout(graph)

    for state in ("s1", "s2"):
        members = [n for n, attrs in graph.nodes(data=True) if attrs["state"] == state]
        hexes = {positions[m] for m in members}
        # every hex in the state must be reachable from some other hex in the
        # same state via a chain of axial adjacencies.
        seen = {next(iter(hexes))}
        frontier = list(seen)
        while frontier:
            current = frontier.pop()
            for neighbour in _axial_neighbors(current):
                if neighbour in hexes and neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append(neighbour)
        assert seen == hexes


def test_compute_hex_layout_leaves_a_water_gap_between_islands():
    graph = _sample_graph()

    positions = compute_hex_layout(graph, water_gap=3)

    s1_hexes = [pos for n, pos in positions.items() if n.startswith("s1")]
    s2_hexes = [pos for n, pos in positions.items() if n.startswith("s2")]
    min_gap = min(_axial_distance(a, b) for a in s1_hexes for b in s2_hexes)
    assert min_gap > 1  # islands must not be hex-adjacent to each other


def test_compute_hex_layout_falls_back_when_no_neighbour_is_placed_yet():
    # b's only graph edge is to c, which is processed *after* it (lower
    # elevation) -- when b is placed, none of its neighbours are placed yet,
    # so it must fall back to any frontier hex rather than crash.
    graph = nx.Graph()
    graph.add_node("a", state="s1", elevation=100)
    graph.add_node("b", state="s1", elevation=50)
    graph.add_node("c", state="s1", elevation=10)
    graph.add_edge("b", "c")

    positions = compute_hex_layout(graph)

    assert set(positions) == {"a", "b", "c"}
    assert len(set(positions.values())) == 3


def test_compute_hex_layout_is_deterministic():
    graph = _sample_graph()

    first = compute_hex_layout(graph)
    second = compute_hex_layout(graph)

    assert first == second


def test_axial_distance_matches_neighbour_offsets():
    origin = (0, 0)
    for offset in HEX_NEIGHBOR_OFFSETS:
        assert _axial_distance(origin, offset) == 1


def test_axial_neighbors_returns_six_hexes():
    assert len(list(_axial_neighbors((2, -3)))) == 6
