import networkx as nx

from open_world.layout.hexgrid import (
    HEX_NEIGHBOR_OFFSETS,
    _axial_distance,
    _axial_neighbors,
    _find_anchor_hex,
    _group_into_landmasses,
    _growth_order,
    _nearest_empty_hex,
    compute_hex_layout,
)


def _sample_graph() -> nx.Graph:
    graph = nx.Graph()
    # s1/p1: elevation-descending chain a->b
    graph.add_node("s1-p1-a", state="s1", province="p1", elevation=100)
    graph.add_node("s1-p1-b", state="s1", province="p1", elevation=80)
    graph.add_edge("s1-p1-a", "s1-p1-b")
    # s1/p2: elevation-descending chain c->d
    graph.add_node("s1-p2-c", state="s1", province="p2", elevation=60)
    graph.add_node("s1-p2-d", state="s1", province="p2", elevation=40)
    graph.add_edge("s1-p2-c", "s1-p2-d")
    # the only link between p1 and p2 -- growth must cross here
    graph.add_edge("s1-p1-b", "s1-p2-c")

    # s2/p3: a separate state, single province, no edge to s1 at all -- must
    # remain its own island.
    graph.add_node("s2-p3-e", state="s2", province="p3", elevation=50)
    graph.add_node("s2-p3-f", state="s2", province="p3", elevation=30)
    graph.add_edge("s2-p3-e", "s2-p3-f")
    return graph


def _is_hex_contiguous(hexes: set) -> bool:
    if not hexes:
        return True
    seen = {next(iter(hexes))}
    frontier = list(seen)
    while frontier:
        current = frontier.pop()
        for neighbour in _axial_neighbors(current):
            if neighbour in hexes and neighbour not in seen:
                seen.add(neighbour)
                frontier.append(neighbour)
    return seen == hexes


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
        assert _is_hex_contiguous({positions[m] for m in members})


def test_compute_hex_layout_keeps_each_province_hex_contiguous():
    # A larger, multi-province state where districts don't arrive in a tidy
    # order -- this is exactly the scenario that used to let provinces end
    # up spatially scattered even though the graph itself stayed connected.
    graph = nx.Graph()
    for i in range(6):
        graph.add_node(f"p1-{i}", state="s1", province="p1", elevation=100 - i)
    for i in range(6):
        graph.add_node(f"p2-{i}", state="s1", province="p2", elevation=90 - i)
    for i in range(5):
        graph.add_edge(f"p1-{i}", f"p1-{i + 1}")
        graph.add_edge(f"p2-{i}", f"p2-{i + 1}")
    graph.add_edge("p1-3", "p2-2")  # the only cross-province link

    positions = compute_hex_layout(graph)

    for province in ("p1", "p2"):
        members = [n for n, attrs in graph.nodes(data=True) if attrs["province"] == province]
        assert _is_hex_contiguous({positions[m] for m in members})


def test_compute_hex_layout_leaves_a_water_gap_between_unconnected_states():
    graph = _sample_graph()

    positions = compute_hex_layout(graph, water_gap=3)

    s1_hexes = [pos for n, pos in positions.items() if n.startswith("s1")]
    s2_hexes = [pos for n, pos in positions.items() if n.startswith("s2")]
    min_gap = min(_axial_distance(a, b) for a in s1_hexes for b in s2_hexes)
    assert min_gap > 1  # unconnected states must not be hex-adjacent


def test_compute_hex_layout_groups_cross_state_connected_states_into_one_landmass():
    graph = _sample_graph()
    # unlike _sample_graph's baseline, link s1 and s2 -- they should now
    # share one landmass instead of getting separate islands.
    graph.add_edge("s1-p2-d", "s2-p3-e")
    # s3 has no edge to anyone and must remain a separate island.
    graph.add_node("s3-a", state="s3", province="p4", elevation=20)
    graph.add_node("s3-b", state="s3", province="p4", elevation=10)
    graph.add_edge("s3-a", "s3-b")

    positions = compute_hex_layout(graph, water_gap=3)

    s1_hexes = {pos for n, pos in positions.items() if n.startswith("s1")}
    s2_hexes = {pos for n, pos in positions.items() if n.startswith("s2")}
    s3_hexes = {pos for n, pos in positions.items() if n.startswith("s3")}

    assert _is_hex_contiguous(s1_hexes | s2_hexes)  # s1+s2 form one landmass
    min_gap_to_s3 = min(_axial_distance(a, b) for a in (s1_hexes | s2_hexes) for b in s3_hexes)
    assert min_gap_to_s3 > 1  # s3 stays a separate island


def test_compute_hex_layout_falls_back_when_no_neighbour_is_placed_yet():
    # b's only graph edge is to c, which is processed *after* it (lower
    # elevation, same province) -- when b is placed, none of its neighbours
    # are placed yet, so it must fall back to any frontier hex rather than
    # crash.
    graph = nx.Graph()
    graph.add_node("a", state="s1", province="p1", elevation=100)
    graph.add_node("b", state="s1", province="p1", elevation=50)
    graph.add_node("c", state="s1", province="p1", elevation=10)
    graph.add_edge("b", "c")

    positions = compute_hex_layout(graph)

    assert set(positions) == {"a", "b", "c"}
    assert len(set(positions.values())) == 3


def test_compute_hex_layout_is_deterministic():
    graph = _sample_graph()

    first = compute_hex_layout(graph)
    second = compute_hex_layout(graph)

    assert first == second


def test_group_into_landmasses_merges_cross_state_connected_states():
    graph = nx.Graph()
    graph.add_node("a", state="s1")
    graph.add_node("b", state="s2")
    graph.add_node("c", state="s3")
    graph.add_edge("a", "b")  # s1 <-> s2 connected; s3 stands alone
    members_by_state = {"s1": ["a"], "s2": ["b"], "s3": ["c"]}

    landmasses = _group_into_landmasses(members_by_state, graph)

    landmass_sets = {frozenset(lm) for lm in landmasses}
    assert landmass_sets == {frozenset({"s1", "s2"}), frozenset({"s3"})}


def test_growth_order_starts_at_the_peak_and_stays_connected():
    graph = nx.Graph()
    graph.add_node("p1-a", state="s1", province="p1", elevation=100)
    graph.add_node("p2-a", state="s1", province="p2", elevation=50)
    graph.add_node("p3-a", state="s1", province="p3", elevation=10)
    graph.add_edge("p1-a", "p2-a")
    graph.add_edge("p2-a", "p3-a")
    provinces = {"p1": ["p1-a"], "p2": ["p2-a"], "p3": ["p3-a"]}

    order = _growth_order(provinces, graph)

    assert order[0] == "p1"  # contains the peak
    assert set(order) == {"p1", "p2", "p3"}
    assert order.index("p2") < order.index("p3")  # p3 only reachable via p2


def test_growth_order_visits_larger_groups_first():
    graph = nx.Graph()
    graph.add_node("hub-a", state="s1", province="hub", elevation=100)
    graph.add_node("small-a", state="s1", province="small", elevation=50)
    graph.add_node("big-a", state="s1", province="big", elevation=50)
    graph.add_node("big-b", state="s1", province="big", elevation=40)
    graph.add_node("big-c", state="s1", province="big", elevation=30)
    graph.add_edge("hub-a", "small-a")
    graph.add_edge("hub-a", "big-a")
    provinces = {
        "hub": ["hub-a"],
        "small": ["small-a"],
        "big": ["big-a", "big-b", "big-c"],
    }

    order = _growth_order(provinces, graph)

    assert order.index("big") < order.index("small")


def test_find_anchor_hex_prefers_the_smallest_elevation_gap():
    graph = nx.Graph()
    graph.add_node("h1", state="s1", province="p1", elevation=100)
    graph.add_node("l1", state="s1", province="p1", elevation=10)
    graph.add_node("h2", state="s1", province="p2", elevation=95)
    graph.add_node("l2", state="s1", province="p2", elevation=5)
    graph.add_edge("h1", "h2")  # gap 5 -- smallest
    graph.add_edge("h1", "l2")  # gap 95
    graph.add_edge("l1", "h2")  # gap 75

    positions = {"h1": (0, 0), "l1": (5, 5)}
    occupied = {(0, 0), (5, 5)}

    anchor = _find_anchor_hex(["h2", "l2"], positions, occupied, graph)

    assert anchor in set(_axial_neighbors((0, 0)))


def test_find_anchor_hex_respects_the_neighbour_filter():
    # h2 has the smallest elevation gap to l1, but it belongs to a different
    # state; with a same-state-only filter, l1 must anchor to h1 (same
    # state) instead, even though its gap is larger.
    graph = nx.Graph()
    graph.add_node("h1", state="s1", province="hub", elevation=20)
    graph.add_node("h2", state="s2", province="other", elevation=12)
    graph.add_node("l1", state="s1", province="p1", elevation=10)
    graph.add_edge("l1", "h1")  # gap 10, same state
    graph.add_edge("l1", "h2")  # gap 2, smaller, but wrong state

    positions = {"h1": (0, 0), "h2": (10, 10)}
    occupied = {(0, 0), (10, 10)}

    def _is_state_1(node: str) -> bool:
        return graph.nodes[node]["state"] == "s1"

    anchor = _find_anchor_hex(["l1"], positions, occupied, graph, is_valid_neighbour=_is_state_1)

    assert anchor in set(_axial_neighbors((0, 0)))  # anchored to h1, not h2


def test_find_anchor_hex_falls_back_when_the_neighbour_is_boxed_in():
    graph = nx.Graph()
    graph.add_node("h1", state="s1", province="p1", elevation=100)
    graph.add_node("h2", state="s1", province="p2", elevation=95)
    graph.add_edge("h1", "h2")

    positions = {"h1": (0, 0)}
    occupied = {(0, 0), *_axial_neighbors((0, 0))}  # h1 has no free hex left around it

    anchor = _find_anchor_hex(["h2"], positions, occupied, graph)

    assert anchor not in occupied


def test_growth_order_appends_stragglers_with_no_inter_group_edge():
    # A hand-crafted case bypassing the normal connectivity guarantee, to
    # exercise the defensive fallback directly: p2 has no edge to anyone.
    graph = nx.Graph()
    graph.add_node("p1-a", state="s1", province="p1", elevation=100)
    graph.add_node("p2-a", state="s1", province="p2", elevation=50)
    provinces = {"p1": ["p1-a"], "p2": ["p2-a"]}

    order = _growth_order(provinces, graph)

    assert set(order) == {"p1", "p2"}


def test_nearest_empty_hex_returns_start_when_free():
    assert _nearest_empty_hex((0, 0), set()) == (0, 0)


def test_nearest_empty_hex_searches_outward_when_boxed_in():
    start = (0, 0)
    occupied = {start, *_axial_neighbors(start)}  # every immediate neighbour taken

    result = _nearest_empty_hex(start, occupied)

    assert result not in occupied
    assert _axial_distance(start, result) == 2


def test_axial_distance_matches_neighbour_offsets():
    origin = (0, 0)
    for offset in HEX_NEIGHBOR_OFFSETS:
        assert _axial_distance(origin, offset) == 1


def test_axial_neighbors_returns_six_hexes():
    assert len(list(_axial_neighbors((2, -3)))) == 6
