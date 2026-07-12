import random

import networkx as nx
import pytest

from open_world.layout.hexgrid import (
    HEX_NEIGHBOR_OFFSETS,
    HOLE_NEIGHBOUR_THRESHOLD,
    HOLE_PENALTY,
    ISOLATION_PENALTY,
    MIN_NEIGHBOURS_FOR_CANDIDACY,
    _axial_distance,
    _axial_neighbors,
    _best_swap_candidate,
    _find_anchor_hex,
    _group_into_landmasses,
    _growth_order,
    _has_same_province_neighbour,
    _hex_defect_cost,
    _hexes_within_radius,
    _is_swap_candidate,
    _nearest_empty_hex,
    _nearest_empty_hex_from_any,
    _swap_hex_contents,
    compute_hex_layout,
    refine_hex_layout,
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


def test_compute_hex_layout_seams_provinces_at_the_matching_district_not_the_peak():
    # p2's own highest-elevation member (p2-peak, 90) is NOT the district
    # that best matches p1's edge (p1-4, 60) -- p2-match (62) is. The seam
    # must form there, not at p2's peak, otherwise a mismatched-elevation
    # district ends up wedged against p1's low edge.
    graph = nx.Graph()
    graph.add_node("p1-0", state="s1", province="p1", elevation=200)
    graph.add_node("p1-1", state="s1", province="p1", elevation=150)
    graph.add_node("p1-2", state="s1", province="p1", elevation=100)
    graph.add_node("p1-3", state="s1", province="p1", elevation=80)
    graph.add_node("p1-4", state="s1", province="p1", elevation=60)
    for i in range(4):
        graph.add_edge(f"p1-{i}", f"p1-{i + 1}")

    graph.add_node("p2-peak", state="s1", province="p2", elevation=90)
    graph.add_node("p2-match", state="s1", province="p2", elevation=62)
    graph.add_node("p2-b", state="s1", province="p2", elevation=55)
    graph.add_node("p2-c", state="s1", province="p2", elevation=50)
    graph.add_edge("p2-peak", "p2-match")
    graph.add_edge("p2-match", "p2-b")
    graph.add_edge("p2-b", "p2-c")

    graph.add_edge("p1-4", "p2-match")  # the only cross-province link, gap 2

    positions = compute_hex_layout(graph)

    assert positions["p2-match"] in set(_axial_neighbors(positions["p1-4"]))


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


def test_growth_order_prefers_the_smallest_elevation_gap_over_size():
    # far is the bigger group, but near has the smaller elevation gap to the
    # peak -- growth should still reach it first, so a mountain range
    # doesn't get a low-elevation group wedged into the middle of it.
    graph = nx.Graph()
    graph.add_node("hub-a", state="s1", province="hub", elevation=100)
    graph.add_node("near-a", state="s1", province="near", elevation=90)
    graph.add_node("far-a", state="s1", province="far", elevation=50)
    graph.add_node("far-b", state="s1", province="far", elevation=40)
    graph.add_node("far-c", state="s1", province="far", elevation=30)
    graph.add_edge("hub-a", "near-a")  # gap 10
    graph.add_edge("hub-a", "far-a")  # gap 50
    provinces = {
        "hub": ["hub-a"],
        "near": ["near-a"],
        "far": ["far-a", "far-b", "far-c"],
    }

    order = _growth_order(provinces, graph)

    assert order.index("near") < order.index("far")


def test_growth_order_breaks_elevation_gap_ties_by_size():
    graph = nx.Graph()
    graph.add_node("hub-a", state="s1", province="hub", elevation=100)
    graph.add_node("small-a", state="s1", province="small", elevation=90)
    graph.add_node("big-a", state="s1", province="big", elevation=90)
    graph.add_node("big-b", state="s1", province="big", elevation=80)
    graph.add_node("big-c", state="s1", province="big", elevation=70)
    graph.add_edge("hub-a", "small-a")  # gap 10
    graph.add_edge("hub-a", "big-a")  # gap 10, tied with small
    provinces = {
        "hub": ["hub-a"],
        "small": ["small-a"],
        "big": ["big-a", "big-b", "big-c"],
    }

    order = _growth_order(provinces, graph)

    assert order.index("big") < order.index("small")


def test_growth_order_honours_an_explicit_start():
    graph = nx.Graph()
    graph.add_node("p1-a", state="s1", province="p1", elevation=100)
    graph.add_node("p2-a", state="s1", province="p2", elevation=50)
    graph.add_node("p3-a", state="s1", province="p3", elevation=10)
    graph.add_edge("p1-a", "p2-a")
    graph.add_edge("p2-a", "p3-a")
    provinces = {"p1": ["p1-a"], "p2": ["p2-a"], "p3": ["p3-a"]}

    order = _growth_order(provinces, graph, start="p2")

    assert order[0] == "p2"  # forced start, not the peak
    assert set(order) == {"p1", "p2", "p3"}


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

    district, anchor = _find_anchor_hex(["h2", "l2"], positions, occupied, graph)

    assert district == "h2"  # smallest gap (5) is h1<->h2, not h1<->l2 or l1<->h2
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

    district, anchor = _find_anchor_hex(
        ["l1"], positions, occupied, graph, is_valid_neighbour=_is_state_1
    )

    assert district == "l1"
    assert anchor in set(_axial_neighbors((0, 0)))  # anchored to h1, not h2


def test_find_anchor_hex_falls_back_when_the_neighbour_is_boxed_in():
    graph = nx.Graph()
    graph.add_node("h1", state="s1", province="p1", elevation=100)
    graph.add_node("h2", state="s1", province="p2", elevation=95)
    graph.add_edge("h1", "h2")

    positions = {"h1": (0, 0)}
    occupied = {(0, 0), *_axial_neighbors((0, 0))}  # h1 has no free hex left around it

    district, anchor = _find_anchor_hex(["h2"], positions, occupied, graph)

    assert district == "h2"
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


def test_nearest_empty_hex_from_any_only_travels_through_allowed_hexes():
    # The whole first ring around start is occupied, so reaching any empty
    # hex requires tunnelling through one ring-1 hex into ring 2. Only one
    # ring-1 hex is allowed -- the search must go through it rather than
    # any of the others, even though they're all equally "occupied".
    start = (0, 0)
    ring1 = set(_axial_neighbors(start))
    occupied = {start, *ring1}
    allowed = (1, 0)

    result = _nearest_empty_hex_from_any({start}, occupied, can_pass=lambda h: h == allowed)

    assert result == (2, 0)  # only reachable by passing through `allowed`


def test_nearest_empty_hex_from_any_raises_when_fully_walled_in():
    start = (0, 0)
    occupied = {start, *_axial_neighbors(start)}

    with pytest.raises(RuntimeError):
        _nearest_empty_hex_from_any({start}, occupied, can_pass=lambda _: False)


def test_axial_distance_matches_neighbour_offsets():
    origin = (0, 0)
    for offset in HEX_NEIGHBOR_OFFSETS:
        assert _axial_distance(origin, offset) == 1


def test_axial_neighbors_returns_six_hexes():
    assert len(list(_axial_neighbors((2, -3)))) == 6


def test_hex_defect_cost_flags_a_district_with_no_same_state_or_edge_neighbour():
    graph = nx.Graph()
    graph.add_node("a", state="s1")
    graph.add_node("b", state="s2")
    graph.add_node("c", state="s2")
    hex_to_node = {(0, 0): "a", (1, 0): "b", (1, -1): "c"}

    cost = _hex_defect_cost((0, 0), hex_to_node, graph)

    assert cost == ISOLATION_PENALTY


def test_hex_defect_cost_ignores_mismatch_when_a_graph_edge_justifies_it():
    graph = nx.Graph()
    graph.add_node("a", state="s1")
    graph.add_node("b", state="s2")
    graph.add_edge("a", "b")
    hex_to_node = {(0, 0): "a", (1, 0): "b"}

    cost = _hex_defect_cost((0, 0), hex_to_node, graph)

    assert cost == 0


def test_hex_defect_cost_does_not_flag_a_normal_state_border():
    # b sits right on the border between s1 and s2: it has a same-state
    # neighbour (a) AND a different-state neighbour (c) with no graph edge.
    # That's ordinary geography (two states sharing a coastline), not a
    # defect -- only *zero* same-state/edge-justified neighbours should be
    # flagged.
    graph = nx.Graph()
    graph.add_node("a", state="s1")
    graph.add_node("b", state="s1")
    graph.add_node("c", state="s2")
    hex_to_node = {(-1, 0): "a", (0, 0): "b", (1, 0): "c"}

    cost = _hex_defect_cost((0, 0), hex_to_node, graph)

    assert cost == 0


def test_hex_defect_cost_flags_a_mostly_enclosed_empty_hex_as_a_hole():
    ring = _axial_neighbors((0, 0))
    graph = nx.Graph()
    hex_to_node = {}
    for i, h in enumerate(ring[:HOLE_NEIGHBOUR_THRESHOLD]):
        graph.add_node(f"n{i}", state="s1")
        hex_to_node[h] = f"n{i}"

    cost = _hex_defect_cost((0, 0), hex_to_node, graph)

    assert cost == HOLE_PENALTY


def test_hex_defect_cost_ignores_a_lightly_enclosed_empty_hex():
    ring = _axial_neighbors((0, 0))
    graph = nx.Graph()
    hex_to_node = {}
    for i, h in enumerate(ring[: HOLE_NEIGHBOUR_THRESHOLD - 1]):
        graph.add_node(f"n{i}", state="s1")
        hex_to_node[h] = f"n{i}"

    cost = _hex_defect_cost((0, 0), hex_to_node, graph)

    assert cost == 0


def test_hexes_within_radius_returns_the_first_ring_only():
    result = _hexes_within_radius((0, 0), 1)

    assert set(result) == set(_axial_neighbors((0, 0)))


def test_hexes_within_radius_expands_with_larger_radius():
    result = _hexes_within_radius((0, 0), 2)

    assert len(result) == 6 + 12  # ring 1 has 6 hexes, ring 2 has 12
    assert all(_axial_distance((0, 0), h) <= 2 for h in result)


def test_swap_hex_contents_exchanges_two_districts():
    positions = {"a": (0, 0), "b": (1, 0)}
    hex_to_node = {(0, 0): "a", (1, 0): "b"}

    _swap_hex_contents((0, 0), (1, 0), hex_to_node, positions)

    assert positions == {"a": (1, 0), "b": (0, 0)}
    assert hex_to_node == {(0, 0): "b", (1, 0): "a"}


def test_swap_hex_contents_relocates_into_an_empty_hex():
    positions = {"a": (0, 0)}
    hex_to_node = {(0, 0): "a"}

    _swap_hex_contents((0, 0), (1, 0), hex_to_node, positions)

    assert positions == {"a": (1, 0)}
    assert hex_to_node == {(1, 0): "a"}


def test_swap_hex_contents_is_its_own_inverse():
    positions = {"a": (0, 0), "b": (1, 0)}
    hex_to_node = {(0, 0): "a", (1, 0): "b"}

    _swap_hex_contents((0, 0), (1, 0), hex_to_node, positions)
    _swap_hex_contents((0, 0), (1, 0), hex_to_node, positions)

    assert positions == {"a": (0, 0), "b": (1, 0)}
    assert hex_to_node == {(0, 0): "a", (1, 0): "b"}


def test_has_same_province_neighbour_true_when_one_is_adjacent():
    graph = nx.Graph()
    graph.add_node("a", province="p1")
    graph.add_node("b", province="p1")
    hex_to_node = {(0, 0): "a", (1, 0): "b"}

    assert _has_same_province_neighbour("a", (0, 0), hex_to_node, graph) is True


def test_has_same_province_neighbour_false_when_none_match():
    graph = nx.Graph()
    graph.add_node("a", province="p1")
    graph.add_node("c", province="p2")
    hex_to_node = {(0, 0): "a", (1, -1): "c"}

    assert _has_same_province_neighbour("a", (0, 0), hex_to_node, graph) is False


def test_has_same_province_neighbour_false_for_an_empty_hex():
    graph = nx.Graph()
    hex_to_node: dict = {}

    assert _has_same_province_neighbour(None, (0, 0), hex_to_node, graph) is False


def test_is_swap_candidate_accepts_occupied_hexes():
    hex_to_node = {(0, 0): "a"}

    assert _is_swap_candidate((0, 0), hex_to_node) is True


def test_is_swap_candidate_rejects_open_water():
    # (5, 5) is far from the only occupied hex, so it has zero occupied
    # neighbours -- swapping into it would be an escape into isolation, not
    # a real fix.
    hex_to_node = {(0, 0): "a"}

    assert _is_swap_candidate((5, 5), hex_to_node) is False


def test_is_swap_candidate_accepts_a_hex_with_enough_occupied_neighbours():
    ring = _axial_neighbors((0, 0))
    hex_to_node = {h: f"n{i}" for i, h in enumerate(ring[:MIN_NEIGHBOURS_FOR_CANDIDACY])}

    assert _is_swap_candidate((0, 0), hex_to_node) is True


def test_best_swap_candidate_fixes_a_stranded_district():
    graph = nx.Graph()
    graph.add_node("s1-1", state="s1", province="p1", elevation=100)
    graph.add_node("s1-2", state="s1", province="p1", elevation=90)
    graph.add_node("s1-3", state="s1", province="p1", elevation=80)
    graph.add_node("s1-x", state="s1", province="p1", elevation=70)
    graph.add_node("s2-bridge", state="s2", province="q1", elevation=60)
    for i, name in enumerate(("s2-a", "s2-b", "s2-c", "s2-d", "s2-e", "s2-f", "s2-g")):
        graph.add_node(name, state="s2", province="q1", elevation=50 - i)

    hex_to_node = {
        (-3, 0): "s1-1",
        (-2, 0): "s1-2",
        (-1, 0): "s1-3",
        (2, 0): "s1-x",  # stranded: every neighbour below is state s2
        (0, 0): "s2-bridge",  # adjacent to s1-3, one mismatch of its own
        (1, 0): "s2-a",
        (1, -1): "s2-b",
        (2, -1): "s2-c",
        (2, 1): "s2-d",
        (1, 1): "s2-e",
        (3, 0): "s2-f",
        (3, -1): "s2-g",
    }
    positions = {node: h for h, node in hex_to_node.items()}
    rng = random.Random(0)  # noqa: S311 -- layout tuning, not a security context

    candidate = _best_swap_candidate(
        (2, 0), hex_to_node, positions, graph, rng, search_radius=3, tolerance=0.0
    )

    assert candidate == (0, 0)  # s2-bridge's hex: fixes s1-x and stays cost-free for s2-bridge


def test_refine_hex_layout_fixes_a_stranded_district():
    graph = nx.Graph()
    graph.add_node("s1-1", state="s1", province="p1", elevation=100)
    graph.add_node("s1-2", state="s1", province="p1", elevation=90)
    graph.add_node("s1-3", state="s1", province="p1", elevation=80)
    graph.add_node("s1-x", state="s1", province="p1", elevation=70)
    graph.add_node("s2-bridge", state="s2", province="q1", elevation=60)
    for i, name in enumerate(("s2-a", "s2-b", "s2-c", "s2-d", "s2-e", "s2-f", "s2-g")):
        graph.add_node(name, state="s2", province="q1", elevation=50 - i)

    positions = {
        "s1-1": (-3, 0),
        "s1-2": (-2, 0),
        "s1-3": (-1, 0),
        "s1-x": (2, 0),
        "s2-bridge": (0, 0),
        "s2-a": (1, 0),
        "s2-b": (1, -1),
        "s2-c": (2, -1),
        "s2-d": (2, 1),
        "s2-e": (1, 1),
        "s2-f": (3, 0),
        "s2-g": (3, -1),
    }

    refined = refine_hex_layout(positions, graph)
    hex_to_node = {h: node for node, h in refined.items()}

    s1x_hex = refined["s1-x"]
    s1x_neighbour_states = {
        graph.nodes[hex_to_node[n]]["state"] for n in _axial_neighbors(s1x_hex) if n in hex_to_node
    }
    assert "s1" in s1x_neighbour_states  # no longer stranded
    assert set(refined) == set(positions)  # every district still placed exactly once
    assert len(set(refined.values())) == len(refined)  # no duplicate hexes


def test_refine_hex_layout_fills_an_interior_hole():
    # (0, 0) is empty but 5 of its 6 neighbours are occupied by the same
    # state/province -- refine should pull one of them inward to fill it,
    # trading a deep interior hole for a harmless recess at the blob's edge.
    graph = nx.Graph()
    ring = _axial_neighbors((0, 0))
    for i in range(5):
        graph.add_node(f"core-{i}", state="s1", province="p1", elevation=100 - i)

    positions = {f"core-{i}": ring[i] for i in range(5)}

    refined = refine_hex_layout(positions, graph, search_radius=1)

    assert (0, 0) in refined.values()
    assert set(refined) == set(positions)


def test_refine_hex_layout_is_a_noop_when_nothing_is_wrong():
    graph = nx.Graph()
    graph.add_node("a", state="s1", province="p1", elevation=100)
    graph.add_node("b", state="s1", province="p1", elevation=90)
    graph.add_edge("a", "b")
    positions = {"a": (0, 0), "b": (1, 0)}

    refined = refine_hex_layout(positions, graph)

    assert refined == positions


def test_refine_hex_layout_does_not_mutate_the_input():
    graph = nx.Graph()
    graph.add_node("a", state="s1", province="p1", elevation=100)
    graph.add_node("b", state="s1", province="p1", elevation=90)
    graph.add_edge("a", "b")
    positions = {"a": (0, 0), "b": (1, 0)}
    original = dict(positions)

    refine_hex_layout(positions, graph)

    assert positions == original
