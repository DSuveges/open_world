"""Hex-grid island placement: real, non-overlapping district positions.

Unlike the graph-diagnostic layouts in :mod:`open_world.layout.clustered`
and :mod:`open_world.layout.placeholder`, this produces genuine spatial
positions meant to look like a map: every district occupies exactly one hex
cell, and islands are packed onto the plane with a water gap between them so
they never touch.

The placement hierarchy has three levels: landmass -> state -> province.

1. **Landmass grouping.** States connected by at least one cross-state graph
   edge (see :mod:`open_world.graph.neighbours`'s ``min_boundary_fraction``)
   are grouped into one shared landmass instead of always getting separate
   islands -- connected components of the state-level adjacency graph. A
   state with no cross-state edges is its own landmass, same as before.
2. **State growth.** Within a landmass, states are visited in an order that
   follows cross-state graph edges (largest-first Prim's-style growth from
   the landmass's overall peak), each seeded next to an *anchor* hex chosen
   to minimize the elevation gap across the seam.
3. **Province growth.** Within a state, provinces are grown the same way,
   one state level down, using same-state edges only (so a province never
   anchors onto a neighbouring state's territory instead of its own state's).
   Each province is grown as a single contiguous hex blob: seed the
   province's highest-elevation district, then walk the rest in
   *descending* elevation order, attaching each to an empty hex next to an
   already-placed same-province graph neighbour.

Elevation decreases outward at every level because growth always proceeds
from high to low and always attaches to the current edge of what's already
placed -- so the coastline (hexes with no empty neighbours left) ends up
low-elevation, matching "the lowest elevation districts should be around the
seashore", and a mountain range spanning a province or state boundary
attaches high-side-to-high-side instead of leaving a low-elevation gap
through the middle.

Landmasses are packed via an expanding-ring search so no two landmass
bounding circles overlap by less than a configurable water gap.
"""

import heapq
import math
from collections import deque
from collections.abc import Callable, Iterable

import networkx as nx
from loguru import logger

from open_world.data.schema import ELEVATION, PROVINCE, STATE

Axial = tuple[int, int]

HEX_NEIGHBOR_OFFSETS: tuple[Axial, ...] = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))

DEFAULT_WATER_GAP = 3.0


def compute_hex_layout(graph: nx.Graph, water_gap: float = DEFAULT_WATER_GAP) -> dict[str, Axial]:
    """Place every district on a hex grid, grouping cross-state-connected states.

    Args:
        graph: Graph whose nodes carry ``state``, ``province`` and
            ``elevation`` attributes.
        water_gap: Minimum hex-distance gap to leave between landmass
            bounding circles.

    Returns:
        Mapping of node id to (q, r) axial hex coordinates.
    """
    members_by_state = _group_by_state(graph)
    landmasses = _group_into_landmasses(members_by_state, graph)

    local_positions: dict[str, dict[str, Axial]] = {}
    radius_by_landmass: dict[str, float] = {}
    for states in landmasses:
        landmass_id, placement = _place_landmass(states, members_by_state, graph)
        local_positions[landmass_id] = placement
        radius_by_landmass[landmass_id] = max(
            (_axial_distance((0, 0), pos) for pos in placement.values()), default=0
        )

    order = sorted(radius_by_landmass, key=lambda name: radius_by_landmass[name], reverse=True)
    offsets = _pack_islands(order, radius_by_landmass, water_gap)

    positions: dict[str, Axial] = {}
    for landmass_id, placement in local_positions.items():
        offset_q, offset_r = offsets[landmass_id]
        for node, (local_q, local_r) in placement.items():
            positions[node] = (local_q + offset_q, local_r + offset_r)

    logger.info(
        "computed hex layout ({} states, {} landmasses, {} districts, water_gap={})",
        len(members_by_state),
        len(landmasses),
        graph.number_of_nodes(),
        water_gap,
    )
    return positions


def apply_hex_positions(graph: nx.Graph, positions: dict[str, Axial]) -> None:
    """Store computed hex coordinates as ``q``/``r`` node attributes, in place.

    Args:
        graph: Graph to mutate.
        positions: Mapping of node id to (q, r), e.g. from
            :func:`compute_hex_layout`.
    """
    for node, (q, r) in positions.items():
        graph.nodes[node]["q"] = q
        graph.nodes[node]["r"] = r


def _group_by_state(graph: nx.Graph) -> dict[str, list[str]]:
    """Bucket node ids by their `state` attribute.

    Args:
        graph: Graph whose nodes carry a ``state`` attribute.

    Returns:
        Mapping of state name to its member node ids.
    """
    members: dict[str, list[str]] = {}
    for node, attrs in graph.nodes(data=True):
        members.setdefault(attrs[STATE], []).append(node)
    return members


def _group_by_province(members: list[str], graph: nx.Graph) -> dict[str, list[str]]:
    """Bucket a state's node ids by their `province` attribute.

    Args:
        members: DistrictIds belonging to a single state.
        graph: Graph whose nodes carry a ``province`` attribute.

    Returns:
        Mapping of province name to its member node ids.
    """
    provinces: dict[str, list[str]] = {}
    for node in members:
        provinces.setdefault(graph.nodes[node][PROVINCE], []).append(node)
    return provinces


def _group_into_landmasses(
    members_by_state: dict[str, list[str]], graph: nx.Graph
) -> list[list[str]]:
    """Group states sharing a cross-state edge into one landmass each.

    A state with no cross-state edge to any other state is its own,
    single-state landmass -- the same "one island per state" behaviour as
    before this existed.

    Args:
        members_by_state: State name to member node ids.
        graph: Full graph, used to find cross-state edges.

    Returns:
        Each landmass as a list of state names.
    """
    state_of = {node: state for state, members in members_by_state.items() for node in members}
    state_graph = nx.Graph()
    state_graph.add_nodes_from(members_by_state)
    for source, target in graph.edges():
        state_a, state_b = state_of[source], state_of[target]
        if state_a != state_b:
            state_graph.add_edge(state_a, state_b)
    return [sorted(component) for component in nx.connected_components(state_graph)]


def _place_landmass(
    states: list[str], members_by_state: dict[str, list[str]], graph: nx.Graph
) -> tuple[str, dict[str, Axial]]:
    """Grow one landmass's districts into a hex island, one blob per state.

    Args:
        states: State names sharing this landmass.
        members_by_state: State name to member node ids.
        graph: Full graph, used to find each district's neighbours.

    Returns:
        (landmass id, positions). The landmass id is the peak state's name,
        so a single-state landmass keeps the old per-state identity.
    """
    groups = {state: members_by_state[state] for state in states}
    order = _growth_order(groups, graph)

    positions: dict[str, Axial] = {}
    occupied: set[Axial] = set()

    for index, state in enumerate(order):
        state_members = groups[state]
        seed = (0, 0) if index == 0 else _find_anchor_hex(state_members, positions, occupied, graph)
        _place_state_into(state_members, graph, positions, occupied, seed)

    return order[0], positions


def _place_state_into(
    members: list[str],
    graph: nx.Graph,
    positions: dict[str, Axial],
    occupied: set[Axial],
    seed: Axial,
) -> None:
    """Grow one state's districts into a hex blob, one contiguous blob per province.

    Mutates `positions` and `occupied` in place, so this can be called
    multiple times to grow several states into one shared landmass.

    Args:
        members: DistrictIds belonging to this state.
        graph: Full graph, used to find each district's neighbours.
        positions: DistrictId to (q, r), shared across the whole landmass.
        occupied: Every hex already taken, shared across the whole landmass.
        seed: Hex to anchor this state's first (peak) province at.
    """
    state = graph.nodes[members[0]][STATE]
    provinces = _group_by_province(members, graph)
    order = _growth_order(provinces, graph)

    def _same_state(node: str) -> bool:
        return graph.nodes[node][STATE] == state

    for index, province in enumerate(order):
        province_members = provinces[province]
        if index == 0:
            province_seed = seed
        else:
            province_seed = _find_anchor_hex(
                province_members, positions, occupied, graph, is_valid_neighbour=_same_state
            )
        _grow_blob_into(province_members, graph, positions, occupied, province_seed, PROVINCE)


def _growth_order(groups: dict[str, list[str]], graph: nx.Graph) -> list[str]:
    """Order groups (provinces or states) so each (after the first) has a visited neighbour.

    Starts from the group containing the overall highest-elevation district
    (the "peak"), then repeatedly visits the *largest* group reachable from
    what's already been visited (a Prim's-algorithm-style growth,
    prioritized by size rather than edge weight). Because connectivity at
    this level is already guaranteed elsewhere (state connectivity by
    :func:`open_world.graph.neighbours.repair_connectivity`; landmass
    connectivity by construction, since a landmass is a connected
    component), this reaches every group. Visiting large groups first,
    while the map is still mostly open, measurably reduces how often a
    group later gets boxed in by its already-placed neighbours with no room
    left to grow into.

    Args:
        groups: Group name (province or state) to member node ids.
        graph: Full graph, used to find inter-group edges.

    Returns:
        Group names in growth order.
    """
    member_of = {node: group for group, members in groups.items() for node in members}
    adjacency: dict[str, set[str]] = {group: set() for group in groups}
    all_members = list(member_of)
    for source, target in graph.edges(all_members):
        group_a, group_b = member_of.get(source), member_of.get(target)
        if group_a is not None and group_b is not None and group_a != group_b:
            adjacency[group_a].add(group_b)
            adjacency[group_b].add(group_a)

    def _by_size_desc(group: str) -> tuple[int, str]:
        return (-len(groups[group]), group)

    start = max(groups, key=lambda g: max(graph.nodes[d][ELEVATION] for d in groups[g]))

    order = [start]
    visited = {start}
    heap = [(-len(groups[g]), g) for g in adjacency[start]]
    heapq.heapify(heap)
    while heap:
        _, current = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        order.append(current)
        for neighbour in adjacency[current]:
            if neighbour not in visited:
                heapq.heappush(heap, (-len(groups[neighbour]), neighbour))

    # Groups with no edge to anything already visited shouldn't happen given
    # the connectivity guarantees above, but appending any stragglers keeps
    # this robust rather than silently dropping districts.
    for group in sorted(groups, key=_by_size_desc):
        if group not in visited:
            order.append(group)
            visited.add(group)

    return order


def _find_anchor_hex(
    members: list[str],
    positions: dict[str, Axial],
    occupied: set[Axial],
    graph: nx.Graph,
    is_valid_neighbour: Callable[[str], bool] | None = None,
) -> Axial:
    """Find where a new group (province or state) should start growing from.

    Looks at every graph edge from this group to an already-placed
    neighbour and picks the smallest-elevation-gap pair, so the new group's
    growth begins right where it's most similar in elevation to its
    neighbour -- keeping a shared mountain range's high side attached to
    the other high side, rather than seaming at an arbitrary point.

    Args:
        members: DistrictIds of the group being placed.
        positions: DistrictId to (q, r) for everything placed so far.
        occupied: Every hex already taken, across all groups so far.
        graph: Full graph, used to find inter-group edges.
        is_valid_neighbour: Optional filter an already-placed neighbour must
            satisfy to be considered. Used when placing a province within a
            state whose `positions` may already contain *other states*
            (sharing a landmass), so a province only anchors onto its own
            state's territory.

    Returns:
        An empty hex to seed this group's growth from.
    """
    best_gap: int | None = None
    best_anchor: Axial | None = None
    for district in members:
        elevation = graph.nodes[district][ELEVATION]
        for neighbour in graph.neighbors(district):
            if neighbour not in positions:
                continue
            if is_valid_neighbour is not None and not is_valid_neighbour(neighbour):
                continue
            gap = abs(graph.nodes[neighbour][ELEVATION] - elevation)
            if best_gap is not None and gap >= best_gap:
                continue
            empty_candidates = [
                h for h in _axial_neighbors(positions[neighbour]) if h not in occupied
            ]
            if empty_candidates:
                best_gap = gap
                best_anchor = empty_candidates[0]

    if best_anchor is not None:
        return best_anchor

    # No already-placed neighbour had room right next to it (fully boxed in);
    # fall back to the nearest empty hex to whatever's already on the map.
    return _nearest_empty_hex(next(iter(positions.values())), occupied)


def _nearest_empty_hex(start: Axial, occupied: set[Axial]) -> Axial:
    """Breadth-first search for the closest unoccupied hex to `start`.

    Args:
        start: Hex to search outward from.
        occupied: Every hex already taken.

    Returns:
        The nearest hex (possibly `start` itself) not in `occupied`.
    """
    if start not in occupied:
        return start
    return _nearest_empty_hex_from_any({start}, occupied)


def _nearest_empty_hex_from_any(sources: set[Axial], occupied: set[Axial]) -> Axial:
    """Multi-source breadth-first search for the closest unoccupied hex.

    Args:
        sources: Hexes to search outward from simultaneously.
        occupied: Every hex already taken.

    Returns:
        The nearest hex not in `occupied`, reachable from any of `sources`.
    """
    seen = set(sources)
    frontier = deque(sources)
    while frontier:
        current = frontier.popleft()
        for neighbour in _axial_neighbors(current):
            if neighbour not in occupied:
                return neighbour
            if neighbour not in seen:
                seen.add(neighbour)
                frontier.append(neighbour)
    msg = "unreachable: an infinite hex grid always has an empty neighbour"
    raise RuntimeError(msg)


def _grow_blob_into(  # noqa: PLR0913 -- each arg is independently necessary here
    members: list[str],
    graph: nx.Graph,
    positions: dict[str, Axial],
    occupied: set[Axial],
    seed: Axial,
    group_key: str,
) -> None:
    """Grow one contiguous blob of `members`, starting at `seed`.

    Mutates `positions` and `occupied` in place so blobs grown one after
    another share a single global occupancy map and never overlap.

    Args:
        members: DistrictIds to place (e.g. one province's districts).
        graph: Full graph, used to find each district's neighbours.
        positions: DistrictId to (q, r), shared across all blobs so far.
        occupied: Every hex already taken, shared across all blobs so far.
        seed: Hex to place the highest-elevation member at (nudged to the
            nearest empty hex if already taken).
        group_key: Node attribute that must match for two districts to
            count as neighbours during this growth (e.g. ``province``).
    """
    group_value = graph.nodes[members[0]][group_key]
    ordered = sorted(members, key=lambda node: graph.nodes[node][ELEVATION], reverse=True)

    first_hex = _nearest_empty_hex(seed, occupied)
    positions[ordered[0]] = first_hex
    occupied.add(first_hex)
    frontier: set[Axial] = {h for h in _axial_neighbors(first_hex) if h not in occupied}
    blob_hexes: set[Axial] = {first_hex}

    for district in ordered[1:]:
        same_group_neighbours = [
            neighbour
            for neighbour in graph.neighbors(district)
            if graph.nodes[neighbour].get(group_key) == group_value and neighbour in positions
        ]
        if not frontier:
            # Boxed in by other already-placed provinces/states with no room
            # left on this blob's own edge. Escape by jumping to the nearest
            # empty hex reachable from the blob's territory -- rare, and
            # means this district ends up spatially detached from the rest
            # of its group, but keeps placement total instead of crashing.
            escape = _nearest_empty_hex_from_any(blob_hexes, occupied)
            frontier.add(escape)
            logger.warning(
                "{} {!r} ran out of room while placing {}; jumped to {}",
                group_key,
                group_value,
                district,
                escape,
            )
        hex_position = _pick_adjacent_empty_hex(same_group_neighbours, positions, frontier)
        positions[district] = hex_position
        occupied.add(hex_position)
        blob_hexes.add(hex_position)
        frontier.discard(hex_position)
        frontier.update(
            neighbour for neighbour in _axial_neighbors(hex_position) if neighbour not in occupied
        )


def _pick_adjacent_empty_hex(
    placed_neighbours: list[str], positions: dict[str, Axial], frontier: set[Axial]
) -> Axial:
    """Choose a hex to place a district in, preferring next to a real neighbour.

    Args:
        placed_neighbours: Already-placed graph neighbours of the district
            being placed, restricted to the relevant group (e.g. province).
        positions: DistrictId to (q, r) for everything placed so far.
        frontier: Empty hexes currently adjacent to the growing blob.

    Returns:
        An empty hex adjacent to one of `placed_neighbours` if any exist
        there, otherwise an arbitrary hex from `frontier`.
    """
    for neighbour in placed_neighbours:
        for candidate in _axial_neighbors(positions[neighbour]):
            if candidate in frontier:
                return candidate
    return next(iter(frontier))


def _axial_neighbors(hex_position: Axial) -> Iterable[Axial]:
    """Yield the 6 axial neighbours of a hex.

    Args:
        hex_position: (q, r) to find neighbours of.

    Returns:
        The 6 adjacent axial coordinates.
    """
    q, r = hex_position
    return [(q + dq, r + dr) for dq, dr in HEX_NEIGHBOR_OFFSETS]


def _axial_distance(a: Axial, b: Axial) -> int:
    """Hex grid distance between two axial coordinates.

    Args:
        a: First hex.
        b: Second hex.

    Returns:
        The number of hex steps between `a` and `b`.
    """
    aq, ar = a
    bq, br = b
    return (abs(aq - bq) + abs(aq + ar - bq - br) + abs(ar - br)) // 2


def _pack_islands(
    order: list[str], radius_by_landmass: dict[str, float], water_gap: float
) -> dict[str, Axial]:
    """Pack landmass bounding circles onto the plane without overlap.

    Places the largest landmass first, then searches outward in expanding
    rings for the nearest position that doesn't overlap any already-placed
    landmass (plus `water_gap`).

    Args:
        order: Landmass ids, largest radius first.
        radius_by_landmass: Mapping of landmass id to its bounding radius
            (in hex units).
        water_gap: Minimum gap to leave between landmass bounding circles.

    Returns:
        Mapping of landmass id to an integer (q, r) hex offset for that
        landmass's center.
    """
    placed: dict[str, Axial] = {}
    placed_circles: list[tuple[float, float, float]] = []

    for landmass_id in order:
        radius = radius_by_landmass[landmass_id]
        center = (
            (0.0, 0.0)
            if not placed_circles
            else _find_free_position(radius, placed_circles, water_gap)
        )
        placed[landmass_id] = (round(center[0]), round(center[1]))
        placed_circles.append((center[0], center[1], radius))

    return placed


def _find_free_position(
    new_radius: float, placed_circles: list[tuple[float, float, float]], gap: float
) -> tuple[float, float]:
    """Find the nearest non-overlapping position for a new circle.

    Args:
        new_radius: Radius of the circle being placed.
        placed_circles: (x, y, radius) of every circle placed so far.
        gap: Minimum required gap between circle edges.

    Returns:
        An (x, y) center that doesn't overlap any placed circle.
    """
    step = max((r for _, _, r in placed_circles), default=1.0) * 0.5 + gap
    search_radius = new_radius + gap
    while True:
        angle_count = max(8, int(search_radius))
        for i in range(angle_count):
            angle = 2 * math.pi * i / angle_count
            candidate = (search_radius * math.cos(angle), search_radius * math.sin(angle))
            if all(
                _distance(candidate, (ox, oy)) >= new_radius + other_radius + gap
                for ox, oy, other_radius in placed_circles
            ):
                return candidate
        search_radius += step


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points.

    Args:
        a: First point.
        b: Second point.

    Returns:
        The straight-line distance between `a` and `b`.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])
