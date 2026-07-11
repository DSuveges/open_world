"""Hex-grid island placement: real, non-overlapping district positions.

Unlike the graph-diagnostic layouts in :mod:`open_world.layout.clustered`
and :mod:`open_world.layout.placeholder`, this produces genuine spatial
positions meant to look like a map: every district occupies exactly one hex
cell, each state grows into a single contiguous island, and islands are
packed onto the plane with a water gap between them so they never touch.

Algorithm, per state:

1. Seed the state's highest-elevation district at the island's local origin
   (0, 0) -- the "peak".
2. Walk the rest of the state's districts in *descending* elevation order,
   placing each into an empty hex adjacent to one of its already-placed
   graph neighbours (falling back to any already-placed same-state
   district if none of its neighbours have been placed yet). Because
   placement always attaches to the current edge of the growing island and
   proceeds from high to low elevation, elevation naturally decreases
   outward -- the coastline (the districts with no empty hex-neighbours
   left, i.e. the rim) ends up low-elevation, matching "the lowest
   elevation districts should be around the seashore".

Only same-state graph edges are used for placement, even though the graph
itself may have a few deliberate cross-state edges (mountain ranges) -- a
literal cross-state hex-adjacency would mean two islands touch, contradicting
"every state is an island surrounded by water". Those edges remain part of
the abstract graph; they just aren't materialized spatially here.

Islands are then packed via an expanding-ring search so no two island
bounding circles overlap by less than a configurable water gap.
"""

import math
from collections.abc import Iterable

import networkx as nx
from loguru import logger

from open_world.data.schema import ELEVATION, STATE

Axial = tuple[int, int]

HEX_NEIGHBOR_OFFSETS: tuple[Axial, ...] = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))

DEFAULT_WATER_GAP = 3.0


def compute_hex_layout(graph: nx.Graph, water_gap: float = DEFAULT_WATER_GAP) -> dict[str, Axial]:
    """Place every district on a hex grid, one island per state.

    Args:
        graph: Graph whose nodes carry ``state`` and ``elevation``
            attributes.
        water_gap: Minimum hex-distance gap to leave between island
            bounding circles.

    Returns:
        Mapping of node id to (q, r) axial hex coordinates.
    """
    members_by_state = _group_by_state(graph)

    local_positions: dict[str, dict[str, Axial]] = {}
    radius_by_state: dict[str, float] = {}
    for state, members in members_by_state.items():
        placement = _place_state(members, graph)
        local_positions[state] = placement
        radius_by_state[state] = max(
            (_axial_distance((0, 0), pos) for pos in placement.values()), default=0
        )

    order = sorted(radius_by_state, key=lambda state: radius_by_state[state], reverse=True)
    offsets = _pack_islands(order, radius_by_state, water_gap)

    positions: dict[str, Axial] = {}
    for state, placement in local_positions.items():
        offset_q, offset_r = offsets[state]
        for node, (local_q, local_r) in placement.items():
            positions[node] = (local_q + offset_q, local_r + offset_r)

    logger.info(
        "computed hex layout ({} states, {} districts, water_gap={})",
        len(members_by_state),
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


def _place_state(members: list[str], graph: nx.Graph) -> dict[str, Axial]:
    """Grow one state's districts into a single contiguous hex island.

    Args:
        members: DistrictIds belonging to this state.
        graph: Full graph, used to find each district's same-state
            neighbours.

    Returns:
        Mapping of node id to (q, r) axial coordinates, local to this
        state's island (seeded at the origin).
    """
    state = graph.nodes[members[0]][STATE]
    ordered = sorted(members, key=lambda node: graph.nodes[node][ELEVATION], reverse=True)

    positions: dict[str, Axial] = {ordered[0]: (0, 0)}
    occupied: set[Axial] = {(0, 0)}
    frontier: set[Axial] = set(_axial_neighbors((0, 0)))

    for district in ordered[1:]:
        same_state_neighbours = [
            neighbour
            for neighbour in graph.neighbors(district)
            if graph.nodes[neighbour][STATE] == state and neighbour in positions
        ]
        hex_position = _pick_adjacent_empty_hex(same_state_neighbours, positions, frontier)
        positions[district] = hex_position
        occupied.add(hex_position)
        frontier.discard(hex_position)
        frontier.update(
            neighbour for neighbour in _axial_neighbors(hex_position) if neighbour not in occupied
        )

    return positions


def _pick_adjacent_empty_hex(
    placed_neighbours: list[str], positions: dict[str, Axial], frontier: set[Axial]
) -> Axial:
    """Choose a hex to place a district in, preferring next to a real neighbour.

    Args:
        placed_neighbours: Already-placed graph neighbours of the district
            being placed, restricted to the same state.
        positions: DistrictId to (q, r) for everything placed so far.
        frontier: Empty hexes currently adjacent to the placed region.

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
    order: list[str], radius_by_state: dict[str, float], water_gap: float
) -> dict[str, Axial]:
    """Pack island bounding circles onto the plane without overlap.

    Places the largest island first, then searches outward in expanding
    rings for the nearest position that doesn't overlap any already-placed
    island (plus `water_gap`).

    Args:
        order: State names, largest radius first.
        radius_by_state: Mapping of state name to its island's bounding
            radius (in hex units).
        water_gap: Minimum gap to leave between island bounding circles.

    Returns:
        Mapping of state name to an integer (q, r) hex offset for that
        island's center.
    """
    placed: dict[str, Axial] = {}
    placed_circles: list[tuple[float, float, float]] = []

    for state in order:
        state_radius = radius_by_state[state]
        center = (
            (0.0, 0.0)
            if not placed_circles
            else _find_free_position(state_radius, placed_circles, water_gap)
        )
        placed[state] = (round(center[0]), round(center[1]))
        placed_circles.append((center[0], center[1], state_radius))

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
