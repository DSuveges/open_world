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

4. **Refinement.** Greedy growth occasionally boxes a province in and has to
   jump it to the nearest free hex (see :func:`_grow_blob_into`'s escape
   hatch), which can strand a district deep inside a different state's
   territory or leave small unreachable pockets ("holes") behind. Each
   landmass's placement is passed through :func:`refine_hex_layout`, a
   bounded, targeted local search that repeatedly looks for nearby hex
   swaps that reduce a small cost function (state-mismatched neighbours,
   holes, and a light elevation-continuity term) before the layout is
   finalized.
"""

import heapq
import math
import random
from collections import deque
from collections.abc import Callable, Iterable

import networkx as nx
from loguru import logger

from open_world.data.schema import ELEVATION, PROVINCE, STATE

Axial = tuple[int, int]

HEX_NEIGHBOR_OFFSETS: tuple[Axial, ...] = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))

DEFAULT_WATER_GAP = 3.0

DEFAULT_REFINE_MAX_PASSES = 6
DEFAULT_REFINE_SEARCH_RADIUS = 3
DEFAULT_REFINE_SEED = 42
ISOLATION_PENALTY = 10.0
HOLE_PENALTY = 10.0
HOLE_NEIGHBOUR_THRESHOLD = 5
ELEVATION_WEIGHT = 0.01
MIN_NEIGHBOURS_FOR_CANDIDACY = 2


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
        placement = refine_hex_layout(placement, graph)
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


def refine_hex_layout(
    positions: dict[str, Axial],
    graph: nx.Graph,
    max_passes: int = DEFAULT_REFINE_MAX_PASSES,
    search_radius: int = DEFAULT_REFINE_SEARCH_RADIUS,
    seed: int = DEFAULT_REFINE_SEED,
) -> dict[str, Axial]:
    """Locally repair stranded districts and interior holes left by greedy growth.

    Greedy hex-by-hex growth (see :func:`_grow_blob_into`) occasionally boxes
    a province in and has to jump it to the nearest free hex, which can land
    a district deep inside a different state's territory (a "stranded"
    district, with zero same-state hex-neighbours) or leave small
    mostly-enclosed empty pockets behind (a "hole"). This runs a bounded,
    targeted local search over just those defects: each pass finds every
    hex with a nonzero defect cost and looks for a nearby hex to swap
    contents with that reduces total cost, repeating until nothing improves
    or `max_passes` is reached. Early passes tolerate a small non-improving
    swap (a mild annealing-style cooling schedule) to escape local optima
    that need two swaps to fix; later passes only accept strict
    improvements. Cost also includes a small elevation-continuity term so a
    fix doesn't undo Stage 3's mountain-range seam matching along the way,
    and every candidate swap is rejected outright if it would strip either
    moved district of a same-province hex-neighbour it currently has, so
    cleanup can't fragment an otherwise-contiguous province as a side
    effect.

    Args:
        positions: DistrictId to (q, r), as produced by growth (e.g. one
            landmass's local placement).
        graph: Full graph, used to check state/province/edge/elevation
            attributes.
        max_passes: Upper bound on refinement passes.
        search_radius: How many hex-steps away from a problem hex to look
            for a swap partner.
        seed: Random seed for candidate ordering and tie-breaking.

    Returns:
        A new positions mapping with the same keys; `positions` itself is
        not mutated.
    """
    positions = dict(positions)
    hex_to_node = {hex_position: node for node, hex_position in positions.items()}
    rng = random.Random(seed)  # noqa: S311 -- layout tuning, not a security context

    total_fixed = 0
    for pass_index in range(max_passes):
        tolerance = ISOLATION_PENALTY * (max_passes - pass_index - 1) / max_passes
        problem_hexes = _find_problem_hexes(hex_to_node, graph)
        if not problem_hexes:
            logger.info("hex layout refine: converged after {} pass(es)", pass_index)
            break
        rng.shuffle(problem_hexes)

        fixed_this_pass = 0
        for bad_hex in problem_hexes:
            candidate = _best_swap_candidate(
                bad_hex, hex_to_node, positions, graph, rng, search_radius, tolerance
            )
            if candidate is None:
                continue
            _swap_hex_contents(bad_hex, candidate, hex_to_node, positions)
            fixed_this_pass += 1
        total_fixed += fixed_this_pass
        logger.info(
            "hex layout refine pass {}: {} problem hexes, {} swaps applied",
            pass_index,
            len(problem_hexes),
            fixed_this_pass,
        )
        if fixed_this_pass == 0:
            break

    logger.info("hex layout refine: {} total swaps applied", total_fixed)
    return positions


def _find_problem_hexes(hex_to_node: dict[Axial, str], graph: nx.Graph) -> list[Axial]:
    """Find every occupied or hole hex with a nonzero defect cost.

    Args:
        hex_to_node: Hex to districtId.
        graph: Full graph, used to evaluate defect cost.

    Returns:
        Hexes worth attempting to fix (state-mismatched districts and
        mostly-enclosed empty hexes).
    """
    problems = []
    checked_holes: set[Axial] = set()
    for occupied_hex in hex_to_node:
        if _hex_defect_cost(occupied_hex, hex_to_node, graph) > 0:
            problems.append(occupied_hex)
        for neighbour in _axial_neighbors(occupied_hex):
            if neighbour in hex_to_node or neighbour in checked_holes:
                continue
            checked_holes.add(neighbour)
            if _hex_defect_cost(neighbour, hex_to_node, graph) > 0:
                problems.append(neighbour)
    return problems


def _hex_defect_cost(hex_position: Axial, hex_to_node: dict[Axial, str], graph: nx.Graph) -> float:
    """Isolation / hole cost of a hex -- the component that flags problems.

    A district touching a neighbouring state along a real border is normal
    geography, not a defect -- only a district with *zero* same-state (or
    graph-edge-justified) neighbours among its occupied neighbours counts as
    stranded, matching the diagnostic definition of "scattered in another
    state". Scoring every mismatched neighbour individually would flag
    thousands of ordinary border hexes across a large landmass, most of
    which no local swap could ever fix anyway.

    Args:
        hex_position: Hex to evaluate.
        hex_to_node: Hex to districtId.
        graph: Full graph, used to check state and edges.

    Returns:
        `ISOLATION_PENALTY` if `hex_position` is occupied, has at least one
        occupied neighbour, and none of them are same-state or
        edge-justified; `HOLE_PENALTY` if `hex_position` is empty and
        mostly enclosed by occupied hexes; otherwise 0.
    """
    occupant = hex_to_node.get(hex_position)
    if occupant is None:
        occupied_neighbours = sum(1 for n in _axial_neighbors(hex_position) if n in hex_to_node)
        return HOLE_PENALTY if occupied_neighbours >= HOLE_NEIGHBOUR_THRESHOLD else 0.0

    occupant_state = graph.nodes[occupant][STATE]
    has_occupied_neighbour = False
    for neighbour_hex in _axial_neighbors(hex_position):
        other = hex_to_node.get(neighbour_hex)
        if other is None:
            continue
        has_occupied_neighbour = True
        if graph.nodes[other][STATE] == occupant_state or graph.has_edge(occupant, other):
            return 0.0
    return ISOLATION_PENALTY if has_occupied_neighbour else 0.0


def _hex_cost(hex_position: Axial, hex_to_node: dict[Axial, str], graph: nx.Graph) -> float:
    """Full swap-evaluation cost: defect cost plus a light elevation regularizer.

    The elevation term never triggers `_find_problem_hexes` on its own (it's
    added only here, not in `_hex_defect_cost`) -- it just discourages a fix
    from landing a district somewhere that reintroduces an elevation seam
    gap, and breaks ties between otherwise-equal candidates.

    Args:
        hex_position: Hex to evaluate.
        hex_to_node: Hex to districtId.
        graph: Full graph, used to check state, edges and elevation.

    Returns:
        `_hex_defect_cost` plus `ELEVATION_WEIGHT` times the summed
        elevation gap to every occupied neighbour.
    """
    cost = _hex_defect_cost(hex_position, hex_to_node, graph)
    occupant = hex_to_node.get(hex_position)
    if occupant is None:
        return cost

    occupant_elevation = graph.nodes[occupant][ELEVATION]
    for neighbour_hex in _axial_neighbors(hex_position):
        other = hex_to_node.get(neighbour_hex)
        if other is None:
            continue
        cost += ELEVATION_WEIGHT * abs(graph.nodes[other][ELEVATION] - occupant_elevation)
    return cost


def _local_cost(hexes: set[Axial], hex_to_node: dict[Axial, str], graph: nx.Graph) -> float:
    """Sum `_hex_cost` over a set of hexes.

    Args:
        hexes: Hexes to evaluate.
        hex_to_node: Hex to districtId.
        graph: Full graph, used to evaluate cost.

    Returns:
        Total cost across `hexes`.
    """
    return sum(_hex_cost(h, hex_to_node, graph) for h in hexes)


def _affected_hexes(hex_position: Axial) -> set[Axial]:
    """Hexes whose cost could change if `hex_position`'s occupant changes.

    A hex's cost only ever depends on its own occupant and its 6 immediate
    neighbours' occupants, so this is exactly the set that needs
    re-evaluating around a single-hex content change.

    Args:
        hex_position: Hex whose contents are about to change.

    Returns:
        `hex_position` plus its 6 immediate neighbours.
    """
    return {hex_position, *_axial_neighbors(hex_position)}


def _is_swap_candidate(hex_position: Axial, hex_to_node: dict[Axial, str]) -> bool:
    """Whether a hex is worth considering as a swap target.

    Either it's already occupied (a genuine district-for-district swap), or
    it's an empty hex with enough occupied neighbours to be a real interior
    hole. Excludes open water far from any existing territory -- otherwise
    a stranded district could "fix" its mismatch cost by escaping into
    isolation instead of reattaching to its own territory, since an
    occupied hex with zero neighbours trivially has nothing to mismatch
    with.

    Args:
        hex_position: Hex to check.
        hex_to_node: Hex to districtId.

    Returns:
        `True` if `hex_position` is a legitimate swap target.
    """
    if hex_position in hex_to_node:
        return True
    occupied_neighbours = sum(1 for n in _axial_neighbors(hex_position) if n in hex_to_node)
    return occupied_neighbours >= MIN_NEIGHBOURS_FOR_CANDIDACY


def _best_swap_candidate(  # noqa: PLR0913 -- each arg is independently necessary here
    bad_hex: Axial,
    hex_to_node: dict[Axial, str],
    positions: dict[str, Axial],
    graph: nx.Graph,
    rng: random.Random,
    search_radius: int,
    tolerance: float,
) -> Axial | None:
    """Search nearby hexes for the best cost-reducing swap partner for `bad_hex`.

    Args:
        bad_hex: Hex flagged as having a nonzero defect cost.
        hex_to_node: Hex to districtId, kept in sync with `positions`.
        positions: DistrictId to (q, r), kept in sync with `hex_to_node`.
        graph: Full graph, used to evaluate cost and province membership.
        rng: Random source; only affects candidate order, and hence which
            equally-good candidate wins ties.
        search_radius: How many hex-steps away to look.
        tolerance: Largest cost delta still accepted (0 = strict
            improvement only; positive values tolerate a small regression).

    Returns:
        The best candidate hex to swap `bad_hex` with, or `None` if nothing
        within `tolerance` was found (after excluding any candidate that
        would strip a currently-attached district of its last same-province
        hex-neighbour, and any empty hex too far out in open water to be a
        real hole -- otherwise a stranded district could "fix" its mismatch
        cost by retreating to total isolation instead of reattaching to its
        own territory, which trivially has zero neighbours to mismatch
        with).
    """
    candidates = [
        h
        for h in _hexes_within_radius(bad_hex, search_radius)
        if _is_swap_candidate(h, hex_to_node)
    ]
    rng.shuffle(candidates)
    affected = _affected_hexes(bad_hex)
    node_bad = hex_to_node.get(bad_hex)

    best: Axial | None = None
    best_delta = tolerance
    for candidate in candidates:
        node_candidate = hex_to_node.get(candidate)
        affected_pair = affected | _affected_hexes(candidate)
        cost_before = _local_cost(affected_pair, hex_to_node, graph)
        had_province_neighbour_bad = _has_same_province_neighbour(
            node_bad, bad_hex, hex_to_node, graph
        )
        had_province_neighbour_candidate = _has_same_province_neighbour(
            node_candidate, candidate, hex_to_node, graph
        )

        _swap_hex_contents(bad_hex, candidate, hex_to_node, positions)
        cost_after = _local_cost(affected_pair, hex_to_node, graph)
        regressed = (
            had_province_neighbour_bad
            and not _has_same_province_neighbour(node_bad, candidate, hex_to_node, graph)
        ) or (
            had_province_neighbour_candidate
            and not _has_same_province_neighbour(node_candidate, bad_hex, hex_to_node, graph)
        )
        _swap_hex_contents(bad_hex, candidate, hex_to_node, positions)  # revert

        delta = cost_after - cost_before
        if not regressed and delta < best_delta:
            best_delta = delta
            best = candidate
    return best


def _has_same_province_neighbour(
    node: str | None, hex_position: Axial, hex_to_node: dict[Axial, str], graph: nx.Graph
) -> bool:
    """Whether `node`, if placed at `hex_position`, has a same-province hex-neighbour.

    Args:
        node: DistrictId to check, or `None` for an empty hex (trivially
            has no neighbour to lose).
        hex_position: Hex `node` currently occupies.
        hex_to_node: Hex to districtId.
        graph: Full graph, used to check province membership.

    Returns:
        `True` if any hex-adjacent occupant belongs to the same province.
    """
    if node is None:
        return False
    province = graph.nodes[node][PROVINCE]
    return any(
        (other := hex_to_node.get(neighbour)) is not None
        and graph.nodes[other][PROVINCE] == province
        for neighbour in _axial_neighbors(hex_position)
    )


def _swap_hex_contents(
    h1: Axial, h2: Axial, hex_to_node: dict[Axial, str], positions: dict[str, Axial]
) -> None:
    """Exchange whatever occupies `h1` and `h2` (a district, or nothing).

    Mutates `hex_to_node` and `positions` in place. Applying this twice
    with the same arguments restores the original state.

    Args:
        h1: First hex.
        h2: Second hex.
        hex_to_node: Hex to districtId, kept in sync with `positions`.
        positions: DistrictId to (q, r), kept in sync with `hex_to_node`.
    """
    occupant1 = hex_to_node.get(h1)
    occupant2 = hex_to_node.get(h2)

    if occupant2 is not None:
        positions[occupant2] = h1
        hex_to_node[h1] = occupant2
    else:
        hex_to_node.pop(h1, None)

    if occupant1 is not None:
        positions[occupant1] = h2
        hex_to_node[h2] = occupant1
    else:
        hex_to_node.pop(h2, None)


def _hexes_within_radius(center: Axial, radius: int) -> list[Axial]:
    """List every hex within `radius` hex-steps of `center` (excluding `center`).

    Args:
        center: Hex to search outward from.
        radius: Maximum hex-distance to include.

    Returns:
        All hexes at hex-distance 1..radius from `center`.
    """
    visited = {center}
    frontier = {center}
    result: list[Axial] = []
    for _ in range(radius):
        next_frontier: set[Axial] = set()
        for h in frontier:
            for neighbour in _axial_neighbors(h):
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.add(neighbour)
        result.extend(next_frontier)
        frontier = next_frontier
    return result


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
    hex_owner: dict[Axial, str] = {}

    for index, state in enumerate(order):
        state_members = groups[state]
        if index == 0:
            seed, anchor_district = (0, 0), None
        else:
            anchor_district, seed = _find_anchor_hex(state_members, positions, occupied, graph)
        _place_state_into(
            state_members, graph, positions, occupied, hex_owner, seed, anchor_district
        )

    return order[0], positions


def _place_state_into(  # noqa: PLR0913 -- each arg is independently necessary here
    members: list[str],
    graph: nx.Graph,
    positions: dict[str, Axial],
    occupied: set[Axial],
    hex_owner: dict[Axial, str],
    seed: Axial,
    anchor_district: str | None = None,
) -> None:
    """Grow one state's districts into a hex blob, one contiguous blob per province.

    Mutates `positions`, `occupied` and `hex_owner` in place, so this can be
    called multiple times to grow several states into one shared landmass.

    Args:
        members: DistrictIds belonging to this state.
        graph: Full graph, used to find each district's neighbours.
        positions: DistrictId to (q, r), shared across the whole landmass.
        occupied: Every hex already taken, shared across the whole landmass.
        hex_owner: Hex to owning state, shared across the whole landmass.
            Lets a boxed-in province's escape search stay within its own
            state's territory instead of tunnelling into a neighbour
            state's hexes.
        seed: Hex to anchor this state's first province at.
        anchor_district: DistrictId :func:`_find_anchor_hex` matched onto
            `seed`'s already-placed neighbour. Its province grows first
            (starting from this exact district), instead of always the
            state's own overall peak province. `None` for the very first
            state placed on a landmass, which has no neighbour to anchor
            onto.
    """
    state = graph.nodes[members[0]][STATE]
    provinces = _group_by_province(members, graph)
    start_province = graph.nodes[anchor_district][PROVINCE] if anchor_district is not None else None
    order = _growth_order(provinces, graph, start=start_province)

    def _same_state(node: str) -> bool:
        return graph.nodes[node][STATE] == state

    for index, province in enumerate(order):
        province_members = provinces[province]
        if index == 0:
            province_seed, first_district = seed, anchor_district
        else:
            first_district, province_seed = _find_anchor_hex(
                province_members, positions, occupied, graph, is_valid_neighbour=_same_state
            )
        _grow_blob_into(
            province_members,
            graph,
            positions,
            occupied,
            hex_owner,
            province_seed,
            PROVINCE,
            first_district=first_district,
        )


def _growth_order(
    groups: dict[str, list[str]], graph: nx.Graph, start: str | None = None
) -> list[str]:
    """Order groups (provinces or states) so each (after the first) has a visited neighbour.

    Starts at `start` if given -- the group containing the district that
    anchored this whole level onto its already-placed parent -- otherwise
    the group containing the overall highest-elevation district (the
    "peak"). From there, repeatedly visits the unvisited group with the
    smallest elevation gap to an already-visited neighbour (a
    Prim's-algorithm-style growth by edge weight, i.e. minimum-spanning-tree
    order), breaking ties by preferring the larger group. Because
    connectivity at this level is already guaranteed elsewhere (state
    connectivity by :func:`open_world.graph.neighbours.repair_connectivity`;
    landmass connectivity by construction, since a landmass is a connected
    component), this reaches every group. Prioritizing the smallest
    elevation gap keeps a mountain range that spans several
    provinces/states attaching high-side-to-high-side all the way through,
    instead of a low-elevation group wedging itself into the middle of the
    range.

    Args:
        groups: Group name (province or state) to member node ids.
        graph: Full graph, used to find inter-group edges.
        start: Group to begin growth from. Defaults to the group containing
            the overall highest-elevation district.

    Returns:
        Group names in growth order.
    """
    member_of = {node: group for group, members in groups.items() for node in members}
    adjacency: dict[str, set[str]] = {group: set() for group in groups}
    pair_gap: dict[tuple[str, str], int] = {}
    all_members = list(member_of)
    for source, target in graph.edges(all_members):
        group_a, group_b = member_of.get(source), member_of.get(target)
        if group_a is not None and group_b is not None and group_a != group_b:
            adjacency[group_a].add(group_b)
            adjacency[group_b].add(group_a)
            gap = abs(graph.nodes[source][ELEVATION] - graph.nodes[target][ELEVATION])
            if gap < pair_gap.get((group_a, group_b), gap + 1):
                pair_gap[group_a, group_b] = gap
                pair_gap[group_b, group_a] = gap

    def _by_size_desc(group: str) -> tuple[int, str]:
        return (-len(groups[group]), group)

    if start is None:
        start = max(groups, key=lambda g: max(graph.nodes[d][ELEVATION] for d in groups[g]))

    order = [start]
    visited = {start}
    heap = [
        (pair_gap[start, neighbour], -len(groups[neighbour]), neighbour)
        for neighbour in adjacency[start]
    ]
    heapq.heapify(heap)
    while heap:
        _, _, current = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        order.append(current)
        for neighbour in adjacency[current]:
            if neighbour not in visited:
                heapq.heappush(
                    heap, (pair_gap[current, neighbour], -len(groups[neighbour]), neighbour)
                )

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
) -> tuple[str, Axial]:
    """Find where -- and with which district -- a new group should start growing.

    Looks at every graph edge from this group to an already-placed
    neighbour and picks the smallest-elevation-gap pair, so the new group's
    growth begins right where it's most similar in elevation to its
    neighbour -- keeping a shared mountain range's high side attached to
    the other high side, rather than seaming at an arbitrary point. Unlike
    just picking a hex, this also identifies *which* district of the new
    group belongs at that seam, since it isn't necessarily the group's own
    highest-elevation member.

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
        (anchor_district, anchor_hex): which member of `members` should
        seed this group's growth, and which empty hex to place it at.
    """
    best_gap: int | None = None
    best_anchor: Axial | None = None
    best_district: str | None = None
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
                best_district = district

    if best_anchor is not None and best_district is not None:
        return best_district, best_anchor

    # No already-placed neighbour had room right next to it (fully boxed in);
    # fall back to the group's own peak, placed nearest whatever's already
    # on the map.
    fallback_district = max(members, key=lambda d: graph.nodes[d][ELEVATION])
    fallback_hex = _nearest_empty_hex(next(iter(positions.values())), occupied)
    return fallback_district, fallback_hex


def _nearest_empty_hex(
    start: Axial, occupied: set[Axial], can_pass: Callable[[Axial], bool] | None = None
) -> Axial:
    """Breadth-first search for the closest unoccupied hex to `start`.

    Args:
        start: Hex to search outward from.
        occupied: Every hex already taken.
        can_pass: Optional filter restricting which *occupied* hexes the
            search may travel through on its way to an empty one (the
            empty destination hex itself is never filtered). Used to keep
            an escape search within one state's own territory. Raises
            `RuntimeError` if no empty hex is reachable without passing
            through a disallowed one.

    Returns:
        The nearest hex (possibly `start` itself) not in `occupied`.
    """
    if start not in occupied:
        return start
    return _nearest_empty_hex_from_any({start}, occupied, can_pass)


def _nearest_empty_hex_from_any(
    sources: set[Axial], occupied: set[Axial], can_pass: Callable[[Axial], bool] | None = None
) -> Axial:
    """Multi-source breadth-first search for the closest unoccupied hex.

    Args:
        sources: Hexes to search outward from simultaneously.
        occupied: Every hex already taken.
        can_pass: Optional filter restricting which *occupied* hexes the
            search may travel through on its way to an empty one (the
            empty destination hex itself is never filtered).

    Returns:
        The nearest hex not in `occupied`, reachable from any of `sources`
        without passing through a hex `can_pass` rejects.

    Raises:
        RuntimeError: No empty hex is reachable under `can_pass`'s
            restriction (the searched region is entirely walled in).
    """
    seen = set(sources)
    frontier = deque(sources)
    while frontier:
        current = frontier.popleft()
        for neighbour in _axial_neighbors(current):
            if neighbour not in occupied:
                return neighbour
            if neighbour not in seen and (can_pass is None or can_pass(neighbour)):
                seen.add(neighbour)
                frontier.append(neighbour)
    msg = "no empty hex reachable without leaving the allowed region"
    raise RuntimeError(msg)


def _grow_blob_into(  # noqa: PLR0913 -- each arg is independently necessary here
    members: list[str],
    graph: nx.Graph,
    positions: dict[str, Axial],
    occupied: set[Axial],
    hex_owner: dict[Axial, str],
    seed: Axial,
    group_key: str,
    first_district: str | None = None,
) -> None:
    """Grow one contiguous blob of `members`, starting at `seed`.

    Mutates `positions`, `occupied` and `hex_owner` in place so blobs grown
    one after another share a single global occupancy map and never
    overlap.

    Args:
        members: DistrictIds to place (e.g. one province's districts).
        graph: Full graph, used to find each district's neighbours.
        positions: DistrictId to (q, r), shared across all blobs so far.
        occupied: Every hex already taken, shared across all blobs so far.
        hex_owner: Hex to owning state, shared across all blobs so far.
            Used to keep a boxed-in province's escape search inside its own
            state's territory rather than tunnelling into a neighbour
            state's hexes.
        seed: Hex to place `first_district` at (nudged to the nearest empty
            hex if already taken).
        group_key: Node attribute that must match for two districts to
            count as neighbours during this growth (e.g. ``province``).
        first_district: DistrictId to place at `seed` -- normally the
            district :func:`_find_anchor_hex` found to best match its
            already-placed neighbour's elevation. Defaults to this group's
            own highest-elevation member (e.g. for the very first group
            placed on the map, which has no neighbour to anchor onto).
    """
    group_value = graph.nodes[members[0]][group_key]
    state_value = graph.nodes[members[0]][STATE]
    ordered = sorted(members, key=lambda node: graph.nodes[node][ELEVATION], reverse=True)
    if first_district is not None and first_district != ordered[0]:
        ordered.remove(first_district)
        ordered.insert(0, first_district)

    first_hex = _nearest_empty_hex(seed, occupied)
    positions[ordered[0]] = first_hex
    occupied.add(first_hex)
    hex_owner[first_hex] = state_value
    frontier: set[Axial] = {h for h in _axial_neighbors(first_hex) if h not in occupied}
    blob_hexes: set[Axial] = {first_hex}

    def _same_state_hex(hex_position: Axial) -> bool:
        return hex_owner.get(hex_position) == state_value

    for district in ordered[1:]:
        same_group_neighbours = [
            neighbour
            for neighbour in graph.neighbors(district)
            if graph.nodes[neighbour].get(group_key) == group_value and neighbour in positions
        ]
        if not frontier:
            # Boxed in by other already-placed provinces/states with no room
            # left on this blob's own edge. First try to escape by tunnelling
            # only through this district's own state's territory, so it
            # lands somewhere still attached to its home state rather than
            # inside a neighbour state's pocket. Only if the whole state is
            # itself walled in does this fall back to jumping anywhere.
            try:
                escape = _nearest_empty_hex_from_any(blob_hexes, occupied, can_pass=_same_state_hex)
                logger.warning(
                    "{} {!r} ran out of room while placing {}; jumped to {} within {!r}",
                    group_key,
                    group_value,
                    district,
                    escape,
                    state_value,
                )
            except RuntimeError:
                escape = _nearest_empty_hex_from_any(blob_hexes, occupied)
                logger.warning(
                    "state {!r} is fully enclosed by other states; {} {!r} crosses into "
                    "foreign territory at {} while placing {}",
                    state_value,
                    group_key,
                    group_value,
                    escape,
                    district,
                )
            frontier.add(escape)
        hex_position = _pick_adjacent_empty_hex(same_group_neighbours, positions, frontier)
        positions[district] = hex_position
        occupied.add(hex_position)
        hex_owner[hex_position] = state_value
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
