"""Randomized, degree-capped neighbour assignment.

Replaces a fixed chain + single-bridge + elevation-affinity combination with
one unified process: each district gets a randomly chosen target degree (in
``[min_degree, max_degree]``), then candidates are found by elevation
closeness in three priority tiers -- same province, then other provinces in
the same state, then other states -- falling through to the next tier only
when the current one runs out of usable candidates. Candidate choice within
a tier is randomized (a random pick among the nearest ``candidate_pool``
elevation neighbours, not always the single nearest), so the resulting graph
isn't perfectly regular, and not every district ends up at the max degree.

Left alone, "prefer same-province" starves boundary-crossing edges almost
entirely whenever provinces are large enough to satisfy demand on their own
(which is most of them, on this dataset -- measured at ~99% same-province,
~0% cross-state). ``min_boundary_fraction`` forces each district to actively
seek a small, randomised quota of same-state/cross-state edges *before*
filling the rest of its budget locally, so provinces and states stay visibly
stitched together rather than only touching as a connectivity-repair
afterthought.

Cross-state edges are further restricted to districts at or above
``cross_state_elevation_percentile`` -- real mountain ranges cross borders
at their peaks, not through the lowlands. This also sidesteps a sharp
random-graph phase transition: letting *every* district roll independently
for a cross-state edge means, past a small threshold, essentially every
state ends up connected to every other state (one giant landmass) rather
than a handful of meaningful mountain bridges. Restricting eligibility to a
small high-elevation slice, and having a district prefer *extending* a
neighbour's existing cross-state bridge over starting an unrelated one (see
``_preferred_cross_state_target``), keeps cross-border connections rare,
clustered, and thematically about mountains.

This alone gives no connectivity guarantee -- see :func:`repair_connectivity`,
which must be run afterwards to guarantee every province and state stays a
single connected region.
"""

import bisect
import itertools
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import networkx as nx
import polars as pl
from loguru import logger

from open_world.data.schema import DISTRICT_ID, ELEVATION, PROVINCE, STATE
from open_world.graph.edge_types import EDGE_KIND, Edge

DEFAULT_MIN_DEGREE = 2
DEFAULT_MAX_DEGREE = 5
DEFAULT_CANDIDATE_POOL = 8
DEFAULT_MIN_BOUNDARY_FRACTION = 0.2
DEFAULT_CROSS_STATE_ELEVATION_PERCENTILE = 0.9
DEFAULT_SEED = 42

KIND_SAME_PROVINCE = "same-province"
KIND_SAME_STATE = "same-state-other-province"
KIND_CROSS_STATE = "cross-state"
KIND_CONNECTIVITY_REPAIR = "connectivity-repair"

BOUNDARY_KINDS = frozenset({KIND_SAME_STATE, KIND_CROSS_STATE})

Pool = tuple[list[str], list[int]]


@dataclass
class _Context:
    """Read-mostly lookups and mutable degree state shared by every pick."""

    province_of: dict[str, str]
    state_of: dict[str, str]
    elevation_of: dict[str, int]
    province_pool: dict[str, Pool]
    state_pool: dict[str, Pool]
    global_pool: Pool
    high_elevation_pool: Pool
    elevation_threshold: float
    max_degree: int
    candidate_pool: int
    rng: random.Random
    degree: dict[str, int] = field(default_factory=dict)
    boundary_count: dict[str, int] = field(default_factory=dict)
    boundary_quota: dict[str, int] = field(default_factory=dict)
    neighbours: dict[str, set[str]] = field(default_factory=dict)
    cross_state_partner: dict[str, str] = field(default_factory=dict)


def assign_edges(  # noqa: PLR0913 -- each strategy knob is meant to be tunable from here
    frame: pl.DataFrame,
    min_degree: int = DEFAULT_MIN_DEGREE,
    max_degree: int = DEFAULT_MAX_DEGREE,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    min_boundary_fraction: float = DEFAULT_MIN_BOUNDARY_FRACTION,
    cross_state_elevation_percentile: float = DEFAULT_CROSS_STATE_ELEVATION_PERCENTILE,
    seed: int = DEFAULT_SEED,
) -> list[tuple[str, str, str]]:
    """Randomly assign each district up to a random number of neighbours.

    Args:
        frame: District table with ``districtId``, ``state``, ``province``
            and ``elevation`` columns.
        min_degree: Lower bound (inclusive) of each district's random target
            degree.
        max_degree: Upper bound (inclusive) of each district's random target
            degree, and a hard cap -- no district ever exceeds this here.
        candidate_pool: How many nearest-elevation candidates to randomly
            choose among at each tier.
        min_boundary_fraction: Fraction (0-1) of each district's target
            degree that must be actively sought from the same-state/
            cross-state tiers before falling back to same-province. Without
            this, boundary edges only happen when a province runs out of
            its own candidates, which is rare once provinces are reasonably
            large.
        cross_state_elevation_percentile: Elevation quantile (0-1); only
            districts at or above it are eligible for cross-state edges, so
            state borders are only crossed by mountain ranges.
        seed: Random seed, for reproducible runs.

    Returns:
        List of (source, target, kind) triples. ``kind`` is one of
        :data:`KIND_SAME_PROVINCE`, :data:`KIND_SAME_STATE` or
        :data:`KIND_CROSS_STATE`, identifying which tier produced the edge.

    Raises:
        ValueError: If ``min_degree`` is not a positive integer, exceeds
            ``max_degree``, or if ``min_boundary_fraction`` or
            ``cross_state_elevation_percentile`` is not in [0, 1].
    """
    if min_degree < 1:
        msg = f"min_degree must be a positive integer, got {min_degree}"
        raise ValueError(msg)
    if min_degree > max_degree:
        msg = f"min_degree ({min_degree}) must not exceed max_degree ({max_degree})"
        raise ValueError(msg)
    if not 0 <= min_boundary_fraction <= 1:
        msg = f"min_boundary_fraction must be in [0, 1], got {min_boundary_fraction}"
        raise ValueError(msg)
    if not 0 <= cross_state_elevation_percentile <= 1:
        msg = (
            "cross_state_elevation_percentile must be in [0, 1], "
            f"got {cross_state_elevation_percentile}"
        )
        raise ValueError(msg)

    ids = frame[DISTRICT_ID].to_list()
    elevation_threshold = float(
        frame[ELEVATION].quantile(cross_state_elevation_percentile, interpolation="linear")
    )
    ctx = _Context(
        province_of=dict(zip(ids, frame[PROVINCE].to_list(), strict=True)),
        state_of=dict(zip(ids, frame[STATE].to_list(), strict=True)),
        elevation_of=dict(zip(ids, frame[ELEVATION].to_list(), strict=True)),
        province_pool=_build_pool(frame, PROVINCE),
        state_pool=_build_pool(frame, STATE),
        global_pool=_sorted_ids_elevations(frame),
        high_elevation_pool=_sorted_ids_elevations(
            frame.filter(pl.col(ELEVATION) >= elevation_threshold)
        ),
        elevation_threshold=elevation_threshold,
        max_degree=max_degree,
        candidate_pool=candidate_pool,
        degree=dict.fromkeys(ids, 0),
        boundary_count=dict.fromkeys(ids, 0),
        neighbours={node_id: set() for node_id in ids},
        rng=random.Random(seed),  # noqa: S311 -- layout randomness, not a security context
    )

    target_degree = {node_id: ctx.rng.randint(min_degree, max_degree) for node_id in ids}
    ctx.boundary_quota = {
        node_id: round(min_boundary_fraction * degree) for node_id, degree in target_degree.items()
    }
    order = list(ids)
    ctx.rng.shuffle(order)

    edges: list[tuple[str, str, str]] = []
    for district in order:
        needed = target_degree[district] - ctx.degree[district]
        if needed <= 0:
            continue
        for candidate, kind in _select_neighbours(district, needed, ctx):
            edges.append((district, candidate, kind))
            ctx.neighbours[district].add(candidate)
            ctx.neighbours[candidate].add(district)
            ctx.degree[district] += 1
            ctx.degree[candidate] += 1
            if kind in BOUNDARY_KINDS:
                ctx.boundary_count[district] += 1
                ctx.boundary_count[candidate] += 1
            if kind == KIND_CROSS_STATE:
                ctx.cross_state_partner[district] = ctx.state_of[candidate]
                ctx.cross_state_partner[candidate] = ctx.state_of[district]

    logger.info(
        "assigned {} randomized edges (min_degree={}, max_degree={}, candidate_pool={}, "
        "min_boundary_fraction={}, cross_state_elevation_percentile={}, seed={})",
        len(edges),
        min_degree,
        max_degree,
        candidate_pool,
        min_boundary_fraction,
        cross_state_elevation_percentile,
        seed,
    )
    return edges


def _select_neighbours(district: str, needed: int, ctx: _Context) -> list[tuple[str, str]]:
    """Pick up to `needed` neighbours for `district`.

    A quota of the pick is reserved for boundary-crossing edges (same-state,
    then cross-state) and sought *first*, regardless of whether same-province
    candidates exist; the remainder falls back to the usual same-province ->
    same-state -> cross-state tier order.

    Args:
        district: DistrictId to find neighbours for.
        needed: Number of additional neighbours wanted.
        ctx: Shared lookups and live degree/neighbour state.

    Returns:
        Up to `needed` (candidate_id, kind) pairs.
    """
    province = ctx.province_of[district]
    state = ctx.state_of[district]
    elevation = ctx.elevation_of[district]
    excluded = set(ctx.neighbours[district])
    excluded.add(district)

    def is_free(candidate: str) -> bool:
        return candidate not in excluded and ctx.degree[candidate] < ctx.max_degree

    def same_state_other_province(candidate: str) -> bool:
        return is_free(candidate) and ctx.province_of[candidate] != province

    def other_state(candidate: str) -> bool:
        return is_free(candidate) and ctx.state_of[candidate] != state

    def cross_state_candidates(count: int) -> list[str]:
        """High-elevation-only search.

        Prefers extending a neighbour's existing cross-state bridge over
        starting an unrelated one.
        """
        if count <= 0 or elevation < ctx.elevation_threshold:
            return []
        found: list[str] = []
        preferred_state = _preferred_cross_state_target(district, ctx)
        if preferred_state is not None:
            preferred = _random_nearest(
                ctx.high_elevation_pool,
                elevation,
                lambda c: other_state(c) and ctx.state_of[c] == preferred_state,
                ctx.candidate_pool,
                count,
                ctx.rng,
            )
            found.extend(preferred)
            excluded.update(preferred)
        if len(found) < count:
            found.extend(
                _random_nearest(
                    ctx.high_elevation_pool,
                    elevation,
                    other_state,
                    ctx.candidate_pool,
                    count - len(found),
                    ctx.rng,
                )
            )
        return found

    picked: list[tuple[str, str]] = []

    quota_remaining = max(0, ctx.boundary_quota.get(district, 0) - ctx.boundary_count[district])
    boundary_needed = min(needed, quota_remaining)
    if boundary_needed > 0:
        forced_state = _random_nearest(
            ctx.state_pool[state],
            elevation,
            same_state_other_province,
            ctx.candidate_pool,
            boundary_needed,
            ctx.rng,
        )
        picked.extend((candidate, KIND_SAME_STATE) for candidate in forced_state)
        excluded.update(forced_state)

        still_needed = boundary_needed - len(forced_state)
        if still_needed > 0:
            forced_cross = cross_state_candidates(still_needed)
            picked.extend((candidate, KIND_CROSS_STATE) for candidate in forced_cross)
            excluded.update(forced_cross)

    if len(picked) < needed:
        tier1 = _random_nearest(
            ctx.province_pool[province],
            elevation,
            is_free,
            ctx.candidate_pool,
            needed - len(picked),
            ctx.rng,
        )
        picked.extend((candidate, KIND_SAME_PROVINCE) for candidate in tier1)
        excluded.update(tier1)

    if len(picked) < needed:
        tier2 = _random_nearest(
            ctx.state_pool[state],
            elevation,
            same_state_other_province,
            ctx.candidate_pool,
            needed - len(picked),
            ctx.rng,
        )
        picked.extend((candidate, KIND_SAME_STATE) for candidate in tier2)
        excluded.update(tier2)

    if len(picked) < needed:
        tier3 = cross_state_candidates(needed - len(picked))
        picked.extend((candidate, KIND_CROSS_STATE) for candidate in tier3)

    return picked


def _preferred_cross_state_target(district: str, ctx: _Context) -> str | None:
    """Check whether any of `district`'s current neighbours already bridges to another state.

    Args:
        district: DistrictId to check.
        ctx: Shared lookups and live neighbour/cross-state-partner state.

    Returns:
        The state name to prefer bridging to, or None if no neighbour has an
        existing cross-state connection.
    """
    for neighbour in ctx.neighbours[district]:
        partner_state = ctx.cross_state_partner.get(neighbour)
        if partner_state is not None:
            return partner_state
    return None


def _build_pool(frame: pl.DataFrame, group_col: str) -> dict[str, Pool]:
    """Bucket districts by `group_col`, each bucket sorted by elevation.

    Args:
        frame: District table.
        group_col: Column to group by, e.g. ``province`` or ``state``.

    Returns:
        Mapping of group name to (districtIds, elevations), both sorted
        ascending by elevation.
    """
    pool: dict[str, Pool] = {}
    for key, group in frame.sort(ELEVATION).group_by(group_col, maintain_order=True):
        pool[key[0]] = (group[DISTRICT_ID].to_list(), group[ELEVATION].to_list())
    return pool


def _sorted_ids_elevations(frame: pl.DataFrame) -> Pool:
    """Return every districtId and elevation, sorted ascending by elevation.

    Args:
        frame: District table.

    Returns:
        (districtIds, elevations), both sorted ascending by elevation.
    """
    ordered = frame.sort(ELEVATION)
    return ordered[DISTRICT_ID].to_list(), ordered[ELEVATION].to_list()


def _nearest_unused(
    pool: Pool,
    target_elevation: int,
    is_valid: Callable[[str], bool],
    count: int,
) -> list[str]:
    """Find up to `count` ids nearest to `target_elevation` satisfying `is_valid`.

    Args:
        pool: (districtIds, elevations), both sorted ascending by elevation.
        target_elevation: Elevation to search around.
        is_valid: Predicate an id must satisfy to be returned.
        count: Maximum number of ids to return.

    Returns:
        Up to `count` ids, nearest-elevation first.
    """
    ids, elevations = pool
    pos = bisect.bisect_left(elevations, target_elevation)
    left, right = pos - 1, pos
    found: list[str] = []
    scanned = 0
    limit = len(ids)
    while len(found) < count and scanned < limit and (left >= 0 or right < len(ids)):
        left_gap = target_elevation - elevations[left] if left >= 0 else float("inf")
        right_gap = elevations[right] - target_elevation if right < len(ids) else float("inf")
        if left_gap <= right_gap:
            candidate = ids[left]
            left -= 1
        else:
            candidate = ids[right]
            right += 1
        scanned += 1
        if is_valid(candidate):
            found.append(candidate)
    return found


def _random_nearest(  # noqa: PLR0913 -- each arg is independently necessary here
    pool: Pool,
    target_elevation: int,
    is_valid: Callable[[str], bool],
    pool_size: int,
    count: int,
    rng: random.Random,
) -> list[str]:
    """Randomly pick up to `count` ids from the `pool_size` nearest valid candidates.

    Args:
        pool: (districtIds, elevations), both sorted ascending by elevation.
        target_elevation: Elevation to search around.
        is_valid: Predicate an id must satisfy to be a candidate.
        pool_size: How many nearest valid candidates to consider.
        count: How many to randomly pick from that pool.
        rng: Random source.

    Returns:
        Up to `count` ids, randomly chosen from the `pool_size` nearest.
    """
    nearby = _nearest_unused(pool, target_elevation, is_valid, pool_size)
    if len(nearby) <= count:
        return nearby
    return rng.sample(nearby, count)


def connect_orphans(graph: nx.Graph, frame: pl.DataFrame) -> None:
    """Guarantee every district has at least one neighbour.

    A district that is the sole member of both its province and its state
    has no repair partner (:func:`repair_connectivity` only merges
    *multiple* components within the same group; a lone node is trivially
    "connected" on its own) and, if its elevation is below the cross-state
    threshold, no cross-state fallback either. This is the true last
    resort: connect any remaining degree-0 district to its nearest
    elevation neighbour, dataset-wide, ignoring every other constraint.

    Args:
        graph: Graph to mutate in place.
        frame: District table, used to find each orphan's nearest elevation
            neighbour.
    """
    orphans = [node for node, degree in graph.degree if degree == 0]
    if not orphans:
        return

    ids, elevations = _sorted_ids_elevations(frame)
    index_by_id = {node_id: index for index, node_id in enumerate(ids)}

    for node in orphans:
        index = index_by_id[node]
        candidates = [i for i in (index - 1, index + 1) if 0 <= i < len(ids)]
        nearest = min(candidates, key=lambda i: abs(elevations[i] - elevations[index]))
        graph.add_edge(node, ids[nearest], **{EDGE_KIND: KIND_CONNECTIVITY_REPAIR})

    logger.warning(
        "{} districts had no same-group or cross-state neighbours; "
        "connected them to their nearest elevation neighbour as a fallback",
        len(orphans),
    )


def repair_connectivity(graph: nx.Graph, frame: pl.DataFrame, max_degree: int) -> None:
    """Guarantee every province and state induces a connected subgraph.

    The randomized assignment gives no connectivity guarantee on its own.
    This finds any province or state that ended up split into multiple
    components and stitches them together with the fewest possible extra
    edges (one nearest-elevation pair per pair of components). Connectivity
    is a harder constraint than the degree cap, so a repair edge is allowed
    to push a district over `max_degree` as a last resort -- this is logged
    whenever it happens.

    Args:
        graph: Graph to mutate in place.
        frame: District table, used to find each component's elevation
            range.
        max_degree: Degree cap to respect where possible.
    """
    _repair_level(graph, frame, PROVINCE, max_degree)
    _repair_level(graph, frame, STATE, max_degree)


def _repair_level(graph: nx.Graph, frame: pl.DataFrame, group_col: str, max_degree: int) -> None:
    """Merge disconnected components within each group of `group_col`.

    Args:
        graph: Graph to mutate in place.
        frame: District table.
        group_col: Column to group by, e.g. ``province`` or ``state``.
        max_degree: Degree cap to respect where possible.
    """
    repaired_groups = 0
    for _, group in frame.group_by(group_col):
        node_ids = group[DISTRICT_ID].to_list()
        subgraph = graph.subgraph(node_ids)
        components = list(nx.connected_components(subgraph))
        if len(components) <= 1:
            continue
        repaired_groups += 1
        elevation_of = dict(
            zip(group[DISTRICT_ID].to_list(), group[ELEVATION].to_list(), strict=True)
        )
        ordered_components = sorted(components, key=lambda comp: min(elevation_of[n] for n in comp))
        for comp_a, comp_b in itertools.pairwise(ordered_components):
            source, target = _closest_pair(comp_a, comp_b, elevation_of)
            if graph.degree[source] >= max_degree or graph.degree[target] >= max_degree:
                logger.warning(
                    "connectivity repair exceeds max_degree for {} <-> {}", source, target
                )
            graph.add_edge(source, target, **{EDGE_KIND: KIND_CONNECTIVITY_REPAIR})
    if repaired_groups:
        logger.info("repaired connectivity for {} {} group(s)", repaired_groups, group_col)


def _closest_pair(
    component_a: set[str], component_b: set[str], elevation_of: dict[str, int]
) -> Edge:
    """Find the pair of districts across two components with the closest elevation.

    Args:
        component_a: DistrictIds in the first component.
        component_b: DistrictIds in the second component.
        elevation_of: DistrictId to elevation lookup.

    Returns:
        The (id_a, id_b) pair minimizing the absolute elevation difference.
    """
    ids_a = sorted(component_a, key=lambda n: elevation_of[n])
    ids_b = sorted(component_b, key=lambda n: elevation_of[n])
    i, j = 0, 0
    best_diff: int | None = None
    best_pair: Edge = (ids_a[0], ids_b[0])
    while i < len(ids_a) and j < len(ids_b):
        elev_a, elev_b = elevation_of[ids_a[i]], elevation_of[ids_b[j]]
        diff = abs(elev_a - elev_b)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_pair = (ids_a[i], ids_b[j])
        if elev_a < elev_b:
            i += 1
        else:
            j += 1
    return best_pair
