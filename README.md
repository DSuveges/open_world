# open-world

Turns a hierarchical tabular dataset (state → province → district, with
elevation/population/tree-count columns) into a data-driven 2D map: a graph
of district neighbours, laid out on a hex grid that looks like real terrain,
with a browser-based viewer for fast visual iteration on the generation
algorithm.

The dataset is, in fact, the Open Targets disease ontology (EFO/MONDO/OTAR
IDs) reinterpreted as fictional geography — "states" are top-level disease
areas, "provinces" are disease groupings, "districts" are individual
diseases, and "elevation" has no real-world meaning beyond being a number
used to drive terrain shape. This is a red herring for the purposes of the
algorithm: everything downstream treats the input as an anonymous
hierarchy + numeric attributes.

## Why it's built this way

The original brief: every district occupies exactly one map location; every
district has at least one neighbour; every province and every state must be
a single connected region ("no islands within an island"); states should
read as separate islands, surrounded by water, unless a state is explicitly
sharing a landmass; elevation should influence which districts end up next
to each other (similar elevations cluster, mountain ranges can cross state
borders, low elevation belongs near the coast).

Two designs were considered: geometry-first (place points in space, then
infer a neighbour graph from proximity) and graph-first (build the
neighbour graph directly from the hierarchy + elevation, then find a spatial
layout that respects it). Graph-first won, because the hard constraints
above are all *graph* properties (connectivity, degree, cross-boundary
edges) — enforcing them post-hoc on a geometric layout is much harder than
building a graph that already satisfies them and then laying it out.

## Pipeline overview

```
parquet (data/states/*.parquet)
        │  open_world.data.loader.load_districts
        ▼
polars DataFrame (one row per district)
        │  open_world.graph.builder.build_graph
        ▼
networkx.Graph (one node per district, "kind"-tagged edges)
        │
        ├─ open_world.graph.validation.validate_graph   (connectivity/degree checks)
        ├─ open_world.graph.metrics.compute_metrics      (quantitative diagnostics)
        └─ open_world.layout.hexgrid.compute_hex_layout  (real map positions)
                │
                ▼
        open_world.viz.api (FastAPI)  →  browser viewer (static/index.html)
```

Everything downstream of `build_graph` is disposable and cheap to recompute
(the viz server rebuilds the whole graph + layout from disk on every
request — see [Iterating on the algorithm](#iterating-on-the-algorithm)),
so the graph-generation step is the thing worth understanding in depth.

## The neighbour graph (`open_world/graph/`)

`build_graph()` (`graph/builder.py`) assembles the graph in three passes:

1. **`neighbours.assign_edges`** — the core randomized algorithm. Each
   district gets a random target degree in `[min_degree, max_degree]`
   (default `[2, 5]`), then candidates are found by *elevation closeness* in
   three priority tiers, falling through to the next tier only when the
   current one runs out of usable candidates:
   1. same province
   2. other provinces in the same state
   3. other states
   Candidate choice within a tier is a random pick among the nearest
   `candidate_pool` (default 8) elevation neighbours, not always the single
   nearest — so the graph isn't perfectly regular, and elevation still
   drives clustering without producing a rigid, deterministic-looking mesh.

   Two things stop this tiered/greedy approach from degenerating:
   - **`min_boundary_fraction`** (default 0.2): left alone, "prefer
     same-province" starves cross-boundary edges almost entirely once
     provinces are large enough to satisfy demand locally (measured at
     ~99% same-province edges on this dataset). Each district is forced to
     actively seek a small, randomized quota of same-state/cross-state
     edges *before* filling the rest of its degree budget locally, so
     provinces and states stay visibly stitched together instead of only
     touching as a connectivity-repair afterthought.
   - **`cross_state_elevation_percentile`** (default 0.9): cross-state
     edges are restricted to districts at or above this elevation
     percentile — real mountain ranges cross borders at their peaks, not
     through the lowlands. This also sidesteps a sharp random-graph phase
     transition: letting *every* district roll independently for a
     cross-state edge means that past a small threshold, essentially every
     state ends up connected to every other state (one giant landmass)
     instead of a handful of meaningful mountain bridges. Restricting
     eligibility to a small high-elevation slice, and having a district
     prefer *extending* a neighbour's existing cross-state bridge over
     starting an unrelated one (`_preferred_cross_state_target`), keeps
     cross-border connections rare, clustered, and thematically about
     mountains.

2. **`neighbours.repair_connectivity`** — `assign_edges` alone gives no
   connectivity guarantee. This runs afterwards and merges disconnected
   components within each province/state by connecting their closest pair
   of districts (by elevation), until every province and state induces a
   single connected subgraph.

3. **`neighbours.connect_orphans`** — last-resort fallback for a district
   that is the sole member of both its province *and* its state (so
   `repair_connectivity` sees it as trivially "connected" — a single-node
   component is still one component). Connects it to its nearest-elevation
   neighbour anywhere, tagged `connectivity-repair`.

Every edge carries a `kind` attribute (`same-province`,
`same-state-other-province`, `cross-state`, or `connectivity-repair`)
recording which tier produced it — this is what the metrics and the
graph-diagnostic viewer break down.

`graph/validation.py` then checks the *hard* constraints (every
province/state connected, no orphan districts) and flags high-degree
districts as a soft warning. `graph/metrics.py` computes the *quantitative*
diagnostics used to judge whether a parameter change actually helped:
cross-province/cross-state edge fractions, elevation-gap statistics per
edge kind, and whether high-elevation districts form cross-state clusters
(the "mountain range crosses a border" signal).

## The hex map (`open_world/layout/hexgrid.py`)

`compute_hex_layout()` places every district on an axial hex grid so it
looks like a real, non-overlapping map (unlike the graph-diagnostic
layouts in `layout/clustered.py` and `layout/placeholder.py`, which exist
only to make the raw graph structure legible, not to look like terrain).
It runs in four stages:

1. **Landmass grouping.** States connected by at least one cross-state edge
   are grouped into one shared landmass (a connected component over the
   state-level adjacency graph) instead of always getting separate islands.
   A state with no cross-state edges is its own landmass.
2. **State growth.** Within a landmass, states are visited in an order that
   follows cross-state edges: a Prim's-MST-style traversal prioritized by
   *smallest elevation gap* to what's already placed (falling back to
   larger-group-first as a tiebreak), so a mountain range spanning several
   states seams high-side-to-high-side instead of a low-elevation state
   wedging into the middle of it. Each new state is seeded next to the
   specific *anchor district* found to have the smallest elevation gap
   across the seam — not just the nearest hex, and not automatically the
   new state's own elevation peak.
3. **Province growth.** Same idea one level down, using only same-state
   edges (so a province never anchors onto a neighbouring state's
   territory). Each province grows as a contiguous hex blob: seed the
   anchor district, then walk the rest in descending elevation order,
   attaching each to an empty hex next to an already-placed same-province
   graph neighbour. Elevation decreases outward at every level because
   growth always proceeds high-to-low and always attaches to the current
   edge of what's already placed — so the coastline naturally ends up
   low-elevation.
4. **Refinement (`refine_hex_layout`).** Greedy growth occasionally boxes a
   province in and has to jump it to the nearest free hex (its escape
   hatch prefers tunnelling through the same state's own territory first,
   only crossing into a neighbour state if that state is itself fully
   enclosed) — this can still strand a handful of districts deep in the
   wrong state, or leave small enclosed "holes" behind. A bounded, targeted
   local search runs per landmass after growth: each pass finds every hex
   with a nonzero defect cost (a district with *zero* same-state or
   graph-edge-justified neighbours; a mostly-enclosed empty hex) and looks
   for a nearby cost-reducing hex swap, with a cooling tolerance schedule
   across passes and a light elevation-continuity term so a fix doesn't
   reopen a mountain-seam gap. Every candidate swap is rejected outright if
   it would strip a district of a same-province neighbour it currently
   has, so cleanup can't fragment an otherwise-contiguous province.

   Note the defect cost only flags *total* isolation, not "touches a
   different state" — two neighbouring states sharing a long coastline is
   normal geography, not a defect; only a district with no legitimate
   connection at all to its own state counts as stranded. (An earlier,
   wrong version of this cost function penalized every mismatched
   neighbour pair individually, which flagged thousands of ordinary border
   hexes and made refinement ~15x slower for no benefit — worth remembering
   if this function is touched again.)

Islands (landmasses) are packed onto the plane via an expanding-ring search
so no two bounding circles overlap by less than `water_gap` (default 3.0
hex units).

**Known residual limitation:** after refinement, a handful of districts
(~10 out of 12,729 on the full dataset) can still end up stranded — this
happens when a state's own territory is entirely walled in by faster-growing
neighbours with zero room left anywhere along its border, so even the
state-aware escape hatch has no legitimate option. Fixing this fully would
mean pre-reserving each state a proportional chunk of space before growth
starts (mirroring what island-packing already does at the landmass level)
— a bigger change that was explicitly deferred as not worth it for a ~0.08%
residual rate.

## The viewer (`open_world/viz/`)

`viz/api.py` is a small FastAPI app. `GET /api/graph` rebuilds the graph and
both layouts **from scratch on every request** (no caching) — this is
deliberate: the whole point is that changing a constant or an algorithm in
`graph/neighbours.py` or `layout/hexgrid.py` and reloading the browser tab
is the entire edit-and-check loop. `GET /` serves the static single-page
viewer (`viz/static/index.html`, canvas + d3.js v7 via CDN, no build step).

The viewer has two independent toggles:
- **View**: `terrain map` (the hex-grid map — what you want most of the
  time) vs. `graph diagnostic` (a province-clustered spectral layout of the
  raw graph, useful for judging graph structure independent of the spatial
  layout algorithm).
- **Color by**: `elevation` (a shaded terrain ramp, percentile-scaled
  because the elevation distribution is skewed) vs. `state / province`
  (a hierarchical color scheme — visually distinct hues per state via
  golden-angle distribution, with lightness variation for provinces within
  a state).

Hovering a hex/node shows its district/province/state name. The sidebar
shows live graph stats (node/edge counts, validity, degree percentiles,
edge-kind breakdown) and metrics from `graph/metrics.py`, plus a legend
that supports isolating a single state/province by clicking its entry.

## Usage

Requires Python ≥3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# install dependencies (creates .venv)
uv sync

# run the graph-generation pipeline once and log a validity/metrics report
uv run open-world

# start the interactive viewer at http://127.0.0.1:8000 (auto-reloads on
# source changes; the graph itself rebuilds on every browser reload)
uv run open-world-viz
```

### Iterating on the algorithm

1. Edit a constant or function in `graph/neighbours.py` or
   `layout/hexgrid.py`.
2. Reload the browser tab pointed at the running `open-world-viz` server —
   no restart needed for graph-generation changes (uvicorn's `reload=True`
   only matters for changes to the server code itself).
3. Check the sidebar metrics and both view modes before trusting a change
   "looks right" — several past regressions (e.g. degree-2 chain topology,
   elevation-affinity over-applied to low elevation, stranded districts)
   were invisible at a glance but obvious in the metrics.
4. For anything not visible in the UI (e.g. exact stranded-district counts,
   timing), write a small one-off script against the library functions
   directly — e.g. call `compute_hex_layout` on a graph from `build_graph`
   and inspect `_axial_neighbors` around each district's hex to count
   defects, or wrap a call in `time.time()` to check performance. These are
   normally throwaway and not worth committing.

### Tests, linting, formatting

```bash
uv run pytest                 # full suite, runs with coverage (see pyproject.toml)
uv run ruff check .           # lint (extensive rule set, see pyproject.toml)
uv run ruff format .          # format
uv run ruff format --check .  # format check only (what CI runs)
```

`.github/workflows/ci.yml` runs `ruff format --check`, `ruff check`, and
`pytest` on every pull request.

Test conventions: Google-style docstrings (enforced by ruff's `D` rules,
except on files under `tests/`, which are exempt from docstring/annotation
requirements), `loguru` for all logging (no `print`/stdlib `logging`), and
the `make_frame` fixture in `tests/conftest.py` for building small synthetic
district tables without hand-writing every schema column.

## Project structure

```
src/open_world/
  __init__.py           entry point (`main`) for the one-shot pipeline run
  data/
    schema.py            column name constants + EXPECTED_SCHEMA
    loader.py             load_districts(): parquet -> validated DataFrame
  graph/
    edge_types.py         Edge type alias + EDGE_KIND constant (avoids circular imports)
    neighbours.py          assign_edges / repair_connectivity / connect_orphans (core algorithm)
    builder.py             build_graph(): orchestrates the above into one nx.Graph
    validation.py          validate_graph(): hard connectivity/degree checks
    metrics.py              compute_metrics(): quantitative diagnostics
  layout/
    hexgrid.py             compute_hex_layout(): the real map (see above)
    clustered.py            province-clustered spectral layout (graph-diagnostic view)
    placeholder.py          plain spectral layout (fast, degenerate for this graph shape)
  viz/
    api.py                  FastAPI app (/, /api/graph)
    export.py                graph_to_json(): nx.Graph -> node-link JSON
    static/index.html        the whole frontend (canvas + d3, no build step)
tests/                    mirrors src/ layout, one test module per source module
data/states/               hive-style flat parquet part-files (the dataset)
.github/workflows/ci.yml   lint + format + test on every PR
```

## Data

`data/states/*.parquet` holds one row per district with columns `state`,
`stateId`, `province`, `provinceId`, `district`, `districtId` (unique),
`elevation` (int32), `population` (int64), `treeCount` (int32) — see
`data/schema.py` for the authoritative schema. On the current dataset:
12,729 districts across 126 provinces and 20 states.

## Status / what's next

Stages 1–4 of the hex-layout rewrite are done (province contiguity,
landmass grouping, elevation seam matching, local-search refinement — see
git history for the staged commits). Stage 5 is not yet scoped as of this
writing.

Ideas raised but explicitly deferred, in case they come up again:
- **Voronoi-based layout** instead of hex grid, for a more organic look.
  Considered before the Stage 1–4 rewrite; decided against switching mid-way
  since it would require re-deriving the same growth-order/anchor-matching
  logic for a different geometry, for uncertain visual benefit.
- **Proportional per-state territory reservation** before growth starts
  (see the hex map's known residual limitation above), to close the last
  ~0.08% of stranded districts.
- **Cosmetic hole-filling** beyond what `refine_hex_layout` already does —
  small interior gaps in province blobs are visually acceptable at present.
