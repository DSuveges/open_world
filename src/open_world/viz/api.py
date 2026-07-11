"""FastAPI app serving the district graph for quick visual iteration.

The graph is rebuilt from disk on every request to `/api/graph` rather than
cached, so changes to the edge-generation algorithms show up on a plain
browser reload -- no server restart needed while iterating.
"""

from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from open_world import DEFAULT_DATA_PATH
from open_world.data.loader import load_districts
from open_world.graph.builder import build_graph
from open_world.graph.metrics import compute_metrics
from open_world.graph.validation import validate_graph
from open_world.layout.clustered import compute_clustered_layout
from open_world.layout.hexgrid import apply_hex_positions, compute_hex_layout
from open_world.layout.placeholder import apply_positions
from open_world.viz.export import graph_to_json

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="open-world map viewer")


@app.get("/api/graph")
def get_graph() -> JSONResponse:
    """Build the district graph and return it as node-link JSON.

    Returns:
        JSON with ``nodes`` (each carrying both a graph-diagnostic ``x``/``y``
        position and a real map ``q``/``r`` hex position), ``edges``, and a
        ``summary`` combining the validation report (connectivity and degree
        checks) with quantitative graph metrics.
    """
    frame = load_districts(DEFAULT_DATA_PATH)
    graph = build_graph(frame)
    report = validate_graph(graph, frame)
    metrics = compute_metrics(graph, frame)
    apply_positions(graph, compute_clustered_layout(graph))
    apply_hex_positions(graph, compute_hex_layout(graph))

    payload: dict[str, Any] = graph_to_json(graph)
    payload["summary"] = {**report.to_summary_dict(), "metrics": metrics.to_dict()}
    return JSONResponse(payload)


@app.get("/")
def index() -> FileResponse:
    """Serve the static visualization page.

    Returns:
        The bundled ``index.html`` file.
    """
    return FileResponse(STATIC_DIR / "index.html")


def serve(host: str = "127.0.0.1", port: int = 8000, *, reload: bool = True) -> None:
    """Run the visualization server.

    Args:
        host: Interface to bind to.
        port: Port to bind to.
        reload: Whether to auto-reload the server on source changes.
    """
    logger.info("starting viz server on http://{}:{}", host, port)
    uvicorn.run("open_world.viz.api:app", host=host, port=port, reload=reload)
