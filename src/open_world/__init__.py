"""Data-driven map generation for the state/province/district dataset."""

from pathlib import Path

from loguru import logger

from open_world.data.loader import load_districts
from open_world.graph.builder import build_graph
from open_world.graph.metrics import compute_metrics
from open_world.graph.validation import validate_graph

DEFAULT_DATA_PATH = Path("data/states")


def main() -> None:
    """Run the graph-generation pipeline against the default dataset and log a report."""
    frame = load_districts(DEFAULT_DATA_PATH)
    graph = build_graph(frame)
    report = validate_graph(graph, frame)
    compute_metrics(graph, frame)

    if report.is_valid:
        logger.info("pipeline finished: graph is valid")
    else:
        logger.warning("pipeline finished: graph has constraint violations, see warnings above")
