"""Loading and validation of the district dataset."""

from pathlib import Path

import polars as pl
from loguru import logger

from open_world.data.schema import DISTRICT_ID, EXPECTED_SCHEMA


class SchemaError(ValueError):
    """Raised when a loaded dataset does not match the expected schema."""


def load_districts(path: str | Path) -> pl.DataFrame:
    """Load the district dataset from a (possibly partitioned) parquet source.

    Args:
        path: Directory containing parquet part-files (e.g. ``data/states``)
            or a glob pattern pointing directly at them.

    Returns:
        One row per district, validated against
        :data:`open_world.data.schema.EXPECTED_SCHEMA`.

    Raises:
        SchemaError: If required columns are missing, mistyped, or if
            ``districtId`` is not unique.
    """
    source = Path(path)
    glob = str(source / "*.parquet") if source.is_dir() else str(source)

    logger.info("loading districts from {}", glob)
    frame = pl.read_parquet(glob)
    _validate_schema(frame)
    logger.info(
        "loaded {} districts across {} provinces and {} states",
        frame.height,
        frame["province"].n_unique(),
        frame["state"].n_unique(),
    )
    return frame


def _validate_schema(frame: pl.DataFrame) -> None:
    """Validate that a frame matches the expected district schema.

    Args:
        frame: Frame to validate.

    Raises:
        SchemaError: If required columns are missing, mistyped, or if
            ``districtId`` is not unique.
    """
    missing = set(EXPECTED_SCHEMA) - set(frame.columns)
    if missing:
        msg = f"dataset is missing required columns: {sorted(missing)}"
        raise SchemaError(msg)

    mismatched = {
        name: (frame.schema[name], dtype)
        for name, dtype in EXPECTED_SCHEMA.items()
        if frame.schema[name] != dtype
    }
    if mismatched:
        msg = f"dataset columns have unexpected dtypes: {mismatched}"
        raise SchemaError(msg)

    n_rows = frame.height
    n_unique = frame[DISTRICT_ID].n_unique()
    if n_unique != n_rows:
        msg = f"districtId must be unique: {n_rows} rows but {n_unique} distinct ids"
        raise SchemaError(msg)
