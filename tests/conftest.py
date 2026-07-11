"""Shared pytest fixtures for building small synthetic district tables."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import polars as pl
import pytest

from open_world.data.schema import EXPECTED_SCHEMA


@pytest.fixture
def make_frame() -> Callable[[Sequence[Mapping[str, Any]]], pl.DataFrame]:
    """Factory fixture building a valid district frame from partial rows.

    Returns:
        A callable that fills in defaults (``population``, ``treeCount``,
        and id columns derived from names) for any missing columns, then
        returns a frame matching :data:`open_world.data.schema.EXPECTED_SCHEMA`.
    """

    def _make(rows: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
        filled = []
        for row in rows:
            record = dict(row)
            record.setdefault("stateId", f"{record['state']}-id")
            record.setdefault("provinceId", f"{record['province']}-id")
            record.setdefault("district", record["districtId"])
            record.setdefault("population", 0)
            record.setdefault("treeCount", 0)
            filled.append(record)
        return pl.DataFrame(filled, schema=EXPECTED_SCHEMA)

    return _make
