import polars as pl
import pytest

from open_world.data.loader import SchemaError, load_districts


def test_load_districts_reads_a_single_directory_of_parts(tmp_path, make_frame):
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "d1", "elevation": 10},
            {"state": "s1", "province": "p1", "districtId": "d2", "elevation": 20},
        ]
    )
    frame.write_parquet(tmp_path / "part-0.parquet")

    loaded = load_districts(tmp_path)

    assert loaded.height == 2
    assert set(loaded["districtId"]) == {"d1", "d2"}


def test_load_districts_reads_multiple_parts(tmp_path, make_frame):
    make_frame(
        [{"state": "s1", "province": "p1", "districtId": "d1", "elevation": 10}]
    ).write_parquet(tmp_path / "part-0.parquet")
    make_frame(
        [{"state": "s1", "province": "p1", "districtId": "d2", "elevation": 20}]
    ).write_parquet(tmp_path / "part-1.parquet")

    loaded = load_districts(tmp_path)

    assert loaded.height == 2


def test_load_districts_rejects_missing_column(tmp_path, make_frame):
    frame = make_frame(
        [{"state": "s1", "province": "p1", "districtId": "d1", "elevation": 10}]
    ).drop("population")
    frame.write_parquet(tmp_path / "part-0.parquet")

    with pytest.raises(SchemaError, match="missing required columns"):
        load_districts(tmp_path)


def test_load_districts_rejects_wrong_dtype(tmp_path, make_frame):
    frame = make_frame(
        [{"state": "s1", "province": "p1", "districtId": "d1", "elevation": 10}]
    ).with_columns(pl.col("elevation").cast(pl.Float64))
    frame.write_parquet(tmp_path / "part-0.parquet")

    with pytest.raises(SchemaError, match="unexpected dtypes"):
        load_districts(tmp_path)


def test_load_districts_rejects_duplicate_district_id(tmp_path, make_frame):
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "d1", "elevation": 10},
            {"state": "s1", "province": "p1", "districtId": "d1", "elevation": 20},
        ]
    )
    frame.write_parquet(tmp_path / "part-0.parquet")

    with pytest.raises(SchemaError, match="districtId must be unique"):
        load_districts(tmp_path)
