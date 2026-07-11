"""Column names and expected dtypes for the district dataset."""

import polars as pl

STATE = "state"
STATE_ID = "stateId"
PROVINCE = "province"
PROVINCE_ID = "provinceId"
DISTRICT = "district"
DISTRICT_ID = "districtId"
TREE_COUNT = "treeCount"
ELEVATION = "elevation"
POPULATION = "population"

EXPECTED_SCHEMA: dict[str, pl.DataType] = {
    STATE: pl.String(),
    STATE_ID: pl.String(),
    PROVINCE: pl.String(),
    PROVINCE_ID: pl.String(),
    DISTRICT: pl.String(),
    DISTRICT_ID: pl.String(),
    TREE_COUNT: pl.Int32(),
    ELEVATION: pl.Int32(),
    POPULATION: pl.Int64(),
}
