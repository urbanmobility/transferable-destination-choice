"""Veraset POI set per tract: every distinct placekey seen in `visits` over the study months, placed in
its most frequent block group, classified by its most frequent NAICS code (naics_map).

Output: derived/multicity/poi_tracts.parquet   tract, n_<cls> x 8, n_other, N (all placekeys)
Usage: python -I poi.py
"""
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DERIVED, MONTHS, VERASET  # noqa: E402
from naics_map import CLASSES, sql_case  # noqa: E402


def build(src, months, out, threads=48, memory_limit="90GB", poi_id="placekey"):
    """poi_id: the POI identifier column (2024 placekey; 2019 safegraph_place_id, whose inline 'home' / 'work' rows
    are not places and are dropped)."""
    assert poi_id in ("placekey", "safegraph_place_id"), poi_id
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{memory_limit}'; SET preserve_insertion_order=false")
    files = ", ".join(f"'{src}/visits/{m}-*.parquet'" for m in months)
    con.execute(f"""
    CREATE TABLE seen AS
    SELECT {poi_id} AS placekey, substr(census_block_group, 1, 11) AS tract, naics_code, count(*) AS n
    FROM read_parquet([{files}])
    WHERE {poi_id} IS NOT NULL AND {poi_id} NOT IN ('home', 'work') AND census_block_group IS NOT NULL
    GROUP BY 1, 2, 3""")
    con.execute("""
    CREATE TABLE pk AS
    SELECT placekey,
           arg_max(tract, n_tract) AS tract,
           arg_max(naics_code, n_naics) AS naics_code
    FROM (SELECT placekey, tract, naics_code,
                 sum(n) OVER (PARTITION BY placekey, tract) AS n_tract,
                 sum(n) OVER (PARTITION BY placekey, naics_code) AS n_naics
          FROM seen)
    GROUP BY placekey""")
    cls_cols = ", ".join(f"count(*) FILTER (WHERE cls = '{c}') AS n_{c}" for c in CLASSES)
    con.execute(f"""
    COPY (
      SELECT tract, {cls_cols}, count(*) FILTER (WHERE cls = 'other') AS n_other, count(*) AS N
      FROM (SELECT tract, {sql_case('naics_code')} AS cls FROM pk)
      GROUP BY tract ORDER BY tract
    ) TO '{out}/poi_tracts.parquet' (FORMAT PARQUET)""")
    n, other = con.execute(f"SELECT count(*), count(*) FILTER (WHERE {sql_case('naics_code')} = 'other') FROM pk").fetchone()
    con.close()
    return {"placekeys": int(n), "classified_share": float(1 - other / n) if n else 0.0}


if __name__ == "__main__":
    from config import POI_ID
    print(json.dumps(build(VERASET, MONTHS, DERIVED, poi_id=POI_ID)), flush=True)
