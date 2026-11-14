"""Home tract per device over the study months: the block group that appears in the most monthly
home-location tables; a tie between block groups drops the device. Only 12-digit numeric block groups count.

Output: derived/multicity/homes.parquet (caid, hcbg, htract, n_months), home_tracts.parquet (tract, home_devices)
Usage: python -I homes.py
"""
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DERIVED, MONTHS, VERASET  # noqa: E402


def build(src, months, out, threads=48, memory_limit="90GB"):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{memory_limit}'; SET preserve_insertion_order=false")
    raw = " UNION ALL ".join(
        f"SELECT CAID AS caid, CENSUS_BLOCK_GROUP AS cbg, '{m}' AS month FROM read_parquet('{src}/home-location/{m}*.parquet')"
        for m in months)
    con.execute(f"CREATE TABLE raw AS {raw}")
    bad = con.execute("SELECT count(*) FROM raw WHERE cbg IS NULL OR NOT regexp_matches(cbg, '^[0-9]{12}$')").fetchone()[0]
    con.execute("""
    CREATE TABLE per AS
    SELECT caid, cbg, count(DISTINCT month) AS n_months
    FROM raw WHERE cbg IS NOT NULL AND regexp_matches(cbg, '^[0-9]{12}$')
    GROUP BY 1, 2""")
    con.execute("""
    CREATE TABLE top AS
    SELECT caid, max(n_months) AS n_months, count(*) FILTER (WHERE n_months = mx) AS n_at_max
    FROM (SELECT *, max(n_months) OVER (PARTITION BY caid) AS mx FROM per)
    GROUP BY caid""")
    ties = con.execute("SELECT count(*) FROM top WHERE n_at_max > 1").fetchone()[0]
    con.execute(f"""
    COPY (
      SELECT p.caid, p.cbg AS hcbg, substr(p.cbg, 1, 11) AS htract, p.n_months
      FROM per p JOIN top t ON p.caid = t.caid AND p.n_months = t.n_months
      WHERE t.n_at_max = 1
    ) TO '{out}/homes.parquet' (FORMAT PARQUET)""")
    con.execute(f"""
    COPY (SELECT htract AS tract, count(*) AS home_devices FROM read_parquet('{out}/homes.parquet') GROUP BY 1 ORDER BY 1)
    TO '{out}/home_tracts.parquet' (FORMAT PARQUET)""")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}/homes.parquet')").fetchone()[0]
    con.close()
    return {"devices": int(n), "ties_dropped": int(ties), "bad_cbg_rows": int(bad)}


if __name__ == "__main__":
    print(json.dumps(build(VERASET, MONTHS, DERIVED)), flush=True)
