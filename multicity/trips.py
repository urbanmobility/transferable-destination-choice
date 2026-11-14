"""National trips table: POI visits + work visits of devices with a home, one count per
(device, local date, destination tract), dwell >= 5 min, destination tract != home tract, device group g from a
seeded hash of the device id. Partitioned by the origin's state so a city reads only its states.

Output: derived/multicity/trips/o_state=SS/<month>_<i>.parquet   date, g, o, d, work, n
        derived/multicity/budget/<month>.parquet                 date, o, active_devices (home devices with any visit)
Usage: python -I trips.py
"""
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DERIVED, GROUP_SEED, MIN_DWELL_MIN, MONTHS, N_GROUPS, VERASET  # noqa: E402


def _month_bounds(m):
    first = pd.Timestamp(f"{m}-01")
    return first.strftime("%Y-%m-%d"), (first + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")


def _next_month(m):
    return (pd.Timestamp(f"{m}-01") + pd.offsets.MonthBegin(1)).strftime("%Y-%m")


def _shards(src, sub, m):
    """(own shards, spill shards) of a month: the shards are cut by UTC date, so the local evening of the month's
    last day sits in the first shard of the next month (plan-1 final review); the last month's spill (06-01) is
    not downloaded and its loss is reported as spill_shards = 0."""
    own = sorted((Path(src) / sub).glob(f"{m}-*.parquet"))
    spill = sorted((Path(src) / sub).glob(f"{_next_month(m)}-01-*.parquet"))
    return own, spill


def _lst(files):
    return "[" + ", ".join(f"'{f}'" for f in files) + "]"


def build(src, months, homes, out, n_groups=N_GROUPS, seed=GROUP_SEED, min_dwell=MIN_DWELL_MIN, threads=48,
          memory_limit="90GB", temp_dir=None, hw_inline=False):
    """hw_inline (2019, line M19): home and work visits are rows of `visits` (safegraph_place_id 'home' / 'work'), the
    work rows carry no usable destination: both are dropped, no work-visits table is read and every trip is non-work."""
    src, out = Path(src), Path(out)
    (out / "trips").mkdir(parents=True, exist_ok=True)
    (out / "budget").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{memory_limit}'; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{temp_dir or out / 'tmp'}'")
    con.execute(f"CREATE TABLE homes AS SELECT caid, htract FROM read_parquet('{homes}')")
    rep = {}
    for m in months:
        d0, d1 = _month_bounds(m)
        # local_timestamp is a BIGINT: the local wall-clock time encoded as a UTC epoch (checked 11-03: CA rows are
        # utc_timestamp - 28800 in March), so the local date is the UTC date of that epoch
        own_v, spill_v = _shards(src, "visits", m)
        own_w, spill_w = ([], []) if hw_inline else _shards(src, "work-visits", m)
        if not own_v:
            raise FileNotFoundError(f"no visits shards for {m} under {src}")
        sel = ("SELECT caid, strftime(make_timestamp(local_timestamp * 1000000), '%Y-%m-%d') AS date, "
               "substr(census_block_group, 1, 11) AS d, minimum_dwell AS dwell, {w} AS work "
               "FROM read_parquet({f}) WHERE census_block_group IS NOT NULL AND local_timestamp IS NOT NULL"
               + (" AND coalesce(safegraph_place_id, '') NOT IN ('home', 'work')" if hw_inline else ""))
        parts = [sel.format(w=0, f=_lst(own_v + spill_v))]
        if own_w + spill_w:
            parts.append(sel.format(w=1, f=_lst(own_w + spill_w)))
        con.execute(f"""
        CREATE OR REPLACE TABLE j AS
        SELECT v.*, h.htract AS o
        FROM ({" UNION ALL ".join(parts)}) v
        JOIN homes h USING (caid)
        WHERE v.date BETWEEN '{d0}' AND '{d1}'""")
        # rows of the boundary shards (first day of this month, first day of the next) whose local date falls
        # outside the month: they belong to the neighbouring months and are counted there (or lost after 05-31)
        bnd = [f for f in own_v + spill_v + own_w + spill_w
               if f.name.startswith(f"{m}-01-") or f.name.startswith(f"{_next_month(m)}-01-")]
        outside = con.execute(
            f"SELECT count(*) FROM read_parquet({_lst(bnd)}) WHERE local_timestamp IS NOT NULL AND "
            f"strftime(make_timestamp(local_timestamp * 1000000), '%Y-%m-%d') NOT BETWEEN '{d0}' AND '{d1}'"
        ).fetchone()[0] if bnd else 0
        con.execute(f"""
        COPY (SELECT date, o, count(DISTINCT caid) AS active_devices FROM j GROUP BY 1, 2 ORDER BY 1, 2)
        TO '{out}/budget/{m}.parquet' (FORMAT PARQUET)""")
        con.execute(f"""
        CREATE OR REPLACE TABLE t AS
        SELECT date, CAST(hash(caid || '-' || '{seed}') % {n_groups} AS INTEGER) AS g, o, d, work::BOOLEAN AS work,
               count(*) AS n, count(DISTINCT caid) AS n_dev
        FROM (SELECT caid, date, o, d, max(work) AS work, max(dwell) AS dwell FROM j WHERE d <> o GROUP BY 1, 2, 3, 4)
        WHERE dwell >= {min_dwell}
        GROUP BY 1, 2, 3, 4, 5""")
        for stale in (out / "trips").glob(f"o_state=*/{m}_*.parquet"):   # OVERWRITE_OR_IGNORE keeps other names
            stale.unlink()
        con.execute(f"""
        COPY (SELECT date, g, o, d, work, n, substr(o, 1, 2) AS o_state FROM t)
        TO '{out}/trips' (FORMAT PARQUET, PARTITION_BY (o_state), OVERWRITE_OR_IGNORE 1, FILENAME_PATTERN '{m}_{{i}}')""")
        rows, trips, dev = con.execute("SELECT count(*), sum(n), sum(n_dev) FROM t").fetchone()
        rep[m] = {"rows": int(rows or 0), "trips": int(trips or 0), "device_day_dest": int(dev or 0),
                  "rows_outside_month": int(outside), "spill_shards": len(spill_v) + len(spill_w)}
        print(m, json.dumps(rep[m]), flush=True)
        con.execute("DROP TABLE j; DROP TABLE t")
    con.close()
    return rep


if __name__ == "__main__":
    from config import HW_INLINE
    print(json.dumps(build(VERASET, MONTHS, DERIVED / "homes.parquet", DERIVED, hw_inline=HW_INLINE)), flush=True)
