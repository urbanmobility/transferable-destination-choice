"""Test setup for the multi-city pipeline: import paths and a synthetic Veraset directory."""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
for p in (SCRIPTS / "multicity", SCRIPTS / "sr", SCRIPTS / "model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

T0, T1, T2 = "06037000100", "06037000200", "06037000300"      # A's home tract, main destination, C's POI tract


def _epoch(s):
    """Veraset stores local_timestamp as a BIGINT: the local wall-clock time encoded as if it were UTC."""
    import pandas as pd
    return int(pd.Timestamp(s).timestamp())


def _visits(rows):
    cols = ["caid", "local_timestamp", "census_block_group", "minimum_dwell", "placekey", "naics_code"]
    rows = [(r[0], _epoch(r[1]), r[2], float(r[3]), r[4], r[5]) for r in rows]
    return pa.table({c: [r[k] for r in rows] for k, c in enumerate(cols)},
                    schema=pa.schema([("caid", pa.string()), ("local_timestamp", pa.int64()),
                                      ("census_block_group", pa.string()), ("minimum_dwell", pa.float64()),
                                      ("placekey", pa.string()), ("naics_code", pa.string())]))


@pytest.fixture
def veraset_dir(tmp_path):
    """Devices: A (home T0 in 03 and 04), B (home tie between two CBGs -> dropped), C (home 06059000400, 03 only),
    D (bad home CBG 'CA000123'). Visits as described in the Task 4 / Task 5 tests."""
    d = tmp_path / "veraset"
    for sub in ("home-location", "visits", "work-visits"):
        (d / sub).mkdir(parents=True)
    hl = lambda rows: pa.table({"CAID": [r[0] for r in rows], "CENSUS_BLOCK_GROUP": [r[1] for r in rows]})
    pq.write_table(hl([("A", "060370001001"), ("B", "060370002001"), ("C", "060590004001"), ("D", "CA000123")]),
                   d / "home-location/2024-03--data_x_0.snappy.parquet")
    pq.write_table(hl([("A", "060370001001"), ("B", "060370003001")]),
                   d / "home-location/2024-04--data_x_0.snappy.parquet")
    pq.write_table(_visits([
        ("A", "2024-03-01 10:00:00", "060370002001", 7, "P1", "722511"),
        ("A", "2024-03-01 12:00:00", "060370002002", 3, "P2", "445110"),     # same device-day-tract as above
        ("A", "2024-03-01 13:00:00", "060370001002", 30, "P1", "722511"),    # A's own home tract: no trip
        ("D", "2024-03-01 11:00:00", "060370002001", 10, "P1", "722511"),    # D has no valid home
    ]), d / "visits/2024-03-01--part-00000.snappy.parquet")
    pq.write_table(_visits([
        ("A", "2024-03-01 23:30:00", "060370002001", 10, "P1", "722511"),    # local date 03-01 in the 03-02 UTC shard
        ("C", "2024-03-02 09:00:00", "060370003001", 4, "P3", "236115"),     # dwell < 5: no trip, POI still counted
        ("B", "2024-03-02 09:00:00", "060370002001", 20, None, None),        # B unhoused; no placekey
    ]), d / "visits/2024-03-02--part-00000.snappy.parquet")
    pq.write_table(_visits([
        ("A", "2024-04-01 10:00:00", "060370002001", 10, "P1", "722511"),
    ]), d / "visits/2024-04-01--part-00000.snappy.parquet")
    pq.write_table(_visits([
        ("A", "2024-03-01 09:00:00", "060370002003", 480, None, None),       # work visit, same day and tract
    ]), d / "work-visits/2024-03-01--part-00000.snappy.parquet")
    pq.write_table(_visits([]), d / "work-visits/2024-03-02--part-00000.snappy.parquet")
    pq.write_table(_visits([]), d / "work-visits/2024-04-01--part-00000.snappy.parquet")
    return d


def pytest_configure(config):
    config.addinivalue_line("markers", "live: talks to the real local LLM service (runs only with LLM_LIVE=1)")


_EXIT = {}


def pytest_sessionfinish(session, exitstatus):
    _EXIT["status"] = int(exitstatus)


def pytest_unconfigure(config):
    """The Julia runtime (juliacall, loaded by the PySR tests) segfaults intermittently in its atexit teardown after
    every test has passed (exit 139 in ~1/3 of runs, already at 7a36f8c). Leave without that teardown, keeping
    pytest's own exit status."""
    import os
    if "juliacall" in sys.modules and "status" in _EXIT:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_EXIT["status"])
