"""Task 5: NAICS -> 8 classes and the Veraset placekey table per tract."""
import pandas as pd
import pytest

from conftest import T0, T1, T2


@pytest.mark.parametrize("code,cls", [
    ("722511", "food"), ("445110", "retail"), ("621111", "health"), ("611110", "education"),
    ("713940", "leisure"), ("721110", "leisure"), ("512131", "leisure"), ("812112", "service"), ("522110", "service"),
    ("485113", "transit"), ("488119", "transit"), ("541110", "office"), ("523110", "office"), ("531110", "office"),
    ("236115", "other"), ("311811", "other"), (None, "other"), ("", "other"), (722511, "food"), ("4851", "transit"),
])
def test_classify(code, cls):
    from naics_map import classify
    assert classify(code) == cls


def test_sql_case_matches_python():
    import duckdb
    from naics_map import classify, sql_case
    codes = ["722511", "445110", "621111", "611110", "713940", "721110", "512131", "812112", "522110", "485113",
             "488119", "541110", "523110", "531110", "236115", "311811", None, "", "4851", "5121", "512110"]
    con = duckdb.connect()
    con.execute("CREATE TABLE c AS SELECT * FROM (VALUES " + ", ".join(
        f"({'NULL' if v is None else repr(v)})" for v in codes) + ") t(naics_code)")
    got = [r[0] for r in con.execute(f"SELECT {sql_case('naics_code')} FROM c").fetchall()]
    assert got == [classify(c) for c in codes]


def test_poi_counts(veraset_dir, tmp_path):
    from poi import build
    r = build(veraset_dir, ("2024-03",), tmp_path, threads=2, memory_limit="2GB")
    p = pd.read_parquet(tmp_path / "poi_tracts.parquet").set_index("tract")
    assert p.loc[T1, "n_food"] == 1 and p.loc[T1, "n_retail"] == 1 and p.loc[T1, "N"] == 2     # P1 (modal CBG in T1), P2
    assert T0 not in p.index                                                                    # P1's minority CBG does not count
    assert p.loc[T2, "n_other"] == 1 and p.loc[T2, "N"] == 1 and p.loc[T2, "n_food"] == 0
    assert r["placekeys"] == 3 and r["classified_share"] == pytest.approx(2 / 3)
