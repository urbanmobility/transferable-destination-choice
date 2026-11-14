"""Task 4: modal homes and the national trips table (dedup by device-day-tract on the local date, work flag, groups)."""
import pandas as pd

from conftest import T0, T1


def test_homes_modal_and_ties(veraset_dir, tmp_path):
    from homes import build
    r = build(veraset_dir, ("2024-03", "2024-04"), tmp_path)
    h = pd.read_parquet(tmp_path / "homes.parquet").set_index("caid")
    assert h.loc["A", "htract"] == T0 and h.loc["A", "n_months"] == 2
    assert "B" not in h.index and r["ties_dropped"] == 1 and r["bad_cbg_rows"] == 1
    assert h.loc["C", "n_months"] == 1 and r["devices"] == 2
    ht = pd.read_parquet(tmp_path / "home_tracts.parquet").set_index("tract")
    assert ht.loc["06059000400", "home_devices"] == 1 and ht.loc[T0, "home_devices"] == 1


def test_trips_dedup_local_date_and_work(veraset_dir, tmp_path):
    from homes import build as hb
    from trips import build
    hb(veraset_dir, ("2024-03", "2024-04"), tmp_path)
    r = build(veraset_dir, ("2024-03",), tmp_path / "homes.parquet", tmp_path, n_groups=20, seed=1, threads=2,
              memory_limit="2GB")
    t = pd.read_parquet(tmp_path / "trips")
    a = t[(t.o == T0) & (t.d == T1)]
    assert len(a) == 1 and a.iloc[0]["date"] == "2024-03-01" and a.iloc[0]["n"] == 1 and bool(a.iloc[0]["work"])
    assert (t.o != t.d).all() and len(t) == 1 and r["2024-03"]["trips"] == 1
    assert set(t.columns) >= {"date", "g", "o", "d", "work", "n"} and 0 <= a.iloc[0]["g"] < 20
    b = pd.read_parquet(tmp_path / "budget").set_index(["date", "o"])
    assert b.loc[("2024-03-01", T0), "active_devices"] == 1
    assert b.loc[("2024-03-02", "06059000400"), "active_devices"] == 1      # C visited (dwell < 5) but was active


def test_group_stable_across_months(veraset_dir, tmp_path):
    from homes import build as hb
    from trips import build
    hb(veraset_dir, ("2024-03", "2024-04"), tmp_path)
    build(veraset_dir, ("2024-03", "2024-04"), tmp_path / "homes.parquet", tmp_path, n_groups=20, seed=1, threads=2,
          memory_limit="2GB")
    t = pd.read_parquet(tmp_path / "trips")
    assert sorted(t["date"]) == ["2024-03-01", "2024-04-01"] and t["g"].nunique() == 1
    assert (tmp_path / "trips/o_state=06").is_dir() and len(list((tmp_path / "trips/o_state=06").glob("*.parquet"))) == 2


def test_month_boundary_spill_is_read_from_the_next_months_first_shard(veraset_dir, tmp_path):
    """Final review (Important 2): a visit in the evening of 03-31 local sits in the 2024-04-01 UTC shard; month 03
    must read that shard too (and count the rows it drops as outside the month)."""
    import pyarrow.parquet as pq
    from conftest import _visits
    from homes import build as hb
    from trips import build
    pq.write_table(_visits([("A", "2024-03-31 23:00:00", "060370002001", 10, "P1", "722511")]),
                   veraset_dir / "visits/2024-04-01--part-00001.snappy.parquet")
    hb(veraset_dir, ("2024-03", "2024-04"), tmp_path)
    r = build(veraset_dir, ("2024-03", "2024-04"), tmp_path / "homes.parquet", tmp_path, n_groups=20, seed=1,
              threads=2, memory_limit="2GB")
    t = pd.read_parquet(tmp_path / "trips")
    assert sorted(t["date"]) == ["2024-03-01", "2024-03-31", "2024-04-01"]
    assert r["2024-03"]["trips"] == 2 and r["2024-03"]["rows_outside_month"] == 1 and r["2024-03"]["spill_shards"] == 3
    assert r["2024-04"]["trips"] == 1 and r["2024-04"]["spill_shards"] == 0      # 2024-05-01 shards absent


def test_rebuild_removes_stale_partition_files(veraset_dir, tmp_path):
    """Final review (Important 3): OVERWRITE_OR_IGNORE keeps files of an earlier run with other names; a rebuild
    must clear the month's files first or counts double."""
    import shutil
    from homes import build as hb
    from trips import build
    hb(veraset_dir, ("2024-03", "2024-04"), tmp_path)
    kw = dict(n_groups=20, seed=1, threads=2, memory_limit="2GB")
    build(veraset_dir, ("2024-03",), tmp_path / "homes.parquet", tmp_path, **kw)
    part = tmp_path / "trips/o_state=06"
    files = sorted(part.glob("2024-03_*.parquet"))
    stale = part / "2024-03_7.parquet"
    shutil.copy(files[0], stale)
    assert len(pd.read_parquet(tmp_path / "trips")) == 2
    build(veraset_dir, ("2024-03",), tmp_path / "homes.parquet", tmp_path, **kw)
    assert not stale.exists() and len(pd.read_parquet(tmp_path / "trips")) == 1
