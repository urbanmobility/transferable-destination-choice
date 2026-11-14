"""Plan 7 Task 3: data lines M (2024, all purposes; unchanged), M_nw (2024 non-work), M19 (2019-04 non-work). Each line
writes its own directories; the 2024 line's constants and paths stay exactly as before."""
import importlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import config
from city_fixture import TRACTS, make_city

SPLITS_M = {"train": ("2024-03-01", "2024-04-30"), "val": ("2024-05-01", "2024-05-15"), "test": ("2024-05-16", "2024-05-31")}


def test_line_M_constants_unchanged():
    c = config.line_constants("M")
    R = config.ROOT
    assert c["MONTHS"] == ("2024-03", "2024-04", "2024-05") and c["SPLITS"] == SPLITS_M
    assert c["MIN_TRAIN_TRIPS"] == 500_000 and c["INCLUDE_WORK"] is True and c["HW_INLINE"] is False
    assert c["POI_ID"] == "placekey" and c["CITIES_CSV"].name == "cities.csv"
    assert c["RESULTS"] == R / "results/multicity" and c["DERIVED"] == R / "derived/multicity"
    assert c["MI"] == R / "derived/model_inputs/M" and c["UNITS_DIR"] == R / "units/M"
    assert config.line_of("M") == "M" and config.line_of("L") == "M" and config.line_of(None) == "M"
    assert config.LINE == "M"                          # the test run has ODSR_LINE unset or M
    assert (config.MONTHS, config.SPLITS, config.MIN_TRAIN_TRIPS) == (c["MONTHS"], c["SPLITS"], 500_000)
    assert (config.RESULTS, config.DERIVED, config.MI, config.CITIES_CSV) == (c["RESULTS"], c["DERIVED"], c["MI"],
                                                                             c["CITIES_CSV"])


def test_line_M19_values():
    c = config.line_constants("M19")
    R = config.ROOT
    assert c["MONTHS"] == ("2019-04",)
    assert c["SPLITS"] == {"train": ("2019-04-01", "2019-04-20"), "val": ("2019-04-21", "2019-04-25"),
                           "test": ("2019-04-26", "2019-04-30")}
    assert c["MIN_TRAIN_TRIPS"] == 163_934 == round(500_000 * 20 / 61)
    assert c["INCLUDE_WORK"] is False and c["HW_INLINE"] is True and c["POI_ID"] == "safegraph_place_id"
    assert c["RESULTS"] == R / "results/multicity19" and c["DERIVED"] == R / "derived/multicity19"
    assert c["MI"] == R / "derived/model_inputs/M19" and c["UNITS_DIR"] == R / "units/M19"
    assert c["CITIES_CSV"].name == "cities_M19.csv" and c["SHARED"] == R / "derived/multicity"


def test_line_M_nw_values():
    c = config.line_constants("M_nw")
    R = config.ROOT
    assert c["MONTHS"] == ("2024-03", "2024-04", "2024-05") and c["SPLITS"] == SPLITS_M
    assert c["MIN_TRAIN_TRIPS"] == 500_000 and c["INCLUDE_WORK"] is False and c["HW_INLINE"] is False
    assert c["RESULTS"] == R / "results/multicity_nw" and c["DERIVED"] == R / "derived/multicity"
    assert c["MI"] == R / "derived/model_inputs/M_nw" and c["CITIES_CSV"].name == "cities.csv"


def test_results_dir_follows_the_root_at_call_time(tmp_path, monkeypatch):
    import sr_data
    monkeypatch.setattr(sr_data, "ROOT", tmp_path)
    assert config.results_dir() == tmp_path / "results/multicity"
    assert config.results_dir("M19") == tmp_path / "results/multicity19"


# ------------------------------------------------------------------------------------------------ units per line
@pytest.fixture
def line_env(tmp_path, monkeypatch):
    mods = []

    def make(line):
        monkeypatch.setenv("ODSR_ROOT", str(tmp_path))
        monkeypatch.setenv("ODSR_LINE", line)
        monkeypatch.setenv("ODSR_CITIES", "TEST1")
        import sr_data, form_vars, units, frozen, p7seal  # noqa: E401
        for m in (sr_data, config, form_vars, units, frozen, p7seal):
            importlib.reload(m)
        mods[:] = [sr_data, config, form_vars, units, frozen, p7seal]
        csv = tmp_path / "cities.csv"
        if not csv.exists():
            make_city(tmp_path, csv)
        return tmp_path, csv, units

    yield make
    monkeypatch.undo()
    for m in mods:
        importlib.reload(m)


def _shared_osm(root):
    from units import OSM_COLS
    base = root / "derived/model_inputs/M/TEST1/tract"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit": TRACTS, **{c: [1.0] * len(TRACTS) for c in OSM_COLS}}).to_parquet(base / "osm_features.parquet")


def _files(d):
    return sorted(str(p.relative_to(d)) for p in d.rglob("*")) if d.exists() else []


@pytest.mark.parametrize("line", ["M_nw", "M19"])
def test_sealed_lines_never_write_2024_paths(line_env, line):
    root, csv, units = line_env(line)
    if line == "M19":                                   # the 2019 national tables live in their own directory
        src, dst = root / "derived/multicity", root / "derived/multicity19"
        dst.mkdir(parents=True)
        for name in ("trips", "budget"):
            shutil.copytree(src / name, dst / name)
        for name in ("home_tracts.parquet", "poi_tracts.parquet"):
            shutil.copy(src / name, dst / name)
    _shared_osm(root)
    before = {d: _files(root / d) for d in ("derived/model_inputs/M", "units/M", "results/multicity")}
    units.build_city("TEST1", root, cities_csv=csv)
    assert {d: _files(root / d) for d in before} == before
    assert (root / f"derived/model_inputs/{line}/TEST1/tract/units.parquet").exists()
    assert (root / f"units/{line}/TEST1_tracts.geojson").exists()


def test_units_nonwork_drops_work_trips(line_env):
    root, csv, units = line_env("M_nw")
    _shared_osm(root)
    units.build_city("TEST1", root, cities_csv=csv)
    t = pd.concat([pd.read_parquet(p) for p in (root / "derived/multicity/trips").rglob("*.parquet")])
    t = t[(t["date"].astype(str) >= "2024-03-01") & (t["date"].astype(str) <= "2024-04-30")
          & t["o"].isin(TRACTS) & t["d"].isin(TRACTS)]
    T = np.load(root / "derived/model_inputs/M_nw/TEST1/tract/T_train.npz")
    assert T["n_work_5"].sum() == 0
    assert T["n_dev_5"].sum() == T["n_nonwork_5"].sum() == t.loc[~t["work"].astype(bool), "n"].sum() > 0
    assert t["work"].astype(bool).any()                 # the fixture has work trips, so something was dropped


# ------------------------------------------------------------------------------------------ 2019 national tables
def _visits19(rows):
    import pandas as pd
    return pa.table({"caid": [r[0] for r in rows], "local_timestamp": [int(pd.Timestamp(r[1]).timestamp()) for r in rows],
                     "census_block_group": [r[2] for r in rows], "minimum_dwell": [float(r[3]) for r in rows],
                     "safegraph_place_id": [r[4] for r in rows], "naics_code": [r[5] for r in rows]})


@pytest.fixture
def veraset19(tmp_path):
    d = tmp_path / "veraset19"
    (d / "visits").mkdir(parents=True)
    pq.write_table(_visits19([
        ("A", "2019-04-02 08:00:00", "060370002001", 600, "home", None),          # inline home row: never a trip
        ("A", "2019-04-02 09:00:00", "060370002002", 480, "work", None),          # inline work row with a CBG
        ("A", "2019-04-02 12:00:00", "060370003001", 30, "sg:abc", "722511"),     # the only trip
        ("A", "2019-04-03 12:00:00", "060370003001", 30, "sg:abc", "722511"),     # same POI again (another day)
    ]), d / "visits/2019-04-02--part-00000.snappy.parquet")
    homes = tmp_path / "homes19.parquet"
    pd.DataFrame({"caid": ["A"], "htract": ["06037000100"]}).to_parquet(homes, index=False)
    return d, homes


def test_trips_2019_drops_inline_home_work(veraset19, tmp_path):
    import trips
    src, homes = veraset19
    trips.build(src, ["2019-04"], homes, tmp_path / "out", hw_inline=True, threads=2, memory_limit="1GB")
    t = pd.concat([pd.read_parquet(p) for p in (tmp_path / "out/trips").rglob("*.parquet")])
    assert sorted(t["d"].unique()) == ["06037000300"] and not t["work"].any() and int(t["n"].sum()) == 2


def test_poi_uses_line_poi_id(veraset19, tmp_path):
    import poi
    src, _ = veraset19
    poi.build(src, ["2019-04"], tmp_path / "out", poi_id="safegraph_place_id", threads=2, memory_limit="1GB")
    p = pd.read_parquet(tmp_path / "out/poi_tracts.parquet").set_index("tract")
    assert list(p.index) == ["06037000300"] and int(p.loc["06037000300", "N"]) == 1


# ---------------------------------------------------------------------------------------------- roles and refs
def test_roles_m19():
    import roles_m19
    qc = pd.DataFrame({"eligible": [False, True, True, False]}, index=pd.Index(["A", "B", "C", "D"], name="city"))
    r24 = pd.DataFrame({"city": ["A", "B", "C", "D"], "role": ["lawtest", "confirm", "excluded", "excluded"]})
    out = roles_m19.assign(qc, r24)
    assert dict(zip(out["city"], out["role"])) == {"A": "old", "B": "old", "C": "new", "D": "excluded"}
    cities = pd.DataFrame({"city": ["A", "B", "Z"], "name": ["a", "b", "z"], "counties": ["1", "2", "3"],
                           "states": ["06", "06", "06"], "n_counties": ["1", "1", "1"], "role": ["lawtest", "x", "y"],
                           "tier": [None] * 3, "substitute_for": [None] * 3})
    init = roles_m19.init_table(cities, ["A", "B"])
    assert list(init["city"]) == ["A", "B"] and set(init["role"]) == {"pending"}
    assert list(init.columns) == list(cities.columns)


def test_refs_summary_of_a_sealed_line_has_no_scores():
    import refs
    res = {"10180": {"models": {"grav_power": {"nll_per_visit": 4.1234}, "M2_glm": {"nll_per_visit": 3.9876}}},
           "10420": {"models": {"grav_power": {"error": "boom"}, "M2_glm": {"nll_per_visit": 3.5}}}}
    s = refs.summary_line(res)
    assert "1 complete" in s and "1 with errors" in s and "10420" in s
    assert not re.search(r"\d\.\d{3}", s)


def test_refs_main_on_a_sealed_line_prints_no_scores(monkeypatch, capsys, tmp_path):
    import p7seal
    import refs
    monkeypatch.setattr(p7seal, "REG", tmp_path / "r.json")
    monkeypatch.setattr(p7seal, "SIGNED", tmp_path / "S")
    monkeypatch.setattr(refs, "_line", lambda: "M19")
    monkeypatch.setattr(refs, "_roles", lambda: pd.Series({"10180": "new"}))
    monkeypatch.setattr(refs, "run_city", lambda c: {"city": c, "models": {"grav_power": {"nll_per_visit": 4.1234},
                                                                           "M2_glm": {"nll_per_visit": 3.9876}},
                                                     "gain_m2": 0.1358})
    refs.main(["10180"])
    out = capsys.readouterr().out
    assert "10180" in out and not re.search(r"\d\.\d{3}", out)


# ------------------------------------------------------------------------------- final review fixes (plan 7)
def test_sealed_line_build_refuses_missing_shared_osm(line_env):
    root, csv, units = line_env("M_nw")
    with pytest.raises(RuntimeError, match="OSM"):
        units.build_city("TEST1", root, cities_csv=csv)


def test_units_main_failure_log_is_the_lines_own(line_env, monkeypatch):
    root, csv, units = line_env("M19")
    monkeypatch.setattr(units, "CITIES_CSV", csv, raising=False)
    before = _files(root / "results/multicity")
    units.main(["TEST1"])                                # no 2019 tables, no shared OSM: the city fails
    assert (root / "results/multicity19/logs/units_failed.txt").read_text().split() == ["TEST1"]
    assert _files(root / "results/multicity") == before


def test_frozen_paths_do_not_follow_the_data_line(line_env):
    root, _, _ = line_env("M19")
    import frozen
    assert frozen.FROZEN == root / "results/multicity/FROZEN" and frozen.FROZEN_DIR == root / "results/multicity/frozen"


def test_pipeline_refuses_non_line_aware_stages_on_sealed_lines(tmp_path):
    import os
    import subprocess
    import sys
    sh = Path(config.__file__).resolve().parent / "pipeline.sh"
    for stage in ("osm", "attrs", "form", "tiers", "units2", "shards"):
        env = {**os.environ, "ODSR_LINE": "M19", "ODSR_ROOT": str(tmp_path), "ODSR_PY": sys.executable}
        r = subprocess.run(["bash", str(sh), stage], env=env, capture_output=True, text=True)
        assert r.returncode == 2 and "not line-aware" in r.stderr, (stage, r.stderr)
    assert list(tmp_path.iterdir()) == []
