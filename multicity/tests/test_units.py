"""Task 7: per-city standard tables isomorphic to round-1 model_inputs, built from the national tables."""
import importlib

import numpy as np
import pandas as pd
import pytest

from city_fixture import TRACTS, make_city


@pytest.fixture
def city_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ODSR_ROOT", str(tmp_path))
    monkeypatch.setenv("ODSR_LINE", "M")
    monkeypatch.setenv("ODSR_CITIES", "TEST1")
    import sr_data, config, form_vars, units  # noqa: E401
    for m in (sr_data, config, form_vars, units):
        importlib.reload(m)
    csv = tmp_path / "cities.csv"
    make_city(tmp_path, csv)
    yield tmp_path, csv, units
    monkeypatch.undo()                 # restore the real ODSR_ROOT, then re-point the reloaded modules at it:
    for m in (sr_data, config, form_vars, units):   # otherwise sr/tests run after this package read the tmp root
        importlib.reload(m)


def test_build_city_shapes_and_sums(city_env):
    root, csv, units = city_env
    r = units.build_city("TEST1", root, cities_csv=csv)
    base = root / "derived/model_inputs/M/TEST1/tract"
    u = pd.read_parquet(base / "units.parquet")
    assert list(u.unit) == TRACTS and set(u.unit.str[:2]) == {"06", "32"}        # both states collected, sorted
    for c in ("x", "y", "area_km2", "population", "income", "hh_no_vehicle_share", "ses", "jobs", "road_km", "pop_potential",
              "home_devices", "origin_included", "n_food", "n_transit", "n_other", "N", "n_transit_osm", "park_share"):
        assert c in u.columns, c
    assert u.loc[u.unit == TRACTS[1], "N"].item() == 6 and u.loc[u.unit == TRACTS[1], "n_retail"].item() == 2
    assert list(u.origin_included) == [True, False, True, True, False, True]
    assert u.loc[u.unit == TRACTS[2], "ses"].item() == 0 and set(u.ses) <= {0, 1, 2, 3}
    T = np.load(base / "T_train.npz")
    assert T["n_dev_5"].sum() == 8 and np.allclose(T["n_work_5"] + T["n_nonwork_5"], T["n_dev_5"])
    assert T["n_dev_5"][0, 1] == 2 and T["n_work_5"][0, 3] == 1 and T["n_dev_5"].shape == (6, 6)
    assert np.load(base / "T_val.npz")["n_dev_5"].sum() == 2 and np.load(base / "T_test.npz")["n_dev_5"].sum() == 1
    G = np.load(base / "T_groups_train.npz")["T"]
    assert G.shape == (20, 6, 6) and np.allclose(G.sum(0), T["n_dev_5"]) and G[2, 3, 4] == 3
    assert r["splits"]["train"]["coverage_out_of_city"] == pytest.approx(1 / 9) and r["o_unmatched_trips"] == 1
    assert r["splits"]["train"]["active_device_days"] == 7 and r["splits"]["val"]["dates"] == 1
    D = np.load(base / "D.npy")
    assert D.shape == (6, 6) and D[0, 0] == 0 and 0.8 < D[0, 1] < 1.2 and D[0, 3] > 100
    assert (base / "form.parquet").exists() and (base / "S.npy").exists()
    assert (root / "units/M/TEST1_tracts.geojson").exists()
    assert (base / "prepare_report.json").exists()


def test_build_city_merges_osm_features(city_env):
    root, csv, units = city_env
    base = root / "derived/model_inputs/M/TEST1/tract"
    base.mkdir(parents=True)
    pd.DataFrame({"unit": TRACTS, "road_km": [2.0] * 6, "n_transit_osm": [1, 0, 0, 0, 0, 0],
                  "park_share": [0.0, 0.25, 0, 0, 0, 0]}).to_parquet(base / "osm_features.parquet", index=False)
    units.build_city("TEST1", root, cities_csv=csv)
    u = pd.read_parquet(base / "units.parquet").set_index("unit")
    assert u.loc[TRACTS[1], "park_share"] == 0.25 and u.loc[TRACTS[0], "n_transit_osm"] == 1 and u.road_km.sum() == 12.0


def test_unit_table_entropy_finite_without_health_class(city_env):
    """Final review (Critical 1): Veraset has no health POI, so the class share of health is 0 everywhere and the
    entropy H_j was 0 * log 0 = NaN in every row; it must be the entropy over the present classes."""
    root, csv, units = city_env
    units.build_city("TEST1", root, cities_csv=csv)
    import sr_data
    u = pd.read_parquet(root / "derived/model_inputs/M/TEST1/tract/units.parquet")
    assert u["n_health"].sum() == 0
    s = sr_data.build(["TEST1"], "tract", "train", "search", None, 0)
    h = s.X[:, sr_data.VARS.index("H_j")]
    assert np.isfinite(h).all() and (h > 0).any()
