"""Plan 7 Task 5: the new city covariates (definitions fixed in config/p7_covariates.json before use)."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

import form_p7 as fp


def _u(**cols):
    n = len(next(iter(cols.values())))
    base = {"unit": [f"t{i}" for i in range(n)], "x": [0.0] * n, "y": [0.0] * n, "area_km2": [1.0] * n,
            "population": [1.0] * n, "jobs": [0.0] * n, "home_devices": [40] * n, "hh_no_vehicle_share": [0.0] * n,
            "road_km": [0.0] * n, "n_transit_osm": [0.0] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_defs_match_config():
    cfg = json.loads((Path(fp.__file__).parent / "config/p7_covariates.json").read_text())
    assert set(fp.DEFS) == set(cfg) and all(fp.DEFS[k] == cfg[k] for k in cfg)
    assert set(fp.DEFS) == {"transit_per_1k", "no_car_share", "road_density", "poi_mix", "jobs_housing",
                            "compactness", "secondary_centres", "circuity", "tract_area"}


def test_jobs_housing():
    assert fp.jobs_housing(_u(population=[100.0, 0.0], jobs=[0.0, 100.0])) == pytest.approx(1.0)
    assert fp.jobs_housing(_u(population=[50.0, 50.0], jobs=[10.0, 10.0])) == pytest.approx(0.0)


def test_compactness():
    u = _u(x=[0.0, 5000.0, 15000.0], y=[0.0, 0.0, 0.0], population=[1.0, 1.0, 2.0])
    assert fp.compactness(u, (0.0, 0.0)) == pytest.approx(0.5)


def test_cbd_centroid_is_job_weighted():
    u = _u(unit=["a", "b", "c"], x=[0.0, 10.0, 99.0], y=[0.0, 0.0, 0.0], jobs=[1.0, 3.0, 50.0])
    assert fp.cbd_xy(u, ["a", "b"]) == pytest.approx((7.5, 0.0))


def test_poi_mix():
    eq = {f"n_{c}": [10] for c in ("food", "retail", "education", "leisure", "service", "office", "transit")}
    assert fp.poi_mix(_u(**eq, n_health=[0])) == pytest.approx(math.log(7), rel=1e-12)
    one = {f"n_{c}": [0] for c in ("retail", "education", "leisure", "service", "office", "transit", "health")}
    assert fp.poi_mix(_u(n_food=[5], **one)) == pytest.approx(0.0)


def test_transit_and_no_car_and_area():
    assert fp.transit_per_1k(_u(population=[4000.0, 6000.0], n_transit_osm=[20.0, 30.0])) == pytest.approx(5.0)
    assert fp.no_car_share(_u(population=[1.0, 3.0], hh_no_vehicle_share=[0.4, 0.0])) == pytest.approx(0.1)
    assert fp.tract_area(_u(population=[1.0, 3.0], area_km2=[2.0, 6.0])) == pytest.approx(5.0)


def test_road_density_counts_built_tracts_only():
    u = _u(road_km=[10.0, 2.0, 100.0], area_km2=[2.0, 2.0, 50.0], home_devices=[40, 30, 5])
    assert fp.road_density(u) == pytest.approx(12.0 / 4.0)


def test_circuity_on_a_square():
    sq = [LineString([(0, 0), (1000, 0)]), LineString([(1000, 0), (1000, 1000)]),
          LineString([(1000, 1000), (0, 1000)]), LineString([(0, 1000), (0, 0)])]
    pts = np.array([[0.0, 0.0], [1000.0, 1000.0]])
    c = fp.circuity_from_lines(sq, pts, n_pairs=10, dmin_km=0.5, dmax_km=5.0, min_valid=1, seed=1)
    assert c == pytest.approx(math.sqrt(2), rel=1e-6)
    assert math.isnan(fp.circuity_from_lines(sq, pts, n_pairs=10, dmin_km=0.5, dmax_km=5.0, min_valid=2, seed=1))
    assert math.isnan(fp.circuity_from_lines(sq, pts, n_pairs=10, dmin_km=2.0, dmax_km=5.0, min_valid=1, seed=1))


def test_main_refuses_lawtest_before_signing(tmp_path, monkeypatch):
    import p7seal
    monkeypatch.setattr(p7seal, "REG", tmp_path / "r.json")
    monkeypatch.setattr(p7seal, "SIGNED", tmp_path / "S")
    monkeypatch.setattr(p7seal.config, "cities_with_role", lambda r: ["10900"] if r == "lawtest" else [])
    with pytest.raises(RuntimeError):
        fp.main(["12060", "10900"])


# ------------------------------------------------------------------ plan 8 Task 2: tagged, write-once covariate files
def test_main_writes_the_tag_file_once(tmp_path, monkeypatch):
    import sr_data
    monkeypatch.setattr(sr_data, "ROOT", tmp_path)
    monkeypatch.setattr(fp, "compute", lambda c: {"city": c, "road_density": 1.0})
    fp.main(["12060"], name="covariates_lawtest.csv")
    p = tmp_path / "results/multicity/p7/covariates_lawtest.csv"
    assert pd.read_csv(p, dtype={"city": str})["city"].tolist() == ["12060"]
    with pytest.raises(FileExistsError):
        fp.main(["12060"], name="covariates_lawtest.csv")
    with pytest.raises(ValueError):
        fp.main(["12060"], name="other.csv")
    assert not (tmp_path / "results/multicity/p7/covariates.csv").exists()


def test_cities_for_tags(monkeypatch):
    monkeypatch.setattr(fp.config, "cities_with_role", lambda r: [r])
    assert fp.cities_for("discovery") == ["search", "discovery"] and fp.cities_for("lawtest") == ["lawtest"]
    with pytest.raises(ValueError):
        fp.cities_for("new19")


def test_cities_for_tags_confirm(monkeypatch):
    monkeypatch.setattr(fp.config, "cities_with_role", lambda r: [r])
    assert fp.cities_for("confirm") == ["confirm"] and fp.COV_FILES["confirm"] == "covariates_confirm.csv"
