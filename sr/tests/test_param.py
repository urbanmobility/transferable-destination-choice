"""Task 1 of the multi-city plan: ROOT / LINE / CITY from the environment, city<N> templates, features(transit_col)."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))


def _reload(monkeypatch, **env):
    for k in ("ODSR_ROOT", "ODSR_LINE", "ODSR_CITIES"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import sr_data
    return importlib.reload(sr_data)


def test_root_and_line_from_env(monkeypatch, tmp_path):
    sr_data = _reload(monkeypatch, ODSR_ROOT=str(tmp_path), ODSR_LINE="M", ODSR_CITIES="31080,BAY9,35620")
    assert sr_data.ROOT == tmp_path and sr_data.LINE == "M"
    assert sr_data.CITY == {"31080": 0, "BAY9": 1, "35620": 2}


def test_defaults_unchanged(monkeypatch):
    sr_data = _reload(monkeypatch)
    assert sr_data.CITY == {"LA": 0, "SFBAY": 1, "SDCA": 2, "SRAA": 3, "RSBO": 4} and sr_data.LINE == "L"
    assert sr_data.ROOT == Path(__file__).resolve().parents[3]      # the parent of the scripts directory


def test_folds_small_city(monkeypatch):
    import warnings
    sr_data = _reload(monkeypatch)
    rng = np.random.default_rng(0)
    d = {"u": pd.DataFrame({"x": rng.random(6), "y": rng.random(6)}), "dest": np.ones(6, bool), "N": 6}
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # 20 clusters on 6 units would warn about empty clusters
        f = sr_data.folds(d)
    assert f.shape == (6,) and set(f) == {0, 1, 2, 3, 4}


def test_n_categories():
    from pysr_setup import n_categories
    assert (n_categories("city2"), n_categories("cityses2"), n_categories("city8"), n_categories("single")) == (2, 6, 8, 1)
    from refit_select import parse_equation
    c = parse_equation("f = #1; g = #2; h = #3; p1 = [1,2,3,4,5,6,7,8]; p2 = [1,2,3,4,5,6,7,8]", "city8", "A", 5)
    assert c.n_cat == 8 and len(c.consts) == 16


def test_features_transit_col():
    from run_gate_l import CLASSES, features
    u = pd.DataFrame({"population": [10, 20], "area_km2": [1, 2], "jobs": [5, 5], "road_km": [1, 1], "pop_potential": [1, 1],
                      "N": [8, 8], "park_share": [0, 0.1], "n_transit_osm": [3, 0], **{f"n_{c}": [1, 1] for c in CLASSES}})
    _, plus, _, _ = features(u, transit_col="n_transit_osm")
    assert np.allclose(plus["log_transit"], np.log1p([3, 0]))
    _, plus0, _, _ = features(u)
    assert np.allclose(plus0["log_transit"], np.log1p([1, 1]))


def test_form_vars_signature_and_paths(monkeypatch):
    import form_vars
    import inspect
    assert "scales" in inspect.signature(form_vars.main).parameters
    sr_data = _reload(monkeypatch, ODSR_LINE="M")
    importlib.reload(form_vars)
    assert form_vars.units_geojson("31080") == sr_data.ROOT / "units/M/31080_tracts.geojson"
    sr_data = _reload(monkeypatch)
    importlib.reload(form_vars)
    assert form_vars.units_geojson("LA") == sr_data.ROOT / "units/LA_tracts.geojson"
