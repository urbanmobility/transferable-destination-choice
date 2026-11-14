"""Task 11: validation-set references per city (gravity, extended radiation, M2/M3 GLM) on the synthetic city."""
import importlib
import json

from test_units import city_env  # noqa: F401
from test_search_param import two_cities  # noqa: F401


def test_run_city_interface(city_env):  # noqa: F811
    root, csv, units = city_env
    units.build_city("TEST1", root, cities_csv=csv)
    import refs
    importlib.reload(refs)
    r = refs.run_city("TEST1")
    assert set(r["models"]) == {"grav_power", "grav_exp", "ext_rad", "M2_glm", "M3_glm", "S0"}   # S0: plan 4 Task 5
    for name, v in r["models"].items():
        assert "nll_per_visit" in v or "error" in v, name
    assert 0 < r["models"]["ext_rad"]["alpha"] < 5
    assert "gain_m2" in r
    saved = json.loads((root / "results/multicity/refs/TEST1.json").read_text())
    assert saved["city"] == "TEST1"


def test_features_skip_class_absent_from_data():
    """Review Focus: a facility class with no POI anywhere (Veraset has no NAICS 62 health places; small cities may
    have no transit POI) must not produce -inf / NaN composition features; its alr column is dropped."""
    import numpy as np
    import pandas as pd
    import run_gate_l
    rng = np.random.default_rng(0)
    u = pd.DataFrame({c: rng.integers(1, 50, 6).astype(float) for c in (f"n_{k}" for k in run_gate_l.CLASSES)})
    u["n_health"] = 0.0
    u["N"] = u[[f"n_{k}" for k in run_gate_l.CLASSES]].sum(1) + 3
    for c in ("population", "area_km2", "jobs", "road_km", "pop_potential", "n_transit", "park_share"):
        u[c] = rng.uniform(0.1, 10, 6)
    f0, plus, total, comp = run_gate_l.features(u)
    assert "alr_health" not in comp and "alr_food" in comp
    for name, v in comp.items():
        assert np.isfinite(np.asarray(v, float)).all(), name
    assert (comp["richness"] <= 7).all()


def test_main_refuses_to_run_without_city_roles(city_env, tmp_path):
    """Review Focus (sealing): refs.main must not silently compute every city when config/cities.csv carries no
    role column (the roles are written by the tiers stage; a stale copy of the file would unseal the confirmation
    cities)."""
    import pytest
    import refs
    importlib.reload(refs)
    csv = tmp_path / "cities_noroles.csv"
    csv.write_text("city,name\nTEST1,Test\n")
    refs.CITIES_CSV = csv
    with pytest.raises(RuntimeError, match="role"):
        refs.main(["TEST1"])
    assert not (tmp_path / "results/multicity/refs/TEST1.json").exists()


def test_main_skips_confirmation_city(city_env, tmp_path):
    root, csv, units = city_env
    units.build_city("TEST1", root, cities_csv=csv)
    import refs
    importlib.reload(refs)
    roles = tmp_path / "cities_roles.csv"
    roles.write_text("city,name,role\nTEST1,Test,confirm\n")
    refs.CITIES_CSV = roles
    refs.main(["TEST1"])
    assert not (root / "results/multicity/refs/TEST1.json").exists()


def test_run_city_refuses_a_confirmation_city(monkeypatch):
    """Final review (Important 4): the seal must not depend on a file — run_city itself refuses config.CONFIRM
    unless ODSR_UNSEAL=1 (set only at the freeze, with a ledger entry)."""
    import pytest
    import config
    import refs
    importlib.reload(refs)
    monkeypatch.delenv("ODSR_UNSEAL", raising=False)
    with pytest.raises(RuntimeError, match="sealed"):
        refs.run_city(config.CONFIRM[0])


def test_present_classes_excludes_an_absent_class():
    import numpy as np
    import pandas as pd
    import run_gate_l
    rng = np.random.default_rng(1)
    u = pd.DataFrame({f"n_{c}": rng.integers(1, 50, 6).astype(float) for c in run_gate_l.CLASSES})
    u["n_health"] = 0.0
    assert run_gate_l.present_classes(u) == [c for c in run_gate_l.CLASSES if c != "health"]


def test_s0_val_refits_corrected_gravity(two_cities):
    """Plan 4 Task 5 (design §5.7): S0 = log(N_j + c) - beta log(d + c0), refitted per city (train x search block,
    multi-start) and scored on val x validation block; the zero point of share_S0."""
    import numpy as np
    import library
    import refs
    importlib.reload(library)
    importlib.reload(refs)
    s0 = refs.s0_val("TEST1")
    assert np.isfinite(s0["nll_per_visit"]) and len(s0["consts"]) == 3
    gp = refs.run_city("TEST1")["models"]
    assert set(gp) >= {"S0", "grav_power"} and gp["S0"]["nll_per_visit"] == s0["nll_per_visit"]


def test_main_s0_only_updates_existing_files_without_touching_confirm(two_cities, monkeypatch):
    """--s0-only adds models.S0 to the existing refs/<city>.json in place (other keys unchanged); a confirmation city
    is neither read (no load / build call) nor written."""
    import sr_data
    import library
    import refs
    importlib.reload(library)
    importlib.reload(refs)
    root = two_cities
    d = root / "results/multicity/refs"
    d.mkdir(parents=True)
    old = {"city": "TEST1", "models": {"grav_power": {"nll_per_visit": 9.0}, "M2_glm": {"nll_per_visit": 8.0}},
           "gain_m2": 1.0}
    (d / "TEST1.json").write_text(json.dumps(old))
    (d / "TEST2.json").write_text(json.dumps({**old, "city": "TEST2"}))
    roles = root / "cities_roles.csv"
    roles.write_text("city,name,role\nTEST1,Test,rest\nTEST2,Test2,confirm\n")
    monkeypatch.setattr(refs, "CITIES_CSV", roles)
    seen, real_load = [], sr_data.load

    def spy(region, scale):
        seen.append(region)
        return real_load(region, scale)
    monkeypatch.setattr(sr_data, "load", spy)
    monkeypatch.setattr(refs, "load", spy)
    refs.main(["--s0-only"])
    new = json.loads((d / "TEST1.json").read_text())
    assert "S0" in new["models"] and new["models"]["grav_power"] == old["models"]["grav_power"] and new["gain_m2"] == 1.0
    assert "S0" not in json.loads((d / "TEST2.json").read_text())["models"]
    assert "TEST2" not in seen and "TEST1" in seen


def _train_everywhere(root):
    """Training-period trips from three origins to every other tract (the fixture has none into some folds), then
    TEST1 is rebuilt."""
    import pandas as pd
    import units
    from city_fixture import TRACTS
    rows, k = [], 0
    for o in (TRACTS[0], TRACTS[3], TRACTS[5]):
        for d in TRACTS:
            if d != o:
                rows.append(("2024-03-05", k % 20, o, d, False, 2 + k % 3))
                k += 1
    t = pd.DataFrame(rows, columns=["date", "g", "o", "d", "work", "n"])
    t["g"] = t["g"].astype("int32")
    for ss in ("06", "32"):
        t[t["o"].str[:2] == ss].to_parquet(root / f"derived/multicity/trips/o_state={ss}/2024-03_9.parquet", index=False)
    b = root / "derived/multicity/budget/2024-03.parquet"
    old = pd.read_parquet(b)
    pd.concat([old, pd.DataFrame({"date": ["2024-03-05"] * 3, "o": [TRACTS[0], TRACTS[3], TRACTS[5]],
                                  "active_devices": [5, 5, 5]})]).to_parquet(b, index=False)
    units.build_city("TEST1", root, cities_csv=root / "cities.csv")


def test_inner_refs_score_fold_4_of_the_training_period(two_cities):
    """Plan 4 Task 6 ruling: the inner-loop floor, delta and share compare fold-4 scores (training period, refits on
    folds 2, 3) with references fitted and scored on the same blocks: grav_power, grav_exp, M2_glm, ext_rad, S0."""
    import numpy as np
    import sr_data
    import library
    import refs
    _train_everywhere(two_cities)
    importlib.reload(library)
    importlib.reload(refs)
    got, real = {}, refs.s0_val

    def spy(city, tr=None, va=None):
        got.update(tr=tr, va=va)
        return real(city, tr=tr, va=va)
    refs.s0_val = spy
    r = refs.inner_refs("TEST1")
    assert set(r) == {"grav_power", "grav_exp", "M2_glm", "ext_rad", "S0", "blocks"}
    assert r["blocks"] == {"train": "search23", "split": "train", "eval": "fold:4"}
    d = sr_data.load("TEST1", "tract")
    m4 = sr_data.dest_mask(d, sr_data.folds(d), "fold:4")
    T = d["T"]["train"].copy()
    np.fill_diagonal(T, 0.0)
    want = T[d["orig"]][:, m4].sum()
    assert want > 0
    for k in ("grav_power", "grav_exp", "M2_glm", "ext_rad"):
        assert r[k]["visits"] == want and np.isfinite(r[k]["nll_per_visit"]), k
    # S0 is refitted on folds 2, 3 and scored on fold 4 (in this 6-tract city fold 4 is one tract and the refitted
    # distance offset makes log(d + c0) undefined there: recorded as an error, the inner loop then has no S0 share)
    f4 = set(np.flatnonzero(m4))
    assert set(got["va"].dest_unit) <= f4 and set(got["tr"].dest_unit).isdisjoint(f4)
    assert np.allclose(got["va"].y, d["T"]["train"][got["va"].origin_unit, got["va"].dest_unit])
    assert "nll_per_visit" in r["S0"] or r["S0"]["error"] == "S0 not fittable"
    saved = json.loads((two_cities / "results/multicity/refs/TEST1.json").read_text())
    assert saved["inner"]["grav_power"] == r["grav_power"]


def test_inner_refs_refuse_a_confirmation_city(monkeypatch):
    import pytest
    import config
    import refs
    importlib.reload(refs)
    monkeypatch.delenv("ODSR_UNSEAL", raising=False)
    with pytest.raises(RuntimeError, match="sealed"):
        refs.inner_refs(config.CONFIRM[0])
