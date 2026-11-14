"""Plan 6 Tasks 3-4: the confirmation stages (refs, fit, rescore, banded) and analyses. Nothing runs before the
freeze; a confirmation city's references are written by one job, run_city before test_refs."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

import confirm
import frozen
from test_fit_library import REFS, _pool
from test_search_param import two_cities  # noqa: F401


@pytest.fixture
def freeze_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, "FROZEN", tmp_path / "FROZEN")
    monkeypatch.setattr(frozen, "FROZEN_DIR", tmp_path / "frozen")

    def do():
        cfg, proto = tmp_path / "lib.json", tmp_path / "p.md"
        cfg.write_text(json.dumps({"entries": [{"cid": "c0001", "role": "main"}, {"cid": "c0003", "role": "reference"}]}))
        proto.write_text("p")
        frozen.freeze(cfg, proto, {x["cid"]: x for x in _pool()}, log=lambda s: None)
    return do


def test_stages_refuse_before_freeze(freeze_tmp):
    for st in ("refs", "fit", "rescore", "banded", "analyze", "report"):
        with pytest.raises(RuntimeError):
            confirm.STAGES[st]()


def test_refs_job_writes_once_per_city_in_order(freeze_tmp, monkeypatch):
    import config
    import refs
    freeze_tmp()
    calls = []
    monkeypatch.setattr(refs, "run_city", lambda c: calls.append(("run_city", c)))
    monkeypatch.setattr(refs, "test_refs", lambda c: calls.append(("test_refs", c)))
    monkeypatch.setenv("ODSR_UNSEAL", "0")
    monkeypatch.delenv("ODSR_UNSEAL")                         # restored (unset) after the test
    c = config.CONFIRM[0]
    assert confirm.refs_job(c) == c and confirm.refs_job("12060") == "12060"
    assert calls == [("run_city", c), ("test_refs", c), ("test_refs", "12060")]
    monkeypatch.setattr(refs, "test_refs", lambda c: (_ for _ in ()).throw(ValueError("boom")))
    assert "boom" in confirm.refs_job("12060")


def test_consts_match_flags_missing_cities():
    old = pd.DataFrame({"cid": ["a", "a", "b"], "city": ["X", "Y", "X"], "consts": ["[1.0, 2.0]", "[3.0]", None]})
    new = pd.DataFrame({"cid": ["a", "a", "b"], "city": ["X", "Y", "X"], "consts": ["[1.0, 2.0000001]", "[3.0]", None]})
    assert confirm.consts_match(new, old, ["a", "b"]) == pytest.approx(1e-7)
    assert confirm.consts_match(new[new.city == "X"], old, ["a"]) == np.inf
    assert confirm.consts_match(new.assign(consts=["[1.0, 2.0]", "[3.0]", "[1.0]"]), old, ["b"]) == np.inf
    assert confirm.consts_match(new.assign(consts=["[1.0, NaN]", "[3.0]", None]), old, ["a"]) == np.inf
    assert confirm.consts_match(new.assign(consts=["[1.0, 2.0]", "[3.0]", "null"]), old, ["b"]) == 0.0


def test_eval_set_is_the_merge_variant_candidates_in_order(tmp_path, monkeypatch):
    (tmp_path / "discovery").mkdir()
    (tmp_path / "loop").mkdir()
    pd.DataFrame({"cid": ["c0003", "c0001", "c0003"], "city": ["X", "X", "Y"]}).to_parquet(tmp_path / "discovery/cities_val.parquet")
    (tmp_path / "loop/pool.json").write_text(json.dumps(_pool()))
    monkeypatch.setattr(confirm, "MV", tmp_path)
    assert [x["cid"] for x in confirm.eval_set()] == ["c0003", "c0001"]


def test_rescore_reproduces_fit_scores(two_cities):
    import fit_library
    importlib.reload(fit_library)
    root = two_cities
    (root / "results/multicity/refs").mkdir(parents=True)
    for c in ("TEST1", "TEST2"):
        (root / f"results/multicity/refs/{c}.json").write_text(json.dumps({"city": c, "models": REFS}))
    out = root / "results/multicity/o"
    fit_library.main(["TEST1", "TEST2"], confirm=set(), procs=1, cands=_pool()[:2], out_dir=out, starts=2)
    cv = pd.read_parquet(out / "cities_val.parquet")
    rs = confirm.rescore(cv, _pool()[:2], "val", procs=1)
    m = cv.merge(rs, on=["cid", "city"], suffixes=("", "_r"))
    assert len(m) == len(cv) and (m.finite == m.finite_r).all()
    ok = m[m.finite.astype(bool)]
    assert np.allclose(ok.val_nll, ok.val_nll_r, atol=1e-10, rtol=0)
    assert np.allclose(ok.se_vs_best.astype(float), ok.se_vs_best_r.astype(float), atol=1e-10, rtol=0)
    assert (m.near_best == m.near_best_r).all() and (m.primary == m.primary_r).all()


# ------------------------------------------------------------------------------------------ Task 4: analyses, report
def test_clopper_pearson_known_values():
    lo, hi = confirm.clopper_pearson(4, 6)
    assert lo == pytest.approx(0.2228, abs=1e-3) and hi == pytest.approx(0.9567, abs=1e-3)
    assert confirm.clopper_pearson(0, 6)[0] == 0.0 and confirm.clopper_pearson(6, 6)[1] == 1.0


def test_prediction_band_contains_fitted_line():
    rng = np.random.default_rng(0)
    x = rng.normal(14.5, 0.8, 50)
    y = -0.15 * x + 0.05 * rng.normal(size=50)
    xn = np.array([14.5, 17.0])
    lo, hi = confirm.prediction_band(x, y, xn)
    b, a = np.polyfit(x, y, 1)
    assert np.all(lo < a + b * xn) and np.all(a + b * xn < hi)
    assert (hi - lo)[1] > (hi - lo)[0] > 2 * 1.9 * 0.05 * 0.9          # wider away from the mean; ~ t * sd


def test_transfer_consts_is_elementwise_median():
    cv = pd.DataFrame({"cid": ["a"] * 3 + ["b"], "city": ["X", "Y", "Z", "X"], "finite": [True, True, False, True],
                       "consts": ["[1.0, 10.0]", "[3.0, 30.0]", "[100.0, 100.0]", "[5.0]"]})
    assert confirm.transfer_consts(cv, "a") == [2.0, 20.0] and confirm.transfer_consts(cv, "b") == [5.0]
    assert confirm.transfer_consts(cv, "zzz") is None


def test_coverage_needs_a_clean_member_near_best_and_below_floor():
    refs = {c: {"grav_power": {"nll_per_visit": 4.2}, "grav_exp": {"nll_per_visit": 4.3}, "ext_rad": {"nll_per_visit": 4.25},
                "M2_glm": {"nll_per_visit": 4.0}} for c in ("A", "B", "C")}                    # delta 0.006 at 0.03
    rows = [("dirty", "A", 4.000), ("clean", "A", 4.004),                                   # A: clean within delta
            ("dirty", "B", 4.000), ("clean", "B", 4.010),                                   # B: clean 0.010 behind
            ("clean", "C", 4.25)]                                                           # C: above the floor
    cv = pd.DataFrame(rows, columns=["cid", "city", "val_nll"]).assign(finite=True, se_vs_best=0.0, within_1se=False,
                                                                       complexity=5)
    assert confirm.coverage(cv, refs, ["clean"], 0.03) == {"A": True, "B": False, "C": False}
    assert confirm.coverage(cv, refs, ["clean"], 0.05)["B"] is True


def test_law_checks_pass_at_five_of_six():
    rng = np.random.default_rng(2)
    x = rng.normal(14.5, 0.8, 60)
    fit = {f"F{k}": (x[k], -0.15 * x[k] + 0.03 * rng.normal()) for k in range(60)}
    good = {f"K{k}": (14.0 + 0.3 * k, -0.15 * (14.0 + 0.3 * k)) for k in range(6)}
    r = confirm.law_checks(fit, good)
    assert r["n_inside"] == 6 and r["pass"] and r["n_confirm"] == 6 and r["slope_confirm"] == pytest.approx(-0.15, abs=0.01)
    bad = {**good, "K0": (14.0, 5.0), "K1": None}
    r2 = confirm.law_checks(fit, bad)
    assert r2["n_inside"] == 4 and not r2["pass"] and r2["inside"]["K1"] is None
    r3 = confirm.law_checks({}, good)
    assert not r3["pass"] and r3["n_inside"] == 0 and r3["n_fit"] == 0 and r3["inside"]["K0"] is None
    s = confirm.sign_check({"K0": 0.1, "K1": 0.2, "K2": -0.01, "K3": 0.3, "K4": 0.1, "K5": None}, "+")
    assert (s["n_ok"], s["n"], s["pass"]) == (4, 6, False)


def test_p2_check_uses_the_discovery_10_90_range():
    r = confirm.p2_check(list(np.linspace(0.5, 0.9, 41)), [0.7, 0.72, 0.68, 0.71, 0.69, 0.7])
    assert r["pass"] and r["disc_q10"] == pytest.approx(0.54) and r["disc_q90"] == pytest.approx(0.86)
    assert not confirm.p2_check(list(np.linspace(0.5, 0.9, 41)), [0.2] * 6)["pass"]


def test_build_report_sections_and_no_lawtest_ids(freeze_tmp, tmp_path, monkeypatch):
    import config
    import report
    freeze_tmp()
    monkeypatch.setattr(report, "_lawtest_ids", lambda: {"99999"})
    c = config.CONFIRM[0]
    a = {"library": {"clean": ["c0001"], "entries": [["c0001", "main"]]},
         "P1": {"test_dest": {"0.03": {"confirm": {"k": 4, "n": 6, "ci": [0.22, 0.96], "cities": {c: True}},
                                       "discovery": {"k": 22, "n": 35, "ci": [0.45, 0.78]}}}},
         "P2": {"c0001": {"conf_median": 0.7, "disc_q10": 0.5, "disc_q90": 0.9, "pass": True}},
         "P3": {"c0001": {c: {"refit": 0.7, "transfer": None, "transfer_finite": False}}},
         "P4": [{"id": "H3-02", "kind": "sign", "n_ok": 6, "n": 6, "pass": True}],
         "P5": {"rmse_d": {"c0001": 6}, "rmse_N_shape": {"c0001": 6}}}
    (tmp_path / "analysis.json").write_text(json.dumps(a))
    s = confirm.build_report(tmp_path / "r.md", root=tmp_path)
    text = (tmp_path / "r.md").read_text()
    assert [ln.split()[1] for ln in text.splitlines() if ln.startswith("## ")] == ["P1", "P2", "P3", "P4", "P5"]
    assert c in text and s == {"pass": 2, "checks": 2}
    assert "Direct transfer not fittable: c0001 1/1" in text
    a["P3"] = {"c0001": {"99999": {"refit": 0.7, "transfer": 0.6}}}
    (tmp_path / "analysis.json").write_text(json.dumps(a))
    with pytest.raises(RuntimeError):
        confirm.build_report(tmp_path / "r.md", root=tmp_path)



def test_rescore_scores_every_row_that_has_constants(two_cities):
    """Review: a row not finite on the source block but with constants is scored on the target block."""
    import fit_library
    importlib.reload(fit_library)
    root = two_cities
    (root / "results/multicity/refs").mkdir(parents=True)
    for c in ("TEST1", "TEST2"):
        (root / f"results/multicity/refs/{c}.json").write_text(json.dumps({"city": c, "models": REFS}))
    out = root / "results/multicity/o"
    fit_library.main(["TEST1"], confirm=set(), procs=1, cands=_pool()[:2], out_dir=out, starts=2)
    cv = pd.read_parquet(out / "cities_val.parquet")
    k = int(np.flatnonzero(cv.finite.astype(bool).to_numpy())[0])
    want = float(cv.val_nll.iloc[k])
    cv2 = cv.copy()
    cv2.loc[cv2.index[k], "finite"] = False
    rs = confirm.rescore(cv2, _pool()[:2], "val", procs=1).set_index("cid")
    assert bool(rs.loc[cv.cid.iloc[k], "finite"]) and rs.loc[cv.cid.iloc[k], "val_nll"] == pytest.approx(want, abs=1e-10)
    cv3 = cv.copy()
    cv3.loc[cv3.index[k], "consts"] = "null"
    rs3 = confirm.rescore(cv3, _pool()[:2], "val", procs=1).set_index("cid")
    assert not bool(rs3.loc[cv.cid.iloc[k], "finite"])


def test_refs_missing_lists_absent_scores():
    good = {"nll_per_visit": 4.0}
    blk = {"grav_power": good, "M2_glm": good, "S0": good, "grav_exp": good, "ext_rad": {"error": "x"}}
    assert confirm.refs_missing({"test_dest": blk, "test_time": blk}, False) == []
    bad = {"test_dest": {**blk, "S0": {"error": "x"}}, "test_time": {"grav_power": good}}
    miss = confirm.refs_missing(bad, True)
    assert "test_dest:S0" in miss and "test_time:M2_glm" in miss and "models" in miss
    assert confirm.refs_missing({"test_dest": {**blk, "grav_power": {"error": "x"}, "grav_exp": {"error": "x"}},
                                 "test_time": blk}, False) == ["test_dest:grav_power", "test_dest:floor"]


def test_freeze_checks_the_protocol_inputs():
    laws_ = [{"id": i, "skeleton": k} for i, k in (("H1-01", "c0413c1"), ("H2-03", "c0079c1"), ("H3-01", "flex_banded"),
                                                    ("H3-02", "flex_banded"), ("H4-01", "c0262"), ("H4-02", "c0514c1"),
                                                    ("H4-03", "c0355r1"))]
    ev = ["c0413c1", "c0079c1", "c0262", "c0514c1", "c0355r1", "c0413c1r1"] + [f"x{k}" for k in range(20)]
    lib = ["c0355r1", "c0262", "c0514c1", "c0413c1r1", "c0079c1"]
    assert confirm.freeze_checks(ev, lib, laws_) == []
    assert any("26" in m for m in confirm.freeze_checks(ev[:-1], lib, laws_))
    assert any("c9999" in m for m in confirm.freeze_checks(ev, lib + ["c9999"], laws_))
    assert any("H4-04" in m for m in confirm.freeze_checks(ev, lib, laws_ + [{"id": "H4-04", "skeleton": "c0262"}]))
    assert any("c0413c1" in m for m in confirm.freeze_checks([e for e in ev if e != "c0413c1"] + ["y"], lib, laws_))


def test_p5_uses_the_v3_gate(tmp_path, monkeypatch):
    (tmp_path / "gates.json").write_text(json.dumps({"variants": {"V3": {"gate": [{"cid": "a"}, {"cid": "b"}]},
                                                                  "V1": {"gate": [{"cid": "z"}]}}}))
    monkeypatch.setattr(confirm, "MV", tmp_path)
    assert confirm.p5_cids() == ["a", "b"]


def test_later_stages_need_a_passed_fit(freeze_tmp, tmp_path, monkeypatch):
    freeze_tmp()
    monkeypatch.setattr(confirm, "ROOT", tmp_path / "confirm")
    for st in ("rescore", "analyze"):
        with pytest.raises(RuntimeError):
            confirm.STAGES[st]()
