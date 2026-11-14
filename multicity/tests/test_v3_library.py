"""Design decision of 2026-11-10 (V3 + item 5): the V3 skeleton library on the discovery cities (shape comparison,
parameter regularities) and cycle-4 hypotheses on the skeletons that enter only under V3, registered from discovery
evidence before any lawtest data is read."""
import json
import math

import numpy as np
import pandas as pd
import pytest

import laws
import v3_library as v3


def _profile():
    prof = {0: (-1.0, 0.50), 1: (-1.2, 0.55), 2: (-1.4, 0.60), 3: (-1.6, 0.70), 4: None, 5: (math.nan, math.nan)}
    refs = {b: {"d": d} for b, d in zip(range(6), (0.5, 1.5, 3.5, 7.0, 14.0, 30.0))}
    visits = {0: 5000, 1: 500, 2: 4000, 3: 3000, 4: 2000, 5: 1500}
    return prof, refs, visits


def test_skel_quantities_use_bands_with_enough_visits():
    prof, refs, visits = _profile()
    q = v3.skel_quantities(prof, refs, visits)
    want = float(np.polyfit(np.log([0.5, 3.5, 7.0]), [0.50, 0.60, 0.70], 1)[0])      # band 1: 500 visits
    assert q["ed_5_10km"] == -1.6 and q["eN_5_10km"] == 0.7 and q["slope_eN_logd"] == pytest.approx(want)
    q2 = v3.skel_quantities(prof, refs, {**visits, 3: 999, 2: 10})
    assert q2["ed_5_10km"] is None and q2["eN_5_10km"] is None and q2["slope_eN_logd"] is None   # < 3 bands


def test_library_from_coverage_keeps_the_clean_picks_of_both_multipliers_in_order():
    cov = {"V3": {"gate": {"0.03": {"clean": [{"cid": "a"}, {"cid": "b"}, {"cid": "c"}]},
                           "0.05": {"clean": [{"cid": "a"}, {"cid": "d"}, {"cid": "b"}]}}}}
    assert v3.library_from_coverage(cov) == ["a", "b", "c", "d"]


def _ev(cid, param, cov, slope, p, n, rho):
    return {"cid": cid, "param": param, "covariate": cov, "slope": slope, "p": p, "n": n, "rho": rho}


def test_select_takes_the_strongest_qualifying_association_per_skeleton():
    ev = [_ev("A", "f:c1", "log_population", 0.2, 0.01, 30, 0.60),
          _ev("A", "ed_5_10km", "log_jobs", -0.3, 0.001, 30, -0.70),
          _ev("A", "f:c0", "water_share", 1.0, 0.0, 10, 0.90),                 # too few cities
          _ev("A", "f:c2", "n_subcentres", -0.1, 0.02, 30, 0.80),              # slope and rho disagree
          _ev("B", "f:c0", "log_population", 0.1, 0.20, 30, 0.30)]             # not significant
    got = v3.select(ev, ["A", "B", "C"])
    assert got == [{"cid": "A", "param": "ed_5_10km", "covariate": "log_jobs", "direction": "-",
                    "evidence": {"slope": -0.3, "p": 0.001, "n": 30, "rho": -0.70}}]


def test_hypothesis_is_well_formed():
    h = v3.hypothesis({"cid": "c0514c1", "param": "f:c1", "covariate": "log_population", "direction": "+",
                       "evidence": {"slope": 0.2, "p": 0.0, "n": 33, "rho": 0.6}})
    assert all(str(h[f]).strip() for f in laws.FIELDS) and h["test"] == "slope_ci"
    assert (h["skeleton"], h["parameter"], h["covariate"], h["direction"]) == ("c0514c1", "f:c1", "log_population", "+")
    assert laws.COV_MEANING["log_population"] in h["statement"] and "增大" in h["statement"]


def test_register_cycle4_assigns_ids_and_refuses_repeats():
    old = [{"skeleton": "c0413c1", "parameter": "elasticity_5km", "covariate": "log_population", "id": "H1-01"}]
    hs = [v3.hypothesis({"cid": c, "param": "f:c1", "covariate": "log_population", "direction": "+",
                         "evidence": {"slope": 0.1, "p": 0.0, "n": 30, "rho": 0.5}}) for c in ("c0262", "c0514c1")]
    lines = []
    new = v3.register_cycle4(hs, old, log=lines.append)
    assert [h["id"] for h in new] == ["H4-01", "H4-02"] and all(h["cycle"] == 4 for h in new)
    assert [h["sha"] for h in new] == [laws._hsha(h) for h in hs] and len(lines) == 1 and "H4-02" in lines[0]
    with pytest.raises(ValueError):
        v3.register_cycle4(hs[:1], old + new, log=lines.append)


def test_band_rows_are_identifiable_when_finite():
    q = {"X1": {"ed_5_10km": -1.5, "eN_5_10km": None, "slope_eN_logd": 0.1}}
    df = v3.band_rows("c0262", q)
    assert set(zip(df.param, df.city)) == {("ed_5_10km", "X1"), ("slope_eN_logd", "X1")} and df.identifiable.all()


def test_lawtest_stage_refuses_before_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "LAWS", tmp_path)
    with pytest.raises(SystemExit):
        v3.require_registered()
    (tmp_path / "registered_v3.json").write_text("[]")
    with pytest.raises(SystemExit):
        v3.require_registered()
    (tmp_path / "registered_v3.json").write_text(json.dumps([{"id": "H4-01", "skeleton": "c0262"}]))
    assert v3.require_registered()[0]["id"] == "H4-01"


def _outputs(root, city):
    (root / "laws").mkdir(parents=True)
    pd.DataFrame({"cid": ["c1"], "city": [city], "n_bands": [5], "rmse_d": [0.2], "rmse_N": [0.1],
                  "rmse_N_shape": [0.08], "slope_N_skel": [0.1], "slope_N_flex": [0.12]}).to_parquet(root / "shape.parquet")
    (root / "params.json").write_text(json.dumps({
        "library": ["c1"], "gate": ["c1"],
        "summary": {"c1": {"f:c0": {"n": 30, "median": 0.5, "q10": 0.2, "q90": 0.9, "cv": 0.4}}},
        "evidence": [_ev("c1", "f:c0", "log_population", 0.1, 0.01, 30, 0.5)],
        "uncovered": {"0.03": [city], "0.05": [city]}}))
    (root / "laws/registered_v3.json").write_text(json.dumps([{"id": "H4-01", "skeleton": "c1", "parameter": "f:c0",
                                                                "covariate": "log_population", "direction": "+",
                                                                "statement": "s"}]))
    (root / "laws/results_v3.json").write_text(json.dumps([{"id": "H4-01", "stat": 0.1, "ci_lo": 0.05, "ci_hi": 0.2,
                                                             "p": 0.001, "n": 20, "m": 13, "holm": True, "pass": True,
                                                             "prev_still_pass": {"H1-01": True}}]))


def test_build_report_writes_sections_and_refuses_sealed_ids(tmp_path, monkeypatch):
    import report
    monkeypatch.setattr(report, "_lawtest_ids", lambda: {"99999"})
    monkeypatch.setattr(report, "_confirm_ids", lambda: set())
    _outputs(tmp_path / "ok", "12060")
    s = v3.build_report(tmp_path / "ok/r.md", root=tmp_path / "ok", laws_dir=tmp_path / "ok/laws")
    text = (tmp_path / "ok/r.md").read_text()
    assert s == {"library": 1, "registered": 1, "passed": 1}
    assert [ln.split()[1] for ln in text.splitlines() if ln.startswith("## ")] == ["1.", "2.", "3.", "4."]
    assert "12060" in text and "H4-01" in text
    _outputs(tmp_path / "bad", "99999")
    with pytest.raises(RuntimeError):
        v3.build_report(tmp_path / "bad/r.md", root=tmp_path / "bad", laws_dir=tmp_path / "bad/laws")
