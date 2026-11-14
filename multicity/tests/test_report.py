"""Plan 4 Task 15: the version-1 result report (design §5.9) — content and sealing."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

S8 = ["35620", "16980", "31080", "BAY9", "26420", "12060", "14460", "42660"]
DISC = ["12420", "12580"]
LAWT = ["10740", "10900"]


def _world(res, confirm_id):
    (res / "loop/c1_discovery").mkdir(parents=True)
    (res / "refs").mkdir()
    (res / "laws").mkdir()
    (res / "agent/r0").mkdir(parents=True)
    pool = [{"cid": "c0209", "equation": "f = log(#6 + 3.0) - 2.0 * log(#1 + 2.1); p1 = []; p2 = []", "complexity": 11,
             "source": ["select_x.json", "pool_r1_4"], "round": 0},
            {"cid": "c0209c1", "equation": "f = log(#6 + 3.0) - 2.0 * log(#1 + 2.0); p1 = []; p2 = []", "complexity": 11,
             "source": ["cleaned_by_agent"], "parent": "c0209", "round": 0, "insight": "displacement distance"}]
    (res / "loop/pool.json").write_text(json.dumps(pool))
    rows = [{"cid": x["cid"], "city": c, "val_nll": 4.0 + 0.001 * k, "finite": True, "consts": "[1, 2, 3]",
             "se_vs_best": 0.0, "within_1se": k == 0} for k, x in enumerate(pool) for c in S8]
    pd.DataFrame(rows).to_parquet(res / "loop/inner_fits.parquet", index=False)
    (res / "loop/flags.json").write_text(json.dumps({"c0209": {"n": 0, "degenerate": []},
                                                      "c0209c1": {"n": 1, "unit_mixing": ["x1 + x6"], "degenerate": []}}))
    (res / "loop/ranked_r5.json").write_text(json.dumps([{"cid": "c0209", "gate1": True, "n_near": 8, "n_flags": 0,
                                                          "complexity": 11, "merged_from": []}]))
    (res / "loop/state.json").write_text(json.dumps({"round": 5, "cycle": 1, "llm_calls": 71,
                                                     "top8_history": [["c0209"], ["c0209"]], "phase": "inner"}))
    for c in S8 + DISC:
        m = {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 3.9}, "S0": {"nll_per_visit": 4.05}}
        (res / f"refs/{c}.json").write_text(json.dumps({"models": m, "inner": m}))
    pd.DataFrame({"cid": "c0209", "param": "f:c1", "city": S8 + DISC, "value": 2.0, "se": 0.01, "sd_starts": 0.0,
                  "identifiable": True}).to_parquet(res / "loop/c1_discovery/param_table.parquet", index=False)
    (res / "loop/c1_discovery/cross_city.json").write_text(json.dumps([{"cid": "c0209", "param": "f:c1",
        "covariate": "log_population", "slope": 0.31, "ci_lo": 0.1, "ci_hi": 0.5, "p": 0.01, "n": 10}]))
    pd.DataFrame({"cid": "c0209", "city": S8 + DISC + LAWT[:1], "finite": True, "near_best": True, "val_nll": 4.0,
                  "primary": True}).to_parquet(res / "loop/c1_discovery/cities_val.parquet", index=False)
    pd.DataFrame({"city": S8 + DISC, "primary": "c0209", "n_near": 1, "covered": True, "share_s0": 0.5,
                  "share_grav": 0.9}).to_csv(res / "loop/c1_discovery/cities_val_summary.csv", index=False)
    (res / "laws/registered.json").write_text(json.dumps([{"id": "H1-01", "statement": "distance decay steepens with population",
        "skeleton": "c0209", "parameter": "f:c1", "covariate": "log_population", "direction": "+", "test": "slope_ci",
        "criterion": "CI > 0", "cycle": 1, "sha": "abc"}]))
    (res / "laws/results.json").write_text(json.dumps([{"id": "H1-01", "p": 0.0123456, "stat": 0.98765, "pass_raw": True,
        "pass": True, "m": 1, "n": 2, "ci_lo": 0.4321, "ci_hi": 1.2345, "lawtest_cities": LAWT}]))
    (res / "loop/c1_laws.json").write_text(json.dumps([{"id": "H1-01", "joint": {"cid": "c0209", "param": "f:c1",
        "covariate": "log_population", "b0": 2.0, "b1": 0.3, "n_params": 4,
        "per_city": {c: {"nll": 4.0, "share_s0": 0.4, "param": 2.0} for c in DISC}}}]))
    (res / "agent/r0/review_c0209.json").write_text(json.dumps({"result": {"insight": "size combined with displacement distance, " + confirm_id,
        "problems": [{"issue": "i", "evidence": "e"}], "cleaned_sympy": "x"}, "verify": {"accepted": True}}))
    (res / "agent/stuck_1.json").write_text(json.dumps({"reason": "no_improvement", "result": {"tried": "t",
        "evidence": "e", "why_framework": "w", "proposal": "p"}}))


def test_report_excludes_sealed_and_lawtest_numbers(tmp_path, monkeypatch):
    import config
    import report
    importlib.reload(report)
    res = tmp_path / "res"
    _world(res, config.CONFIRM[0])
    monkeypatch.setattr(report, "_lawtest_ids", lambda: set(LAWT))
    out = tmp_path / "report.md"
    summ = report.build(out, results=res)
    md = out.read_text()
    for c in config.CONFIRM:
        assert c not in md
    for c in LAWT:
        assert c not in md
    for num in ("0.0123", "0.9876", "0.4321", "1.2345"):                  # lawtest statistics never printed
        assert num not in md
    assert "H1-01" in md and "| pass |" in md and "c0209" in md and "c0209c1" in md and "displacement distance" in md
    assert "size combined with displacement distance" in md and "no_improvement" in md and "log_population" in md
    assert summ["n_candidates"] == 2 and summ["n_laws"] == 1
    assert "share_S0" in md and "fold 4" in md
    assert "Coverage curve" in md and "| 1 | c0209 | 10 | 10 / 10 | 10 |" in md


def test_scan_refuses_sealed_ids():
    import config
    import report
    importlib.reload(report)
    with pytest.raises(ValueError):
        report._scan(f"text {config.CONFIRM[0]} more", lawtest=set())
    with pytest.raises(ValueError):
        report._scan("city 10740", lawtest={"10740"})
    report._scan("city 107400 and 35620", lawtest={"10740"})


def test_coverage_curve_greedy_over_near_best_and_floor():
    """Design decision of 2026-11-05: the library size is decided from coverage — greedily add the skeleton that covers most uncovered
    cities (covered = near-best and above the floor); ties go to the simpler skeleton."""
    import report
    importlib.reload(report)
    rows = []
    cov = {"A": ["c1", "c2"], "B": ["c1"], "C": ["c1", "c3"], "D": ["c2"], "E": ["c3"], "F": []}
    for city, cids in cov.items():
        for cid in ("c1", "c2", "c3"):
            rows.append({"cid": cid, "city": city, "finite": True, "near_best": cid in cids,
                         "val_nll": 4.0 if city != "E" else 4.95, "primary": cids[:1] == [cid]})
    cv = pd.DataFrame(rows)
    refs = {c: {"grav_power": {"nll_per_visit": 5.0}, "ext_rad": {"nll_per_visit": 4.9}} for c in cov}
    curve = report.coverage_curve(cv, refs, {"c1": 9, "c2": 7, "c3": 7})
    assert [(s["cid"], s["new"], s["covered"]) for s in curve] == [("c1", 3, 3), ("c2", 1, 4)]
    assert curve[0]["n_cities"] == 6 and curve[0]["primary_in"] == 3          # E: c3 is near-best but below the floor
    assert report.coverage_curve(cv.iloc[:0], refs, {}) == []


def test_scan_ignores_decimal_constants():
    import config
    import report
    importlib.reload(report)
    report._scan(f"`log(x6 + {config.CONFIRM[0]}.0)`", lawtest={"10740"})
    report._scan("`0.10740*x1`", lawtest={"10740"})
    with pytest.raises(ValueError):
        report._scan("City 10740.", lawtest={"10740"})
