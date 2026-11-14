"""Plan 8 Task 3 (design decision A1): the primary holdout test of the signed plan-7 hypotheses on the 26 law-test
cities, Holm over the registered family, write-once results and the report."""
import json

import numpy as np
import pandas as pd
import pytest

import config
import p7seal
import p8_test as pt
import sr_data

HYP = {"id": "P7-01", "statement": "y increases with x", "outcome": "y", "covariate": "x", "control": None, "direction": "+",
       "evidence": {"n": 35, "stat": 1.0, "ci": [0.5, 1.5], "rank_r": 0.6, "p": 0.0002, "plain_slope": 1.0}}


@pytest.fixture
def sealed_p7(tmp_path, monkeypatch):
    monkeypatch.setattr(p7seal, "REG", tmp_path / "registered_p7.json")
    monkeypatch.setattr(p7seal, "SIGNED", tmp_path / "P7_SIGNED")
    p7seal.REG.write_text(json.dumps([HYP], ensure_ascii=False))
    return tmp_path


def _ledger(monkeypatch):
    import ledger
    logged = []
    monkeypatch.setattr(ledger, "add", logged.append)
    return logged


def test_primary_frames_uses_exactly_the_lawtest_cities(tmp_path, monkeypatch):
    monkeypatch.setattr(sr_data, "ROOT", tmp_path)
    monkeypatch.setattr(pt.config, "cities_with_role", lambda r: ["L1", "L2"] if r == "lawtest" else [])
    p7 = tmp_path / "results/multicity/p7"
    p7.mkdir(parents=True)
    pd.DataFrame({"city": ["L1", "L2", "D1"], "block": "test_dest", "y": [1.0, 2.0, 3.0],
                  "log_population": [14.0, 15.0, 16.0]}).to_parquet(p7 / "city_outcomes_lawtest.parquet")
    pd.DataFrame({"city": ["L1", "L2"], "x": [0.1, 0.2]}).to_csv(p7 / "covariates_lawtest.csv", index=False)
    monkeypatch.setattr(pt.ex.laws, "covariates", lambda cs: pd.DataFrame({"city": cs, "log_population": [14.0, 15.0]}))
    out, cov, cities = pt.primary_frames()
    assert cities == ["L1", "L2"] and out["city"].tolist() == ["L1", "L2"] and cov["city"].tolist() == ["L1", "L2"]


def test_check_complete_refuses_missing_covariate_and_city():
    h = [{"id": "P7-01", "outcome": "y", "covariate": "x", "control": "log_population"}]
    out = pd.DataFrame({"city": ["a", "b"], "y": [1.0, np.nan]})
    cov = pd.DataFrame({"city": ["a", "b"], "x": [1.0, 2.0], "log_population": [5.0, 6.0]})
    pt.check_complete(h, out, cov, ["a", "b"])                                       # NaN outcome is allowed
    with pytest.raises(ValueError):
        pt.check_complete(h, out, cov.assign(log_population=[5.0, np.nan]), ["a", "b"])
    with pytest.raises(ValueError):
        pt.check_complete(h, out, cov, ["a", "b", "c"])


def test_run_tier_holm_over_ten(monkeypatch):
    ps = [0.0002, 0.004, 0.006, 0.03, 0.2, 0.5, 0.9, 1.0, 1.0, 0.0001]
    raw = [True] * 9 + [False]
    monkeypatch.setattr(pt.ex, "run_test", lambda h, out, cov, B=5000: {"id": h["id"], "p": ps[h["k"]],
                                                                         "pass_raw": raw[h["k"]], "n": 26})
    hyps = [{"id": f"P7-{k + 1:02d}", "k": k, "outcome": "y", "covariate": "x", "control": None, "direction": "+"}
            for k in range(10)]
    rows = pt.run_tier(hyps, None, None)
    assert [r["supported"] for r in rows] == [True, True, True] + [False] * 7 and {r["m"] for r in rows} == {10}
    assert rows[9]["holm"] is True and rows[9]["pass"] is False                      # Holm alone is not a pass


def test_test_stage_writes_once_and_refuses_changes(sealed_p7, tmp_path, monkeypatch):
    monkeypatch.setattr(sr_data, "ROOT", tmp_path)
    _ledger(monkeypatch)
    cities = [f"c{i:02d}" for i in range(30)]
    x = np.linspace(0.0, 1.0, 30)
    monkeypatch.setattr(pt, "primary_frames", lambda: (pd.DataFrame({"city": cities, "y": x}),
                                                       pd.DataFrame({"city": cities, "x": x}), cities))
    with pytest.raises(RuntimeError):
        pt.stage_test()                                                              # not signed
    p7seal.sign(log=lambda s: None)
    pt.stage_test()
    p = tmp_path / "results/multicity/laws/results_p7.json"
    res = json.loads(p.read_text())
    assert res["registration_sha"] == p7seal._sha(p7seal.REG) and res["rows"][0]["supported"] and len(res["cities"]) == 30
    with pytest.raises(FileExistsError):
        pt.stage_test()
    p.unlink()
    p7seal.REG.write_text(p7seal.REG.read_text() + " ")
    with pytest.raises(RuntimeError):
        pt.stage_test()                                                              # registration changed after signing
    monkeypatch.setattr(sr_data, "LINE", "M19")
    with pytest.raises(ValueError):
        pt.stage_test()


def test_render_has_every_id_and_no_nan():
    hyps = [HYP, dict(HYP, id="P7-02", statement="y decreases as z increases", covariate="z", direction="-")]
    results = {"cities": ["a"] * 26, "registration_sha": "79f476458c85" + "0" * 52, "B": 5000,
               "rows": [{"id": "P7-01", "n": 26, "stat": 0.9, "ci_lo": 0.4, "ci_hi": 1.3, "p": 0.0002, "m": 2,
                         "holm": True, "pass": True, "supported": True},
                        {"id": "P7-02", "n": 24, "stat": float("nan"), "ci_lo": None, "ci_hi": None, "p": 1.0, "m": 2,
                         "holm": False, "pass": False, "supported": False}]}
    md = pt.render(results, hyps)
    assert "P7-01" in md and "P7-02" in md and "nan" not in md.lower() and "Supported 1/2" in md and "A1" in md


def test_check_complete_refuses_absent_outcome_column():
    h = [{"id": "P7-02", "outcome": "transfer_loss_c0355r1", "covariate": "x", "control": None}]
    out = pd.DataFrame({"city": ["a", "b"], "y": [1.0, 2.0]})
    cov = pd.DataFrame({"city": ["a", "b"], "x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        pt.check_complete(h, out, cov, ["a", "b"])
