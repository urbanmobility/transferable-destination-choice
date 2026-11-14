"""Plan 7 Task 6: exploration on the discovery cities, the selection rule fixed before evidence, the registration
draft and the test function plan 8 will use on the holdout cities."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import explore_p7 as ex


def _world(n=40, seed=0, coef=2.0):
    rng = np.random.default_rng(seed)
    logpop = 15 + rng.normal(size=n)
    x = 0.6 * (logpop - 15) + 0.8 * rng.normal(size=n)
    y = coef * x - 3 * logpop + rng.normal(scale=0.1, size=n)
    cities = [f"c{i:02d}" for i in range(n)]
    out = pd.DataFrame({"city": cities, "ed_5_10km": y})
    cov = pd.DataFrame({"city": cities, "circuity": x, "log_population": logpop})
    return out, cov, cities


def test_relations_partial_slope_recovers_coefficient(monkeypatch):
    out, cov, cities = _world()
    monkeypatch.setattr(ex, "discovery", lambda: cities)
    rel = ex.relations(out, cov, cities, B=500)
    r = rel[(rel["outcome"] == "ed_5_10km") & (rel["covariate"] == "circuity")].iloc[0]
    assert r["partial_slope"] == pytest.approx(2.0, abs=0.2) and abs(r["slope"] - 2.0) > 0.5
    assert r["pci_lo"] > 0 and r["n"] == 40 and r["r_size"] == pytest.approx(np.corrcoef(cov["circuity"], cov["log_population"])[0, 1])
    s = rel[(rel["outcome"] == "ed_5_10km") & (rel["covariate"] == "log_population")].iloc[0]
    assert np.isnan(s["partial_slope"]) and s["slope"] < 0


def test_relations_refuses_non_discovery(monkeypatch):
    out, cov, cities = _world()
    monkeypatch.setattr(ex, "discovery", lambda: cities[:-1])
    with pytest.raises(ValueError):
        ex.relations(out, cov, cities, B=50)


def _rel_rows():
    rows = []
    for o in ("o1", "o2", "o3", "o4"):
        for k, c in enumerate(("a", "b", "c", "d")):
            rows.append({"outcome": o, "covariate": c, "n": 35, "spearman_rho": 0.5, "slope": 1.0, "ci_lo": 0.2,
                         "ci_hi": 1.8, "partial_slope": 1.0 + k, "pci_lo": 0.1, "pci_hi": 2.0, "partial_rho": 0.9 - 0.1 * k,
                         "r_size": 0.1, "p": 0.001})
    df = pd.DataFrame(rows)
    df.loc[(df["outcome"] == "o4") & (df["covariate"] == "d"), ["pci_lo", "pci_hi"]] = [-0.5, 0.5]      # CI has 0
    df.loc[df["covariate"] == "c", "r_size"] = 0.9                                                        # size proxy
    df.loc[(df["outcome"] == "o1") & (df["covariate"] == "d"), "n"] = 20                                   # too few
    return df


def test_select_caps_known_and_size_proxies():
    known = [{"outcome": "o2", "covariate": "a", "exclude": True}, {"outcome": "o*", "covariate": "zzz", "exclude": True},
             {"outcome": "o3", "covariate": "b", "exclude": False}]
    sel = ex.select(_rel_rows(), known, max_total=10, max_per_outcome=2, max_per_covariate=10)
    pairs = [(s["outcome"], s["covariate"]) for s in sel]
    assert len(sel) <= 10 and all(sum(o == x for x, _ in pairs) <= 2 for o in ("o1", "o2", "o3", "o4"))
    assert ("o2", "a") not in pairs and not any(c == "c" for _, c in pairs) and ("o4", "d") not in pairs
    assert ("o1", "d") not in pairs and ("o3", "b") in pairs
    assert pairs[0] == ("o1", "a")                                         # ranked by |partial_rho|
    assert all(s["direction"] == "+" and s["control"] == "log_population" and s["test"] == "partial_slope_ci" for s in sel)
    assert len(ex.select(_rel_rows(), [], max_total=3)) == 3
    few = ex.select(_rel_rows(), [], max_total=10, max_per_outcome=2, max_per_covariate=1)
    assert len({s["covariate"] for s in few}) == len(few)


def test_select_uses_the_plain_slope_for_log_population():
    rel = pd.DataFrame([{"outcome": "gain_grav_m2", "covariate": "log_population", "n": 35, "spearman_rho": -0.6,
                         "slope": -0.05, "ci_lo": -0.08, "ci_hi": -0.02, "partial_slope": np.nan, "pci_lo": np.nan,
                         "pci_hi": np.nan, "partial_rho": np.nan, "r_size": 1.0, "p": 0.001, "pearson_r": -0.6}])
    s = ex.select(rel, [])
    assert len(s) == 1 and s[0]["control"] is None and s[0]["test"] == "slope_ci" and s[0]["direction"] == "-"


def test_run_test_pass_and_fail():
    out, cov, _ = _world(n=30, seed=4, coef=1.0)
    h = {"id": "P7-01", "outcome": "ed_5_10km", "covariate": "circuity", "control": "log_population",
         "test": "partial_slope_ci", "direction": "+"}
    r = ex.run_test(h, out, cov, B=1000)
    assert r["pass_raw"] and r["n"] == 30 and r["ci_lo"] > 0 and r["stat"] == pytest.approx(1.0, abs=0.2)
    assert not ex.run_test({**h, "direction": "-"}, out, cov, B=1000)["pass_raw"]
    s = ex.run_test({**h, "covariate": "log_population", "control": None, "test": "slope_ci", "direction": "-"}, out, cov, B=1000)
    assert s["pass_raw"] and s["stat"] < 0


def test_draft_fields_and_ids(tmp_path, monkeypatch):
    import p7seal
    sel = ex.select(_rel_rows(), [], max_total=3)
    hyps = ex.draft(sel)
    assert [h["id"] for h in hyps] == ["P7-01", "P7-02", "P7-03"]
    for h in hyps:
        assert set(h) >= {"id", "statement", "outcome", "covariate", "control", "direction", "test", "block", "primary",
                          "replication", "criterion", "evidence"}
        assert (h["block"], h["primary"], h["replication"]) == ("test_dest", "lawtest_2024", "M19_new")
        assert "5000" in h["criterion"] and "Holm" in h["criterion"]
    monkeypatch.setattr(p7seal, "REG", tmp_path / "registered_p7.json")
    monkeypatch.setattr(p7seal, "SIGNED", tmp_path / "P7_SIGNED")
    ex.write_draft(hyps, tmp_path / "memo.md")
    assert json.loads(p7seal.REG.read_text())[0]["id"] == "P7-01" and "P7-03" in (tmp_path / "memo.md").read_text()
    p7seal.sign(log=lambda s: None)
    with pytest.raises(FileExistsError):
        ex.write_draft(hyps, tmp_path / "memo.md")


def test_relations_and_run_test_refuse_duplicate_cities(monkeypatch):
    out, cov, cities = _world(n=30)
    monkeypatch.setattr(ex, "discovery", lambda: cities)
    dup = pd.concat([out, out], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        ex.relations(dup, cov, cities, B=50)
    h = {"outcome": "ed_5_10km", "covariate": "circuity", "control": "log_population", "direction": "+"}
    with pytest.raises(ValueError, match="unique"):
        ex.run_test(h, dup, cov, B=50)


def test_memo_flags_prior_observations(tmp_path, monkeypatch):
    import p7seal
    known = [{"outcome": "o1", "covariate": "a", "exclude": False,
              "source": "own descriptive observation on 41 cities incl. the 6 confirmation cities"}]
    hyps = ex.draft(ex.select(_rel_rows(), known, max_total=1))
    monkeypatch.setattr(p7seal, "REG", tmp_path / "registered_p7.json")
    monkeypatch.setattr(p7seal, "SIGNED", tmp_path / "P7_SIGNED")
    ex.write_draft(hyps, tmp_path / "memo.md")
    memo = (tmp_path / "memo.md").read_text()
    assert "Prior observation" in memo and "confirmation cities" in memo


def test_known_relations_disclose_own_prior_observations():
    k = json.loads((Path(ex.__file__).parent / "config/p7_known.json").read_text())["known"]
    own = [x for x in k if "own" in x["source"]]
    assert {(x["outcome"], x["covariate"]) for x in own} >= {("theta_turn_c0514c1", "log_population"),
                                                           ("ed_5_10km_resid", "circuity")}
    assert all(not x["exclude"] for x in own)


# ------------------------------------------------------------------ plan 8 Task 3: one merge for exploration and test
def test_frames_matches_the_exploration_merge(tmp_path, monkeypatch):
    oc = pd.DataFrame({"city": ["a", "a", "b", "z"], "block": ["test_dest", "val", "test_dest", "test_dest"],
                       "ed_1_2km": [1.0, 9.0, 2.0, 3.0], "log_population": [5.0, 5.0, 6.0, 7.0]})
    monkeypatch.setattr(ex.laws, "covariates", lambda cs: pd.DataFrame({"city": cs, "log_population": [5.0, 6.0]}))
    pd.DataFrame({"city": ["a", "b"], "circuity": [1.2, 1.3]}).to_csv(tmp_path / "c.csv", index=False)
    out, cov = ex.frames(oc, ["a", "b"], tmp_path / "c.csv")
    assert out["city"].tolist() == ["a", "b"] and out["ed_1_2km"].tolist() == [1.0, 2.0]
    assert "block" not in out and "log_population" not in out
    assert cov.columns.tolist() == ["city", "log_population", "circuity"]
