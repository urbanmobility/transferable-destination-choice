"""Plan 4 Tasks 11-13: cross-city analysis (parameter table, identifiability, covariates, regression summaries,
composites, term existence, agent payload), hypothesis registration / tests, parameterised skeletons."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from test_search_param import two_cities  # noqa: F401

EQ = "f = log(#6 + 1.0) - 1.5 * log(#1 + 1.0) - 0.3 * log(#26 + 1.0); p1 = []; p2 = []"
EQ2 = "f = log(#6 + 2.0) - 1.9 * #2; p1 = []; p2 = []"


def _cities(n=10):
    return [f"D{k:02d}" for k in range(n)]


def _cov(n=10, seed=0):
    rng = np.random.default_rng(seed)
    c = pd.DataFrame({"city": _cities(n), "log_population": np.linspace(13, 16, n)})
    for k in ("pop_density_gini", "cbd_job_share", "subcentre_job_share", "n_subcentres", "orientation_entropy",
              "water_share", "log_jobs"):
        c[k] = rng.normal(size=n)
    return c


def _cities_val(n=10, seed=0):
    """3 candidates x n cities: c1's distance coefficient (consts[1], 'f:c1') = 1.0 + 0.4 * log_population + noise;
    c1's S_ij coefficient has start dispersion above its SE (not identifiable); c2 is a 2-parameter power form."""
    rng = np.random.default_rng(seed)
    cov = _cov(n, seed)
    rows = []
    for k, city in enumerate(_cities(n)):
        lp = cov.loc[k, "log_population"]
        beta = 1.0 + 0.4 * lp + rng.normal(0, 0.02)
        rows.append({"cid": "c1", "city": city, "finite": True, "val_nll": 4.0,
                     "consts": json.dumps([3.0, beta, 1.2, 0.3, 1.0]), "consts_se": json.dumps([0.5, 0.01, 0.1, 0.05, 0.6]),
                     "consts_sd": json.dumps([0.0, 0.0, 0.0, 0.2, 0.0]),
                     "consts_cov": json.dumps(np.diag([0.25, 1e-4, 0.01, 0.0025, 0.36]).tolist())})
        rows.append({"cid": "c2", "city": city, "finite": True, "val_nll": 4.1, "consts": json.dumps([2.0, 1.9]),
                     "consts_se": json.dumps([0.3, 0.02]), "consts_sd": json.dumps([0.0, 0.0]),
                     "consts_cov": json.dumps(np.diag([0.09, 4e-4]).tolist())})
        rows.append({"cid": "c3", "city": city, "finite": False, "val_nll": None, "consts": "null", "consts_se": "null",
                     "consts_sd": "null", "consts_cov": "null"})
    return pd.DataFrame(rows), cov


def _pool():
    return [{"cid": "c1", "equation": EQ, "complexity": 14}, {"cid": "c2", "equation": EQ2, "complexity": 7},
            {"cid": "c3", "equation": EQ2, "complexity": 7}]


def test_param_table_and_identifiability():
    import laws
    importlib.reload(laws)
    cv, _ = _cities_val()
    pt = laws.param_table(cv, _pool())
    assert set(pt.columns) >= {"cid", "param", "city", "value", "se", "sd_starts", "identifiable"}
    assert set(pt.cid) == {"c1", "c2"} and len(pt[pt.cid == "c1"]) == 50
    assert sorted(pt[pt.cid == "c1"].param.unique()) == ["f:c0", "f:c1", "f:c2", "f:c3", "f:c4"]
    by = pt.groupby(["cid", "param"]).identifiable.all()
    assert by[("c1", "f:c1")] and not by[("c1", "f:c3")] and not by[("c1", "f:c4")]      # sd > se; |v| < 2 se


def test_cross_city_summary_detects_a_slope_and_skips_unidentifiable():
    import laws
    importlib.reload(laws)
    cv, cov = _cities_val()
    pt = laws.param_table(cv, _pool())
    summ = laws.cross_city_summary(pt, cov, B=200)
    hit = next(s for s in summ if (s["cid"], s["param"], s["covariate"]) == ("c1", "f:c1", "log_population"))
    assert hit["ci_lo"] > 0 and abs(hit["slope"] - 0.4) < 0.05 and hit["n"] == 10 and 0 <= hit["p"] < 0.05
    assert not any(s["param"] in ("f:c3", "f:c4") and s["cid"] == "c1" for s in summ)
    assert {s["covariate"] for s in summ} == set(laws.COVARIATES)
    assert laws.cross_city_summary(pt, cov, B=200) == summ                                # seeded


def test_composite_elasticity_at_reference_distance():
    import laws
    importlib.reload(laws)
    cv, _ = _cities_val(n=2)
    ref = {c: {"d": 5.0, "S_ij": 9.0, "dlogS_dlogd": 2.0} for c in _cities(2)}
    comp = laws.composites("c1", cv, _pool(), ref)
    r = comp.iloc[0]
    c = json.loads(cv[(cv.cid == "c1") & (cv.city == r.city)].consts.iloc[0])
    want = -c[1] * 5.0 / (5.0 + c[2]) - c[3] * 9.0 / (9.0 + c[4]) * 2.0
    assert r.param == "elasticity_5km" and r.value == pytest.approx(want, rel=1e-6) and r.se > 0
    assert laws.composites("c2", cv, _pool(), ref).empty                                 # no S_ij term


def test_term_existence_uses_delta_city():
    import laws
    importlib.reload(laws)
    refs = {"A": {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 4.0}},
            "B": {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 4.5}}}
    abl = {("c1", "A"): [{"comp": "f", "key": "-c3*log(c4 + x26)", "term": "-0.3*log(x26 + 1)", "delta_nll": 0.02}],
           ("c1", "B"): [{"comp": "f", "key": "-c3*log(c4 + x26)", "term": "-0.2*log(x26 + 3)", "delta_nll": 0.02}]}
    ex = laws.term_existence(abl, refs)
    assert list(ex.columns) == ["cid", "comp", "term", "city", "delta_nll", "exists"]
    assert dict(zip(ex.city, ex.exists)) == {"A": False, "B": True}                      # delta 0.03 vs 0.015
    assert set(ex.term) == {"-c3*log(c4 + x26)"}


def test_covariates_from_form_metrics(two_cities, monkeypatch):
    import config
    import laws
    importlib.reload(laws)
    d = two_cities / "results/multicity"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"city": ["A", "B", config.CONFIRM[0]], "population": [1e6, 2e6, 3e6], "jobs": [5e5, 1e6, 1e6],
                  "pop_density_gini": [0.3, 0.4, 0.5], "cbd_job_share": 0.1, "subcentre_job_share": 0.3,
                  "n_subcentres": 4, "orientation_entropy": 3.2, "orientation_order": 0.2, "water_share": 0.01}
                 ).to_csv(d / "form_metrics.csv", index=False)
    monkeypatch.setattr(laws, "RESULTS", d)
    cov = laws.covariates(["B", "A"])
    assert list(cov.city) == ["B", "A"] and cov.log_population.iloc[1] == pytest.approx(np.log(1e6))
    assert list(cov.columns) == ["city"] + list(laws.COVARIATES)
    with pytest.raises(ValueError, match="sealed"):
        laws.covariates(["A", config.CONFIRM[0]])


def test_analysis_payload_is_city_level_without_lawtest_or_confirm(monkeypatch):
    import llm
    import laws
    importlib.reload(laws)
    cv, cov = _cities_val()
    pt = laws.param_table(cv, _pool())
    summ = laws.cross_city_summary(pt, cov, B=50)
    ex = pd.DataFrame({"cid": "c1", "comp": "f", "term": "-c3*log(c4 + x26)", "city": _cities(), "delta_nll": 0.05,
                       "exists": [True] * 6 + [False] * 4})
    monkeypatch.setattr(laws, "_lawtest_ids", lambda: {"LT01"})
    p = laws.analysis_payload(summ, ex, pt, _pool())
    llm.check_input(p)
    s = json.dumps(p, ensure_ascii=False)
    assert "LT01" not in s and "c1" in s and p["term_existence"][0]["n_exists"] == 6 and p["n_cities"] == 10
    assert set(p["methods"]) == {"slope_ci", "group_diff", "term_exist_prop"} and "log_population" in p["covariates"]
    assert any(x["param"] == "f:c1" for x in p["identifiable"]["c1"])
    bad = pd.concat([ex, ex.head(1).assign(city="LT01")])
    with pytest.raises(ValueError, match="lawtest"):
        laws.analysis_payload(summ, bad, pt, _pool())


def test_cross_city_summary_is_reproducible_across_processes():
    """The loop must be replayable from its logs: the bootstrap seed may not depend on Python's per-process hash."""
    import subprocess
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    code = ("import sys; sys.path[:0] = [%r, %r, %r]; import test_laws as t, laws; cv, cov = t._cities_val(); "
            "s = laws.cross_city_summary(laws.param_table(cv, t._pool()), cov, B=30); "
            "h = [x for x in s if x['param'] == 'f:c1' and x['cid'] == 'c1']; print([(x['ci_lo'], x['ci_hi']) for x in h])"
            % (str(here), str(here.parent), str(here.parents[1] / "sr")))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": h,
                           "PATH": "/usr/bin"}).stdout for h in ("1", "2")}
    assert len(outs) == 1 and next(iter(outs)).strip()


# ---- Task 12: registration and tests on the lawtest cities ----

def _h(**k):
    h = {"statement": "distance decay steepens as city population grows", "skeleton": "c1", "parameter": "f:c1", "covariate": "log_population",
         "direction": "+", "test": "slope_ci", "criterion": "95% CI of the slope excludes 0, positive"}
    h.update(k)
    return h


def test_register_refuses_unidentifiable_and_duplicate(tmp_path):
    """Review Focus 4: an unidentifiable parameter, an unknown test, an illegal covariate or a pair already registered
    is refused with its reason and never reaches the test."""
    import laws
    importlib.reload(laws)
    cv, _ = _cities_val()
    pt = laws.param_table(cv, _pool())
    log = []
    h_ok, h_unident, h_dup, h_bad_test = _h(), _h(parameter="f:c3"), _h(direction="-"), _h(covariate="water_share",
                                                                                          test="anova")
    h_bad_cov = _h(covariate="median_income")
    acc, rej = laws.register([h_ok, h_unident, h_dup, h_bad_test, h_bad_cov], 1, pt, registered=[], out_dir=tmp_path,
                             log=log.append)
    assert [h["id"] for h in acc] == ["H1-01"] and acc[0]["cycle"] == 1 and acc[0]["sha"]
    assert [r["reason"] for r in rej] == ["parameter not identifiable", "pair already registered", "unknown test",
                                          "unknown covariate"]
    saved = json.loads((tmp_path / "registered.json").read_text())
    assert [h["id"] for h in saved] == ["H1-01"] and len(log) == 1 and "H1-01" in log[0]
    acc2, rej2 = laws.register([_h(), _h(parameter="f:c2")], 2, pt, registered=saved, out_dir=tmp_path, log=log.append)
    assert [r["reason"] for r in rej2] == ["pair already registered"] and [h["id"] for h in acc2] == ["H2-01"]
    assert [h["id"] for h in json.loads((tmp_path / "registered.json").read_text())] == ["H1-01", "H2-01"]
    _, rej3 = laws.register([{k: v for k, v in _h(covariate="water_share").items() if k != "criterion"}], 2, pt,
                            registered=[], out_dir=tmp_path, log=log.append)
    assert rej3[0]["reason"].startswith("missing fields")


def test_register_caps_five_per_cycle(tmp_path):
    import laws
    importlib.reload(laws)
    cv, _ = _cities_val()
    pt = laws.param_table(cv, _pool())
    hs = [_h(covariate=k) for k in laws.COVARIATES]                                       # 8 distinct pairs
    acc, rej = laws.register(hs, 1, pt, registered=[], out_dir=tmp_path, log=lambda s: None)
    assert len(acc) == 5 and [r["reason"] for r in rej] == ["cycle cap reached"] * 3
    assert [h["id"] for h in acc] == [f"H1-0{k}" for k in range(1, 6)]


def test_holm_steps_down():
    import laws
    assert laws.holm([0.001, 0.011, 0.02, 0.2, 0.5, 0.6, 0.004]) == [True, False, False, False, False, False, True]
    assert laws.holm([0.01, 0.04], alpha=0.05) == [True, True] and laws.holm([]) == []


def test_holm_uses_cumulative_count(tmp_path):
    """7 hypotheses over two cycles: the stored raw p of the first six are reused (each is tested once), the seventh is
    tested now on the lawtest cities; 0.012 passes on its own but fails Holm at m = 7 (third smallest > 0.05 / 5)."""
    import laws
    importlib.reload(laws)
    cv, cov = _cities_val(n=12, seed=3)
    pt = laws.param_table(cv, _pool())
    reg = [dict(_h(covariate=k), id=f"H1-0{i + 1}", cycle=1) for i, k in enumerate(laws.COVARIATES[:6])]
    reg.append(dict(_h(), id="H2-01", cycle=2))
    prev = {f"H1-0{i + 1}": {"p": p, "stat": 0.1, "pass_raw": True} for i, p in
            enumerate([0.001, 0.012, 0.02, 0.2, 0.5, 0.6])}
    res = laws.test_registered(reg, pt, cov, pd.DataFrame(columns=["cid", "comp", "term", "city", "delta_nll", "exists"]),
                               previous=prev, out_dir=tmp_path, log=lambda s: None)
    by = {r["id"]: r for r in res}
    assert by["H1-02"]["pass_raw"] and not by["H1-02"]["pass"] and by["H1-01"]["pass"]
    assert by["H2-01"]["pass_raw"] and by["H2-01"]["p"] < 0.007 and by["H2-01"]["pass"] and by["H2-01"]["m"] == 7
    assert json.loads((tmp_path / "results.json").read_text())[0]["id"] == "H1-01"
    assert laws.agent_view(res) == {r["id"]: r["pass"] for r in res}


def test_group_diff_and_term_exist_prop():
    import laws
    importlib.reload(laws)
    cv, cov = _cities_val(n=12, seed=3)
    pt = laws.param_table(cv, _pool())
    r = laws.run_test(_h(test="group_diff"), pt, cov, None)
    assert r["pass_raw"] and r["stat"] > 0 and r["p"] < 0.05
    cities = list(cov.sort_values("log_population").city)
    ex = pd.DataFrame({"cid": "c1", "comp": "f", "term": "-c3*log(c4 + x26)", "city": cities, "delta_nll": 0.05,
                       "exists": [False] * 6 + [True] * 6})                                  # exists in the large half
    h = _h(test="term_exist_prop", parameter="-c3*log(c4 + x26)")
    r = laws.run_test(h, pt, cov, ex)
    assert r["stat"] == pytest.approx(1.0) and r["pass_raw"] and r["p"] < 0.05
    assert not laws.run_test(dict(h, direction="-"), pt, cov, ex)["pass_raw"]


# ---- Task 13: parameterised skeletons ----

def _city_logit(beta, seed, n_orig=80, n_dest=12):
    from sr_data import SRData, VARS
    rng = np.random.default_rng(seed)
    n = n_orig * n_dest
    X = np.zeros((n, len(VARS)), np.float32)
    X[:, 0] = rng.uniform(0, 3, n)
    X[:, 5] = rng.integers(1, 50, n)
    y = np.zeros(n)
    for i in range(n_orig):
        s = slice(i * n_dest, (i + 1) * n_dest)
        u = -beta * X[s, 0] + np.log(X[s, 5] + 1.0)
        p = np.exp(u - u.max())
        y[s] = rng.multinomial(40, p / p.sum())
    d = SRData(X=X, names=list(VARS), y=y, w=np.ones(n), offsets=np.arange(0, n + 1, n_dest), city=np.zeros(n, int),
               ses=np.zeros(n, int), offset_B=np.zeros(n), origin_unit=np.repeat(np.arange(n_orig), n_dest),
               dest_unit=np.tile(np.arange(n_dest), n_orig))
    return {"tr": d, "va": d}


def test_joint_refit_recovers_the_covariate_slope():
    import laws
    import library
    from refit_select import parse_equation
    importlib.reload(laws)
    lx = laws.parameterize({"cid": "c9", "equation": "f = log(#6 + 1.0) - 1.0 * #1; p1 = []; p2 = []",
                            "complexity": 7, "consts_refit": [1.0, 1.0]}, "f:c1", "log_population")
    assert lx["law"] == {"param": "f:c1", "covariate": "log_population"} and lx["cid"] == "c9"
    datas = {"A": _city_logit(0.5, 1), "B": _city_logit(1.0, 2), "C": _city_logit(1.5, 3)}
    cov = pd.DataFrame({"city": ["A", "B", "C"], "log_population": [13.0, 14.0, 15.0]})
    refs = {c: {"grav_power": {"nll_per_visit": 3.0}, "M2_glm": {"nll_per_visit": 2.0}, "S0": {"nll_per_visit": 2.5}}
            for c in datas}
    jr = laws.joint_refit(lx, datas, cov, refs=refs, starts=2)
    assert jr["b1"] > 0.3 and abs(jr["b0"] - 1.0) < 0.15 and jr["n_params"] == 3
    single = parse_equation(lx["equation"], None, "A", 7)
    for c in datas:
        r = library.refit_city(single, c, starts=1, tr=datas[c]["tr"], va=datas[c]["va"])
        assert jr["per_city"][c]["nll"] <= r["val_nll"] + 0.03 * 1.0
        assert jr["per_city"][c]["share_s0"] is not None
    seeds = laws.substituted_seeds(lx, jr, cov, ["A", "C"])
    assert set(seeds) == {"A", "C"} and len(seeds["A"]) == 1 and seeds["A"][0]["g"] == "1.0"
    fA, fC = seeds["A"][0]["f"], seeds["C"][0]["f"]
    assert "N_j" in fA and "d" in fA and "#" not in fA and fA != fC
    import seed_agent
    ok, _, e = seed_agent.validate_expr(fA)
    import sympy
    assert ok and abs(-float(e.coeff(sympy.Symbol("d"))) - (jr["b0"] + jr["b1"] * jr["z"]["A"])) < 1e-6


def test_untested_registered_hypotheses_and_bootstrap_resolution(tmp_path):
    """Final review #6: the registered tests use enough replicates to resolve Holm thresholds (alpha / m >= 0.005)."""
    import laws
    importlib.reload(laws)
    assert laws.TEST_B >= 2000
    cv, cov = _cities_val(n=12, seed=3)
    r = laws.run_test(_h(), laws.param_table(cv, _pool()), cov, None)
    assert r["B"] == laws.TEST_B


def test_analysis_payload_lists_registered_hypotheses_with_pass_only(monkeypatch):
    """Final review #5: the agent sees what is already registered and its pass / fail (booleans only)."""
    import llm
    import laws
    importlib.reload(laws)
    cv, cov = _cities_val()
    pt = laws.param_table(cv, _pool())
    ex = pd.DataFrame(columns=["cid", "comp", "term", "city", "delta_nll", "exists"])
    monkeypatch.setattr(laws, "_lawtest_ids", lambda: set())
    reg = [dict(_h(), id="H1-01", cycle=1, sha="s")]
    p = laws.analysis_payload([], ex, pt, _pool(), registered=reg, results={"H1-01": {"pass": True, "p": 0.001234}})
    llm.check_input(p)
    assert p["registered"] == [{"id": "H1-01", "skeleton": "c1", "parameter": "f:c1", "covariate": "log_population",
                                "direction": "+", "test": "slope_ci", "pass": True}]
    assert "0.001234" not in json.dumps(p)
