"""Plan 9, Task 5: summaries over cities, paired share differences, the opponent rule and the verdicts."""
import numpy as np
import pandas as pd
import pytest

import p9_stats as st


def _t(rows):
    """rows: (city, model, level, share or None for a failure)."""
    return pd.DataFrame([{"city": c, "model": m, "level": lv, "share": -np.inf if s is None else s,
                          "nll": np.nan if s is None else 4.0 - s, "cpc": np.nan if s is None else 0.5 + s / 10,
                          "failed": s is None} for c, m, lv, s in rows])


def test_summary_counts_failures_as_worst():
    t = _t([("a", "m", "local", 0.9), ("b", "m", "local", 0.8), ("c", "m", "local", None),
            ("a", "m", "uniform", 0.5), ("b", "m", "uniform", None), ("c", "m", "uniform", None)])
    s = st.summary(t).set_index(["model", "level"])
    r = s.loc[("m", "local")]
    assert r["n"] == 3 and r["n_failed"] == 1 and r["share_med"] == pytest.approx(0.8)
    assert r["nll_med"] == pytest.approx(3.15) and r["cpc_med"] == pytest.approx(0.585)       # the two finite cities
    assert s.loc[("m", "uniform"), "share_med"] == -np.inf and s.loc[("m", "uniform"), "n_failed"] == 2


def test_paired_failure_rules():
    t = _t([("a", "x", "local", 0.9), ("a", "y", "local", 0.7), ("b", "x", "local", None), ("b", "y", "local", 0.7),
            ("c", "x", "local", 0.9), ("c", "y", "local", None), ("d", "x", "local", None), ("d", "y", "local", None)])
    d = st.paired(t, ("x", "local"), ("y", "local"))
    assert d["a"] == pytest.approx(0.2) and d["b"] == -np.inf and d["c"] == np.inf and d["d"] == 0.0
    assert list(d.index) == ["a", "b", "c", "d"]


def test_paired_requires_same_cities():
    t = _t([("a", "x", "local", 0.9), ("a", "y", "local", 0.7), ("b", "x", "local", 0.9)])
    with pytest.raises(ValueError, match="cities"):
        st.paired(t, ("x", "local"), ("y", "local"))
    with pytest.raises(ValueError, match="no rows"):
        st.paired(t, ("x", "local"), ("z", "local"))


def test_contrast_known_values():
    r = st.contrast(pd.Series([0.1] * 9, index=list("abcdefghi")), B=200, seed=1)
    assert r["n"] == 9 and r["median"] == pytest.approx(0.1) and r["ci_lo"] == pytest.approx(0.1)
    assert r["ci_hi"] == pytest.approx(0.1) and (r["wins"], r["losses"], r["ties"]) == (9, 0, 0)
    assert r["p"] == pytest.approx(2 / 201)                 # never exactly 0: (k + 1) / (B + 1)


def test_contrast_is_reproducible_and_two_sided():
    d = pd.Series(np.random.default_rng(0).normal(0.0, 1.0, 30))
    a, b = st.contrast(d, B=500, seed=3), st.contrast(d, B=500, seed=3)
    assert a == b and a["ci_lo"] < 0 < a["ci_hi"] and a["p"] > 0.05


def test_contrast_with_infinite_differences():
    d = pd.Series([0.2, 0.3, 0.1, -np.inf, np.inf, 0.25, 0.15])
    r = st.contrast(d, B=300, seed=0)
    assert r["median"] == pytest.approx(0.2) and not np.isnan([r["ci_lo"], r["ci_hi"], r["p"]]).any()
    assert (r["wins"], r["losses"], r["ties"]) == (6, 1, 0)


META = {"kwon": {"original": True, "needs_trips": True, "n_free": 110},
        "grav_P": {"original": True, "needs_trips": False, "n_free": 2},
        "rad_P": {"original": True, "needs_trips": False, "n_free": 0},
        "str_N": {"original": False, "needs_trips": False, "n_free": 3},
        "s0_N": {"original": False, "needs_trips": False, "n_free": 3}}


def _s(local, uniform):
    return pd.DataFrame([{"model": m, "level": lv, "share_med": v}
                         for lv, d in (("local", local), ("uniform", uniform)) for m, v in d.items()]
                        + [{"model": "ours", "level": "local", "share_med": 0.99}])


def test_select_opponents_rules():
    s = _s({"kwon": 0.95, "grav_P": 0.1, "rad_P": -3.0, "str_N": 0.93, "s0_N": 0.90},
           {"kwon": 0.96, "grav_P": 0.05, "rad_P": -3.0, "str_N": 0.80, "s0_N": 0.89})
    o = st.select_opponents(s, META)
    assert o["fit"] == {"model": "kwon", "share_med": 0.95}                       # Kwon may be the fit opponent
    assert o["zero_shot"] == {"model": "s0_N", "share_med": 0.89}                 # ... but never the zero-shot one
    assert o["fit_original"]["model"] == "kwon" and o["zero_shot_original"]["model"] == "grav_P"
    assert "ours" not in {v["model"] for v in o.values()}                         # only models in meta are eligible


def test_select_opponents_tie_break():
    s = _s({"kwon": 0.5, "grav_P": 0.9, "rad_P": 0.9, "str_N": 0.9, "s0_N": 0.9},
           {"kwon": 0.5, "grav_P": 0.1, "rad_P": 0.1, "str_N": 0.7, "s0_N": 0.7})
    o = st.select_opponents(s, META)
    assert o["fit"]["model"] == "rad_P"                    # fewest free parameters
    assert o["zero_shot"]["model"] == "s0_N"               # same parameter count: alphabetical


def test_select_opponents_needs_a_candidate():
    with pytest.raises(ValueError, match="zero_shot"):
        st.select_opponents(_s({"kwon": 0.5}, {"kwon": 0.5}), {"kwon": META["kwon"]})


def test_judge_holm_and_verdicts():
    res = {"C1": {"ci_lo": 0.01, "ci_hi": 0.05, "p": 0.0004}, "C2": {"ci_lo": 0.001, "ci_hi": 0.05, "p": 0.04},
           "C3": {"ci_lo": -0.02, "ci_hi": 0.03, "p": 0.6}, "C4": {"ci_lo": -0.06, "ci_hi": -0.01, "p": 0.001},
           "D1": {"ci_lo": 0.5, "ci_hi": 0.9, "p": 0.0004}}
    out = st.judge(res, ["C1", "C2", "C3", "C4"], alpha=0.05)
    assert out["C1"]["verdict"] == "better" and out["C4"]["verdict"] == "worse" and out["C3"]["verdict"] == "no distinguishable difference"
    assert out["C2"]["verdict"] == "no distinguishable difference" and out["C2"]["ci_excludes_zero"] and not out["C2"]["holm"]
    assert out["D1"]["verdict"] is None and out["D1"]["holm"] is None and out["D1"]["ci_excludes_zero"]
    assert "verdict" not in res["C1"]                      # the input is not modified


def test_paired_refuses_nan_share_on_a_row_that_did_not_fail():
    t = _t([("a", "x", "local", 0.9), ("a", "y", "local", 0.7)])
    t.loc[0, "share"] = np.nan
    with pytest.raises(ValueError, match="share"):
        st.paired(t, ("x", "local"), ("y", "local"))
