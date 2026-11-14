"""Plan 2 Task 4: clean.py — degeneracy rules, canonical form (additive constants), term contributions and the
code-side check of an LLM simplification proposal."""
import importlib

import numpy as np
import sympy

from test_search_param import two_cities  # noqa: F401


def _data():
    import sr_data
    tr = sr_data.build(["TEST1"], "tract", "train", "search", None, 0)
    va = sr_data.build(["TEST1"], "tract", "val", "val", None, 0)
    return tr, va


def P(eq, template=None, cx=5):
    from refit_select import parse_equation
    return parse_equation(eq, template, "A", cx)


def test_degenerate_rules(two_cities):
    import clean
    importlib.reload(clean)
    tr, _ = _data()
    assert "tiny_exponent" in clean.degenerate(P("f = #1 ^ 0.001; p1 = []; p2 = []"), tr)
    assert "huge_const" in clean.degenerate(P("f = log(#1 + 1.0) + 100000.0 * #4; p1 = []; p2 = []"), tr)
    assert "tiny_const" in clean.degenerate(P("f = log(#1 + 1.0) + 0.00001 * #4; p1 = []; p2 = []"), tr)
    assert "zero_denominator" in clean.degenerate(P("f = 1.0 / (#5 - 35.0); p1 = []; p2 = []"), tr)   # E_j 30..60: sign change
    assert "flat_term" in clean.degenerate(P("f = log(#1 + 1.0) + 0.5 * #7; p1 = []; p2 = []"), tr)    # a_j = 1 everywhere
    assert clean.degenerate(P("f = log(#5 + 1.0) - 2.0 * log(#1 + 1.0); p1 = []; p2 = []"), tr) == []
    assert set(clean.degenerate(P("f = log(#1 + 1.0); g = 1.0 / (#5 - 35.0); h = #1; p1 = [0.1]; p2 = [0.2]"), tr)) == {"zero_denominator"}


def test_canonical_merges_constants():
    import clean
    from refit_select import XS
    c = clean.canonical(P("f = log(#5 + 2.0 - 3.0) + 5.0; p1 = []; p2 = []"))
    assert c.n_lit == 1 and np.allclose(c.consts, [1.0])
    assert c.comps["f"].subs({sympy.Symbol("c0"): 1.0}) == sympy.log(XS[4] - 1.0)
    c2 = clean.canonical(P("f = log(#5 + 1.0) + 2.0; g = nexp(#5) + 1.0; h = log(#1 + 2.0); p1 = [0.1, 0.2]; p2 = [-2.0, -2.5]",
                           "city2", 10))
    assert c2.n_lit == 2 and c2.n_cat == 2 and np.allclose(c2.consts, [1.0, 2.0, 0.1, 0.2, -2.0, -2.5])
    assert c2.comps["g"] == sympy.exp(-XS[4]) and c2.template == "city2" and c2.complexity == 10
    assert clean.render(c2).startswith("f = log(#5 + 1.0")


def test_term_contrib_marks_removable(two_cities):
    import clean
    importlib.reload(clean)
    tr, _ = _data()
    c = P("f = log(#5 + 1.0) + 0.001 * #7; p1 = []; p2 = []")
    refs = {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 4.0}}
    out = clean.term_contrib(c, tr, refs)
    assert len(out) == 2 and all(k in out[0] for k in ("term", "delta_nll", "share_of_gap", "removable"))
    flat = next(o for o in out if "x7" in o["term"])
    main = next(o for o in out if "log" in o["term"])
    assert flat["removable"] and flat["share_of_gap"] < 0.01
    assert not main["removable"] and main["delta_nll"] > 0.1 and main["share_of_gap"] == main["delta_nll"] / 1.0


def test_verify_simplification(two_cities):
    import clean
    importlib.reload(clean)
    tr, va = _data()
    c = P("f = -2.0 * log(#1 + 0.5 + 0.5); p1 = []; p2 = []")          # distance: positive on every row
    r = clean.verify_simplification(c, "log(y1)", tr, va)
    assert r["accepted"] is False and "y1" in r["reason"]
    assert clean.verify_simplification(c, "log(x1 +", tr, va)["accepted"] is False
    r = clean.verify_simplification(c, "1.0 / (x1 + 1.0)", tr, va)       # structure changed
    assert r["accepted"] is False and r["corr"] < 0.999
    r = clean.verify_simplification(c, "-2.0 * log(x1 + 1.0)", tr, va)
    assert r["accepted"] and r["corr"] > 0.999 and r["dnll"] <= 0.5 * r["se"] + 1e-9
    assert r["candidate"].n_lit == 2 and r["sympy"] == "-2.0*log(x1 + 1.0)"
    # a proposal that is not finite on the validation block is rejected, never raised
    c5 = P("f = log(#5 + 0.5 + 0.5); p1 = []; p2 = []")                   # E_j = 0 on the validation destination
    r = clean.verify_simplification(c5, "log(x5 + 1.0)", tr, va)
    assert r["accepted"] is False and "finite" in r["reason"]


def test_clean_pool_parallel_matches_sequential(two_cities):
    """ETA review 11-03: term contributions refit every term of every candidate; clean_pool runs the candidates in a
    fork pool and must give the same output as the sequential run."""
    import json
    import clean
    import library
    import sr_data
    importlib.reload(library)
    importlib.reload(clean)
    pool = [{"cid": "c0001", "equation": "f = log(#5 + 1.0) + 0.001 * #7; p1 = []; p2 = []", "template": None, "complexity": 8},
            {"cid": "c0002", "equation": "f = -2.0 * log(#1 + 0.5 + 0.5); p1 = []; p2 = []", "template": None, "complexity": 6}]
    import pandas as pd
    fits = pd.DataFrame([{"cid": c, "city": "TEST1", "finite": True} for c in ("c0001", "c0002")])
    data = {"TEST1": sr_data.build(["TEST1"], "tract", "train", "search", None, 0)}
    refs = {"TEST1": {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 4.0}}}
    a = clean.clean_pool(pool, fits, data, refs, ["TEST1"], procs=1)
    b = clean.clean_pool(pool, fits, data, refs, ["TEST1"], procs=2)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) and set(a) == {"c0001", "c0002"}
    assert a["c0002"]["canonical"].startswith("f = ") and a["c0001"]["term_city"] == "TEST1"


def test_term_contrib_covers_g_and_h(two_cities):
    """Plan 4 Task 4 (design §5.8.1): every top-level term of f, g and h is ablated (template candidates via
    to_single); removing a component's only term leaves that component 0."""
    import clean
    importlib.reload(clean)
    tr, _ = _data()
    c = P("f = log(#6 + 1.0); g = #2 + #17 * #2; h = #1; p1 = [-1.5]; p2 = [0.01]", "city1", 9)
    refs = {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 4.0}}
    out = clean.term_contrib(c, tr, refs)
    assert sorted(t["comp"] for t in out) == ["f", "g", "g", "h"]
    assert all(np.isfinite(t["delta_nll"]) for t in out)
    assert {t["term"] for t in out if t["comp"] == "g"} == {"x2", "x17*x2"}
    assert {t["key"] for t in out if t["comp"] == "f"} == {"log(c0 + x6)"}                 # constants as symbols
