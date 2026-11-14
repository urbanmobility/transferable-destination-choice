"""Plan 2 Task 6: equivalence merging of candidates across the search cities, ranking, and the GATE 1 package."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

CITIES = ["T1", "T2"]
CIDS = ["c0001", "c0002", "c0003"]
B = 50


def _fits(vals, within):
    rows = []
    for city in CITIES:
        for k, cid in enumerate(CIDS):
            v = vals[city][k]
            rows.append({"cid": cid, "city": city, "train_nll": v, "val_nll": v, "finite": v is not None,
                         "consts": "[1.0]", "se_vs_best": 0.01, "within_1se": within[city][k]})
    return pd.DataFrame(rows)


def _etas(rng, offsets):
    """c0001 and c0002: the same ordering within every origin group (a monotone transform); c0003: reversed."""
    n = offsets[-1]
    base = rng.normal(size=n)
    e2 = 3.0 * base + 1.0
    e3 = -base
    return {"offsets": offsets, "eta": {"c0001": base, "c0002": e2, "c0003": e3}}


@pytest.fixture
def world():
    rng = np.random.default_rng(0)
    offsets = np.array([0, 5, 11, 20])
    etas = {c: _etas(rng, offsets) for c in CITIES}
    boot = {}
    for c in CITIES:
        b = rng.normal(size=(B, 3)) * 0.01 + np.array([4.0, 4.0, 4.3])
        b[:, 1] = b[:, 0] + rng.normal(size=B) * 0.002        # c0002 paired with c0001: tiny difference SD
        boot[c] = b
    vals = {"T1": [4.000, 4.001, 4.300], "T2": [4.000, 4.001, 4.300]}
    within = {"T1": [True, True, False], "T2": [True, True, False]}
    pool = [{"cid": "c0001", "equation": "f = log(#5 + 1.0); p1 = []; p2 = []", "template": None, "complexity": 6, "source": ["a"]},
            {"cid": "c0002", "equation": "f = log(#5 + 2.0) + 1.0; p1 = []; p2 = []", "template": None, "complexity": 8, "source": ["b"]},
            {"cid": "c0003", "equation": "f = 1.0 / (#5 + 1.0); p1 = []; p2 = []", "template": None, "complexity": 6, "source": ["b"]}]
    return pool, _fits(vals, within), boot, etas


def test_equivalent_pairs(world):
    import merge
    importlib.reload(merge)
    pool, fits, boot, etas = world
    assert merge.equivalent("c0001", "c0002", fits, boot, etas, CITIES)
    assert not merge.equivalent("c0001", "c0003", fits, boot, etas, CITIES)
    # a city where one candidate's eta is constant, or where one is not fittable -> not equivalent
    etas2 = {c: {"offsets": e["offsets"], "eta": dict(e["eta"])} for c, e in etas.items()}
    etas2["T2"]["eta"]["c0002"] = np.zeros(20)
    assert not merge.equivalent("c0001", "c0002", fits, boot, etas2, CITIES)
    f2 = fits.copy()
    f2.loc[(f2.cid == "c0002") & (f2.city == "T1"), ["finite", "val_nll"]] = [False, np.nan]
    assert not merge.equivalent("c0001", "c0002", f2, boot, etas, CITIES)
    # the same ordering but a validation loss more than one bootstrap SD apart -> not equivalent
    f3 = fits.copy()
    f3.loc[(f3.cid == "c0002") & (f3.city == "T1"), "val_nll"] = 4.2
    assert not merge.equivalent("c0001", "c0002", f3, boot, etas, CITIES)


def test_merge_and_rank(world):
    import merge
    importlib.reload(merge)
    pool, fits, boot, etas = world
    reps = merge.merge(pool, fits, boot, etas, CITIES)
    assert [r["cid"] for r in reps] == ["c0001", "c0003"]
    assert reps[0]["merged_from"] == ["c0002"] and reps[1]["merged_from"] == []
    refs = {c: {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 3.75}} for c in CITIES}
    ranked = merge.rank(reps, fits, refs, {}, min_cities=1, top=8)          # plan 4: merge.rank is scoring.rank
    assert [r["cid"] for r in ranked] == ["c0001", "c0003"] and ranked[0]["n_1se"] == 2 and ranked[1]["n_1se"] == 0
    assert ranked[0]["gate1"] is True and ranked[1]["gate1"] is False
    assert merge.rank(reps, fits, refs, {}, min_cities=3, top=8)[0]["gate1"] is False


def test_gate1_package(world, tmp_path):
    import llm
    import merge
    importlib.reload(merge)
    pool, fits, boot, etas = world
    refs = {c: {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 3.75}} for c in CITIES}
    reps = merge.rank(merge.merge(pool, fits, boot, etas, CITIES), fits, refs, {"c0003": {"n": 1}}, min_cities=1, top=8)
    clean_out = {"c0001": {"degenerate": [], "canonical": "f = log(#5 + 1.0); p1 = []; p2 = []",
                           "term_contrib": [{"term": "log(x5 + 1.0)", "delta_nll": 0.3, "share_of_gap": 0.24, "removable": False}]},
                 "c0003": {"degenerate": ["zero_denominator"], "canonical": None, "term_contrib": []}}
    llm_out = {"simplify": {"c0001": {"sympy": "log(x5 + 1)", "accepted": True, "reason": "r"}},
               "readability": {"c0001": {"score": 4, "terms": [{"term": "log(x5 + 1)", "meaning": "log attraction of job size"}],
                                         "uses_variables": ["x5"], "invalid": False}}}
    b_best = {"T1": 3.9, "T2": 3.95}
    jp, mp = merge.gate1_package(reps, pool, fits, refs, clean_out, llm_out, b_best, out_dir=tmp_path, cities=CITIES)
    g = json.loads(jp.read_text())
    assert [x["cid"] for x in g["candidates"]] == ["c0001", "c0003"] and g["candidates"][0]["gate1"] is True
    c1 = g["candidates"][0]
    assert c1["simplified"] == "log(x5 + 1)" and c1["readability"]["score"] == 4 and c1["cities"]["T1"]["share"] == pytest.approx(0.8)
    assert c1["cities"]["T1"]["b_gap"] == pytest.approx(0.1) and c1["n_1se"] == 2 and c1["merged_from"] == ["c0002"]
    assert c1["n_near"] == 2 and c1["cities"]["T1"]["near_best"] is True
    assert g["candidates"][1]["n_near"] == 0 and g["candidates"][1]["cities"]["T2"]["near_best"] is False
    assert g["candidates"][1]["degenerate"] == ["zero_denominator"] and g["candidates"][1]["simplified"] is None
    assert c1["n_flags"] == 0 and g["candidates"][1]["n_flags"] == 1
    llm.check_input(g)                                      # no confirmation city anywhere in the package
    md = mp.read_text()
    assert "c0001" in md and "| " in md and "GATE 1" in md
    assert "near-best" in md and "5%" in md and "Flags" in md                          # V3 gate (2026-11-10)
    import scoring
    with scoring.registered_rule():
        assert "3%" in merge._markdown(g)

def test_merge_computes_etas_only_for_loss_close_pairs(world):
    """ETA review 11-03: the validation etas of 100+ candidates in New York do not fit in memory at once; merge asks
    for them only for candidates in pairs whose losses are within one bootstrap SD in every search city."""
    import merge
    importlib.reload(merge)
    pool, fits, boot, etas = world
    asked = []

    def lazy(cids):
        asked.append(sorted(cids))
        return {c: {"offsets": e["offsets"], "eta": {k: v for k, v in e["eta"].items() if k in cids}} for c, e in etas.items()}
    reps = merge.merge(pool, fits, boot, lazy, CITIES)
    assert asked == [["c0001", "c0002"]] and [r["cid"] for r in reps] == ["c0001", "c0003"]
    assert merge.loss_close("c0001", "c0002", fits, boot, CITIES) and not merge.loss_close("c0001", "c0003", fits, boot, CITIES)


def test_rank_counts_a_city_when_the_regret_is_within_3_percent_of_the_grav_m2_gap():
    """Design decision of 2026-11-05: the 1-SE sets hold only each city's best, so a city counts for a skeleton when one of its
    members is in the city's 1-SE set or its validation NLL is at most 3% of (NLL_grav - NLL_M2) above the city's best."""
    import merge
    importlib.reload(merge)
    import scoring
    assert scoring.MARGIN == 0.03 and not hasattr(merge, "NEAR_SHARE")
    vals = {"T1": [4.000, 4.001, 4.030], "T2": [4.000, None, 4.050]}           # gap 1.25 -> margin 0.0375
    within = {"T1": [True, False, False], "T2": [True, False, False]}
    fits = _fits(vals, within)
    refs = {c: {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 3.75}} for c in CITIES}
    reps = [{"cid": "c0002", "complexity": 8, "merged_from": []}, {"cid": "c0003", "complexity": 6, "merged_from": []}]
    v3 = {r["cid"]: r for r in merge.rank(reps, fits, refs, {}, min_cities=1, top=8)}   # V3 gate: 5% -> 0.0625
    assert v3["c0003"]["near_cities"] == ["T1", "T2"] and v3["c0002"]["near_cities"] == ["T1"]
    with scoring.registered_rule():
        ranked = merge.rank(reps, fits, refs, {}, min_cities=1, top=8)
        ranked2 = merge.rank(reps, fits, refs, {}, min_cities=2, top=8)
    by = {r["cid"]: r for r in ranked}
    assert by["c0002"]["n_near"] == 1 and by["c0002"]["near_cities"] == ["T1"]       # not fittable in T2
    assert by["c0003"]["n_near"] == 1 and by["c0003"]["near_cities"] == ["T1"]       # T2: 0.05 > 0.0375
    assert by["c0002"]["n_1se"] == 0 and by["c0003"]["n_1se"] == 0
    assert [r["cid"] for r in ranked] == ["c0003", "c0002"]                          # tie on n_near: lower complexity
    assert all(r["gate1"] for r in ranked)
    assert not any(r["gate1"] for r in ranked2)


def test_load_llm_skips_call_inputs_and_failed_calls(tmp_path):
    import merge
    importlib.reload(merge)
    w = lambda name, obj: (tmp_path / name).write_text(json.dumps(obj))  # noqa: E731
    w("simplify_c0001.in.json", {"cid": "c0001", "equation": "log(x6)"})
    w("simplify_c0001.json", {"point": "simplify", "result": {"cid": "c0001", "sympy": "log(x6)", "accepted": True}})
    w("simplify_c0002.in.json", {"cid": "c0002", "equation": "x1"})
    w("simplify_c0002.json", {"point": "simplify", "result": None, "error": "LLM call failed after 3 attempts"})
    w("readability_c0001.in.json", {"cid": "c0001", "equation": "log(x6)"})
    w("readability_c0001.json", {"point": "readability", "result": {"score": 4, "terms": []}})
    w("readability_c0003.json", {"point": "readability", "result": None, "error": "x"})
    out = merge.load_llm(tmp_path)
    assert out["simplify"] == {"c0001": {"cid": "c0001", "sympy": "log(x6)", "accepted": True}}
    assert out["readability"] == {"c0001": {"score": 4, "terms": []}}
    assert merge.load_llm(tmp_path / "missing") == {"simplify": {}, "readability": {}}


def test_loss_close_uses_delta_city_when_refs_are_given(world):
    """Plan 4 (equivalence): with refs, two candidates are loss-close in a city when their losses differ
    by at most max(bootstrap SD, delta_city = 3% of the grav->M2 gap); without refs the old SD rule stands."""
    import merge
    importlib.reload(merge)
    pool, fits, boot, etas = world
    f3 = fits.copy()
    f3.loc[(f3.cid == "c0002") & (f3.city == "T1"), "val_nll"] = 4.03            # 0.03 apart: >> SD, < delta 0.0375
    refs = {c: {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 3.75}} for c in CITIES}
    assert not merge.loss_close("c0001", "c0002", f3, boot, CITIES)
    assert merge.loss_close("c0001", "c0002", f3, boot, CITIES, refs)
    assert merge.equivalent("c0001", "c0002", f3, boot, etas, CITIES, refs)
    assert [r["cid"] for r in merge.merge(pool, f3, boot, etas, CITIES, refs)] == ["c0001", "c0003"]
    f3.loc[(f3.cid == "c0002") & (f3.city == "T1"), "val_nll"] = 4.05
    assert not merge.loss_close("c0001", "c0002", f3, boot, CITIES, refs)


def _chain_world():
    """c0001 ~ c0002 ~ c0003 by loss at 0.03 (neighbours 0.004 apart, ends 0.008; delta 0.006 / 0.010), every eta a
    monotone transform of one base (rank-close)."""
    rng = np.random.default_rng(7)
    offsets = np.array([0, 6, 13, 21])
    vals = {"c0001": 4.000, "c0002": 4.004, "c0003": 4.008}
    rows, boot, etas = [], {}, {}
    for c in CITIES:
        common = rng.normal(size=B) * 0.01
        boot[c] = np.column_stack([v + common + rng.normal(size=B) * 0.0005 for v in vals.values()])
        rows += [{"cid": k, "city": c, "train_nll": v, "val_nll": v, "finite": True, "consts": "[1.0]",
                  "se_vs_best": 0.0005, "within_1se": k == "c0001"} for k, v in vals.items()]
        base = rng.normal(size=offsets[-1])
        etas[c] = {"offsets": offsets, "eta": {"c0001": base, "c0002": 2 * base, "c0003": np.exp(base)}}
    pool = [{"cid": k, "equation": "f = log(#5 + 1.0); p1 = []; p2 = []", "template": None, "complexity": cx}
            for k, cx in (("c0001", 5), ("c0002", 7), ("c0003", 9))]
    refs = {c: {"grav_power": {"nll_per_visit": 4.2}, "M2_glm": {"nll_per_visit": 4.0}} for c in CITIES}
    return pool, pd.DataFrame(rows), boot, etas, refs


def test_merge_default_is_the_v3_leader_and_chain_the_registered_rule():
    import merge
    import scoring
    importlib.reload(merge)
    pool, fits, boot, etas, refs = _chain_world()
    sets = lambda reps: {r["cid"]: r["merged_from"] for r in reps}  # noqa: E731
    assert sets(merge.merge(pool, fits, boot, etas, CITIES, refs)) == {"c0001": ["c0002"], "c0003": []}
    assert sets(merge.merge(pool, fits, boot, etas, CITIES, refs, linkage="chain")) == {"c0001": ["c0002", "c0003"]}
    assert sets(merge.merge(pool, fits, boot, etas, CITIES, refs, margin=0.05)) == {"c0001": ["c0002", "c0003"]}
    with scoring.registered_rule():
        assert sets(merge.merge(pool, fits, boot, etas, CITIES, refs)) == {"c0001": ["c0002", "c0003"]}
    with pytest.raises(ValueError):
        merge.merge(pool, fits, boot, etas, CITIES, refs, linkage="star")
