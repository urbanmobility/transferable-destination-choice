"""Plan 4 Task 10: agent review and code-verified cleaning (inner-loop step 2)."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from test_search_param import two_cities  # noqa: F401

CITIES8 = ["35620", "16980", "31080", "BAY9", "26420", "12060", "14460", "42660"]
EQ = "f = log(#6 + 17.0) - 0.88 * log(#1 + 0.64) - 0.6 * log(#26 + #6 + 14909.0); p1 = []; p2 = []"
REFS = {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 4.0}, "S0": {"nll_per_visit": 4.1},
        "grav_exp": {"nll_per_visit": 5.1}, "ext_rad": {"nll_per_visit": 4.9}}


def _entry(cid="c0122", rnd=0, eq=EQ, cx=27):
    return {"cid": cid, "equation": eq, "template": None, "variant": "A", "complexity": cx,
            "consts_refit": [17.0, 0.88, 0.64, 0.6, 14909.0], "source": ["select_x.json"], "round": rnd}


def _fits(cids, nll=4.2, cities=CITIES8):
    rows = []
    for k, cid in enumerate(cids):
        for c in cities:
            rows.append({"cid": cid, "city": c, "train_nll": nll, "val_nll": nll + 0.001 * k, "finite": True,
                         "consts": json.dumps([17.0 + k, 0.88, 0.64, 0.6, 14909.0]), "se_vs_best": 0.0005,
                         "within_1se": k == 0, "n_converged": 3, "loss_sd_starts": 0.0,
                         "consts_sd": json.dumps([0.0] * 5)})
    return pd.DataFrame(rows)


@pytest.fixture
def city_datas_small(two_cities):
    import library
    importlib.reload(library)
    return {c: library.city_data(c, n_orig=None) for c in ("TEST1", "TEST2")}


@pytest.fixture
def refs_small():
    return {c: REFS for c in ("TEST1", "TEST2")}


def test_review_payload_has_no_validation_numbers_and_passes_check_input():
    import llm
    import review
    importlib.reload(review)
    x = _entry()
    fl = {"origin_only": [], "unit_mixing": ["x26 + x6 + 14909.0"], "disguised_linear": [], "n": 1,
          "degenerate": ["huge_const"]}
    abl = [{"comp": "f", "term": "log(x6 + 17.0)", "delta_nll": 0.3, "share_of_gap": 0.3, "removable": False}]
    cr = review.const_ranges(x, _fits(["c0122"]))
    p = review.review_payload(x, _fits(["c0122"]), fl, abl, cr, refs={c: REFS for c in CITIES8})
    s = json.dumps(p, ensure_ascii=False)
    assert "val_nll" not in s and "fold4_nll" in s and "validation" not in s.lower()
    llm.check_input(p)
    assert p["cid"] == "c0122" and p["code_flags"]["unit_mixing"] == ["x26 + x6 + 14909.0"]
    assert set(p["fold4_nll"]) == set(CITIES8) and set(p["fold4_nll"]["12060"]) == {"nll", "gap_to_city_best",
                                                                                   "share_s0", "share_grav"}
    assert p["const_ranges"]["c0"] == [17.0, 17.0] and "c0" in p["eta_parametric"]
    assert set(p["variables"]) == {"x1", "x6", "x26"} and p["term_ablation"][0]["comp"] == "f"


def test_const_ranges_and_xmax(city_datas_small):
    import review
    importlib.reload(review)
    f = _fits(["c0122"], cities=["A", "B"])
    f.loc[f.city == "B", "consts"] = json.dumps([20.0, 1.0, 0.5, 0.6, 100.0])
    assert review.const_ranges(_entry(), f)["c4"] == [100.0, 14909.0]
    xm = review.city_xmax(city_datas_small)
    assert set(xm) == {"TEST1", "TEST2"} and xm["TEST1"]["d"] > 0 and "S_ij" in xm["TEST1"]


def test_verify_cleaned_accepts_within_delta(city_datas_small, refs_small):
    import review
    importlib.reload(review)
    orig = _entry(eq="f = log(#6 + 17.0) - 0.88 * log(#1 + 0.64); p1 = []; p2 = []", cx=11)
    orig["consts_refit"] = [17.0, 0.88, 0.64]
    cities = ["TEST1", "TEST2"]
    v = review.verify_cleaned(orig, "log(x6 + 17) - 0.88*log(x1 + 0.64) - 0.6*log(x26/10000 + 1.5)", cities,
                              city_datas_small, refs_small)
    assert v["accepted"] and set(v["per_city"]) == set(cities) and v["new_nonfittable"] == []
    assert all(v["per_city"][c]["finite"] for c in cities) and v["mean_norm_delta"] <= 1.0
    assert len(v["consts"]) == 6 and v["complexity"] > 0          # 17, 0.88, 0.64, 0.6, 10000, 1.5


def test_verify_cleaned_rejects_new_nonfittable(city_datas_small, refs_small):
    """Review Focus 1: a cleaned version whose eta is not finite in a search city (log of a negative argument) is
    recorded as not fittable there and rejected — never an exception."""
    import review
    importlib.reload(review)
    orig = _entry(eq="f = log(#6 + 17.0) - 0.88 * log(#1 + 0.64); p1 = []; p2 = []", cx=11)
    orig["consts_refit"] = [17.0, 0.88, 0.64]
    v = review.verify_cleaned(orig, "log(x6 - 1e6)", ["TEST1", "TEST2"], city_datas_small, refs_small)
    assert not v["accepted"] and v["new_nonfittable"] == ["TEST1", "TEST2"] and "not fittable" in v["reason"]
    for bad, why in (("log(x6 +", "parse error"), ("log(y + 1)", "illegal symbols"), ("", "empty")):
        v = review.verify_cleaned(orig, bad, ["TEST1"], city_datas_small, refs_small)
        assert not v["accepted"] and why in v["reason"]


def test_select_for_review_quota_and_new_first():
    import review
    import scoring
    importlib.reload(review)
    pool = [_entry(f"c{k:04d}", rnd=(5 if k >= 65 else 0), cx=10 + k % 7) for k in range(70)]
    fits = _fits([x["cid"] for x in pool])
    refs = {c: REFS for c in CITIES8}
    flags = {x["cid"]: {"n": 0} for x in pool}
    sel = review.select_for_review(pool, fits, refs, flags, reviewed=set(), quota=60)
    assert len(sel) == 60 and sel[0]["round"] == max(x["round"] for x in pool)
    assert [x["round"] for x in sel[:5]] == [5] * 5 and all(x["round"] == 0 for x in sel[5:])
    order = [r["cid"] for r in scoring.rank([dict(x, merged_from=[]) for x in pool], fits, refs, flags)]
    old = [x["cid"] for x in sel[5:]]
    assert old == [c for c in order if c in set(old)]                      # rank order within the old pool
    sel2 = review.select_for_review(pool, fits, refs, flags, reviewed={x["cid"] for x in pool[65:]}, quota=60)
    assert all(x["round"] == 0 for x in sel2)
    above = fits.assign(val_nll=4.95)                                       # every candidate above the floor 4.9
    assert review.select_for_review(pool, above, refs, flags, reviewed=set()) == []


def test_review_round_caps_two_cleanings_per_parent(two_cities, monkeypatch, tmp_path):
    import library
    import review
    importlib.reload(library)
    importlib.reload(review)
    eq = "f = log(#6 + 17.0) - 0.88 * log(#1 + 0.64); p1 = []; p2 = []"
    P = _entry("c0001", eq=eq, cx=11)
    Q = _entry("c0002", eq="f = log(#6 + 3.0) - 1.5 * log(#1 + 1.0); p1 = []; p2 = []", cx=11)
    for x in (P, Q):
        x["consts_refit"] = [17.0, 0.88, 0.64] if x is P else [3.0, 1.5, 1.0]
    kids = [dict(_entry(f"c0001c{k}", eq=eq, cx=11), parent="c0001", source=["cleaned_by_agent"], round=1,
                 consts_refit=[17.0, 0.88, 0.64]) for k in (1, 2)]
    pool = [P, Q] + kids
    cities = ["TEST1", "TEST2"]
    datas = {c: library.city_data(c, n_orig=None) for c in cities}
    fits = _fits([x["cid"] for x in pool], cities=cities)
    refs = {c: REFS for c in cities}
    calls = []

    def agent(payload):
        calls.append(payload["cid"])
        return {"problems": [{"issue": "shift constant", "evidence": "x"}], "cleaned_sympy": "log(x6 + 10) - 1.2*log(x1 + 0.5)",
                "changes": "changed initial values", "expected_fit_cost": "0", "insight": "size and shift distance"}
    flags = {x["cid"]: {"n": 0} for x in pool}
    out = review.review_round(pool, fits, refs, flags, {}, cities, datas, round_no=6, out_dir=tmp_path, agent=agent,
                              quota=60, reviewed=set())
    assert sorted(calls) == ["c0001", "c0001c1", "c0001c2", "c0002"]
    new = out["new"]
    assert {x["parent"] for x in new} == {"c0002", "c0001c1", "c0001c2"}             # c0001 has 2 cleanings already
    q = next(x for x in new if x["parent"] == "c0002")
    assert q["cid"] == "c0002c1" and q["source"] == ["cleaned_by_agent"] and q["round"] == 6
    assert q["insight"] == "size and shift distance" and parse_ok(q["equation"])
    rec = json.loads((tmp_path / "review_c0001.json").read_text())
    assert rec["result"]["cleaned_sympy"] and rec["verify"]["skipped"] == "parent has 2 cleanings"
    assert json.loads((tmp_path / "review_c0002.json").read_text())["verify"]["accepted"] is True
    assert out["llm_calls"] == 4 and sorted(out["reviewed"]) == ["c0001", "c0001c1", "c0001c2", "c0002"]
    out2 = review.review_round(pool, fits, refs, flags, {}, cities, datas, round_no=6, out_dir=tmp_path,
                               agent=agent, quota=60, reviewed=set(), max_calls=1)
    assert out2["llm_calls"] == 1


def parse_ok(eq):
    from refit_select import parse_equation
    return parse_equation(eq, None, "A", 1) is not None


def test_verify_cleaned_refits_the_original_when_fits_lack_it(city_datas_small, refs_small):
    import review
    importlib.reload(review)
    orig = _entry(eq="f = log(#6 + 17.0) - 0.88 * log(#1 + 0.64); p1 = []; p2 = []", cx=11)
    orig["consts_refit"] = [17.0, 0.88, 0.64]
    v = review.verify_cleaned(orig, "log(x6 + 17) - 0.88*log(x1 + 0.64)", ["TEST1", "TEST2"], city_datas_small,
                              refs_small, inner_fits=_fits(["other"], cities=["TEST1", "TEST2"]))
    assert v["accepted"] and all(v["per_city"][c]["nll_orig"] is not None for c in ("TEST1", "TEST2"))
