"""Plan 4 Task 6: scoring rules (design §5.7) — floor, tolerance delta_city, S0 / classic shares, near-best,
term existence and the lexicographic rank."""
import pandas as pd
import pytest

from scoring import MARGIN, delta_city, near_best, passes_floor, rank, share, term_exists

CITIES = ["A", "B", "C", "D"]


def _refs(grav=5.0, m2=4.0, s0=4.1, gexp=5.2, rad=4.9):
    return {"grav_power": {"nll_per_visit": grav}, "M2_glm": {"nll_per_visit": m2}, "S0": {"nll_per_visit": s0},
            "grav_exp": {"nll_per_visit": gexp}, "ext_rad": {"nll_per_visit": rad}}


def _fits(vals, se=0.0005):
    """vals: {cid: {city: nll or None}}; best per city gets within_1se True."""
    rows = []
    for cid, per in vals.items():
        for city, v in per.items():
            rows.append({"cid": cid, "city": city, "val_nll": v, "finite": v is not None, "se_vs_best": se,
                         "within_1se": False})
    df = pd.DataFrame(rows)
    ok = df[df.finite]
    best = ok.groupby("city")["val_nll"].idxmin()
    df.loc[best.to_numpy(), "within_1se"] = True
    return df


def test_delta_and_share_zero_points():
    r = _refs()
    assert MARGIN == 0.03 and delta_city(r) == pytest.approx(0.03)
    assert share(4.05, r, "S0") == pytest.approx(0.5) and share(4.5, r, "grav_power") == pytest.approx(0.5)
    assert share(None, r) is None and share(4.05, {"grav_power": {"nll_per_visit": 5.0}}) is None
    assert passes_floor(4.85, r) and not passes_floor(4.95, r) and not passes_floor(None, r)
    assert passes_floor(4.95, {"grav_power": {"nll_per_visit": 5.0}, "ext_rad": {"error": "x"}})   # missing ignored
    assert delta_city({"grav_power": {"nll_per_visit": 4.0}, "M2_glm": {"nll_per_visit": 4.5}}) == 0.0


def test_term_exists_uses_delta_city():
    r = _refs()
    assert term_exists(0.031, r) and term_exists(0.03, r) and not term_exists(0.029, r)


def test_near_best_uses_max_of_se_and_delta():
    refs = {c: _refs() for c in CITIES}
    fits = _fits({"X": {c: 4.10 for c in CITIES}, "Y": {"A": 4.125, "B": 4.135, "C": None, "D": 4.10}})
    nb = near_best(fits, refs)
    assert ("Y", "A") in nb and ("Y", "B") not in nb and ("Y", "C") not in nb and ("Y", "D") in nb
    wide = _fits({"X": {"A": 4.10}, "Y": {"A": 4.135}}, se=0.04)                 # 1 SE > delta: the SE decides
    assert ("Y", "A") in near_best(wide, {"A": _refs()})


def test_rank_prefers_clean_over_slightly_better_dirty():
    # two candidates near-best in 4 cities; A has one disguised-linear flag and is 2 milli-nats better; B is clean
    refs = {c: _refs() for c in CITIES}
    fits = _fits({"A": {c: 4.100 for c in CITIES}, "B": {c: 4.102 for c in CITIES}})
    reps = [{"cid": "A", "complexity": 9, "merged_from": []}, {"cid": "B", "complexity": 11, "merged_from": []}]
    ranked = rank(reps, fits, refs, flags={"A": {"n": 1}, "B": {"n": 0}})
    assert [r["cid"] for r in ranked][:2] == ["B", "A"] and ranked[0]["gate1"] and ranked[1]["gate1"]
    assert ranked[0]["n_near"] == 4 and ranked[0]["n_flags"] == 0 and ranked[1]["n_flags"] == 1
    assert ranked[1]["n_1se"] == 4 and ranked[0]["near_cities"] == CITIES


def test_rank_order_near_count_then_cleanliness_then_complexity():
    refs = {c: _refs() for c in CITIES}
    fits = _fits({"P": {c: 4.10 for c in CITIES},                                  # near 4, clean, cx 15
                  "Q": {"A": 4.10, "B": 4.10, "C": 4.20, "D": 4.20},               # near 2, clean, cx 5
                  "R": {c: 4.10 for c in CITIES},                                  # near 4, clean, cx 12
                  "S": {c: 4.10 for c in CITIES}})                                 # near 4, degenerate, cx 5
    reps = [{"cid": k, "complexity": cx, "merged_from": []} for k, cx in (("P", 15), ("Q", 5), ("R", 12), ("S", 5))]
    flags = {"P": {"n": 0}, "Q": {"n": 0}, "R": {"n": 0}, "S": {"n": 0, "degenerate": ["huge_const"]}}
    ranked = rank(reps, fits, refs, flags, min_cities=4, top=2)
    assert [r["cid"] for r in ranked] == ["R", "P", "S", "Q"]
    assert [r["gate1"] for r in ranked] == [True, True, False, False]


def test_rank_counts_merged_members():
    refs = {c: _refs() for c in CITIES}
    fits = _fits({"A": {"A": 4.10, "B": 4.10, "C": 4.30, "D": 4.30}, "A2": {"A": 4.30, "B": 4.30, "C": 4.10, "D": 4.10}})
    ranked = rank([{"cid": "A", "complexity": 5, "merged_from": ["A2"]}], fits, refs, flags={})
    assert ranked[0]["n_near"] == 4 and ranked[0]["gate1"]


def test_rank_drops_candidates_below_floor():
    refs = {c: _refs() for c in CITIES}
    fits_all_above_floor = _fits({"A": {c: 4.95 for c in CITIES}, "B": {c: None for c in CITIES}})
    reps = [{"cid": "A", "complexity": 5, "merged_from": []}, {"cid": "B", "complexity": 5, "merged_from": []}]
    assert rank(reps, fits_all_above_floor, refs, flags={}) == []
    assert rank([], fits_all_above_floor, refs, flags={}) == []


def test_v3_margins_and_the_registered_rule():
    """V3 rule of 2026-11-10: the search-city gate at 0.05, the merge at 0.03 without chaining, conclusions at 0.03;
    registered_rule(m) restores the one-multiplier chain rule of plans 2-5 for reproduction."""
    import scoring
    v3 = (0.03, 0.05, 0.03, "leader")
    assert (scoring.MARGIN, scoring.GATE_MARGIN, scoring.MERGE_MARGIN, scoring.MERGE_LINKAGE) == v3
    r = _refs()                                                                          # gap 1.0
    assert delta_city(r) == pytest.approx(0.03) and delta_city(r, 0.05) == pytest.approx(0.05)
    with scoring.registered_rule():
        assert (scoring.MARGIN, scoring.GATE_MARGIN, scoring.MERGE_MARGIN, scoring.MERGE_LINKAGE) == (0.03, 0.03, 0.03, "chain")
    with scoring.registered_rule(0.05):
        assert (scoring.MARGIN, scoring.GATE_MARGIN, scoring.MERGE_MARGIN) == (0.05, 0.05, 0.05)
        assert delta_city(r) == pytest.approx(0.05)
    assert (scoring.MARGIN, scoring.GATE_MARGIN, scoring.MERGE_MARGIN, scoring.MERGE_LINKAGE) == v3


def test_rank_uses_the_gate_margin_and_near_best_the_conclusion_margin():
    import scoring
    refs = {c: _refs() for c in CITIES}                                                  # delta 0.03 / 0.05
    fits = _fits({"X": {c: 4.10 for c in CITIES}, "Y": {c: 4.14 for c in CITIES}})
    reps = [{"cid": "X", "complexity": 9, "merged_from": []}, {"cid": "Y", "complexity": 5, "merged_from": []}]
    by = {r["cid"]: r for r in rank(reps, fits, refs, flags={})}
    assert by["Y"]["n_near"] == 4 and by["Y"]["gate1"]                                   # 0.04 <= 0.05
    assert ("Y", "A") not in near_best(fits, refs) and ("Y", "A") in near_best(fits, refs, 0.05)
    assert {r["cid"]: r for r in rank(reps, fits, refs, flags={}, margin=0.03)}["Y"]["n_near"] == 0
    with scoring.registered_rule():
        assert {r["cid"]: r for r in rank(reps, fits, refs, flags={})}["Y"]["n_near"] == 0
