"""Plan 5 (pre-freeze consolidation): pre-freeze reforms, frozen-rule ranking on a copy of the loop state, discovery
fits and coverage, comparison with the distance-band flexible model, report."""
import math

import pytest

import checks
import review
from refit_select import XS, parse_equation
from sr_data import VARS

import prefreeze


def _cand(r):
    e, why = review._parse(r["sympy"])
    assert why is None
    eq, _ = review._new_entry_fields({"cid": r["parent"], "equation": "f = log(#6 + 1.0); p1 = []; p2 = []",
                                      "template": None, "variant": "A", "complexity": 10}, e, None)
    return parse_equation(eq, None, "A", 1)


def _at(e, N_j):
    sub = {x: 1.0 for x in e.free_symbols}
    sub[XS[VARS.index("N_j")]] = N_j
    return e.subs(sub)


def _entry(cid, eq="f = log(#6 + 1.0); p1 = []; p2 = []", cx=20):
    return {"cid": cid, "equation": eq, "template": None, "variant": "A", "complexity": cx, "consts_refit": [1.0],
            "source": ["select_x.json"], "round": 6}


def _verify(orig, s):
    return {"accepted": "x18*x7" not in s, "reason": "ok", "mean_norm_delta": 0.2, "new_nonfittable": [],
            "per_city": {}, "equation": "f = log(#6 + 1); p1 = []; p2 = []", "complexity": 20, "consts": [1.0]}


# ---------------------------------------------------------------------------------------------------------- Task 1
def test_reforms_config_parses_flags_and_zero_safety():
    rs = prefreeze.load_reforms(prefreeze.REFORMS)
    assert [r["id"] for r in rs] == ["R1", "R2", "R3", "R4", "R5", "R6"]
    assert [r["parent"] for r in rs] == ["c0413c1"] * 4 + ["c0353c1", "c0355"]
    for r in rs:
        e, why = review._parse(r["sympy"])
        assert why is None and r["rationale"]
    mix = {r["id"]: bool(checks.unit_mixing(_cand(r))) for r in rs}
    assert mix == {"R1": True, "R2": False, "R3": False, "R4": True, "R5": False, "R6": False}
    for r in rs:                      # every reform is finite when N_j = 0 (literal constants, other vars = 1)
        e, _ = review._parse(r["sympy"])
        assert math.isfinite(float(_at(e, N_j=0.0)))


def test_reform_entries_names_sources_and_skips_rejected():
    rs = [r for r in prefreeze.load_reforms(prefreeze.REFORMS) if r["id"] in ("R1", "R2", "R4", "R6")]
    pool = [_entry("c0413c1"), _entry("c0355")]
    acc, res = prefreeze.reform_entries(rs, pool, _verify)
    assert [x["cid"] for x in acc] == ["c0413c1r1", "c0413c1r2", "c0355r1"]       # R4 (x18*x7) rejected
    assert all(x["source"] == ["prefreeze_reform"] and x["round"] == 8 for x in acc)
    assert all(x["template"] is None and x["variant"] == "A" and x["complexity"] == 20 for x in acc)
    assert acc[0]["parent"] == "c0413c1" and acc[0]["consts_refit"] == [1.0] and acc[0]["insight"]
    assert [(r["cid"], r["accepted"]) for r in res] == [("c0413c1r1", True), ("c0413c1r2", True),
                                                        ("c0413c1r3", False), ("c0355r1", True)]
    assert res[2]["id"] == "R4" and res[2]["parent"] == "c0413c1"


def test_reform_with_unknown_parent_raises():
    with pytest.raises(KeyError):
        prefreeze.reform_entries([{"id": "R9", "parent": "c9999", "sympy": "log(x6 + 1)", "rationale": "x"}], [],
                                 _verify)


def test_cities_are_search_and_discovery_only(tmp_path, monkeypatch):
    import config
    csv = tmp_path / "cities.csv"
    csv.write_text("city,role\nS1,search\nD1,discovery\nL1,lawtest\nC1,confirm\nR1,rest\n")
    monkeypatch.setattr(config, "CITIES_CSV", csv)
    assert prefreeze.cities() == ["S1", "D1"]


# ---------------------------------------------------------------------------------------------------------- Task 2
def _fake_loop_dir(d):
    import numpy as np
    import pandas as pd
    (d / "inner_boot").mkdir(parents=True)
    (d / "pool.json").write_text('[{"cid": "c0001"}]')
    (d / "flags.json").write_text('{"c0001": {"n": 0}}')
    pd.DataFrame({"cid": ["c0001"], "city": ["A"], "val_nll": [1.0]}).to_parquet(d / "inner_fits.parquet")
    np.save(d / "inner_boot/A.npy", np.ones((3, 1)))
    (d / "state.json").write_text("{}")                     # not part of the copy
    return d


def _digest(d):
    import hashlib
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


class _RecordingSteps:
    def __init__(self, pool):
        self._pool, self.calls, self.ranked = list(pool), [], [{"cid": "c0001"}]

    def pool(self):
        return list(self._pool)

    def save_pool(self, pool):
        self.calls.append(("save_pool", [x["cid"] for x in pool]))
        self._pool = list(pool)

    def fit_new(self, r):
        self.calls.append(("fit_new", r))

    def rank(self, r):
        self.calls.append(("rank", r))
        return self.ranked


def test_prepare_copy_copies_and_never_touches_source(tmp_path):
    src = _fake_loop_dir(tmp_path / "loop")
    before = _digest(src)
    prefreeze.prepare_copy(src, tmp_path / "pf")
    assert _digest(src) == before
    got = _digest(tmp_path / "pf")
    assert set(got) == {"pool.json", "flags.json", "inner_fits.parquet", "inner_boot/A.npy"}
    assert all(got[k] == before[k] for k in got)
    with pytest.raises(FileExistsError):
        prefreeze.prepare_copy(src, tmp_path / "pf")


def test_rank_with_reforms_calls_in_order():
    s = _RecordingSteps(pool=[{"cid": "c0001"}])
    out = prefreeze.rank_with_reforms(s, [{"cid": "c0413c1r1"}])
    assert s.calls == [("save_pool", ["c0001", "c0413c1r1"]), ("fit_new", 8), ("rank", 8)] and out == s.ranked


def test_reform_fate_reports_merges():
    ranked = [{"cid": "c0413c1", "merged_from": ["c0413c1r1"], "n_near": 6, "n_flags": 1, "complexity": 27,
               "gate1": True},
              {"cid": "c0355r1", "merged_from": [], "n_near": 5, "n_flags": 0, "complexity": 15, "gate1": True}]
    f = {x["cid"]: x for x in prefreeze.reform_fate(ranked, [{"cid": "c0413c1r1"}, {"cid": "c0355r1"},
                                                             {"cid": "c0353c1r1"}])}
    assert f["c0413c1r1"]["rep"] == "c0413c1" and f["c0413c1r1"]["position"] == 1
    assert f["c0413c1r1"]["n_near"] == 6 and f["c0413c1r1"]["n_flags"] == 1 and f["c0413c1r1"]["gate1"] is True
    assert f["c0355r1"]["rep"] == "c0355r1" and f["c0355r1"]["position"] == 2 and f["c0355r1"]["complexity"] == 15
    assert f["c0353c1r1"]["rep"] is None and f["c0353c1r1"]["position"] is None   # not ranked (no floor city)


# ---------------------------------------------------------------------------------------------------------- Task 3
REFS3 = {c: {"grav_power": {"nll_per_visit": 5.0}, "grav_exp": {"nll_per_visit": 5.1}, "ext_rad": {"nll_per_visit": 4.9},
             "M2_glm": {"nll_per_visit": 4.0}, "S0": {"nll_per_visit": 4.5}} for c in ("A", "B", "C")}


def _cv(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["cid", "city", "val_nll", "finite"])


def _cv_near(near):
    """Every (cid, city) pair of c1, c2 x A, B, C; near-best (and primary) where listed; all above the floor."""
    import pandas as pd
    rows = [{"cid": cid, "city": c, "val_nll": 4.2, "finite": True, "near_best": (cid, c) in near,
             "primary": (cid, c) in near} for cid in ("c1", "c2") for c in ("A", "B", "C")]
    return pd.DataFrame(rows)


def test_check_reproduction():
    old = _cv([("c1", "A", 3.0, True), ("c1", "B", 2.0, True), ("c2", "A", 1.0, False)])
    new = _cv([("c1", "A", 3.0, True), ("c1", "B", 2.0 + 5e-7, True), ("c2", "A", 1.1, True), ("c3", "A", 1.0, True)])
    d = prefreeze.check_reproduction(new, old, ["c1", "c2"])
    assert set(d) == {"c1", "c2"}
    assert d["c1"] == pytest.approx(5e-7) and d["c2"] == float("inf")
    assert prefreeze.check_reproduction(new, old, ["c3"])["c3"] == float("inf")      # nothing to compare with


def test_coverage_tables_clean_excludes_flagged():
    cv = _cv_near([("c1", "A"), ("c1", "B"), ("c2", "C")])          # c1 near-best in A, B; c2 in C
    t = prefreeze.coverage_tables(cv, REFS3, {"c1": {"complexity": 10}, "c2": {"complexity": 12}},
                                  {"c1": {"n": 1}, "c2": {"n": 0}})
    assert [r["cid"] for r in t["all"]] == ["c1", "c2"] and t["all"][-1]["covered"] == 3
    assert [r["cid"] for r in t["clean"]] == ["c2"] and t["clean"][-1]["covered"] == 1
    assert t["clean"][-1]["n_cities"] == 3                          # the uncovered cities still count
    t2 = prefreeze.coverage_tables(cv, REFS3, {"c1": {"complexity": 10}, "c2": {"complexity": 12}},
                                   {"c1": {"n": 0, "degenerate": ["huge_const"]}, "c2": {"n": 0}})
    assert [r["cid"] for r in t2["clean"]] == ["c2"]                 # degenerate counts as a flag (scoring.n_flags)


# ---------------------------------------------------------------------------------------------------------- Task 6
def test_compare_rmse_and_band_filter():
    import numpy as np
    skel = {("cA", "X"): {0: (-1.0, 0.7), 1: (-1.4, 0.8), 2: (-1.8, 0.9), 3: (-2.0, 1.0), 4: None}}
    flex = {"X": {0: (-1.2, 0.7), 1: (-1.4, 0.8), 2: (-1.8, 0.9), 3: (-2.4, 0.5), 4: (-2.0, 1.0)}}
    vis = {"X": {0: 5000, 1: 5000, 2: 5000, 3: 10, 4: 5000}}     # band 3 under MIN_BAND_VISITS, band 4 None in skel
    dref = {"X": {0: 0.7, 1: 1.4, 2: 3.2, 3: 7.0, 4: 14.0}}
    r = prefreeze.compare(skel, flex, vis, dref).iloc[0]
    assert (r.cid, r.city, r.n_bands) == ("cA", "X", 3)
    assert r.rmse_N == pytest.approx(0.0) and r.rmse_d == pytest.approx(0.2 / math.sqrt(3))
    s = np.polyfit(np.log([0.7, 1.4, 3.2]), [0.7, 0.8, 0.9], 1)[0]
    assert r.slope_N_skel == pytest.approx(s) and r.slope_N_flex == pytest.approx(s)


def test_compare_needs_three_bands():
    skel = {("cA", "X"): {0: (-1.0, 0.7), 1: (-1.4, 0.8)}}
    flex = {"X": {0: (-1.2, 0.7), 1: (-1.4, 0.8)}}
    r = prefreeze.compare(skel, flex, {"X": {0: 5000, 1: 5000}}, {"X": {0: 0.7, 1: 1.4}}).iloc[0]
    assert r.n_bands == 2 and math.isnan(r.rmse_d) and math.isnan(r.rmse_N) and math.isnan(r.slope_N_flex)


def test_closest_counts_only_cities_with_two_candidates():
    import pandas as pd
    df = pd.DataFrame({"cid": ["a", "b", "a"], "city": ["X", "X", "Y"], "rmse_N": [0.1, 0.2, 0.05],
                       "rmse_d": [0.3, 0.1, 0.2]})
    assert prefreeze.closest(df) == {"rmse_N": {"a": 1}, "rmse_N_shape": {}, "rmse_d": {"b": 1}}


def _write_minimal_outputs(root, include_city):
    import json
    import pandas as pd
    (root / "discovery").mkdir(parents=True)
    (root / "loop").mkdir()
    (root / "banded").mkdir()
    (root / "reforms.json").write_text(json.dumps({"entries": [], "results": [
        {"id": "R1", "cid": "c1r1", "parent": "c1", "sympy": "log(x6 + 1)", "accepted": False, "reason": "x",
         "mean_norm_delta": 2.0, "new_nonfittable": [], "per_city": {}, "equation": None, "complexity": None}]}))
    (root / "fate.json").write_text("[]")
    pd.DataFrame([{"cid": "c1", "city": include_city, "val_nll": 4.2, "finite": True, "near_best": False,
                   "primary": False, "share_s0": None, "share_grav": None, "consts": "[1.0]"}]).to_parquet(
        root / "discovery/cities_val.parquet")
    (root / "discovery/coverage.json").write_text(json.dumps({"all": [], "clean": [], "c2_all": [], "c2_clean": [],
                                                              "reproduction": {}}))
    (root / "loop/pool.json").write_text(json.dumps([_entry("c1")]))
    (root / "loop/flags.json").write_text("{}")


def test_build_report_refuses_sealed_city_ids(tmp_path, monkeypatch):
    import report
    monkeypatch.setattr(report, "_lawtest_ids", lambda: {"99999"})
    monkeypatch.setattr(report, "_confirm_ids", lambda: set())
    _write_minimal_outputs(tmp_path, include_city="99999")
    with pytest.raises(RuntimeError):
        prefreeze.build_report(tmp_path / "r.md", root=tmp_path)
    assert not (tmp_path / "r.md").exists()


def test_build_report_writes_the_seven_sections(tmp_path, monkeypatch):
    import report
    monkeypatch.setattr(report, "_lawtest_ids", lambda: {"99999"})
    monkeypatch.setattr(report, "_confirm_ids", lambda: set())
    _write_minimal_outputs(tmp_path, include_city="12060")
    s = prefreeze.build_report(tmp_path / "r.md", root=tmp_path)
    text = (tmp_path / "r.md").read_text()
    assert [ln.split()[1] for ln in text.splitlines() if ln.startswith("## ")] == [f"{k}." for k in range(1, 8)]
    assert "12060" in text and s["n_cities"] == 1


def test_build_report_compares_with_the_flexible_model(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import report
    from sr_data import VARS as V
    monkeypatch.setattr(report, "_lawtest_ids", lambda: set())
    monkeypatch.setattr(report, "_confirm_ids", lambda: set())
    _write_minimal_outputs(tmp_path, include_city="12060")
    eq = "f = 0.219*log(#7) - 2.19*log(#1 + 1.12) + 0.781*log(#6 + 1); p1 = []; p2 = []"
    (tmp_path / "loop/pool.json").write_text(json.dumps([_entry("c1", eq=eq, cx=13), _entry("c2", eq=eq, cx=13)]))
    pd.DataFrame([{"cid": k, "city": "12060", "val_nll": 4.2, "finite": True, "near_best": True, "primary": k == "c1",
                   "share_s0": 0.8, "share_grav": 0.99, "consts": json.dumps(c)}
                  for k, c in (("c1", [0.18, 2.23, 2.23, 0.88, 6.93]), ("c2", [0.18, 1.0, 0.5, 0.5, 6.93]))]
                 ).to_parquet(tmp_path / "discovery/cities_val.parquet")
    refs = {str(b): {**{v: 1.0 for v in V}, "d": d, "logd": math.log(d), "N_j": 40.0, "dlogS_dlogd": 1.5, "n_rows": 9}
            for b, d in ((0, 0.7), (1, 1.4), (2, 3.2))}
    prof = {"0": [-2.23 * 0.7 / (0.7 + 2.23), 0.88 * 40 / 46.93], "1": [-2.23 * 1.4 / (1.4 + 2.23), 0.88 * 40 / 46.93],
            "2": [-2.23 * 3.2 / (3.2 + 2.23), 0.88 * 40 / 46.93], "3": None, "4": None, "5": None}
    (tmp_path / "banded/12060.json").write_text(json.dumps(
        {"city": "12060", "refs": {**refs, "3": None, "4": None, "5": None}, "profile": prof,
         "visits": {str(b): 5000.0 for b in range(6)}, "val_nll": 4.0, "m2_val_nll": 4.01}))
    s = prefreeze.build_report(tmp_path / "r.md", root=tmp_path)
    assert s["closest"] == {"rmse_N": {"c1": 1}, "rmse_N_shape": {"c1": 1}, "rmse_d": {"c1": 1}}  # c1 = flex
    text = (tmp_path / "r.md").read_text()
    row = next(ln for ln in text.splitlines() if ln.startswith("| c1 | 1 |"))
    assert row.split("|")[3].strip() == "0.000"                              # rmse_d median of c1


# ---------------------------------------------------------------------------------------------- final-review fixes
def test_check_reproduction_counts_cities_missing_from_the_new_fit():
    old = _cv([("c1", "A", 3.0, True), ("c1", "B", 2.0, True), ("c1", "C", 1.0, True)])
    new = _cv([("c1", "A", 3.0, True)])                       # B, C dropped (a city error in fit_library)
    assert prefreeze.check_reproduction(new, old, ["c1"])["c1"] == float("inf")


def test_reproduction_gate_stops_on_mismatch_empty_or_city_error():
    import pandas as pd
    ok = pd.DataFrame({"city": ["A", "B"], "primary": ["c1", "c1"]})
    assert prefreeze.reproduction_gate({"c1": 0.0, "c2": 5e-7}, ok) == []
    assert prefreeze.reproduction_gate({}, ok)                                    # nothing compared
    assert prefreeze.reproduction_gate({"c1": 1e-3}, ok)                          # mismatch
    err = pd.DataFrame({"city": ["A", "B"], "primary": ["c1", None], "error": [None, "ValueError('x')"]})
    assert any("B" in r for r in prefreeze.reproduction_gate({"c1": 0.0}, err))  # a city with a fit error


def test_compare_shape_metric_ignores_a_constant_level_offset():
    skel = {("cA", "X"): {b: (-1.0 - 0.3 * b, 0.6 + 0.1 * b + 0.3) for b in range(4)},     # e_N shifted by +0.3
            ("cB", "X"): {b: (-1.0 - 0.3 * b, 0.9) for b in range(4)}}                     # flat e_N
    flex = {"X": {b: (-1.0 - 0.3 * b, 0.6 + 0.1 * b) for b in range(4)}}
    vis, dref = {"X": {b: 5000 for b in range(4)}}, {"X": {0: 0.7, 1: 1.4, 2: 3.2, 3: 7.0}}
    df = prefreeze.compare(skel, flex, vis, dref).set_index("cid")
    assert df.loc["cA", "rmse_N"] == pytest.approx(0.3) and df.loc["cA", "rmse_N_shape"] == pytest.approx(0.0)
    assert df.loc["cB", "rmse_N_shape"] > 0.05
    assert prefreeze.closest(df.reset_index())["rmse_N_shape"] == {"cA": 1}


def test_build_report_shows_the_unit_mixing_flag_of_every_reform(tmp_path, monkeypatch):
    import json
    import report
    monkeypatch.setattr(report, "_lawtest_ids", lambda: set())
    monkeypatch.setattr(report, "_confirm_ids", lambda: set())
    _write_minimal_outputs(tmp_path, include_city="12060")
    rs = {r["id"]: r for r in prefreeze.load_reforms(prefreeze.REFORMS)}
    res = [{"id": k, "cid": f"c0413c1r{i}", "parent": "c0413c1", "sympy": rs[k]["sympy"], "accepted": k != "R4",
            "reason": "ok", "mean_norm_delta": 0.0, "new_nonfittable": [], "per_city": {}, "equation": None,
            "complexity": None} for i, k in enumerate(("R1", "R2", "R4"), 1)]
    (tmp_path / "reforms.json").write_text(json.dumps({"entries": [], "results": res}))
    prefreeze.build_report(tmp_path / "r.md", root=tmp_path)
    rows = {ln.split("|")[1].strip(): ln.split("|") for ln in (tmp_path / "r.md").read_text().splitlines()
            if ln.startswith("| R")}
    col = lambda k: rows[k][-2].strip()  # noqa: E731 — last column = unit mixing
    assert (col("R1"), col("R2"), col("R4")) == ("yes", "no", "yes")


# ---------------------------------------------------------------------------------- 11-09 design decisions: helpers
def _bjson(profile, d, visits):
    return {"profile": {str(k): v for k, v in profile.items()},
            "refs": {str(k): (None if v is None else {"d": v}) for k, v in d.items()},
            "visits": {str(k): v for k, v in visits.items()}}


def test_flex_quantities_match_the_plan5_definitions():
    import numpy as np
    b = _bjson({0: [-1.0, 0.3], 1: [-1.2, 0.35], 2: [-1.4, 0.5], 3: [-1.6, 0.6], 4: [-1.8, 0.7], 5: None},
               {0: 0.7, 1: 1.4, 2: 3.2, 3: 7.0, 4: 14.0, 5: None}, {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 500, 5: 0})
    q = prefreeze.flex_quantities(b)
    assert q["ed_5_10km"] == pytest.approx(-1.6)
    assert q["slope_eN_logd"] == pytest.approx(np.polyfit(np.log([0.7, 1.4, 3.2, 7.0]), [0.3, 0.35, 0.5, 0.6], 1)[0])
    b2 = _bjson({0: [-1.0, 0.3], 1: [-1.2, 0.35], 3: [-1.6, 0.6]}, {0: 0.7, 1: 1.4, 3: 7.0}, {0: 5000, 1: 5000, 3: 999})
    q2 = prefreeze.flex_quantities(b2)
    assert q2["ed_5_10km"] is None and q2["slope_eN_logd"] is None        # band 3 under MIN_BAND_VISITS, 2 bands


def test_skeleton_profiles_counts_failures():
    import json
    import pandas as pd
    eq = "f = 0.219*log(#7) - 2.19*log(#1 + 1.12) + 0.781*log(#6 + 1); p1 = []; p2 = []"
    pool = {"c1": _entry("c1", eq=eq, cx=13), "c2": _entry("c2", eq=eq, cx=13)}
    cv = pd.DataFrame([{"cid": "c1", "city": "X", "finite": True, "consts": json.dumps([0.18, 2.23, 2.23, 0.88, 6.93])},
                       {"cid": "c2", "city": "X", "finite": True, "consts": json.dumps([1.0])},       # wrong length
                       {"cid": "c1", "city": "Y", "finite": True, "consts": json.dumps([0.18, 2.23, 2.23, 0.88, 6.93])}])
    from sr_data import VARS as V
    ref = {**{v: 1.0 for v in V}, "d": 3.0, "logd": math.log(3.0), "N_j": 40.0, "dlogS_dlogd": 1.5}
    prof, failed = prefreeze.skeleton_profiles(cv, pool, {"X": {2: ref}})
    assert set(prof) == {("c1", "X")} and failed == 1                     # Y has no band refs: not attempted


def test_rank_with_reforms_runs_under_the_registered_rule():
    """Plan 5 ranked under the registered rule (0.03 everywhere, chain merge); the V3 rule change of 2026-11-10 changed the
    defaults, so the plan-5 step pins the old rule to stay reproducible."""
    import scoring
    seen = []

    class S(_RecordingSteps):
        def rank(self, r):
            seen.append((scoring.GATE_MARGIN, scoring.MERGE_MARGIN, scoring.MERGE_LINKAGE))
            return super().rank(r)
    prefreeze.rank_with_reforms(S(pool=[{"cid": "c0001"}]), [{"cid": "c0413c1r1"}])
    assert seen == [(0.03, 0.03, "chain")] and (scoring.GATE_MARGIN, scoring.MERGE_LINKAGE) == (0.05, "leader")
