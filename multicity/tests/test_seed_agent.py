"""Plan 2 addendum (11-04): call point 0 — LLM-proposed seed skeletons for PySR, verified and screened by code, made
diverse, allocated to the search runs, and written as a selection-format file for the library."""
import importlib
import json

import numpy as np
import pytest

from test_search_param import two_cities  # noqa: F401


@pytest.fixture
def sa():
    import seed_agent
    return importlib.reload(seed_agent)


def test_validate_expr_whitelist(sa):
    assert sa.validate_expr("log(N_j + 1.0) - 1.7 * logd")[0]
    assert sa.validate_expr("nexp(0.1 * d) + E_j ^ 0.3")[0]
    assert sa.validate_expr("d ^ logd")[0]                                   # exponent = one variable: allowed
    for bad, why in (("log(foo + 1)", "symbol"), ("sqrt(d)", "function"), ("exp(d)", "function"),
                     ("d ^ (2.0 * logd)", "exponent"), ("log(d +", "parse"), ("", "parse"), ("s_health * d", "symbol")):
        ok, reason, _ = sa.validate_expr(bad)
        assert not ok and why in reason, (bad, reason)


def test_payload_is_whitelisted_and_names_every_lens(sa):
    import llm
    summ = {"d": {"q10": 1.0, "q50": 8.0, "q90": 30.0}, "N_j": {"q10": 2.0, "q50": 20.0, "q90": 90.0}}
    for lens in sa.LENSES:
        p = sa.payload(lens, summ)
        llm.check_input(p)
        assert p["lens"] == lens["name"] and "N_j" in p["variables"] and "s_health" not in p["variables"]
        assert "nexp" in p["operators"] and p["n"] == sa.N_PER_LENS
    assert len(sa.LENSES) == 4 and len({l["name"] for l in sa.LENSES}) == 4


def test_screen_rejects_and_fits(two_cities, sa):
    import sr_data
    data = sr_data.build(["TEST1", "TEST2"], "tract", "train", "search", None, 0)
    props = [{"id": "p1", "lens": "distance", "family": "power", "f": "-1.5 * logd", "g": "1.0", "h": "1.0"},
             {"id": "p2", "lens": "mass", "family": "gravity", "f": "log(N_j + 1.0)", "g": "logd", "h": "d"},
             {"id": "p3", "lens": "mass", "family": "bad", "f": "log(P_j - 1000.0)", "g": "1.0", "h": "1.0"},  # NaN on every row
             {"id": "p4", "lens": "form", "family": "bad", "f": "sqrt(d)", "g": "1.0", "h": "1.0"},
             {"id": "p5", "lens": "mass", "family": "gravity", "f": "log(N_j + 2.0)", "g": "logd", "h": "d"}]    # = p2 structurally
    out = {r["id"]: r for r in sa.screen(props, data)}
    assert out["p1"]["status"] == "ok" and np.isfinite(out["p1"]["loss"]) and len(out["p1"]["consts"]) >= 2
    assert out["p2"]["status"] == "ok" and out["p2"]["equation"].startswith("f = log(#6 + 1.0); g = #2; h = #1; p1 = [")
    assert out["p3"]["status"].startswith("rejected") and "finite" in out["p3"]["status"]
    assert out["p4"]["status"].startswith("rejected") and "function" in out["p4"]["status"]
    assert out["p5"]["status"] == "duplicate of p2"
    assert out["p1"]["backbone"] == "power" and out["p2"]["backbone"] == "power"


def test_diverse_skips_near_copies(sa):
    rng = np.random.default_rng(0)
    base = rng.normal(size=50)
    rows = [{"id": "a", "status": "ok", "loss": 1.0, "eta": base, "family": "f1"},
            {"id": "b", "status": "ok", "loss": 1.1, "eta": 2 * base + 0.001 * rng.normal(size=50), "family": "f1"},
            {"id": "c", "status": "ok", "loss": 1.2, "eta": rng.normal(size=50), "family": "f2"},
            {"id": "d", "status": "rejected: x", "loss": None, "eta": None, "family": "f3"}]
    assert [r["id"] for r in sa.diverse(rows, k=3)] == ["a", "c"]
    assert [r["id"] for r in sa.diverse(rows, k=1)] == ["a"]


def test_allocate_gives_each_run_hand_seeds_and_a_different_subset(sa):
    hand = {"single": ["log(N_j + 1.0) - 1.7 * logd"], "city": [{"f": "log(N_j + 1.0)", "g": "logd", "h": "d", "p1": -1.7, "p2": 0.0}],
            "single_B": ["log(N_j + 1.0)"], "city_B": [{"f": "log(N_j + 1.0)", "g": "N_j", "h": "E_j", "p1": 0.0, "p2": 0.0}]}
    sel = [{"id": f"s{k}", "family": f"fam{k % 3}", "f": f"log(N_j + {k}.0)", "g": "logd", "h": "d", "p": [-1.5, 0.1]}
           for k in range(1, 10)]
    tags = ["tract_A_city8_s20", "tract_A_city8_s21", "tract_B_city8_s20", "tract_A_single-31080_s20"]
    out = sa.allocate(sel, tags, hand, per_run=4)
    a20, a21 = out["tract_A_city8_s20"], out["tract_A_city8_s21"]
    assert a20["city"][0] == hand["city"][0] and a21["city"][0] == hand["city"][0]           # hand seeds everywhere
    ids20 = {g["seed_id"] for g in a20["city"][1:]}
    ids21 = {g["seed_id"] for g in a21["city"][1:]}
    assert len(ids20) == 4 and ids20 != ids21
    assert len({s["family"] for s in sel if s["id"] in ids20}) == 3                          # families rotate
    assert a20["city"][1]["p1"] == -1.5 and a20["city"][1]["p2"] == 0.1
    s = out["tract_A_single-31080_s20"]["single"]
    assert s[0] == hand["single"][0] and all(isinstance(x, str) for x in s) and "(-1.5) * (logd)" in s[1]
    assert "city_B" in out["tract_B_city8_s20"] and out["tract_B_city8_s20"]["city_B"][0] == hand["city_B"][0]
    import pysr_setup
    import sr_data
    active = [v for v in sr_data.VARS if v != "s_health"]
    g = pysr_setup.guesses_for(a20, "A", "city8", active)
    assert g[1]["f"].startswith("log(#6 + ") and len(g[1]["p1"]) == 8 and "seed_id" not in g[1]


def test_select_file_feeds_the_library_pool(two_cities, sa, tmp_path):
    import library
    import sr_data
    importlib.reload(library)
    data = sr_data.build(["TEST1"], "tract", "train", "search", None, 0)
    rows = sa.screen([{"id": "p2", "lens": "mass", "family": "gravity", "f": "log(N_j + 1.0)", "g": "logd", "h": "d"}], data)
    p = sa.select_file(rows, tmp_path / "select_seeds_agent.json")
    s = json.loads(p.read_text())
    assert s["variant"] == "A" and s["candidates"][0]["source_seed"] == "p2" and s["candidates"][0]["template"] is None
    pool = library.pool([p])
    assert len(pool) == 1 and pool[0]["source"] == ["select_seeds_agent.json"]


def test_propose_uses_the_llm_and_keeps_provenance(sa, monkeypatch):
    import llm
    calls = []

    def fake(messages, schema, effort="low", max_tokens=2048, timeout=300, name="x"):
        calls.append((name, effort))
        return {"proposals": [{"family": "power", "rationale": "r", "f": "-1.5 * logd", "g": "1.0", "h": "1.0"}]}
    monkeypatch.setattr(llm, "call", fake)
    out = sa.propose({"d": {"q10": 1.0, "q50": 8.0, "q90": 30.0}})
    assert len(out) == len(sa.LENSES) and len(calls) == len(sa.LENSES) and calls[0][1] == "medium"
    assert {p["lens"] for p in out} == {l["name"] for l in sa.LENSES} and out[0]["id"] == "L1-01"


# --- Round 4 (design decision of 11-04, after the challenger diagnostic): structural classes, one class per run -------------

def test_backbone_classes(sa):
    """The class is decided by code from how distance and S_ij enter eta, not by the LLM's family label."""
    one = {"g": "1.0", "h": "1.0"}
    assert sa.backbone({"f": "log(N_j + 1.0) - 1.7 * logd", **one}) == "power"
    assert sa.backbone({"f": "log(N_j + 1.0)", "g": "log(d + 2.5)", "h": "d"}) == "power"
    assert sa.backbone({"f": "log(N_j + 1.0) - log(S_ij + 1.0)", **one}) == "opportunity"
    assert sa.backbone({"f": "log(N_j + 1.0) - log(S_ij + 1.0)", "g": "logd", "h": "1.0"}) == "gravity_opportunity"
    assert sa.backbone({"f": "log(N_j + 1.0)", "g": "nexp(d / 8.0)", "h": "d ^ 0.5"}) == "kernel_other"
    assert sa.backbone({"f": "log(N_j + 1.0)", "g": "H_j", "h": "1.0"}) == "no_distance"


def test_diverse_by_class_compares_eta_only_within_a_class(sa):
    """The shared distance component makes eta correlate across structurally different skeletons, so the correlation
    filter runs inside each class; every class keeps up to per_class skeletons."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=50)
    rows = [{"id": "a", "status": "ok", "loss": 1.0, "eta": base, "backbone": "power"},
            {"id": "b", "status": "ok", "loss": 1.1, "eta": base * 1.01, "backbone": "opportunity"},
            {"id": "c", "status": "ok", "loss": 1.2, "eta": base * 0.99, "backbone": "opportunity"},
            {"id": "d", "status": "ok", "loss": 1.3, "eta": rng.normal(size=50), "backbone": "opportunity"},
            {"id": "e", "status": "ok", "loss": 1.4, "eta": rng.normal(size=50), "backbone": "opportunity"},
            {"id": "f", "status": "rejected: x", "loss": None, "eta": None, "backbone": None}]
    out = sa.diverse_by_class(rows, per_class=2)
    assert {k: [r["id"] for r in v] for k, v in out.items()} == {"power": ["a"], "opportunity": ["b", "d"]}


def test_allocate_family_gives_each_run_only_its_class(sa):
    by = {"opportunity": [{"id": "o1", "f": "log(N_j + 1.0) - log(S_ij + 1.0)", "g": "1.0", "h": "1.0", "p": [0.0, 0.0]}],
          "gravity_opportunity": [{"id": "g1", "f": "log(N_j + 1.0) - log(S_ij + 1.0)", "g": "logd", "h": "1.0", "p": [-1.2, 0.0]}],
          "kernel_other": [{"id": "k1", "f": "log(N_j + 1.0)", "g": "d", "h": "1.0", "p": [-0.1, 0.0]}]}
    assert sa.FAMILY_SEEDS == {"40": "opportunity", "41": "gravity_opportunity", "42": "kernel_other"}
    out = sa.allocate_family(by, ["tract_A_single-35620_s40", "tract_A_single-35620_s41", "tract_A_single-BAY9_s42"])
    assert out["tract_A_single-35620_s40"]["seed_ids"] == ["o1"] and out["tract_A_single-35620_s40"]["family"] == "opportunity"
    assert out["tract_A_single-35620_s41"]["seed_ids"] == ["g1"]
    s = out["tract_A_single-BAY9_s42"]["single"]
    assert len(s) == 1 and "(-0.1) * (d)" in s[0] and "logd" not in s[0]          # no hand gravity seeds
    import pysr_setup
    import sr_data
    g = pysr_setup.guesses_for(out["tract_A_single-BAY9_s42"], "A", "single", [v for v in sr_data.VARS if v != "s_health"])
    assert len(g) == 1 and "#" in g[0]["f"]
    with pytest.raises(ValueError):
        sa.allocate_family(by, ["tract_A_city8_s40"])                               # family rounds are single-city only
    with pytest.raises(KeyError):
        sa.allocate_family(by, ["tract_A_single-BAY9_s20"])                         # seed number not mapped to a class


def test_family_lenses_carry_no_gravity_reference(sa):
    import llm
    summ = {"d": {"q10": 1.0, "q50": 8.0, "q90": 30.0}}
    assert [l["name"] for l in sa.FAMILY_LENSES] == ["opportunity", "gravity_opportunity", "kernel_other"]
    for lens in sa.FAMILY_LENSES:
        p = sa.payload(lens, summ)
        llm.check_input(p)
        assert p["references"] == [] and p["lens"] == lens["name"]
    assert sa.payload(sa.LENSES[0], summ)["references"] == sa.REFERENCES


def test_challengers_are_screened_and_written_with_their_source(two_cities, sa, tmp_path):
    import sr_data
    data = sr_data.build(["TEST1", "TEST2"], "tract", "train", "search", None, 0)
    assert [c["id"] for c in sa.CHALLENGERS] == ["D-grav_disp", "D-io_rank", "D-grav_disp_io"]
    rows = sa.screen(sa.challenger_props(), data)
    assert [r["backbone"] for r in rows] == ["power", "opportunity", "gravity_opportunity"]
    assert all(r["status"] == "ok" and r["lens"] == "diagnostic" for r in rows)
    s = json.loads(sa.select_file(rows, tmp_path / "select_challengers_diag.json", source="diagnostic").read_text())
    assert s["source"] == "diagnostic" and [c["source_seed"] for c in s["candidates"]] == ["D-grav_disp", "D-io_rank", "D-grav_disp_io"]


def test_pysr_safe_has_no_unary_minus_on_variables_or_functions():
    """Round 5 (11-06): PySR's parser has no unary '-' operator, and the LLM seeds wrote '-log(...)' / '* -H_j', so
    every run failed at start. pysr_safe re-prints a seed so that only numeric literals carry a sign
    ('(-1.0) * log(...)'), equivalent to the original; _single uses it."""
    import re
    import sympy
    import seed_agent
    unary = re.compile(r"(^|[(+*/^,]\s*)-\s*[A-Za-z_(]")
    for s in ("-log(d + 1.0) + q_i * (-H_j)", "log(N_j + 1.0) - 0.5 * nexp(-d / 3.0)", "-(N_j + 1.0) ^ (-0.5)",
              "log(S_ij + 1.0) * -logd"):
        out = seed_agent.pysr_safe(s)
        assert out and not unary.search(out), out
        a, b = seed_agent.validate_expr(s)[2], seed_agent.validate_expr(out)[2]
        syms = sorted(a.free_symbols | b.free_symbols, key=str)
        fa, fb = (sympy.lambdify(syms, x, "numpy") for x in (a, b))
        rng = __import__("numpy").random.default_rng(0)
        for _ in range(5):                                     # equal up to float printing (1/3 -> 0.333...)
            v = rng.uniform(0.5, 3.0, len(syms))
            assert abs(fa(*v) - fb(*v)) < 1e-9 * max(1.0, abs(fa(*v))), (s, out)
    assert seed_agent.pysr_safe("sqrt(d)") is None                                    # not a legal seed
    one = seed_agent._single({"f": "-log(d + 1.0)", "g": "-H_j", "h": "1.0", "p": [-0.5, 0.2]})
    assert not unary.search(one)


def test_pysr_complexity_matches_pysr_node_count():
    """Calibrated 11-06 on the 8 round-5 seeds of 12060 (PySR warned 59 53 73 66 58 67 58 69 > maxsize 30)."""
    import seed_agent
    assert seed_agent.pysr_complexity("log(N_j + 1.0)") == 4                       # log, +, 1.0, N_j
    assert seed_agent.pysr_complexity("log(N_j + 1.0) - 1.5 * log(d + 1.0)") == 11
    assert seed_agent.pysr_complexity("sqrt(d)") is None
