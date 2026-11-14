"""Plan 4 Task 9: structural-class coverage, gap list, round planning, queue file and per-run class seeds
(step 0 of the inner loop)."""
import importlib
import json

import pytest

from test_search_param import two_cities  # noqa: F401

from diversity import CLASSES, classify, coverage, gaps, plan_round, write_queue

POW = "f = log(#6 + 1.0) - 1.8 * log(#1 + 0.7); p1 = []; p2 = []"
POW2 = "f = log(#6 + 3.0) - 2.1 * #2 + 0.1 * log(#24 + 0.2); p1 = []; p2 = []"
GO = "f = log(#6 + 1.0) - 1.8 * log(#1 + 0.7) - 0.3 * log(#26 + 1.0); p1 = []; p2 = []"
OPP = "f = log(#6 + 1.0) - 1.0 * log(#26 + 1.0); p1 = []; p2 = []"
KER = "f = log(#6 + 1.0) - 0.4 * #1; p1 = []; p2 = []"
NOD = "f = log(#6 + 1.0) + 0.2 * #16; p1 = []; p2 = []"


def _pool():
    return [{"cid": f"c{k:04d}", "equation": e, "complexity": 7} for k, e in enumerate([POW, POW2, GO, OPP, KER], 1)]


def test_classify_pool_equations():
    assert [classify(x["equation"]) for x in _pool()] == ["power", "power", "gravity_opportunity", "opportunity",
                                                          "kernel_other"]
    assert classify(NOD) == "no_distance" and set(CLASSES) == {"power", "kernel_other", "opportunity",
                                                               "gravity_opportunity", "no_distance"}
    assert classify("f = log(#6 + 1.0); g = #2; h = #1; p1 = [1.0]; p2 = [1.0]") == "power"     # logd in g


def test_coverage_counts_pool_front_and_searched():
    cov = coverage(_pool(), {"top8": ["c0001", "c0003"], "run_x": [KER]},
                   searched={"power", "kernel_other", "opportunity", "gravity_opportunity"})
    assert cov["power"] == {"n_pool": 2, "n_front": 1, "searched": True, "rho_max": None}
    assert cov["gravity_opportunity"]["n_front"] == 1 and cov["kernel_other"]["n_front"] == 1
    assert cov["no_distance"] == {"n_pool": 0, "n_front": 0, "searched": False, "rho_max": None}


def test_gaps_puts_never_searched_classes_first():
    cov = {"power": {"n_pool": 163, "searched": True}, "no_distance": {"n_pool": 0, "searched": False},
           "kernel_other": {"n_pool": 7, "searched": True}, "opportunity": {"n_pool": 23, "searched": True},
           "gravity_opportunity": {"n_pool": 31, "searched": True}}
    assert gaps(cov)[:1] == ["no_distance"] and "kernel_other" in gaps(cov) and "power" not in gaps(cov)
    assert gaps(cov) == ["no_distance", "kernel_other"]


def test_plan_round_forces_a_never_searched_class():
    rows = plan_round(["kernel_other", "opportunity"], ["35620", "31080"], 5,
                      searched={"power", "kernel_other", "opportunity", "gravity_opportunity"})
    assert any(r["cls"] == "no_distance" for r in rows) and len(rows) == 6 and {r["seed_no"] for r in rows} <= {50, 51, 52}
    assert rows[0] == {"tag": "tract_A_single-35620_s50", "city": "35620", "cls": "no_distance", "seed_no": 50}
    one = plan_round(["kernel_other"], ["35620"], 6, searched=set(CLASSES))           # fill, never repeat a class
    assert [r["cls"] for r in one] == ["kernel_other", "opportunity", "gravity_opportunity"]
    assert [r["seed_no"] for r in one] == [60, 61, 62]
    cov = {"power": {"n_pool": 206, "n_front": 6}, "kernel_other": {"n_pool": 8, "n_front": 0},
           "opportunity": {"n_pool": 24, "n_front": 0}, "gravity_opportunity": {"n_pool": 37, "n_front": 2},
           "no_distance": {"n_pool": 0, "n_front": 0}}
    r5 = plan_round(["no_distance"], ["35620"], 5, searched=set(CLASSES) - {"no_distance"}, cov=cov)
    assert [r["cls"] for r in r5] == ["no_distance", "gravity_opportunity", "kernel_other"]   # 11-06 round 5
    none = plan_round([], ["35620"], 6, searched=set(CLASSES))                         # no gap: the non-power classes
    assert [r["cls"] for r in none] == ["kernel_other", "opportunity", "gravity_opportunity"]


def test_write_queue_format_matches_round_4(tmp_path):
    rows = plan_round(["kernel_other", "opportunity"], ["35620", "31080"], 5, searched={"power"})
    write_queue(rows, tmp_path / "q.txt")
    t = (tmp_path / "q.txt").read_text()
    assert "SR_BLOCK=search23" in t and "tract A single:35620 50 40 100" in t
    body = [l.split() for l in t.splitlines() if l.strip() and not l.startswith("#")]
    assert len(body) == 6 and all(len(r) == 6 for r in body)
    head = "\n".join(l for l in t.splitlines() if l.startswith("#"))
    assert "MAXP 24" in head and "SR_THREADS=5" in head and "SR_TIMEOUT_SINGLE=10800" in head


def test_lens_for_class_has_the_origin_rule():
    from agent_points import lens_for_class
    for c in CLASSES:
        lens = lens_for_class(c)
        assert set(lens) == {"name", "focus", "prefix", "references"} and lens["name"] == c and lens["references"] == []
        assert "Origin variables" in lens["focus"] and "interaction" in lens["focus"]
    with pytest.raises(KeyError):
        lens_for_class("other")


def test_seeds_for_round_one_class_per_run_and_extra_seeds(two_cities, monkeypatch, tmp_path):
    import sr_data
    import seed_agent
    import diversity
    importlib.reload(seed_agent)
    importlib.reload(diversity)
    calls = []

    def fake_propose(summary, effort="medium", lenses=None):
        calls.append(lenses[0]["name"])
        if lenses[0]["name"] == "opportunity":
            return [{"id": "O-01", "lens": "opportunity", "family": "io", "rationale": "r",
                     "f": "log(N_j + 1.0) - 1.0 * log(S_ij + 1.0)", "g": "1.0", "h": "1.0"}]
        return [{"id": "K-01", "lens": "kernel_other", "family": "exp", "rationale": "r",
                 "f": "log(N_j + 1.0) - 0.3 * d", "g": "1.0", "h": "1.0"},
                {"id": "K-02", "lens": "kernel_other", "family": "bad", "rationale": "r",
                 "f": "log(N_j + 1.0) - 0.3 * logd", "g": "1.0", "h": "1.0"}]             # power: not this class
    monkeypatch.setattr(seed_agent, "propose", fake_propose)
    rows = [{"tag": "tract_A_single-TEST1_s50", "city": "TEST1", "cls": "kernel_other", "seed_no": 50},
            {"tag": "tract_A_single-TEST1_s51", "city": "TEST1", "cls": "opportunity", "seed_no": 51}]
    data = sr_data.build(["TEST1"], "tract", "train", "search", None, 0)
    extra = {"TEST1": [{"f": "log(N_j + 2.0) - 1.9 * log(d + 0.5)", "g": "1.0", "h": "1.0"}]}
    out = diversity.seeds_for_round(rows, {}, extra, data, gdir=tmp_path)
    # one seed per class < MIN_SEEDS: each class is asked a second time (11-06 size rule)
    assert sorted(calls) == ["kernel_other"] * 2 + ["opportunity"] * 2 and set(out["files"]) == {r["tag"] for r in rows}
    assert out["llm_calls"] == 4
    k = json.loads(out["files"]["tract_A_single-TEST1_s50"].read_text())
    o = json.loads(out["files"]["tract_A_single-TEST1_s51"].read_text())
    assert k["family"] == "kernel_other" and k["seed_ids"] == ["K-01"] and o["seed_ids"] == ["O-01"]
    has = lambda seeds: any("d + 0.5" in s or "0.5 + d" in s for s in seeds)                       # noqa: E731
    assert has(k["single"]) and has(o["single"])                    # extra seed in both (PySR-safe re-print, 11-06)
    assert all(isinstance(s, str) for s in k["single"]) and not any("logd" in s for s in k["single"])


def test_seeds_over_maxsize_are_dropped_and_the_class_is_asked_again(two_cities, monkeypatch, tmp_path):
    """11-06: every round-5 seed had PySR complexity 53-73 > maxsize 30; seeds above maxsize are dropped and a class
    left with fewer than 3 seeds gets one more call whose lens states the node limit."""
    import sr_data
    import seed_agent
    import diversity
    importlib.reload(seed_agent)
    importlib.reload(diversity)
    big = " + ".join(f"log(N_j + {k}.0) * H_j" for k in range(1, 8))
    def propose(summary, effort="medium", lenses=None):
        calls.append(lenses[0]["focus"])
        if len(calls) == 1:
            return [{"id": "K-01", "lens": "kernel_other", "family": "f", "rationale": "r", "f": big + " - 0.3 * d",
                     "g": "1.0", "h": "1.0"}]
        return [{"id": "K-02", "lens": "kernel_other", "family": "f", "rationale": "r",
                 "f": "log(N_j + 1.0) - 0.3 * d", "g": "1.0", "h": "1.0"}]
    calls = []
    monkeypatch.setattr(seed_agent, "propose", propose)
    rows = [{"tag": "tract_A_single-TEST1_s50", "city": "TEST1", "cls": "kernel_other", "seed_no": 50}]
    data = sr_data.build(["TEST1"], "tract", "train", "search", None, 0)
    out = diversity.seeds_for_round(rows, {}, {}, data, gdir=tmp_path)
    spec = json.loads(out["files"]["tract_A_single-TEST1_s50"].read_text())
    assert out["llm_calls"] == 2 and spec["seed_ids"] == ["K-02"] and "previous batch" in calls[1]
    assert all(seed_agent.pysr_complexity(s) <= 30 for s in spec["single"])
