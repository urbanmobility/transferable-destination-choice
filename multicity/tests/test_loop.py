"""Plan 4 Task 14: the agent-loop controller — stability, stuck detection, budget, night window, state, run (every
heavy step replaced by a stub)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

import loop
from test_search_param import two_cities  # noqa: F401

CFG = {"inner_rounds_max": 3, "outer_cycles_max": 2, "llm_calls_max": 600, "review_quota": 60, "hyps_per_cycle": 5,
       "stable_rounds": 2,
       "schedule": {"outer_after": ["2026-11-06T21:00:00Z", "2026-11-07T18:00:00Z"],
                    "freeze_request_after": "2026-11-07T23:00:00Z"},
       "search": {"maxp": 24, "threads": 5, "timeout_s": 10800, "night_only_utc": [3, 15]}}
T0 = datetime(2026, 11, 5, 23, 0, tzinfo=timezone.utc)                  # day 1, 15:00 PST: daytime


class FakeSteps:
    def __init__(self, ranks=None, passes=None, t0=T0):
        self.t = t0
        self.calls = []
        self.ranks = list(ranks or [])          # top-8 cid lists returned by rank(), one per round (last repeats)
        self.passes = list(passes or [])        # passed ids per outer cycle
        self.review_quota = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += timedelta(seconds=s)

    def init(self):
        self.calls.append(("init",))

    def search(self, st, r, max_calls):
        self.calls.append(("search", r, self.t.isoformat()))
        self.t += timedelta(hours=3)
        return {"new": [f"n{r}"], "llm_calls": 3, "classes": ["no_distance", "kernel_other"]}

    def fit_new(self, r):
        self.calls.append(("fit_new", r))

    def review(self, r, quota, reviewed):
        self.calls.append(("review", r))
        self.review_quota.append(quota)
        self.t += timedelta(hours=1)
        return {"new": [], "reviewed": [f"x{r}_{k}" for k in range(min(quota, 2))], "llm_calls": quota}

    def rank(self, r):
        self.calls.append(("rank", r))
        top = self.ranks.pop(0) if len(self.ranks) > 1 else (self.ranks[0] if self.ranks else ["a"])
        return [{"cid": c, "gate1": True, "n_near": 4, "n_flags": 0} for c in top]

    def outer(self, cycle, top8, max_calls):
        self.calls.append(("outer", cycle))
        self.t += timedelta(hours=2)
        p = self.passes.pop(0) if self.passes else []
        return {"passed": p, "registered": [f"H{cycle}-01"], "llm_calls": 1, "seeds": {"A": [{"f": "x"}]} if p else {}}

    def coverage(self, st):
        return {"power": {"n_pool": 1}}

    def stuck_report(self, payload, n):
        self.calls.append(("stuck_report", n))
        return {"tried": "t", "evidence": "e", "why_framework": "w", "proposal": "p"}, 1


@pytest.fixture
def L(tmp_path):
    def make(steps=None, cfg=None):
        lines = []
        lp = loop.Loop(cfg or json.loads(json.dumps(CFG)), steps or FakeSteps(), root=tmp_path / "loop",
                       agent_dir=tmp_path / "agent", log=lines.append)
        lp.lines = lines
        return lp
    return make


def test_state_roundtrip(tmp_path):
    st = loop.State(round=5, cycle=1, llm_calls=70, top8_history=[["a", "b"]], reviewed=["c1"],
                    searched_classes=["power", "no_distance"], extra_seeds={"A": [{"f": "x"}]}, search_nights=["2026-11-06"])
    st.save(tmp_path / "state.json")
    assert loop.State.load(tmp_path / "state.json") == st
    assert loop.State.load(tmp_path / "nothing.json") == loop.State()
    assert set(loop.State().searched_classes) == {"power", "kernel_other", "opportunity", "gravity_opportunity"}


def test_config_file_is_the_owner_budget():
    cfg = json.loads((loop.CONFIG).read_text())
    assert (cfg["inner_rounds_max"], cfg["outer_cycles_max"], cfg["llm_calls_max"], cfg["review_quota"],
            cfg["hyps_per_cycle"], cfg["stable_rounds"]) == (3, 2, 600, 60, 5, 2)
    assert cfg["search"] == {"maxp": 24, "threads": 5, "timeout_s": 10800, "night_only_utc": [3, 15]}


def test_stable_three_conditions(L):
    lp = L()
    st = loop.State(round=5, top8_history=[["a"], ["b"]], rounds_since_outer=1, rounds_searched=1)
    assert not lp.stable(st)
    assert lp.stable(loop.State(round=5, top8_history=[["a"], ["a"]], rounds_since_outer=1, rounds_searched=1))
    assert lp.stable(loop.State(round=7, top8_history=[["a"], ["b"]], rounds_since_outer=1, rounds_searched=3))
    lp.steps.t = datetime(2026, 11, 6, 21, 30, tzinfo=timezone.utc)                     # past outer_after[0]
    assert lp.stable(st)
    after = loop.State(round=5, cycle=1, top8_history=[["a"], ["b"]], rounds_since_outer=0, rounds_searched=1)
    assert not lp.stable(after)                       # an outer cycle must be followed by an inner round first
    after.rounds_since_outer = 1
    assert not lp.stable(after)                       # cycle 2's deadline is outer_after[1]
    lp.steps.t = datetime(2026, 11, 7, 18, 30, tzinfo=timezone.utc)
    assert lp.stable(after)


def test_stuck_after_a_full_cycle_without_improvement(L):
    lp = L()
    st = loop.State(round=5, cycle=1, top8_history=[["a", "b"], ["a", "b"]], top8_at_cycle_start=["a", "b"],
                    cycle_passed=[])
    assert lp.stuck(st, "outer") == "no_improvement"
    assert lp.stuck(st, "inner") is None                                   # only the inner loop is stable
    st.cycle_passed = ["H1-01"]
    assert lp.stuck(st, "outer") is None
    st.cycle_passed, st.top8_history = [], [["a", "b"], ["a", "c"]]
    assert lp.stuck(st, "outer") is None
    empty = loop.State(round=5, last_ranked_empty=True, searched_classes=list(loop.CLASSES))
    assert lp.stuck(empty, "inner") == "nothing_passes_floor"
    empty.searched_classes = ["power"]
    assert lp.stuck(empty, "inner") is None                                # unsearched classes remain


def test_loop_stuck_when_nothing_passes_floor(L, tmp_path):
    """Review Focus 5: no candidate passes the floor after every class was searched -> the stuck path: the diagnosis
    input file, one stuck_report call, a STUCK ledger line, exit code 3."""
    steps = FakeSteps(ranks=[[]])
    lp = L(steps)
    lp.state = loop.State(searched_classes=list(loop.CLASSES))
    lp.state.save(lp.state_path)
    assert lp.run() == 3
    p = tmp_path / "agent/stuck_1.in.json"
    assert p.exists() and json.loads(p.read_text())["reason"] == "nothing_passes_floor"
    assert (tmp_path / "agent/stuck_1.json").exists() and ("stuck_report", 1) in steps.calls
    assert any(l.startswith("STUCK") for l in lp.lines)
    assert loop.State.load(lp.state_path).stuck_n == 1


def test_budget_stops_llm_calls(L):
    steps = FakeSteps()
    lp = L(steps)
    loop.State(llm_calls=600).save(lp.state_path)
    assert lp.run() == 0 and not any(c[0] == "review" for c in steps.calls)
    assert any(l.startswith("BUDGET") for l in lp.lines)
    steps2 = FakeSteps()
    lp2 = L(steps2)
    loop.State(llm_calls=590).save(lp2.state_path)
    lp2.cfg["outer_cycles_max"] = 0
    lp2.run()
    assert steps2.review_quota[0] == 10                                    # only the remaining budget


def test_full_run_one_search_per_night_and_freeze_request(L):
    """Round 0 (review only) by day; round 5 in night 1; round 6 must wait for night 2 even though night 1 is still
    open; outer cycles at the scheduled deadlines; the freeze request after the last cycle; extra seeds kept."""
    steps = FakeSteps(ranks=[["a", "b"], ["a", "c"], ["a", "d"], ["a", "e"]], passes=[["H1-01"], []])
    lp = L(steps)
    assert lp.run() == 0
    kinds = [c for c in steps.calls if c[0] in ("search", "outer", "review")]
    searches = [c for c in steps.calls if c[0] == "search"]
    assert [c[1] for c in searches] == [5, 6]
    nights = [datetime.fromisoformat(c[2]) for c in searches]
    assert all(3 <= t.hour < 15 for t in nights) and nights[1] - nights[0] > timedelta(hours=12)
    assert [c[1] for c in steps.calls if c[0] == "outer"] == [1, 2]
    assert kinds.index(("outer", 1)) < [i for i, c in enumerate(kinds) if c[0] == "search" and c[1] == 6][0]
    st = loop.State.load(lp.state_path)
    assert st.cycle == 2 and st.round == 6 and st.extra_seeds == {"A": [{"f": "x"}]} and st.llm_calls > 0
    assert any("FREEZE" in l for l in lp.lines) and ("init",) in steps.calls
    assert sorted(set(st.searched_classes)) == sorted(loop.CLASSES)


def test_until_pauses_and_resume_skips_finished_rounds(L):
    steps = FakeSteps(ranks=[["a"], ["b"]])
    lp = L(steps)
    assert lp.run(until=T0 + timedelta(minutes=30)) == 0                    # round 0 done, paused before round 5
    st = loop.State.load(lp.state_path)
    assert st.round == 0 and [c for c in steps.calls if c[0] == "review"] == [("review", 0)]
    assert any("paused" in l for l in lp.lines)
    lp2 = L(steps)
    lp2.run(until=steps.t + timedelta(hours=1))
    assert [c for c in steps.calls if c[0] == "review"].count(("review", 0)) == 1


def test_real_steps_plumbing_on_fixture_cities(two_cities, tmp_path, monkeypatch):
    """The real Steps on the two fixture cities (validation-mode data stands in for the inner blocks, which the
    6-tract fixture cannot fill): incremental inner fits keep rows and bootstrap columns aligned and recompute the
    1-SE flags; flags, rank and a review round with a stubbed agent run end to end."""
    import importlib
    import numpy as np
    import library
    import agent_points
    importlib.reload(library)
    steps = loop.Steps(CFG, root=tmp_path / "loop", procs=1)
    cities = ["TEST1", "TEST2"]
    monkeypatch.setattr(steps, "cities", lambda role: cities if role == "search" else [])
    datas = {c: library.city_data(c, None) for c in cities}
    monkeypatch.setattr(steps, "datas", lambda: datas)
    refs = {c: {"grav_power": {"nll_per_visit": 50.0}, "grav_exp": {"nll_per_visit": 50.0},
                "ext_rad": {"nll_per_visit": 50.0}, "M2_glm": {"nll_per_visit": 0.0}, "S0": {"nll_per_visit": 10.0}}
            for c in cities}
    monkeypatch.setattr(steps, "refs", lambda: refs)
    E = lambda cid, eq, cx: {"cid": cid, "equation": eq, "template": None, "variant": "A", "complexity": cx,  # noqa: E731
                             "consts_refit": None, "source": ["t"], "round": 0}
    pool = [E("c0001", "f = log(#6 + 1.0) - 1.5 * log(#1 + 1.0); p1 = []; p2 = []", 9),
            E("c0002", "f = log(#6 + 2.0) - 0.5 * #1; p1 = []; p2 = []", 7)]
    (tmp_path / "loop").mkdir(parents=True)
    steps.save_pool(pool)
    steps.fit_new(0)
    fits, boot = steps._fits()
    assert len(fits) == 4 and boot["TEST1"].shape[1] == 2 and set(steps.flags()) == {"c0001", "c0002"}
    first = {c: boot[c].copy() for c in cities}
    steps.save_pool(pool + [E("c0003", "f = log(#6 + 1.0) - 1.4 * log(#1 + 0.5) - 0.01 * #7; p1 = []; p2 = []", 13)])
    steps.fit_new(5)
    fits, boot = steps._fits()
    assert len(fits) == 6 and boot["TEST2"].shape[1] == 3 and set(steps.flags()) == {"c0001", "c0002", "c0003"}
    for c in cities:
        f = fits[fits.city == c]
        assert list(f.cid) == ["c0001", "c0002", "c0003"] and f.within_1se.sum() >= 1
        assert np.allclose(boot[c][:, :2], first[c], equal_nan=True)          # old columns kept, new one appended
    ranked = steps.rank(5)
    assert ranked and (tmp_path / "loop/ranked_r5.json").exists() and all(r["n_near"] >= 1 for r in ranked[:1])
    monkeypatch.setattr(agent_points, "review", lambda p: {"problems": [], "cleaned_sympy": "log(x6 + 1) - 1.5*log(x1 + 1)",
                                                          "changes": "", "expected_fit_cost": "0", "insight": "i"})
    monkeypatch.setattr(loop.config, "RESULTS", tmp_path / "res")
    out = steps.review(5, 2, set())
    assert out["llm_calls"] == 2 and len(out["reviewed"]) == 2
    assert len(steps.pool()) == 3 + len(out["new"]) and len(steps._fits()[0]) == 2 * len(steps.pool())
    assert (tmp_path / "res/agent/r5").exists()


def _outer_world(tmp_path, monkeypatch, hyps):
    """Synthetic city-level fits for Steps.outer: 10 discovery + 10 lawtest cities where c1's distance coefficient
    grows with log population; fit_library / covariates / reference points / ablations / validation data stubbed."""
    import pandas as pd
    import agent_points
    import fit_library
    import laws
    import test_laws as TL
    res = tmp_path / "res"
    monkeypatch.setattr(loop.config, "RESULTS", res)
    monkeypatch.setattr(laws, "RESULTS", res)
    disc = [f"D{k:02d}" for k in range(10)]
    law_c = [f"L{k:02d}" for k in range(10)]
    roles = {"search": disc[:2], "discovery": disc[2:], "lawtest": law_c}
    steps = loop.Steps(CFG, root=res / "loop", procs=1)
    monkeypatch.setattr(steps, "cities", lambda role: roles.get(role, []))
    monkeypatch.setattr(laws, "_lawtest_ids", lambda: set(law_c))
    (res / "loop").mkdir(parents=True)
    steps.save_pool(TL._pool())
    (res / "refs").mkdir(parents=True)
    for c in disc + law_c:
        (res / f"refs/{c}.json").write_text(json.dumps({"models": {"grav_power": {"nll_per_visit": 5.0},
                                                                     "M2_glm": {"nll_per_visit": 4.0},
                                                                     "S0": {"nll_per_visit": 4.5}}}))
    cvd, covd = TL._cities_val(n=10, seed=0)
    cvl, covl = TL._cities_val(n=10, seed=7)
    cvl["city"] = cvl["city"].str.replace("D", "L")
    covl["city"] = covl["city"].str.replace("D", "L")
    cov_all = pd.concat([covd, covl], ignore_index=True)
    fit_calls = []

    def fake_fit(cities, procs=8, cands=None, out_dir=None, **k):
        fit_calls.append((cities[0][0], procs))
        out_dir.mkdir(parents=True, exist_ok=True)
        src = cvl if cities[0].startswith("L") else cvd
        src[src.cid.isin([x["cid"] for x in cands])].to_parquet(out_dir / "cities_val.parquet", index=False)
    monkeypatch.setattr(fit_library, "main", fake_fit)
    monkeypatch.setattr(laws, "covariates", lambda cities: cov_all.set_index("city").loc[list(cities)].reset_index())
    monkeypatch.setattr(laws, "reference_points", lambda datas: {})
    monkeypatch.setattr(loop, "_small_datas", lambda cities: {})
    monkeypatch.setattr(loop, "_ablations_val", lambda *a, **k: {})
    data = {c: TL._city_logit(1.0, k) for k, c in enumerate(disc)}
    monkeypatch.setattr(loop, "_city_val", lambda c: data[c])
    monkeypatch.setattr(agent_points, "hypotheses", lambda p: {"hypotheses": hyps.pop(0) if hyps else []})
    lines = []
    monkeypatch.setattr(laws, "register", lambda *a, **k: TRUE_REGISTER(*a, **{**k, "log": lines.append}))
    monkeypatch.setattr(laws, "test_registered", lambda *a, **k: TRUE_TEST(*a, **{**k, "log": lines.append}))
    return {"steps": steps, "res": res, "disc": disc, "lines": lines, "fit_calls": fit_calls}


HYP = {"statement": "distance decay steepens with population", "skeleton": "c1", "parameter": "f:c1", "covariate": "log_population",
       "direction": "+", "test": "slope_ci", "criterion": "CI > 0", "rationale": "r"}


def test_real_steps_outer_cycle_glue(tmp_path, monkeypatch):
    """Steps.outer end to end: analysis payload -> hypotheses (stub agent) -> registration -> lawtest fits -> Holm
    test -> pass -> joint refit -> substituted seeds for the search cities."""
    import pandas as pd
    w = _outer_world(tmp_path, monkeypatch, [[HYP, dict(HYP, parameter="f:c3")]])
    steps, res, disc, lines = w["steps"], w["res"], w["disc"], w["lines"]
    o = steps.outer(1, ["c1", "c2"], max_calls=10)
    assert o["registered"] == ["H1-01"] and o["passed"] == ["H1-01"] and o["llm_calls"] == 1
    assert set(o["seeds"]) == set(disc[:2]) and "N_j" in o["seeds"]["D00"][0]["f"]
    inp = json.loads((res / "agent/c1/hypotheses.in.json").read_text())
    assert "L00" not in json.dumps(inp) and inp["n_cities"] == 10
    assert json.loads((res / "agent/c1/lawtest_pass.json").read_text()) == {"H1-01": True}
    j = json.loads((res / "loop/c1_laws.json").read_text())
    assert j[0]["id"] == "H1-01" and j[0]["joint"]["param"] == "f:c1"
    assert any("registered H1-01" in l for l in lines) and any("H1-01 PASS" in l for l in lines)
    pt = pd.read_parquet(res / "loop/c1_discovery/param_table.parquet")
    assert set(pt.city) <= set(disc) and json.loads((res / "loop/c1_discovery/cross_city.json").read_text())
    assert w["fit_calls"] == [("D", 24), ("L", 24)]                         # final review #8: outer procs 24


def test_outer_tests_registered_hypotheses_left_untested_by_a_crash(tmp_path, monkeypatch):
    """Final review #4: a hypothesis registered in a cycle that crashed before its test is tested in the next outer
    call even when the agent proposes nothing new; the next cycle's payload lists it with its verdict (#5)."""
    import laws
    w = _outer_world(tmp_path, monkeypatch, [[HYP], [], []])
    steps, res = w["steps"], w["res"]
    real = laws.test_registered
    monkeypatch.setattr(laws, "test_registered", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        steps.outer(1, ["c1", "c2"], max_calls=10)
    assert not (res / "laws/results.json").exists()
    monkeypatch.setattr(laws, "test_registered", real)
    o = steps.outer(1, ["c1", "c2"], max_calls=10)
    assert o["registered"] == [] and o["passed"] == ["H1-01"]
    assert [r["id"] for r in json.loads((res / "laws/results.json").read_text())] == ["H1-01"]
    steps.outer(2, ["c1", "c2"], max_calls=10)
    inp = json.loads((res / "agent/c2/hypotheses.in.json").read_text())
    assert inp["registered"][0]["id"] == "H1-01" and inp["registered"][0]["pass"] is True


def test_dry_run_accepts_the_library_pool_before_the_first_run(tmp_path, monkeypatch, capsys):
    """Final review #9: before the first run only results/multicity/library/pool.json exists."""
    res = tmp_path / "res"
    (res / "library").mkdir(parents=True)
    (res / "library/pool.json").write_text("[]")
    monkeypatch.setattr(loop.config, "RESULTS", res)
    monkeypatch.setattr(loop.config, "cities_with_role", lambda r: [])
    loop._dry_run(CFG)
    out = json.loads(capsys.readouterr().out)
    assert out["checks"]["pool (library or loop)"] is True and "loop/pool.json" not in out["checks"]


import laws as _laws  # noqa: E402
TRUE_REGISTER, TRUE_TEST = _laws.register, _laws.test_registered


def test_real_steps_search_round_glue(two_cities, tmp_path, monkeypatch):
    """Steps.search on the fixture cities: queue + class seed files written, the queue runner and the fold-4
    selection stubbed; selected equations are canonicalised (origin-only terms stripped) and deduplicated against the
    pool before they get new cids with round and source."""
    import importlib
    import library
    import orchestrate
    import seed_agent
    importlib.reload(library)
    cities = ["TEST1", "TEST2"]
    steps = loop.Steps(CFG, root=tmp_path / "loop", procs=1)
    monkeypatch.setattr(steps, "cities", lambda role: cities if role == "search" else [])
    datas = {c: library.city_data(c, None) for c in cities}
    monkeypatch.setattr(steps, "datas", lambda: datas)
    (tmp_path / "loop").mkdir(parents=True)
    steps.save_pool([{"cid": "c0224", "equation": "f = log(#6 + 1.0) - 1.5 * log(#1 + 1.0); p1 = []; p2 = []",
                      "complexity": 9, "round": 0, "source": ["x"]}])
    monkeypatch.setattr(seed_agent, "propose", lambda summary, effort="medium", lenses=None: [
        {"id": f"{lenses[0]['prefix']}-01", "lens": lenses[0]["name"], "family": "f", "rationale": "r",
         "f": "log(N_j + 1.0) + 0.1 * H_j", "g": "1.0", "h": "1.0"}])
    gdir = tmp_path / "guesses"
    import diversity
    real_seeds = diversity.seeds_for_round
    monkeypatch.setattr(diversity, "seeds_for_round", lambda rows, s, e, d, gdir_=None, **k: real_seeds(rows, s, e, d, gdir=gdir))
    queued = []
    monkeypatch.setattr(steps, "_run_queue", lambda q, r: queued.append(Path_(q).read_text()))
    srm = tmp_path / "srm"
    (srm).mkdir()
    steps.srm = srm

    def fake_run_all(jobs):
        for name, script, args, env in jobs:
            (srm / f"{name}.json").write_text(json.dumps({"candidates": [
                {"equation": "f = log(#6 + 1.0) - 1.5 * log(#1 + 1.0) - 2.0 * #18; p1 = []; p2 = []", "template": "single",
                 "complexity": 11, "consts_refit": [1.0, 1.5, 1.0, 2.0]},                   # = c0224 after canonicalisation
                {"equation": "f = log(#6 + 1.0) + 0.2 * #16; p1 = []; p2 = []", "template": "single", "complexity": 7,
                 "consts_refit": [1.0, 0.2]}]}))
            assert env["SR_EVAL_BLOCK"] == "fold:4" and env["SR_TRAIN_BLOCK"] == "search23" and env["SR_EVAL_SPLIT"] == "train"
    monkeypatch.setattr(orchestrate, "_run_all", fake_run_all)
    st = loop.State(top8_history=[["c0224"]])
    out = steps.search(st, 5, max_calls=10)
    assert "SR_BLOCK=search23" in queued[0] and "single:TEST1 50 40 100" in queued[0]
    assert out["llm_calls"] == 6 and out["classes"] == ["kernel_other", "no_distance", "power"]     # 1-member power: sparse
    # (one stub seed per class < MIN_SEEDS -> every class asked twice)
    pool = steps.pool()
    assert [x["cid"] for x in pool] == ["c0224", "c0225"] and pool[1]["round"] == 5
    assert pool[1]["source"] == ["select_loop_r5_TEST1.json"] and "#16" in pool[1]["equation"]
    assert len(list(gdir.glob("*.json"))) == 6 and (tmp_path / "loop/r5/queue.txt").exists()


from pathlib import Path as Path_  # noqa: E402


# ---- final-review fix pass ----

class CrashingSteps(FakeSteps):
    """review() raises once (after a successful search), then works."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.boom = True

    def review(self, r, quota, reviewed):
        if r > 0 and self.boom:
            self.boom = False
            raise RuntimeError("disk full")
        return super().review(r, quota, reviewed)


def test_crash_after_search_resumes_without_searching_again(L):
    """Final review #1, #3: a step that raises after the night's search writes a CRASH ledger line, keeps the
    sub-step markers in state.json and exits 4; the rerun continues with fit / review / rank of the same round and
    never re-runs the search."""
    steps = CrashingSteps(ranks=[["a"], ["b"], ["c"]])
    lp = L(steps)
    assert lp.run() == 4
    st = loop.State.load(lp.state_path)
    assert st.cur_round == 5 and "search" in st.cur_done and st.rounds_searched == 1 and "disk full" in st.last_error
    assert any(l.startswith("CRASH") and "review" in l for l in lp.lines)
    lp2 = L(steps)
    lp2.cfg["outer_cycles_max"] = 0
    lp2.run(until=steps.t + timedelta(hours=2))
    assert [c[1] for c in steps.calls if c[0] == "search"] == [5]
    st = loop.State.load(lp2.state_path)
    assert st.round == 5 and st.cur_round is None and st.rounds_searched == 1


def test_top8_is_the_gate1_set_of_the_ranking(L):
    """Rule of 11-06 (replaces final review #2 "first 8 of the ranking, gate1 or not"): the loop's top 8 holds
    only the gate1 candidates (near-best in >= 4 search cities, at most 8, scoring.rank); fewer than 8 are not padded
    with candidates near-best nowhere (round 5 put c0310c1 = -1.75 log d in slot 8). A ranking with candidates above
    the floor but no gate1 gives an empty top 8: not "nothing passes the floor" in the inner loop, but a stuck stop at
    the outer gate instead of an outer cycle on nothing."""
    class S(FakeSteps):
        def rank(self, r):
            self.calls.append(("rank", r))
            return [{"cid": f"k{i}", "gate1": i < 2, "n_near": 4 if i < 2 else 1, "n_flags": 0} for i in range(10)]
    steps = S()
    lp = L(steps)
    lp.inner_round(lp.state)
    assert lp.state.top8_history[-1] == ["k0", "k1"] and not lp.state.last_ranked_empty

    class N(FakeSteps):
        def rank(self, r):
            self.calls.append(("rank", r))
            return [{"cid": f"k{i}", "gate1": False, "n_near": 1, "n_flags": 0} for i in range(3)]
    none = N()
    lp1 = L(none)
    lp1.state = loop.State()                                   # tmp_path is shared with lp's saved state
    lp1.inner_round(lp1.state)
    assert lp1.state.top8_history[-1] == [] and not lp1.state.last_ranked_empty
    assert lp1.stuck(lp1.state, "inner") is None

    empty = FakeSteps(ranks=[[]])
    lp2 = L(empty)
    loop.State(round=5, rounds_since_outer=1, rounds_searched=3, top8_history=[[]], last_ranked_empty=True,
               phase="inner").save(lp2.state_path)
    assert lp2.run() == 3 and not any(c[0] == "outer" for c in empty.calls)
    assert any("no_candidate_for_outer" in l for l in lp2.lines)


def test_queue_is_rerun_when_its_runs_have_no_front(tmp_path):
    """11-06: the failed round-5 queue ended with QUEUE_DONE but no run had a hall of fame; a resume must rerun it."""
    st = loop.Steps(CFG, root=tmp_path / "loop", procs=1)
    st.srm = tmp_path / "srm"
    (st.srm / "logs").mkdir(parents=True)
    q = tmp_path / "q.txt"
    q.write_text("# x\ntract A single:A 50 40 100\ntract A single:B 51 40 100\n")
    (st.srm / "logs/queue_loop_r5.log").write_text("... QUEUE_DONE q\n")
    assert not st._queue_finished(q, 5)
    for tag in ("tract_A_single-A_s50", "tract_A_single-B_s51"):
        (st.srm / f"runs/{tag}").mkdir(parents=True)
        (st.srm / f"runs/{tag}/hall_of_fame.csv").write_text("complexity,loss,equation\n")
    assert st._queue_finished(q, 5)


def test_select_tolerates_a_failed_city_but_not_all(two_cities, tmp_path, monkeypatch):
    """11-06: one city's selection failing must not lose the other cities' candidates; all failing is a crash."""
    import importlib
    import library
    import orchestrate
    importlib.reload(library)
    cities = ["TEST1", "TEST2"]
    steps = loop.Steps(CFG, root=tmp_path / "loop", procs=1)
    monkeypatch.setattr(steps, "cities", lambda role: cities if role == "search" else [])
    datas = {c: library.city_data(c, None) for c in cities}
    monkeypatch.setattr(steps, "datas", lambda: datas)
    (tmp_path / "loop").mkdir(parents=True)
    steps.save_pool([{"cid": "c0224", "equation": "f = log(#6 + 1.0) - 1.5 * log(#1 + 1.0); p1 = []; p2 = []",
                      "complexity": 9, "round": 0, "source": ["x"]}])
    steps.srm = tmp_path / "srm"
    steps.srm.mkdir()
    rows = [{"tag": f"tract_A_single-{c}_s50", "city": c, "cls": "no_distance", "seed_no": 50} for c in cities]

    def half(jobs):
        name = jobs[0][0]
        (steps.srm / f"{name}.json").write_text(json.dumps({"candidates": [
            {"equation": "f = log(#6 + 1.0) + 0.2 * #16; p1 = []; p2 = []", "template": "single", "complexity": 7}]}))
        raise RuntimeError(f"failed: {[j[0] for j in jobs[1:]]}")
    monkeypatch.setattr(orchestrate, "_run_all", half)
    import ledger
    before = ledger.LEDGER.read_text()
    lines = []
    steps.log = lines.append                                   # tests never write the real ledger
    new = steps._select_and_add(rows, 5)
    assert [x["cid"] for x in new] == ["c0225"]
    assert ledger.LEDGER.read_text() == before and any("partly failed" in l for l in lines)
    monkeypatch.setattr(orchestrate, "_run_all", lambda jobs: (_ for _ in ()).throw(RuntimeError("failed: all")))
    with pytest.raises(RuntimeError):
        steps._select_and_add(rows, 6)
