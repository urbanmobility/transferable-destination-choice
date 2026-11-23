"""Agent-in-the-loop controller (plan 4 Task 14).

Inner loop (search cities, training period): round 0 = review of the initial pool (no search); rounds 5, 6, 7 =
diversity seeds -> one night search (search23) -> fold-4 selection -> pool -> inner fits -> review / cleaning -> rank.
Stable (enter the outer loop) = the top 8 unchanged over `stable_rounds` rounds, or the search rounds used up, or the
cycle's scheduled deadline passed — always after at least one inner round since the last outer cycle.
Outer cycle (validation period): top 8 on discovery + search cities -> cross-city summary -> hypotheses (agent) ->
registration -> test on the lawtest cities (Holm, cumulative) -> passed laws: joint refit + substituted seeds.
Stuck: (a) a full cycle with no change of the top 8 and no passed hypothesis; (b) every structural class
searched and nothing passes the floor -> diagnosis input + stuck_report call + STUCK ledger line, exit 3.
Budget (config/loop.json, edited only by hand): search rounds, outer cycles, LLM calls, review quota, hypotheses per cycle; the
search runs only inside the night window (UTC hours) and at most once per night (a 3-day schedule).
Every step is ledgered; state.json makes the loop resumable; the heavy work sits in Steps (tests stub it).

CLI (compute server, scripts/multicity): python -I loop.py run [--until ISO] [--dry-run] | python -I loop.py status
"""
import dataclasses
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import config  # noqa: E402

CONFIG = Path(__file__).resolve().parent / "config/loop.json"
CLASSES = ("power", "kernel_other", "opportunity", "gravity_opportunity", "no_distance")
INITIAL_SEARCHED = ("power", "kernel_other", "opportunity", "gravity_opportunity")       # rounds 1-4
FIRST_SEARCH_ROUND = 5
WAIT_S = 600
TOP_K = 8


@dataclasses.dataclass
class State:
    round: int = -1                      # last finished inner round (-1 none, 0 initial-pool review, 5.. searches)
    cycle: int = 0                       # finished outer cycles
    llm_calls: int = 0
    rounds_searched: int = 0
    rounds_since_outer: int = 0
    top8_history: list = dataclasses.field(default_factory=list)
    reviewed: list = dataclasses.field(default_factory=list)
    searched_classes: list = dataclasses.field(default_factory=lambda: list(INITIAL_SEARCHED))
    search_nights: list = dataclasses.field(default_factory=list)
    extra_seeds: dict = dataclasses.field(default_factory=dict)
    top8_at_cycle_start: list = dataclasses.field(default_factory=list)
    cycle_passed: list = dataclasses.field(default_factory=list)
    registered: list = dataclasses.field(default_factory=list)
    last_ranked_empty: bool = False
    stuck_n: int = 0
    phase: str = "inner"
    pool_path: str = ""
    cur_round: int | None = None         # the inner round in progress (resume after a crash: final review #1)
    cur_done: list = dataclasses.field(default_factory=list)      # its finished sub-steps: search, fit, review
    last_error: str = ""

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=1))
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class Loop:
    def __init__(self, cfg, steps, root=None, agent_dir=None, log=None, cfg_path=None):
        self.cfg, self.steps, self.cfg_path = cfg, steps, cfg_path
        self.root = Path(root) if root else config.RESULTS / "loop"
        self.agent_dir = Path(agent_dir) if agent_dir else config.RESULTS / "agent"
        if log is None:
            import ledger
            log = ledger.add
        self.log = log
        self.state_path = self.root / "state.json"
        self.state = State.load(self.state_path)
        self._step = "start"

    # ------------------------------------------------------------------------------------------- rules
    def _deadline(self, cycle):
        oa = self.cfg["schedule"]["outer_after"]
        oa = oa if isinstance(oa, list) else [oa]
        return _ts(oa[min(cycle, len(oa) - 1)])

    def stable(self, st):
        if st.round < 0 or st.rounds_since_outer < 1:
            return False
        h, k = st.top8_history, int(self.cfg["stable_rounds"])
        if len(h) >= k and all(sorted(x) == sorted(h[-1]) for x in h[-k:]):
            return True
        if st.rounds_searched >= int(self.cfg["inner_rounds_max"]):
            return True
        return self.steps.now() >= self._deadline(st.cycle)

    def stuck(self, st, phase):
        if phase == "inner":
            if st.last_ranked_empty and set(CLASSES) <= set(st.searched_classes):
                return "nothing_passes_floor"
            return None
        if st.cycle >= 1 and st.top8_history and sorted(st.top8_history[-1]) == sorted(st.top8_at_cycle_start) \
                and not st.cycle_passed:
            return "no_improvement"
        return None

    def _night(self):
        lo, hi = self.cfg["search"]["night_only_utc"]
        now = self.steps.now()
        return now.date().isoformat() if lo <= now.hour < hi else None

    def _calls_left(self, st):
        return int(self.cfg["llm_calls_max"]) - int(st.llm_calls)

    # ------------------------------------------------------------------------------------------- steps
    def inner_round(self, st):
        """One inner round, resumable by sub-step (st.cur_round / st.cur_done); None = a search is due but the night
        window is closed or this night already had a finished search."""
        r = st.cur_round if st.cur_round is not None else (0 if st.round < 0 else max(st.round + 1, FIRST_SEARCH_ROUND))
        if r > 0 and "search" not in st.cur_done:
            night = self._night()
            if night is None or night in st.search_nights:
                return None
            st.cur_round, st.cur_done = r, []
            st.save(self.state_path)
            self._step = f"search r{r}"
            s = self.steps.search(st, r, self._calls_left(st))
            st.search_nights.append(night)
            st.llm_calls += int(s["llm_calls"])
            st.searched_classes = sorted(set(st.searched_classes) | set(s["classes"]))
            st.rounds_searched += 1
            st.cur_done.append("search")
            self.log(f"Plan 4 loop round {r}: search done — {len(s['new'])} new pool candidates, classes "
                     f"{sorted(set(s['classes']))}, LLM calls {st.llm_calls}")
            st.save(self.state_path)
        st.cur_round = r
        if "fit" not in st.cur_done:
            self._step = f"fit r{r}"
            self.steps.fit_new(r)
            st.cur_done.append("fit")
            st.save(self.state_path)
        left = self._calls_left(st)
        if "review" not in st.cur_done:
            if left > 0:
                self._step = f"review r{r}"
                rv = self.steps.review(r, min(int(self.cfg["review_quota"]), left), set(st.reviewed))
                st.llm_calls += int(rv["llm_calls"])
                st.reviewed = sorted(set(st.reviewed) | set(rv["reviewed"]))
                self.log(f"Plan 4 loop round {r}: review {len(rv['reviewed'])} candidates, {len(rv['new'])} cleaned "
                         f"versions accepted, LLM calls {st.llm_calls}")
            st.cur_done.append("review")
            st.save(self.state_path)
        self._step = f"rank r{r}"
        ranked = self.steps.rank(r)
        # rule of 11-06 (replaces the original "the first 8 of the ranking"): only the gate1 candidates (near-best in
        # >= 4 search cities, at most TOP_K, scoring.rank); fewer are not padded with candidates near-best nowhere
        top = sorted(x["cid"] for x in ranked if x.get("gate1"))[:TOP_K]
        st.top8_history.append(top)
        st.last_ranked_empty = not ranked
        st.round = r
        st.cur_round, st.cur_done = None, []
        st.rounds_since_outer += 1
        if r == 0 and st.cycle == 0:
            st.top8_at_cycle_start = top
        self.log(f"Plan 4 loop round {r} done: {len(ranked)} candidates above the floor, top 8 {top}")
        st.save(self.state_path)
        return st

    def outer_cycle(self, st):
        cyc = st.cycle + 1
        top = st.top8_history[-1] if st.top8_history else []
        o = self.steps.outer(cyc, top, self._calls_left(st))
        st.llm_calls += int(o["llm_calls"])
        st.cycle = cyc
        st.cycle_passed = list(o["passed"])
        st.registered = sorted(set(st.registered) | set(o.get("registered", [])))
        for city, seeds in (o.get("seeds") or {}).items():
            st.extra_seeds.setdefault(city, []).extend(seeds)
        self.log(f"Plan 4 loop outer cycle {cyc} done: top 8 {top}; registered {o.get('registered', [])}; passed "
                 f"{list(o['passed'])}; LLM calls {st.llm_calls}")
        st.save(self.state_path)
        return st

    def stuck_exit(self, st, reason):
        st.stuck_n += 1
        n = st.stuck_n
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        payload = {"reason": reason, "rounds": st.round, "rounds_searched": st.rounds_searched, "cycles": st.cycle,
                   "searched_classes": st.searched_classes, "top8_history": st.top8_history,
                   "registered": st.registered, "passed_last_cycle": st.cycle_passed,
                   "coverage": self.steps.coverage(st), "llm_calls": st.llm_calls}
        (self.agent_dir / f"stuck_{n}.in.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        res, calls = (None, 0)
        if self._calls_left(st) > 0:
            res, calls = self.steps.stuck_report(payload, n)
        st.llm_calls += int(calls)
        (self.agent_dir / f"stuck_{n}.json").write_text(json.dumps({"reason": reason, "result": res}, ensure_ascii=False,
                                                                   indent=1))
        st.phase = "stuck"
        st.save(self.state_path)
        self.log(f"STUCK {n} ({reason}) after round {st.round}, cycle {st.cycle}: diagnosis in "
                 f"agent/stuck_{n}.json; the loop stops for a manual decision")
        return 3

    # ------------------------------------------------------------------------------------------- main loop
    def _reload_cfg(self):
        if self.cfg_path:
            self.cfg = json.loads(Path(self.cfg_path).read_text())

    def _freeze(self, st, why):
        st.phase = "done"
        st.save(self.state_path)
        self.log(f"FREEZE request (plan 4 loop): {why}; rounds {st.round}, cycles {st.cycle}, LLM calls {st.llm_calls}; "
                 "the freeze is decided manually (confirmation cities and test period stay sealed)")
        return 0

    def run(self, until=None):
        """Main loop with an exception boundary (final review #3): a failing step writes a CRASH ledger line, keeps
        the resume markers in state.json and exits 4; `loop.py run` resumes from there."""
        st = self.state = State.load(self.state_path)
        try:
            return self._run(st, until)
        except Exception as e:  # noqa: BLE001 — every crash is ledgered; the state keeps the sub-step markers
            import traceback
            st.last_error = f"{self._step}: {type(e).__name__}: {str(e)[:500]}"
            st.save(self.state_path)
            self.log(f"CRASH in {self._step} (round {st.round}, in progress {st.cur_round} {st.cur_done}, cycle "
                     f"{st.cycle}): {type(e).__name__}: {str(e)[:300]}; resume with loop.py run")
            traceback.print_exc()
            return 4

    def _run(self, st, until):
        self._step = "init"
        self.steps.init()
        if st.phase == "stuck":
            self.log("Plan 4 loop: state is STUCK — waiting for a manual decision (edit state.json phase to resume)")
            return 3
        while True:
            self._reload_cfg()
            now = self.steps.now()
            if until is not None and now >= until:
                st.save(self.state_path)
                self.log(f"Plan 4 loop paused at {now.isoformat()} (--until); resume with loop.py run")
                return 0
            if self._calls_left(st) <= 0:
                st.save(self.state_path)
                self.log(f"BUDGET: LLM calls {st.llm_calls} >= {self.cfg['llm_calls_max']}; the loop stops")
                return 0
            if st.phase == "done":
                return 0
            if st.phase == "inner":
                if st.round < 0 or (not self.stable(st) and st.rounds_searched < int(self.cfg["inner_rounds_max"])):
                    if self.inner_round(st) is None:
                        if self.stable(st):
                            continue
                        self.steps.sleep(WAIT_S)
                        continue
                    reason = self.stuck(st, "inner")
                    if reason:
                        return self.stuck_exit(st, reason)
                    continue
                if st.cycle >= int(self.cfg["outer_cycles_max"]):
                    return self._freeze(st, "outer cycles used up")
                if not (st.top8_history and st.top8_history[-1]):
                    return self.stuck_exit(st, "no_candidate_for_outer")
                st.phase = "outer"
                st.save(self.state_path)
            if st.phase == "outer":
                self._step = f"outer c{st.cycle + 1}"
                self.outer_cycle(st)
                reason = self.stuck(st, "outer")
                if reason:
                    return self.stuck_exit(st, reason)
                st.phase = "inner"
                st.rounds_since_outer = 0
                st.top8_at_cycle_start = list(st.top8_history[-1]) if st.top8_history else []
                st.save(self.state_path)
                if st.cycle >= int(self.cfg["outer_cycles_max"]):
                    return self._freeze(st, f"{st.cycle} outer cycles done")
                if self.steps.now() >= _ts(self.cfg["schedule"]["freeze_request_after"]):
                    return self._freeze(st, "schedule: freeze_request_after passed")


# ================================================================================================ real steps
class Steps:
    """The heavy compute-server work of each loop step. Data artifacts live under results/multicity/loop/ (pool.json,
    inner_fits.parquet + inner_boot/<city>.npy, flags.json, ranked_r<k>.json, c<k>_discovery/, c<k>_lawtest/,
    c<k>_laws.json) and results/multicity/agent/ (LLM inputs / outputs; calls/ = the call log)."""

    def __init__(self, cfg, root=None, procs=None):
        import sr_data
        self.cfg = cfg
        self.root = Path(root) if root else config.RESULTS / "loop"
        self.procs = int(procs or os.environ.get("LOOP_PROCS", "48"))                # inner refits (shared data)
        self.outer_procs = int(os.environ.get("LOOP_OUTER_PROCS", "24"))  # outer fit_library: one dataset per worker
        self.srm = sr_data.ROOT / "results/sr/M"
        self._datas = None
        self._refs = None
        self.log = None                     # ledger writer; None = ledger.add (tests inject a list's append)

    # -- basics
    def now(self):
        return datetime.now(timezone.utc)

    def sleep(self, s):
        time.sleep(s)

    def cities(self, role):
        return config.cities_with_role(role)

    def init(self):
        import llm
        self.root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("LLM_LOG_DIR", str(config.RESULTS / "agent/calls"))
        p = self.root / "pool.json"
        if not p.exists():
            base = json.loads((config.RESULTS / "library/pool.json").read_text())
            for x in base:
                x["round"] = 0
                x["source"] = list(x.get("source", [])) + ["pool_r1_4"]
            p.write_text(json.dumps(base, ensure_ascii=False, indent=1))
        if not llm.ping():
            raise RuntimeError(f"LLM endpoint {llm.endpoint()[0]} not reachable")

    def pool(self):
        return json.loads((self.root / "pool.json").read_text())

    def save_pool(self, pool):
        (self.root / "pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1))

    def refs(self):
        if self._refs is None:
            out = {}
            for c in self.cities("search"):
                r = json.loads((config.RESULTS / f"refs/{c}.json").read_text())
                if "inner" not in r:
                    raise RuntimeError(f"refs/{c}.json has no inner references (run refs.py --inner)")
                out[c] = r["inner"]
            self._refs = out
        return self._refs

    def datas(self):
        if self._datas is None:
            import library
            self._datas = {c: library.city_data(c, None, mode="inner") for c in self.cities("search")}
        return self._datas

    # -- inner fits (incremental: new candidates only; 1-SE recomputed over all columns)
    def _fits(self):
        import pandas as pd
        import numpy as np
        p = self.root / "inner_fits.parquet"
        if not p.exists():
            return pd.DataFrame(), {}
        fits = pd.read_parquet(p)
        boot = {c: np.load(self.root / f"inner_boot/{c}.npy") for c in self.cities("search")}
        return fits, boot

    def fit_new(self, r):
        import numpy as np
        import pandas as pd
        import library
        pool = self.pool()
        fits, boot = self._fits()
        have = set(fits["cid"]) if len(fits) else set()
        new = [x for x in pool if x["cid"] not in have]
        if not new:
            return
        res = library.fit_cities(new, self.cities("search"), procs=self.procs, starts=3, data=self.datas())
        rows = []
        (self.root / "inner_boot").mkdir(exist_ok=True)
        for c in self.cities("search"):
            nr, nb = res[c]
            old = fits[fits["city"] == c].to_dict("records") if len(fits) else []
            allrows = old + nr
            mat = np.column_stack([boot[c], nb]) if c in boot and boot[c].size else nb
            rows += library._one_se(allrows, mat)
            np.save(self.root / f"inner_boot/{c}.npy", mat)
        pd.DataFrame(rows).to_parquet(self.root / "inner_fits.parquet", index=False)
        self._flags_for(new)

    def _flags_for(self, new):
        import checks
        import clean
        import review
        from library import candidate
        fits, _ = self._fits()
        xm = review.city_xmax(self.datas())
        fl = self.flags()
        for x in new:
            rows = fits[(fits["cid"] == x["cid"]) & fits["finite"].astype(bool)]
            consts = {r.city: json.loads(r.consts) for r in rows.itertuples()}
            c = candidate(x)
            f = checks.flags(c, consts, xm)
            deg = set()
            for city in consts:
                deg |= set(clean.degenerate(clean.canonical(c), self.datas()[city]["tr"]))
            f["degenerate"] = sorted(deg)
            fl[x["cid"]] = f
        (self.root / "flags.json").write_text(json.dumps(fl, ensure_ascii=False, indent=1))

    def flags(self):
        p = self.root / "flags.json"
        return json.loads(p.read_text()) if p.exists() else {}

    # -- search round
    def search(self, st, r, max_calls):
        import diversity
        import seed_agent
        import sr_data
        pool = self.pool()
        search = self.cities("search")
        front = {"top8": st.top8_history[-1]} if st.top8_history else {}
        cov = diversity.coverage(pool, front, set(st.searched_classes))
        rows = diversity.plan_round(diversity.gaps(cov), search, r, set(st.searched_classes), cov=cov)
        d = self.root / f"r{r}"
        d.mkdir(parents=True, exist_ok=True)
        s = self.cfg["search"]
        q = d / "queue.txt"
        diversity.write_queue(rows, q, maxp=s["maxp"], threads=s["threads"], timeout=s["timeout_s"])
        data = sr_data.build(search, "tract", "train", "search23", 15, 0)
        classes = sorted({x["cls"] for x in rows})
        calls = 0
        if max_calls >= len(classes):
            sd = diversity.seeds_for_round(rows, seed_agent.var_summary(data), st.extra_seeds, data)
            calls = sd["llm_calls"]
            (d / "seeds.json").write_text(json.dumps({k: v for k, v in sd.items() if k != "files"}, ensure_ascii=False,
                                                     indent=1, default=str))
        (d / "coverage.json").write_text(json.dumps({"coverage": cov, "rows": rows}, ensure_ascii=False, indent=1))
        self._run_queue(q, r)
        new = self._select_and_add(rows, r)
        return {"new": new, "llm_calls": calls, "classes": classes}

    def _run_queue(self, q, r):
        import sr_data
        s = self.cfg["search"]
        log = self.srm / f"logs/queue_loop_r{r}.log"
        if self._queue_finished(q, r):                                 # resume: the round's queue already finished
            return
        env = dict(os.environ, SR_BLOCK="search23", SR_THREADS=str(s["threads"]), SR_TIMEOUT_SINGLE=str(s["timeout_s"]),
                   ODSR_ROOT=str(sr_data.ROOT))
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            subprocess.run(["bash", str(Path(__file__).resolve().parent / "run_queue.sh"), str(q), str(s["maxp"])],
                           env=env, stdout=f, stderr=subprocess.STDOUT, check=True)

    def _queue_finished(self, q, r):
        """The round's queue log says QUEUE_DONE and every run of the queue left a final front (11-06: a queue whose
        runs all died at start also ends with QUEUE_DONE)."""
        log = self.srm / f"logs/queue_loop_r{r}.log"
        if not (log.exists() and "QUEUE_DONE" in log.read_text()):
            return False
        rows = [l.split() for l in Path(q).read_text().splitlines() if l.strip() and not l.startswith("#")]
        tags = [f"{x[0]}_{x[1]}_{x[2].replace(':', '-')}_s{x[3]}" for x in rows]
        return all((self.srm / f"runs/{t}/hall_of_fame.csv").exists() for t in tags)

    def _select_and_add(self, rows, r):
        import checks
        import orchestrate
        import review
        from refit_select import parse_equation
        from clean import render
        jobs = []
        for c in sorted({x["city"] for x in rows}):
            runs = [self.srm / f"runs/{x['tag']}" for x in rows if x["city"] == c]
            name = f"select_loop_r{r}_{c}"
            jobs.append((name, orchestrate.SCRIPTS / "sr/refit_select.py", ["tract", "A", *runs],
                         dict(SELECT_OUT=name, SELECT_REGIONS=c, SELECT_N_ORIG=300, SELECT_PROCS=10,
                              SR_TRAIN_BLOCK="search23", SR_EVAL_SPLIT="train", SR_EVAL_BLOCK="fold:4")))
        for name, *_ in jobs:
            (self.srm / f"{name}.json").unlink(missing_ok=True)       # no stale selection of an earlier attempt
        try:
            orchestrate._run_all(jobs)
        except RuntimeError as e:                                      # one city's failure keeps the others (11-06)
            ok = [j[0] for j in jobs if (self.srm / f"{j[0]}.json").exists()]
            if not ok:
                raise
            import ledger
            (self.log or ledger.add)(f"Plan 4 loop round {r}: selection partly failed ({str(e)[:200]}); kept {ok}")
            jobs = [j for j in jobs if j[0] in ok]
        pool = self.pool()
        keys = {_canon_key(x["equation"], x.get("template")) for x in pool}
        n = max(int(x["cid"][1:5]) for x in pool if x["cid"][1:5].isdigit())
        xm = review.city_xmax(self.datas())
        new = []
        for name, _, _, env in jobs:
            city = env["SELECT_REGIONS"]
            s = json.loads((self.srm / f"{name}.json").read_text())
            for x in s["candidates"]:
                c = parse_equation(x["equation"], x.get("template"), "A", int(x["complexity"]))
                try:
                    k = checks.canonicalize(c, xm[city])
                except Exception:  # noqa: BLE001 — an equation the canonicaliser cannot handle enters as it is
                    k = c
                key = _canon_key(render(k), None)
                if key in keys:
                    continue
                keys.add(key)
                n += 1
                new.append({"cid": f"c{n:04d}", "equation": render(k), "template": None, "variant": "A",
                            "complexity": int(x["complexity"]), "consts_refit": [float(v) for v in k.consts],
                            "source": [f"{name}.json"], "round": r})
        self.save_pool(pool + new)
        return new

    # -- review
    def review(self, r, quota, reviewed):
        import multiprocessing as mp
        import review
        pool, (fits, _), refs, fl = self.pool(), self._fits(), self.refs(), self.flags()
        sel = review.select_for_review(pool, fits, refs, fl, reviewed, quota)
        _A.update(datas=self.datas(), refs=refs, fits=fits)
        with mp.get_context("fork").Pool(min(self.procs, max(1, len(sel)))) as P:
            abl = dict(P.map(_ablate, sel)) if sel else {}
        out = review.review_round(pool, fits, refs, fl, abl, self.cities("search"), self.datas(), r,
                                  config.RESULTS / f"agent/r{r}", quota=quota, reviewed=reviewed, procs=self.procs)
        if out["new"]:
            self.save_pool(self.pool() + out["new"])
            self.fit_new(r)
        return out

    # -- rank (equivalence merge on fold-4 etas, lexicographic rank)
    def rank(self, r):
        import merge
        import scoring
        pool, (fits, boot), refs, fl = self.pool(), self._fits(), self.refs(), self.flags()
        cities = self.cities("search")
        etas = lambda cids: _inner_etas([x for x in pool if x["cid"] in set(cids)], fits, self.datas())  # noqa: E731
        reps = merge.merge(pool, fits, boot, etas, cities, refs)
        ranked = scoring.rank(reps, fits, refs, fl)
        (self.root / f"ranked_r{r}.json").write_text(json.dumps(ranked, ensure_ascii=False, indent=1, default=str))
        return ranked

    def coverage(self, st):
        import diversity
        return diversity.coverage(self.pool(), {"top8": st.top8_history[-1]} if st.top8_history else {},
                                  set(st.searched_classes))

    # -- outer cycle
    def outer(self, cyc, top8, max_calls):
        import pandas as pd
        import agent_points
        import fit_library
        import laws
        import llm
        pool = self.pool()
        by = {x["cid"]: x for x in pool}
        cands = [by[c] for c in top8 if c in by]
        disc = self.cities("search") + self.cities("discovery")
        dd = self.root / f"c{cyc}_discovery"
        fit_library.main(disc, procs=self.outer_procs, cands=cands, out_dir=dd)
        cv = pd.read_parquet(dd / "cities_val.parquet")
        vrefs = {c: json.loads((config.RESULTS / f"refs/{c}.json").read_text())["models"] for c in disc}
        small = _small_datas(disc)
        pt = _with_composites(laws.param_table(cv, pool), cv, pool, cands, laws.reference_points(small))
        cov = laws.covariates(disc)
        summ = laws.cross_city_summary(pt, cov)
        ex = laws.term_existence(_ablations_val(cands, cv, disc, vrefs, self.procs), vrefs)
        pt.to_parquet(dd / "param_table.parquet", index=False)
        ex.to_parquet(dd / "term_existence.parquet", index=False)
        (dd / "cross_city.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
        reg_path = config.RESULTS / "laws/registered.json"
        res_path = config.RESULTS / "laws/results.json"
        registered = json.loads(reg_path.read_text()) if reg_path.exists() else []
        prev = {r["id"]: r for r in json.loads(res_path.read_text())} if res_path.exists() else {}
        payload = laws.analysis_payload(summ, ex, pt, pool, registered=registered, results=prev)
        ad = config.RESULTS / f"agent/c{cyc}"
        ad.mkdir(parents=True, exist_ok=True)
        inp = ad / "hypotheses.in.json"
        inp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        calls, hyps = 0, []
        if max_calls > 0:
            o = llm.run_point("hypotheses", inp, ad / "hypotheses.json", agent_points.hypotheses)
            calls = 1
            hyps = (o.get("result") or {}).get("hypotheses", [])
        acc, rej = laws.register(hyps, cyc, pt, registered, terms={(e.cid, e.term) for e in ex.itertuples()},
                                 cap=int(self.cfg["hyps_per_cycle"]))
        passed, seeds, laws_out = [], {}, []
        allreg = registered + acc
        untested = [h for h in allreg if h["id"] not in prev]           # final review #4: also left over by a crash
        if untested:
            law_c = self.cities("lawtest")
            sk = sorted({h["skeleton"] for h in allreg})
            ld = self.root / f"c{cyc}_lawtest"
            fit_library.main(law_c, procs=self.outer_procs, cands=[by[c] for c in sk], out_dir=ld)
            cvl = pd.read_parquet(ld / "cities_val.parquet")
            lrefs = {c: json.loads((config.RESULTS / f"refs/{c}.json").read_text())["models"] for c in law_c}
            ptl = _with_composites(laws.param_table(cvl, pool), cvl, pool, [by[c] for c in sk],
                                   laws.reference_points(_small_datas(law_c)))
            exl = laws.term_existence(_ablations_val([by[c] for c in sk], cvl, law_c, lrefs, self.procs), lrefs) \
                if any(h["test"] == "term_exist_prop" for h in allreg) else None
            res = laws.test_registered(allreg, ptl, laws.covariates(law_c), exl)
            view = laws.agent_view(res)
            (ad / "lawtest_pass.json").write_text(json.dumps(view, indent=1))
            passed = [h["id"] for h in untested if view.get(h["id"])]
            big = {c: _city_val(c) for c in disc}
            for h in untested:
                if not view.get(h["id"]) or h["test"] == "term_exist_prop" or not h["parameter"].split(":")[-1].startswith("c"):
                    continue
                lx = laws.parameterize(by[h["skeleton"]], h["parameter"], h["covariate"])
                jr = laws.joint_refit(lx, big, cov, refs=vrefs, starts=2)
                s = laws.substituted_seeds(lx, jr, laws.covariates(self.cities("search")), self.cities("search"))
                for c, v in s.items():
                    seeds.setdefault(c, []).extend(v)
                laws_out.append({"id": h["id"], "joint": jr})
            (self.root / f"c{cyc}_laws.json").write_text(json.dumps(laws_out, ensure_ascii=False, indent=1, default=str))
        _VAL.clear()                                                    # final review #8: free the validation data
        return {"passed": passed, "registered": [h["id"] for h in acc], "llm_calls": calls, "seeds": seeds}

    def stuck_report(self, payload, n):
        import agent_points
        import llm
        d = config.RESULTS / "agent"
        o = llm.run_point("stuck_report", d / f"stuck_{n}.in.json", d / f"stuck_{n}.call.json", agent_points.stuck_report)
        return o.get("result"), 1


def _canon_key(eq, template):
    """Structure key that does not depend on the literal order of the string: the equation is canonicalised and
    re-rendered (sympy term order) before the constants become symbols (library.structure_key)."""
    import checks
    from clean import render
    from library import structure_key
    from refit_select import parse_equation
    c = parse_equation(eq, template, "A", 1)
    try:
        return structure_key(parse_equation(render(checks.canonicalize(c, None)), None, "A", 1))
    except Exception:  # noqa: BLE001 — a form the canonicaliser cannot handle keeps its raw key
        return structure_key(c)


_A = {}


def _ablate(x):
    import clean
    from library import candidate
    fits = _A["fits"]
    rows = fits[(fits["cid"] == x["cid"]) & fits["finite"].astype(bool)]
    if not len(rows):
        return x["cid"], []
    city = rows["city"].iloc[0]
    try:
        return x["cid"], clean.term_contrib(clean.canonical(candidate(x)), _A["datas"][city]["tr"], _A["refs"][city])
    except Exception:  # noqa: BLE001 — a candidate whose ablation fails is reviewed without it
        return x["cid"], []


def _abl_pair(job):
    import clean
    from library import candidate
    x, city, consts = job
    D = _A["vdatas"][city]
    try:
        c = dataclasses.replace(candidate({**x, "consts_refit": None}), consts=__import__("numpy").asarray(consts, float))
        return (x["cid"], city), clean.term_contrib(c, D["tr"], _A["vrefs"][city])
    except Exception:  # noqa: BLE001
        return (x["cid"], city), []


def _ablations_val(cands, cv, cities, vrefs, procs):
    """{(cid, city): term_contrib on the city's training rows from its validation-fit constants}."""
    import multiprocessing as mp
    jobs = []
    for x in cands:
        for r in cv[(cv["cid"] == x["cid"]) & cv["finite"].astype(bool)].itertuples():
            if r.city in cities:
                jobs.append((x, r.city, json.loads(r.consts)))
    _A.update(vdatas={c: _city_val(c) for c in cities}, vrefs=vrefs)
    if not jobs:
        return {}
    with mp.get_context("fork").Pool(min(procs, len(jobs))) as P:
        return dict(P.map(_abl_pair, jobs, chunksize=1))


_VAL = {}


def _city_val(city):
    import library
    if city not in _VAL:
        _VAL[city] = library.city_data(city, None)
    return _VAL[city]


def _small_datas(cities):
    import sr_data
    return {c: {"tr": sr_data.build([c], "tract", "train", "search", 50, 0)} for c in cities}


def _with_composites(pt, cv, pool, cands, ref):
    import pandas as pd
    import laws
    parts = [pt] + [laws.composites(x["cid"], cv, pool, ref) for x in cands]
    return pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pt


def _inner_etas(xs, fits, datas):
    import numpy as np
    from library import candidate
    from refit_select import eta
    out = {}
    for city, D in datas.items():
        es = {}
        for x in xs:
            r = fits[(fits["cid"] == x["cid"]) & (fits["city"] == city)]
            if not len(r) or not bool(r["finite"].iloc[0]):
                continue
            c = candidate({**x, "consts_refit": None})
            cs = np.asarray(json.loads(r["consts"].iloc[0]), float)
            if len(cs) != len(c.consts):
                continue
            e = eta(c, D["va"], consts=cs)
            if np.isfinite(e).all():
                es[x["cid"]] = e.astype(np.float32)
        out[city] = {"offsets": D["va"].offsets, "eta": es}
    return out


# ================================================================================================ CLI
def _dry_run(cfg):
    st = State.load(config.RESULTS / "loop/state.json")
    checks = {"pool (library or loop)": (config.RESULTS / "library/pool.json").exists()
              or (config.RESULTS / "loop/pool.json").exists(),
              "cities discovery / lawtest": bool(config.cities_with_role("discovery")) and bool(config.cities_with_role("lawtest")),
              "LLM key file": bool(os.environ.get("LLM_KEY_FILE")) and Path(os.path.expanduser(os.environ["LLM_KEY_FILE"])).exists()}
    for c in config.cities_with_role("search"):
        p = config.RESULTS / f"refs/{c}.json"
        r = json.loads(p.read_text()) if p.exists() else {}
        checks[f"refs {c} inner+S0"] = "inner" in r and "S0" in r.get("models", {})
    print(json.dumps({"state": dataclasses.asdict(st), "config": cfg, "checks": checks}, ensure_ascii=False, indent=1))
    return 0 if all(checks.values()) else 1


def main(argv):
    cmd = argv[0] if argv else "status"
    cfg = json.loads(CONFIG.read_text())
    if cmd == "status":
        print(json.dumps(dataclasses.asdict(State.load(config.RESULTS / "loop/state.json")), ensure_ascii=False, indent=1))
        return 0
    if cmd != "run":
        sys.exit(f"unknown command {cmd}")
    if "--dry-run" in argv:
        return _dry_run(cfg)
    until = None
    if "--until" in argv:
        until = _ts(argv[argv.index("--until") + 1])
    lp = Loop(cfg, Steps(cfg), cfg_path=CONFIG)
    return lp.run(until=until)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
