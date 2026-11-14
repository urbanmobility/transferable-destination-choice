"""Deterministic orchestrator of plan 2 (Task 8): a fixed list of stages, each with declared
inputs and outputs; before a stage the inputs must exist and their shas go to the ledger, after it the outputs'
shas go to the ledger; a stage whose outputs are unchanged since its last "done" line is skipped; a failure writes
FAILED and stops the chain. Stages that need a human gate run only when the ledger holds "GATE n: 批准" (the approval
marker, "批准" = approved; kept verbatim because the ledger is matched against it).

Stages (in order): search_wait -> select_pooled -> select_single -> library -> clean -> llm_simplify ->
llm_readability -> merge -> fit_library -> gate1 (writes the GATE 1 entry and stops).
CLI: python -I orchestrate.py run [STAGE ...] [--dry-run]      (no stage = every stage up to gate1)
     python -I orchestrate.py gate 1 [--approve]
Scripts run as subprocesses (Julia stays out of this process); the LLM points run in-process.
"""
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import config  # noqa: E402
import ledger  # noqa: E402
import sr_data  # noqa: E402

PY = os.environ.get("ODSR_PY", sys.executable)
SCRIPTS = Path(__file__).resolve().parents[1]
ROOT, RES = sr_data.ROOT, config.RESULTS
LIB, SRM = RES / "library", ROOT / "results/sr/M"
SELECT_N_ORIG, LLM_SAMPLE_ORIG = 300, 300


class StageError(RuntimeError):
    pass


@dataclasses.dataclass
class Stage:
    name: str
    inputs: list
    outputs: list
    run: Callable
    needs_gate: Optional[int] = None


# ----------------------------------------------------------------------------------------------- ledger helpers
def _add(text, ledger_file=None):
    if ledger_file is None:
        ledger.add(text)
    else:
        with open(ledger_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M %Z')} {text}\n")


def _read(ledger_file=None):
    p = Path(ledger_file) if ledger_file else ledger.LEDGER
    return p.read_text() if p.exists() else ""


def _sha(p):
    p = Path(p)
    if p.is_dir():
        import hashlib
        h = hashlib.sha256()
        for f in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(f"{f.relative_to(p)}:{f.stat().st_size}:{ledger.sha(f)}\n".encode())
        return h.hexdigest()[:12]
    return ledger.sha(p)


def _shas(paths):
    return {Path(p).name: _sha(p) for p in paths if Path(p).exists()}


def _done_shas(name, ledger_file):
    lines = [l for l in _read(ledger_file).splitlines() if f"stage {name} done" in l and "outputs sha " in l]
    if not lines:
        return None
    m = re.search(r"outputs sha (\{.*\})\s*$", lines[-1])
    return json.loads(m.group(1)) if m else None


def gate_approved(n, ledger_file=None):
    return any(re.search(rf"GATE {n}: 批准", l) for l in _read(ledger_file).splitlines())


def approve(n, ledger_file=None, note=""):
    _add(f"GATE {n}: 批准 {note}".rstrip(), ledger_file)


# ----------------------------------------------------------------------------------------------- the runner
def run(names, stages=None, dry_run=False, ledger_file=None):
    stages = STAGES if stages is None else stages
    by = {s.name: s for s in stages}
    unknown = [n for n in names if n not in by]
    if unknown:
        _add(f"orchestrate: unknown stage {unknown}", ledger_file)
        raise StageError(f"unknown stage {unknown}")
    for n in names:
        s = by[n]
        missing = [str(p) for p in s.inputs if not Path(p).exists()]
        if dry_run:
            gate = f" [needs GATE {s.needs_gate}]" if s.needs_gate is not None else ""
            print(f"[dry-run] stage {n}{gate}: inputs {[str(p) for p in s.inputs]} -> outputs {[str(p) for p in s.outputs]}"
                  + (f" (missing now: {len(missing)})" if missing else ""))
            continue
        if s.needs_gate is not None and not gate_approved(s.needs_gate, ledger_file):
            _add(f"orchestrate: stage {n} refused — GATE {s.needs_gate} not approved", ledger_file)
            raise StageError(f"stage {n} needs GATE {s.needs_gate} approval")
        if missing:
            _add(f"stage {n} missing input {missing}", ledger_file)
            raise StageError(f"stage {n}: missing input {missing}")
        prev, cur = _done_shas(n, ledger_file), _shas(s.outputs)
        if prev is not None and s.outputs and all(Path(p).exists() for p in s.outputs) and cur == prev:
            _add(f"stage {n} skipped (outputs unchanged, sha {json.dumps(cur)})", ledger_file)
            continue
        _add(f"stage {n} start inputs sha {json.dumps(_shas(s.inputs))}", ledger_file)
        t0 = time.time()
        try:
            s.run()
        except Exception as e:  # noqa: BLE001 — every failure is ledgered and stops the chain
            _add(f"stage {n} FAILED after {time.time() - t0:.0f} s: {type(e).__name__}: {str(e)[:300]}", ledger_file)
            raise StageError(f"stage {n} failed: {type(e).__name__}: {e}") from e
        _add(f"stage {n} done ({time.time() - t0:.0f} s) outputs sha {json.dumps(_shas(s.outputs))}", ledger_file)


# ----------------------------------------------------------------------------------------------- real stages
def _cities():
    return pd.read_csv(config.CITIES_CSV, dtype=str)


def search_cities():
    c = _cities()
    return list(c[c["role"] == "search"]["city"]) if "role" in c.columns else []


def _env(**kw):
    e = dict(os.environ)
    e.update(ODSR_ROOT=str(ROOT), ODSR_LINE="M", ODSR_CITIES=",".join(_cities()["city"]), SR_DROP_VARS="s_health")
    e.update({k: str(v) for k, v in kw.items()})
    return e


def _run(script, *args, **env):
    subprocess.run([PY, "-u", str(script), *map(str, args)], check=True, env=_env(**env), cwd=str(ROOT))


def _run_all(jobs):
    """Run several (script, args, env) concurrently; each logs to results/sr/M/logs/<name>.log; any failure fails."""
    procs = []
    (SRM / "logs").mkdir(parents=True, exist_ok=True)
    for name, script, args, env in jobs:
        f = open(SRM / f"logs/{name}.log", "w")
        procs.append((name, f, subprocess.Popen([PY, "-u", str(script), *map(str, args)], env=_env(**env), cwd=str(ROOT),
                                                stdout=f, stderr=subprocess.STDOUT)))
    bad = []
    for name, f, p in procs:
        if p.wait() != 0:
            bad.append(name)
        f.close()
    if bad:
        raise RuntimeError(f"failed: {bad} (logs results/sr/M/logs/<name>.log)")


QUEUES = [SCRIPTS / f"multicity/config/{q}"
          for q in ("queue_search.txt", "queue_search_r2.txt", "queue_search_r3.txt", "queue_search_r4.txt")]
WAIT_LOGS = [(SRM / "logs/queue_r3.log", SRM / "logs/snap_hof_r3.log"),     # rounds still running when relaunched
             (SRM / "logs/queue_r4.log", SRM / "logs/snap_hof_r4.log")]
SEEDS_SELECT = SRM / "select_seeds_agent.json"          # call point 0: LLM seed skeletons that passed the code verdict
EXTRA_SELECT = [SEEDS_SELECT, SRM / "select_seeds_agent_r4.json",           # round-4 class seeds (seed_agent families)
                SRM / "select_challengers_diag.json"]                       # the 11-04 diagnostic's three structures


def _queue_rows(q):
    return [l.split() for l in Path(q).read_text().splitlines() if l.strip() and not l.startswith("#")]


def _tag(r):
    return f"{r[0]}_{r[1]}_{r[2].replace(':', '-')}_s{r[3]}"


def pooled_runs(variant):
    """Pooled runs of every queue (rounds 1-2 were stopped early; their pooled fronts and snapshots are kept)."""
    return [SRM / f"runs/{_tag(r)}" for q in QUEUES for r in _queue_rows(q) if r[2].startswith("city") and r[1] == variant]


def single_runs(city):
    """Single-city runs of rounds 2-4 (round 1's never started; round 2's were stopped after ~1 h with fronts; round 4 =
    one structural class per run)."""
    return [SRM / f"runs/{_tag(r)}" for q in QUEUES[1:] for r in _queue_rows(q) if r[2] == f"single:{city}"]


def _search_done():
    return all(q.exists() and "QUEUE_DONE" in q.read_text() and s.exists() and "SNAP_DONE" in s.read_text()
               for q, s in WAIT_LOGS)


def _search_wait():
    while not _search_done():
        time.sleep(600)


def _sel(name, variant, runs, regions, procs):
    return (name, SCRIPTS / "sr/refit_select.py", ["tract", variant, *runs],
            dict(SELECT_OUT=name, SELECT_REGIONS=regions, SELECT_N_ORIG=SELECT_N_ORIG, SELECT_SNAPSHOTS=1, SELECT_PROCS=procs))


def _select_pooled():
    sc = ",".join(search_cities())
    _run_all([_sel("select_tract_A", "A", pooled_runs("A"), sc, 24), _sel("select_tract_B", "B", pooled_runs("B"), sc, 24)])


def _select_single():
    _run_all([_sel(f"select_tract_A_single{c}", "A", single_runs(c), c, 10) for c in search_cities()])


def _library():
    _run(SCRIPTS / "multicity/library.py", SRM / "select_tract_A.json",
         *[SRM / f"select_tract_A_single{c}.json" for c in search_cities()],
         *[p for p in EXTRA_SELECT if p.exists()], LIB_PROCS=48)


def _clean():
    _run(SCRIPTS / "multicity/clean.py", CLEAN_PROCS=48)


def _llm_stage(point):
    """Proposals for the whole pool in 2 threads (resumable), then — for simplify — code-side verification of every
    pending proposal in a fork pool. A failed call is recorded per candidate, never fatal."""
    import llm
    import llm_stage
    if not llm.ping():
        raise RuntimeError(f"LLM endpoint {llm.endpoint()[0]} not reachable (LLM service paused? SSH tunnel down?)")
    pool = json.loads((LIB / "pool.json").read_text())
    fits = pd.read_parquet(LIB / "search_city_fits.parquet")
    cj = json.loads((LIB / "clean.json").read_text()) if (LIB / "clean.json").exists() else {}
    llm_stage.propose(point, pool, cj, fits, LIB / "llm", threads=2)
    if point == "simplify":
        llm_stage.verify_pending(pool, fits, search_cities(), LIB / "llm", procs=32)


def _merge():
    _run(SCRIPTS / "multicity/merge.py")


def _fit_library():
    _run(SCRIPTS / "multicity/fit_library.py", FIT_PROCS=24)


def _gate1():
    pkg, md, summ, unc = LIB / "candidates_gate1.json", LIB / "gate1.md", RES / "cities_val_summary.csv", RES / "uncovered.txt"
    uncovered = unc.read_text().split() if unc.exists() else []
    g = json.loads(pkg.read_text())
    ledger.add(f"GATE 1: candidate package candidates_gate1.json sha {ledger.sha(pkg)} ({g['n_gate1']} candidates / {g['n_candidates']} after merging), "
               f"gate1.md sha {ledger.sha(md)}, cities_val_summary.csv sha {ledger.sha(summ)}, "
               f"uncovered {len(uncovered)} cities {uncovered}; waiting for approval (continues only once the ledger holds 'GATE 1: 批准')")
    print("GATE 1 package written; waiting for approval")


STAGES = [
    Stage("search_wait", QUEUES, [p for pair in WAIT_LOGS for p in pair], _search_wait),
    Stage("select_pooled", [r / "config.json" for v in ("A", "B") for r in pooled_runs(v)],
          [SRM / "select_tract_A.json", SRM / "select_tract_B.json"], _select_pooled),
    Stage("select_single", [r / "config.json" for c in search_cities() for r in single_runs(c)],
          [SRM / f"select_tract_A_single{c}.json" for c in search_cities()], _select_single),
    Stage("library", [SRM / "select_tract_A.json"] + [SRM / f"select_tract_A_single{c}.json" for c in search_cities()]
          + [p for p in EXTRA_SELECT if p.exists()],
          [LIB / "pool.json", LIB / "search_city_fits.parquet"] + [LIB / f"boot_{c}.npy" for c in search_cities()], _library),
    Stage("clean", [LIB / "pool.json", LIB / "search_city_fits.parquet"], [LIB / "clean.json"], _clean),
    Stage("llm_simplify", [LIB / "pool.json", LIB / "clean.json"], [LIB / "llm"], lambda: _llm_stage("simplify")),
    Stage("llm_readability", [LIB / "pool.json", LIB / "clean.json"], [LIB / "llm"], lambda: _llm_stage("readability")),
    Stage("merge", [LIB / "pool.json", LIB / "search_city_fits.parquet", LIB / "clean.json"],
          [LIB / "candidates_gate1.json", LIB / "gate1.md"], _merge),
    Stage("fit_library", [LIB / "candidates_gate1.json", LIB / "pool.json"],
          [RES / "cities_val.parquet", RES / "cities_val_summary.csv", RES / "uncovered.txt"], _fit_library),
    Stage("gate1", [LIB / "candidates_gate1.json", LIB / "gate1.md", RES / "cities_val_summary.csv"], [], _gate1),
]


def cli_parse(argv):
    flags = {"dry_run": "--dry-run" in argv, "approve": "--approve" in argv}
    rest = [a for a in argv if not a.startswith("--")]
    return (rest[0] if rest else "run"), rest[1:], flags


if __name__ == "__main__":
    cmd, args, flags = cli_parse(sys.argv[1:])
    if cmd == "run":
        run(args or [s.name for s in STAGES], dry_run=flags["dry_run"])
    elif cmd == "gate":
        n = int(args[0])
        if flags["approve"]:
            approve(n, note="(approved via orchestrate.py gate --approve)")
        print(f"GATE {n} approved: {gate_approved(n)}")
    else:
        sys.exit(f"unknown command {cmd}")
