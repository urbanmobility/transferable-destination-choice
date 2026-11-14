"""Inner-loop step 0 — structural diversity (plan 4 Task 9).

classify(): structural class of a pool equation (PySR #k form), the rule of seed_agent.backbone on the parsed
  components: power (log distance), kernel_other (distance only non-logarithmically), opportunity (S_ij, no
  distance), gravity_opportunity (both), no_distance.
coverage(): per class the pool count, the count in the given fronts (cids or equations), whether it was ever searched,
  and the largest within-class eta correlation on the given rows (None without rows).
gaps(): never-searched classes first, then classes with fewer than min_pool pool candidates (CLASSES order).
plan_round(): 3 single-city runs per search city; classes = never-searched classes (code-enforced) + gaps, wrapped to
  3; no gap at all -> the three non-power searched classes; seed = 10 x round + k (k = position, same class on every
  city, as in round 4).
write_queue(): the run_queue.sh queue file (round-4 format, header records the environment).
seeds_for_round(): one diversity_seeds call (seed_agent.propose with lens_for_class) per class of the round, screened
  on the given rows, diversity-filtered within the class; each run's config/guesses/<tag>.json holds only its class's
  seeds plus the city's extra (substituted) seeds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import sr_data  # noqa: E402
from refit_select import XS, parse_equation  # noqa: E402

CLASSES = ("power", "kernel_other", "opportunity", "gravity_opportunity", "no_distance")
RUNS_PER_CITY, NITER, N_ORIG = 3, 40, 100
MAXSIZE = 30            # = pysr_setup.make_model maxsize: a seed above it never reaches the front (11-06)
MIN_SEEDS = 3           # fewer seeds of a class after the size filter -> one more call with the node limit stated
_D, _LOGD, _S = XS[sr_data.VARS.index("d")], XS[sr_data.VARS.index("logd")], XS[sr_data.VARS.index("S_ij")]


def _comps(eq):
    return list(parse_equation(eq, None, "A", 1).comps.values())


def classify(eq: str) -> str:
    exprs = _comps(eq)
    syms = set().union(*[e.free_symbols for e in exprs]) if exprs else set()
    has_s, has_d = _S in syms, bool(syms & {_D, _LOGD})
    if has_s:
        return "gravity_opportunity" if has_d else "opportunity"
    if not has_d:
        return "no_distance"
    logdist = _LOGD in syms or any(_D in a.free_symbols for e in exprs for a in e.atoms(sympy.log))
    return "power" if logdist else "kernel_other"


def _rho_max(members, data):
    from library import candidate
    from refit_select import eta
    es = []
    for x in members:
        try:
            e = eta(candidate(x), data)
        except Exception:  # noqa: BLE001 — a member that cannot be evaluated is left out of the statistic
            continue
        if np.isfinite(e).all() and np.std(e) > 0:
            es.append(e)
    if len(es) < 2:
        return None
    c = np.abs(np.corrcoef(np.vstack(es)))
    np.fill_diagonal(c, 0.0)
    return float(c.max())


def coverage(pool: list, fronts: dict, searched: set, data=None) -> dict:
    by_cid = {x["cid"]: x for x in pool}
    cls = {x["cid"]: classify(x["equation"]) for x in pool}
    front = []
    for items in fronts.values():
        for it in items:
            front.append(cls[it] if it in cls else classify(by_cid[it]["equation"] if it in by_cid else it))
    out = {}
    for c in CLASSES:
        members = [x for x in pool if cls[x["cid"]] == c]
        out[c] = {"n_pool": len(members), "n_front": sum(f == c for f in front), "searched": c in searched,
                  "rho_max": _rho_max(members, data) if data is not None else None}
    return out


def gaps(cov: dict, min_pool: int = 8) -> list:
    never = [c for c in CLASSES if c in cov and not cov[c].get("searched")]
    sparse = [c for c in CLASSES if c in cov and cov[c].get("searched") and cov[c].get("n_pool", 0) < min_pool]
    return never + sparse


def plan_round(gaps_: list, search_cities: list, round_no: int, searched: set, cov: dict | None = None) -> list:
    """Never-searched classes first (code-enforced), then the gaps; fewer than 3 -> filled with the other non-power
    classes, most front members first, then fewest pool members (11-06: round 5 had one gap and repeated it in all
    24 runs); only when every non-power class is in use are classes repeated."""
    classes = [c for c in CLASSES if c not in searched]
    classes += [c for c in gaps_ if c not in classes]
    if len(classes) < RUNS_PER_CITY:
        rest = [c for c in CLASSES if c != "power" and c not in classes]
        key = (lambda c: (-cov.get(c, {}).get("n_front", 0), cov.get(c, {}).get("n_pool", 0), CLASSES.index(c))) \
            if cov else CLASSES.index
        classes += sorted(rest, key=key)
    classes = (classes * RUNS_PER_CITY)[:RUNS_PER_CITY]
    return [{"tag": f"tract_A_single-{city}_s{10 * round_no + k}", "city": city, "cls": cls,
             "seed_no": 10 * round_no + k} for city in search_cities for k, cls in enumerate(classes)]


def write_queue(rows: list, path, block: str = "search23", maxp: int = 24, threads: int = 5, timeout: int = 10800):
    cls = {r["seed_no"]: r["cls"] for r in rows}
    head = [f"# Multi-city search queue, agent loop (plan 4 Task 9): single-city runs, one",
            f"# structural class per run (seed -> class: {json.dumps(cls, ensure_ascii=False)}); each run's",
            "# config/guesses/<tag>.json holds only its class's seeds (diversity_seeds call + substituted law seeds).",
            f"# Environment: SR_BLOCK={block} (destination folds 2, 3; fold 4 ranks), MAXP {maxp}, SR_THREADS={threads},",
            f"# SR_TIMEOUT_SINGLE={timeout} s, {NITER} iterations, {N_ORIG} origins; search space, operators and loss as round 4."]
    body = [f"tract A single:{r['city']} {r['seed_no']} {NITER} {N_ORIG}" for r in rows]
    Path(path).write_text("\n".join(head + body) + "\n")
    return None


def _seed_string(s):
    if isinstance(s, str):
        from seed_agent import pysr_safe
        return pysr_safe(s) or s
    from seed_agent import _single
    return _single({**s, "p": [float(s.get("p1", 1.0)), float(s.get("p2", 1.0))]})


def _fit_seeds(props, data, classes, per_class):
    """Screen, keep the seeds whose single-run form fits PySR's maxsize, diversity-filter within each class."""
    import seed_agent
    screened = seed_agent.screen(props, data)
    small = [r for r in screened if r["backbone"] in classes and r["status"] == "ok"
             and (seed_agent.pysr_complexity(seed_agent._single(r)) or MAXSIZE + 1) <= MAXSIZE]
    return seed_agent.diverse_by_class(small, per_class=per_class), screened


def seeds_for_round(rows: list, summary: dict, extra_seeds: dict, data, gdir=None, per_class: int = 8) -> dict:
    """{"files": {tag: path}, "llm_calls": n, "by_class": {cls: [seed ids]}, "screen": [rows without eta]}."""
    import seed_agent
    from agent_points import lens_for_class
    gdir = Path(gdir) if gdir else Path(__file__).resolve().parent / "config/guesses"
    gdir.mkdir(parents=True, exist_ok=True)
    classes = list(dict.fromkeys(r["cls"] for r in rows))
    props, calls = [], 0
    for c in classes:
        props += seed_agent.propose(summary, lenses=[lens_for_class(c)])
        calls += 1
    by, screened = _fit_seeds(props, data, classes, per_class)
    for c in classes:
        if len(by.get(c, [])) < MIN_SEEDS:                           # too few small seeds: ask once more, stricter
            lens = lens_for_class(c)
            lens["focus"] += (f" Most seeds of the previous batch exceeded the single-city search complexity limit of "
                              f"{MAXSIZE} nodes and could not be used by the search; this time each whole η "
                              "must not exceed 20 nodes.")
            props += seed_agent.propose(summary, lenses=[lens])
            calls += 1
    by, screened = _fit_seeds(props, data, classes, per_class)
    files = {}
    for r in rows:
        pick = by.get(r["cls"], [])
        spec = {"note": f"agent loop {r['tag']} class {r['cls']}", "family": r["cls"],
                "single": [seed_agent._single(s) for s in pick] + [_seed_string(s) for s in extra_seeds.get(r["city"], [])],
                "seed_ids": [s["id"] for s in pick]}
        p = gdir / f"{r['tag']}.json"
        p.write_text(json.dumps(spec, ensure_ascii=False, indent=1))
        files[r["tag"]] = p
    return {"files": files, "llm_calls": calls, "by_class": {c: [s["id"] for s in v] for c, v in by.items()},
            "screen": [{k: v for k, v in s.items() if k != "eta"} for s in screened]}
