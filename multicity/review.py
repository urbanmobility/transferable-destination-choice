"""Inner-loop step 2 — agent review and code-verified cleaning (plan 4 Task 10).

review_payload(): one candidate's city-level material for the agent: the canonical eta (literal values) and its
  parametric form, the variables it uses, the code flags (degeneracy + origin-only / unit mixing / disguised linear),
  the f/g/h term ablation, fold-4 scores per search city (training period; never the validation period) and the range
  of each constant over the 8 search-city refits (identifiability). Passes llm.check_input.
select_for_review(): candidates above the floor in >= 1 city, not reviewed yet, in scoring.rank order, the current
  round's new candidates first; at most `quota`.
verify_cleaned(): the agent's cleaned eta refitted (multi-start) on every search city's inner data and scored on fold
  4; accepted when mean_city[(NLL_new - NLL_orig) / delta_city] <= 1 and no city becomes not fittable. A parse error,
  an illegal symbol or a non-finite eta is a rejection with a reason, never an exception (Review Focus 1).
review_round(): the calls (llm.run_point, files agent/r<round>/review_<cid>.{in,}.json; MAX_CONCURRENT threads),
  then the verification of every cleaned proposal (fork pool over proposal x city); accepted versions become pool
  entries {cid: <parent>c<k>, source [cleaned_by_agent], parent, round, complexity, insight}; a parent with 2
  cleanings already gets no more.

The prompts were issued in Chinese in the original runs; the text here is a translation.
"""
import dataclasses
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import llm  # noqa: E402
import scoring  # noqa: E402
import sr_data  # noqa: E402
from clean import _pysr, canonical, eta_string  # noqa: E402
from library import candidate, refit_city  # noqa: E402
from llm_points import VAR_MEANING, XVAR  # noqa: E402
from refit_select import XS, parse_equation  # noqa: E402

MAX_CLEANINGS, STARTS = 2, 3
SETUP = ("Data: tract-level trip OD for 8 US search cities (mobile-device locations, training period March–April 2024). "
         "Model: conditional logit of destination choice, P(j|i) = exp(η_ij) / Σ_k exp(η_ik). Score fold4_nll: mean "
         "negative log-likelihood per visit on the held-out destination blocks (fold 4) in the training period (lower "
         "is better; constants fitted on folds 2 and 3). share_s0 = (NLL_S0 − NLL) / (NLL_S0 − NLL_M2), S0 = modified "
         "gravity log(N_j + c) − β·log(d + c0), M2 = flexible GLM with 22 parameters; share_grav uses classic power-law "
         "gravity as the zero point. gap_to_city_best = difference from the best candidate in that city. const_ranges: "
         "range of each constant after refitting the same structure separately in each of the 8 cities "
         "(identifiability).")


def _used(c):
    syms = set().union(*[e.free_symbols for e in c.comps.values()])
    return [f"x{k + 1}" for k in range(len(XS)) if XS[k] in syms]


def _parametric(c):
    f = str(c.comps["f"])
    if "g" in c.comps:
        n = c.n_lit
        f += f" + c{n}*({c.comps['g']}) + c{n + 1}*({c.comps['h']})"
    return f


def const_ranges(x: dict, fits) -> dict:
    rows = fits[(fits["cid"] == x["cid"]) & fits["finite"].astype(bool)]
    cs = [json.loads(v) for v in rows["consts"] if isinstance(v, str) and v not in ("null", "None")]
    cs = [v for v in cs if v is not None]
    if not cs:
        return {}
    n = min(len(v) for v in cs)
    a = np.array([v[:n] for v in cs], float)
    return {f"c{i}": [float(a[:, i].min()), float(a[:, i].max())] for i in range(n)}


def city_xmax(datas: dict) -> dict:
    """{city: {variable: max over the city's training rows}} (for checks.disguised_linear)."""
    return {c: {v: float(np.nanmax(D["tr"].X[:, k])) for k, v in enumerate(sr_data.VARS)} for c, D in datas.items()}


def review_payload(x: dict, inner_fits, flags: dict, ablation: list, const_ranges_: dict, refs: dict | None = None) -> dict:
    c = candidate(x)
    rows = inner_fits[inner_fits["cid"] == x["cid"]]
    ok = inner_fits[inner_fits["finite"].astype(bool) & inner_fits["val_nll"].notna()]
    best = ok.groupby("city")["val_nll"].min()
    f4 = {}
    for r in rows.itertuples():
        v = float(r.val_nll) if bool(r.finite) and r.val_nll == r.val_nll else None
        rc = (refs or {}).get(r.city)
        f4[r.city] = {"nll": None if v is None else round(v, 5),
                      "gap_to_city_best": None if v is None else round(v - float(best[r.city]), 5),
                      "share_s0": None if v is None or rc is None else _r(scoring.share(v, rc, "S0")),
                      "share_grav": None if v is None or rc is None else _r(scoring.share(v, rc, "grav_power"))}
    p = {"cid": x["cid"], "setup": SETUP, "complexity": int(x["complexity"]),
         "eta": eta_string(canonical(c)), "eta_parametric": _parametric(c),
         "variables": {v: f"{XVAR[v]}: {VAR_MEANING[XVAR[v]]}" for v in _used(c)},
         "code_flags": {"degenerate": list((flags or {}).get("degenerate", [])),
                        **{k: list((flags or {}).get(k, [])) for k in ("origin_only", "unit_mixing", "disguised_linear")}},
         "term_ablation": [{"comp": t.get("comp", "f"), "term": t["term"], "delta_nll": _r(t["delta_nll"]),
                            "share_of_gap": _r(t.get("share_of_gap"))} for t in (ablation or [])],
         "fold4_nll": f4, "const_ranges": const_ranges_, "source": list(x.get("source", []))}
    llm.check_input(p)
    return p


def _r(v, nd=4):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def select_for_review(pool, inner_fits, refs, flags, reviewed: set, quota: int = 60) -> list:
    todo = [x for x in pool if x["cid"] not in reviewed]
    if not todo:
        return []
    ranked = scoring.rank([dict(x, merged_from=[]) for x in todo], inner_fits, refs, flags, top=len(todo))
    cur = max(int(x.get("round", 0)) for x in pool)
    order = {r["cid"]: k for k, r in enumerate(ranked)}
    by = {x["cid"]: x for x in todo}
    sel = sorted((by[c] for c in order), key=lambda x: (int(x.get("round", 0)) != cur, order[x["cid"]]))
    return sel[:quota]


def _parse(s):
    s = (s or "").strip()
    if not s:
        return None, "empty proposal"
    try:
        e = sympy.sympify(s.replace("^", "**"), locals={f"x{k + 1}": XS[k] for k in range(len(XS))})
    except Exception as ex:  # noqa: BLE001 — anything the agent wrote that sympy cannot read is a rejection
        return None, f"parse error: {type(ex).__name__}"
    if not isinstance(e, sympy.Basic):
        return None, "parse error: not an expression"
    bad = sorted(str(z) for z in e.free_symbols if z not in set(XS))
    if bad:
        return None, f"illegal symbols {bad}"
    if not (e.free_symbols & set(XS)):
        return None, "no variable"
    return e, None


def _nodes(e):
    return len(list(sympy.preorder_traversal(e)))


_V = {}       # shared with forked verification workers: {"datas": {...}}


def _refit_new(job):
    key, eq, city = job
    D = _V["datas"][city]
    try:
        r = refit_city(parse_equation(eq, None, "A", 1), city, starts=_V["starts"], tr=D["tr"], va=D["va"])
    except Exception as e:  # noqa: BLE001 — a cleaned version that breaks the refit is not fittable there
        return key, city, None, None, repr(e)[:200]
    return key, city, (r["val_nll"] if r["finite"] else None), r["consts"], None


def _decide(orig_nll, new, cities, refs):
    per, norm, nonfit = {}, [], []
    for c in cities:
        o, n = orig_nll.get(c), new.get(c)
        per[c] = {"nll_orig": o, "nll_new": n, "finite": n is not None}
        if o is not None and n is None:
            nonfit.append(c)
        if o is not None and n is not None:
            norm.append((n - o) / max(scoring.delta_city(refs[c]), 1e-12))
    mean = float(np.mean(norm)) if norm else None
    acc = bool(norm) and not nonfit and mean <= 1.0
    reason = ("ok" if acc else f"not fittable in {nonfit}" if nonfit else "no city to compare" if not norm
              else f"mean loss increase {mean:.2f} x delta_city > 1")
    return {"accepted": acc, "per_city": per, "mean_norm_delta": mean, "new_nonfittable": nonfit, "reason": reason}


def _orig_nll(orig, cities, datas, starts, inner_fits=None):
    rows = inner_fits[inner_fits["cid"] == orig["cid"]] if inner_fits is not None else None
    if rows is not None and len(rows):
        return {r.city: (float(r.val_nll) if bool(r.finite) and r.val_nll == r.val_nll else None)
                for r in rows.itertuples() if r.city in cities}
    out = {}
    for c in cities:
        r = refit_city(candidate(orig), c, starts=starts, tr=datas[c]["tr"], va=datas[c]["va"])
        out[c] = r["val_nll"] if r["finite"] else None
    return out


def _new_entry_fields(orig, e, consts):
    eq = f"f = {_pysr(e)}; p1 = []; p2 = []"
    c0 = candidate(orig)
    base = sum(_nodes(v) for v in c0.comps.values())
    cx = max(1, int(round(int(orig["complexity"]) * _nodes(e) / max(base, 1))))      # on the PySR complexity scale
    return eq, cx


def verify_cleaned(orig: dict, sympy_str: str, cities: list, datas: dict, refs: dict, starts: int = STARTS,
                   inner_fits=None) -> dict:
    e, why = _parse(sympy_str)
    if e is None:
        return {"accepted": False, "per_city": {}, "mean_norm_delta": None, "new_nonfittable": [], "reason": why}
    eq, cx = _new_entry_fields(orig, e, None)
    _V.update(datas=datas, starts=starts)
    res = [_refit_new(("k", eq, c)) for c in cities]
    new = {c: v for _, c, v, _, _ in res}
    out = _decide(_orig_nll(orig, cities, datas, starts, inner_fits), new, cities, refs)
    consts = next((cs for _, c, v, cs, _ in res if cs is not None), None)
    out.update(equation=eq, complexity=cx, consts=consts, sympy=str(e))
    return out


def _call_one(args):
    x, payload, out_dir, agent = args
    inp, outp = out_dir / f"review_{x['cid']}.in.json", out_dir / f"review_{x['cid']}.json"
    inp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return x["cid"], llm.run_point("review", inp, outp, agent)


def review_round(pool, inner_fits, refs, flags, ablations, cities, datas, round_no, out_dir, agent=None,
                 quota: int = 60, reviewed: set = frozenset(), max_calls: int | None = None, procs: int = 8,
                 starts: int = STARTS) -> dict:
    """{"new": accepted pool entries, "reviewed": cids, "llm_calls": n, "verified": {cid: verdict}}."""
    import agent_points
    agent = agent or agent_points.review
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = quota if max_calls is None else max(0, min(quota, max_calls))
    sel = select_for_review(pool, inner_fits, refs, flags, set(reviewed), q)
    jobs = [(x, review_payload(x, inner_fits, flags.get(x["cid"], {}), ablations.get(x["cid"], []),
                               const_ranges(x, inner_fits), refs), out_dir, agent) for x in sel]
    with ThreadPoolExecutor(llm.MAX_CONCURRENT) as ex:
        outs = dict(ex.map(_call_one, jobs))
    n_kids = {}
    for x in pool:
        if x.get("parent"):
            n_kids[x["parent"]] = n_kids.get(x["parent"], 0) + 1
    props, verdicts = {}, {}
    for x in sel:
        res = outs[x["cid"]].get("result") or {}
        s = (res.get("cleaned_sympy") or "").strip()
        if not s:
            verdicts[x["cid"]] = {"skipped": "no cleaned version" if outs[x["cid"]].get("result") else "call failed"}
            continue
        if n_kids.get(x["cid"], 0) >= MAX_CLEANINGS:
            verdicts[x["cid"]] = {"skipped": f"parent has {MAX_CLEANINGS} cleanings"}
            continue
        e, why = _parse(s)
        if e is None:
            verdicts[x["cid"]] = {"accepted": False, "reason": why}
            continue
        n_kids[x["cid"]] = n_kids.get(x["cid"], 0) + 1
        props[x["cid"]] = (x, e, n_kids[x["cid"]], res)
    _V.update(datas=datas, starts=starts)
    vjobs = [(cid, _new_entry_fields(x, e, None)[0], c) for cid, (x, e, _, _) in props.items() for c in cities]
    if procs == 1 or len(vjobs) <= 1:
        vres = list(map(_refit_new, vjobs))
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(procs, len(vjobs))) as P:
            vres = list(P.imap_unordered(_refit_new, vjobs, chunksize=1))
    new = []
    for cid, (x, e, k, res) in props.items():
        got = {c: v for key, c, v, _, _ in vres if key == cid}
        v = _decide(_orig_nll(x, cities, datas, starts, inner_fits), got, cities, refs)
        eq, cx = _new_entry_fields(x, e, None)
        consts = next((cs for key, c, val, cs, _ in sorted(vres, key=lambda t: cities.index(t[1]))
                       if key == cid and cs is not None), None)
        v.update(equation=eq, complexity=cx, sympy=str(e))
        verdicts[cid] = v
        if v["accepted"]:
            new.append({"cid": f"{cid}c{k}", "equation": eq, "template": None, "variant": "A", "complexity": cx,
                        "consts_refit": consts, "source": ["cleaned_by_agent"], "parent": cid, "round": round_no,
                        "insight": res.get("insight", ""), "changes": res.get("changes", "")})
    for x in sel:                                    # the verdict goes next to the call record
        p = out_dir / f"review_{x['cid']}.json"
        rec = json.loads(p.read_text())
        rec["verify"] = verdicts.get(x["cid"])
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
    return {"new": new, "reviewed": [x["cid"] for x in sel], "llm_calls": len(sel), "verified": verdicts}
