"""Rule-based cleaning of candidate skeletons (plan 2 Task 4; design §5.4.2, round-1 lessons §9).

degenerate(c, data): rules fired on the city's training rows — tiny_exponent (|power| < 0.01), huge_const (|literal|
  > 1e4), tiny_const (0 < |literal| < 1e-4), zero_denominator (a negative-power base below 1e-6 in magnitude or
  changing sign on the rows), flat_term (an additive term of f / g / h whose values on the rows have variance
  < 1e-3). Any rule marks the candidate degenerate; the orchestrator records the names.
canonical(c): literal values substituted, additive constants inside the same expression merged (round 1: -c1 + c2
  inside a log), top-level numeric terms of f / g / h removed (an additive constant of the utility is not
  identified in a conditional logit), then reparsed so the literal count and consts agree.
term_contrib(c, data, refs): each top-level additive term of f, g and h removed in turn (the rest refitted; plan 4
  Task 4 added g, h and the "comp" field; a component whose only term is removed becomes 0); delta_nll on
  the training rows against the full refit; share_of_gap = delta / (NLL_grav - NLL_M2); share < 1% marks the term
  removable (a flag for the GATE 1 table, never an automatic deletion).
verify_simplification(c, s, tr, va): an LLM proposal s (sympy string in x1..x26) is accepted only if it parses,
  uses no other symbol, its eta correlates > 0.999 with the candidate's on the training rows, and after a refit its
  validation loss is at most 0.5 device-group-bootstrap SE above the candidate's; otherwise rejected with a reason
  and the candidate stays as it was.
"""
import dataclasses
import re
import sys
from pathlib import Path

import numpy as np
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
from library import _boot_weights, to_single  # noqa: E402
from refit_select import PENALTY, XS, _device_group_y, boot_nll, eta, loss, parse_equation, refit  # noqa: E402

RULES = ("tiny_exponent", "huge_const", "tiny_const", "zero_denominator", "flat_term")
TINY_EXP, HUGE_CONST, TINY_CONST, DEN_EPS, FLAT_VAR = 0.01, 1e4, 1e-4, 1e-6, 1e-3
CORR_MIN, SE_FACTOR, REMOVABLE_SHARE = 0.999, 0.5, 0.01
XSET = set(XS)


def _c(i):
    return sympy.Symbol(f"c{i}")


def _lits(c, exact=False):
    return {_c(i): (sympy.Float(float(v)) if exact else float(v)) for i, v in enumerate(c.consts[:c.n_lit])}


def _rows(expr, c, data):
    """Values of a sympy expression (in x*, c*) on the rows of data."""
    cs = [_c(i) for i in range(max(c.n_lit, 1))]
    f = sympy.lambdify(list(XS) + cs, expr, "numpy")
    cols = [data.X[:, k].astype(np.float64) for k in range(len(XS))]
    lit = list(c.consts[:c.n_lit]) if c.n_lit else [0.0]
    with np.errstate(all="ignore"):
        return np.broadcast_to(np.asarray(f(*cols, *lit), float), (len(data.y),))


def degenerate(c, data):
    flags = set()
    lits = _lits(c)
    for e in c.comps.values():
        for p in e.atoms(sympy.Pow):
            ex = p.exp.subs(lits)
            if ex.is_number:
                if abs(float(ex)) < TINY_EXP:
                    flags.add("tiny_exponent")
                if float(ex) < 0:
                    den = _rows(p.base, c, data)
                    fin = den[np.isfinite(den)]
                    if len(fin) and (np.abs(fin).min() < DEN_EPS or (fin.min() < 0 < fin.max())):
                        flags.add("zero_denominator")
        for t in sympy.Add.make_args(e):
            if t.is_number:                    # an additive constant: canonical() removes it
                continue
            v = _rows(t, c, data)
            if np.isfinite(v).all() and len(v) > 1 and float(np.var(v)) < FLAT_VAR:
                flags.add("flat_term")
    for v in c.consts[:c.n_lit]:
        if abs(v) > HUGE_CONST:
            flags.add("huge_const")
        if 0 < abs(v) < TINY_CONST:
            flags.add("tiny_const")
    return sorted(flags)


def _pysr(expr):
    """sympy expression (x1..x26, numbers) -> PySR-style right-hand side (#k variables, ^ powers)."""
    return re.sub(r"\bx(\d+)\b", r"#\1", str(expr)).replace("**", "^")


def render(c, consts=None):
    """PySR-style equation string of a candidate with its literal values substituted (inverse of parse_equation)."""
    consts = np.asarray(c.consts if consts is None else consts, float)
    lits = {_c(i): sympy.Float(float(v)) for i, v in enumerate(consts[:c.n_lit])}
    parts = [f"{name} = {_pysr(c.comps[name].subs(lits))}" for name in ("f", "g", "h") if name in c.comps]
    if "g" in c.comps:
        p1 = consts[c.n_lit:c.n_lit + c.n_cat]
        p2 = consts[c.n_lit + c.n_cat:c.n_lit + 2 * c.n_cat]
        parts += [f"p1 = [{', '.join(repr(float(v)) for v in p1)}]", f"p2 = [{', '.join(repr(float(v)) for v in p2)}]"]
    return "; ".join(parts)


def canonical(c):
    lits = _lits(c, exact=True)
    comps = {}
    for name, e in c.comps.items():
        terms = [t for t in sympy.Add.make_args(sympy.expand_mul(e.subs(lits), deep=False)) if not t.is_number]
        comps[name] = sympy.Add(*terms) if terms else sympy.Integer(0)
    tmp = dataclasses.replace(c, comps=comps, n_lit=0, consts=np.asarray(c.consts[c.n_lit:], float))
    return parse_equation(render(tmp), c.template, c.variant, c.complexity)


def term_contrib(c, data, refs):
    c = to_single(c)
    gap = float(refs["grav_power"]["nll_per_visit"] - refs["M2_glm"]["nll_per_visit"])
    full = refit(c, data)
    base = loss(full, data)
    lits = _lits(full, exact=True)
    out = []
    for name in ("f", "g", "h"):
        if name not in full.comps:
            continue
        terms = sympy.Add.make_args(full.comps[name])
        for t in terms:
            if t.is_number:          # an additive constant (also p * const in g, h) is not identified: nothing to ablate
                continue
            rest = sympy.Add(*[u for u in terms if u is not t])           # empty -> 0 (the component vanishes)
            cc = dataclasses.replace(full, comps={**full.comps, name: rest})
            r = refit(cc, data)
            nll = loss(r, data)
            delta = float(nll - base) if nll < PENALTY else float("inf")
            share = delta / gap if gap > 0 else float("nan")
            out.append({"comp": name, "term": str(t.subs(lits)), "key": str(t), "delta_nll": delta,
                        "share_of_gap": float(share),
                        "removable": bool(share < REMOVABLE_SHARE)})
    return out


def verify_simplification(c, s, tr, va=None, B=200):
    c = to_single(c)
    out = {"accepted": False, "corr": None, "dnll": None, "se": None, "reason": "", "sympy": None, "candidate": None}
    try:
        expr = sympy.sympify(s, locals={f"x{k + 1}": XS[k] for k in range(len(XS))})
    except Exception as e:  # noqa: BLE001 — anything the LLM wrote that sympy cannot read is a rejection, not a crash
        out["reason"] = f"parse error: {type(e).__name__}"
        return out
    bad = sorted(str(z) for z in getattr(expr, "free_symbols", set()) if z not in XSET)
    if bad or not isinstance(expr, sympy.Basic):
        out["reason"] = f"illegal symbols {bad or [str(expr)]}"
        return out
    cs = parse_equation(f"f = {_pysr(expr)}; p1 = []; p2 = []", None, c.variant, c.complexity)
    e0, e1 = eta(c, tr), eta(cs, tr)
    if not (np.isfinite(e0).all() and np.isfinite(e1).all()):
        out["reason"] = "eta not finite on the training rows"
        return out
    if np.allclose(e0, e0[0]) or np.allclose(e1, e1[0]):
        corr = 1.0 if np.allclose(e0 - e0.mean(), e1 - e1.mean()) else 0.0
    else:
        corr = float(np.corrcoef(e0, e1)[0, 1])
    out["corr"] = corr
    if not corr > CORR_MIN:
        out["reason"] = f"structure changed (corr {corr:.4f})"
        return out
    r0, r1 = refit(c, tr), refit(cs, tr)
    out["sympy"], out["candidate"] = str(expr), r1
    if va is None:
        out["accepted"], out["reason"] = True, "corr only (no validation data)"
        return out
    v0, v1 = loss(r0, va), loss(r1, va)
    if v1 >= PENALTY or v0 >= PENALTY:
        out["reason"] = "validation loss not finite"
        return out
    Yg = _device_group_y(va, None, "tract", "val")
    W = _boot_weights(B)
    d = boot_nll(eta(r1, va), va, Yg, W) - boot_nll(eta(r0, va), va, Yg, W)
    d = d[np.isfinite(d)]
    se = float(np.std(d)) if len(d) else 0.0
    out.update(dnll=float(v1 - v0), se=se)
    out["accepted"] = bool(v1 - v0 <= SE_FACTOR * se + 1e-12)
    out["reason"] = "ok" if out["accepted"] else f"validation loss {v1 - v0:.5f} above {SE_FACTOR} SE ({se:.5f})"
    return out


def eta_string(c):
    """sympy string of eta of the single-category form with literal values substituted (the LLM's input)."""
    c = to_single(c)
    lits = _lits(c, exact=True)
    e = c.comps["f"].subs(lits)
    if "g" in c.comps:
        p1, p2 = float(c.consts[c.n_lit]), float(c.consts[c.n_lit + 1])
        e = e + sympy.Float(p1) * c.comps["g"].subs(lits) + sympy.Float(p2) * c.comps["h"].subs(lits)
    return str(e)


_P = {}     # shared with forked workers of clean_pool


def _clean_one(x):
    from library import candidate
    pool_fits, data, refs, cities = _P["fits"], _P["data"], _P["refs"], _P["cities"]
    cc = canonical(candidate(x))
    flags = sorted(set().union(*[set(degenerate(cc, data[c])) for c in cities]))
    fin = set(pool_fits[(pool_fits["cid"] == x["cid"]) & pool_fits["finite"].astype(bool)]["city"])
    fin = [c for c in cities if c in fin]
    tc = term_contrib(cc, data[fin[0]], refs[fin[0]]) if fin else []
    print(x["cid"], flags, [t["term"] for t in tc if t["removable"]], flush=True)
    return x["cid"], {"canonical": render(cc), "canonical_eta": eta_string(cc), "degenerate": flags,
                      "term_contrib": tc, "term_city": fin[0] if fin else None}


def clean_pool(pool, fits, data, refs, cities, procs=1):
    """{cid: canonical form, eta string, degeneracy flags (union over cities), term contributions (first city where
    the candidate is fittable)}; candidates in a fork pool, the data shared copy-on-write."""
    _P.update(fits=fits, data=data, refs=refs, cities=list(cities))
    if procs == 1 or len(pool) == 1:
        res = list(map(_clean_one, pool))
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(procs, len(pool))) as P:
            res = list(P.imap_unordered(_clean_one, pool, chunksize=1))
    _P.clear()
    order = {x["cid"]: k for k, x in enumerate(pool)}
    return dict(sorted(res, key=lambda t: order[t[0]]))


def main(n_orig=300, procs=None):
    """Orchestrator stage `clean`: clean_pool over the whole pool on the search cities (n_orig sampled origins each)
    -> results/multicity/library/clean.json."""
    import json
    import os
    import pandas as pd
    import sr_data
    from config import CITIES_CSV, RESULTS
    from library import lib_dir
    import ledger
    procs = int(os.environ.get("CLEAN_PROCS", "48")) if procs is None else procs
    lib = lib_dir()
    pool = json.loads((lib / "pool.json").read_text())
    fits = pd.read_parquet(lib / "search_city_fits.parquet")
    cities_tbl = pd.read_csv(CITIES_CSV, dtype=str)
    cities = list(cities_tbl[cities_tbl["role"] == "search"]["city"])
    data = {c: sr_data.build([c], "tract", "train", "search", n_orig, 0) for c in cities}
    refs = {c: json.loads((RESULTS / f"refs/{c}.json").read_text())["models"] for c in cities}
    out = clean_pool(pool, fits, data, refs, cities, procs)
    (lib / "clean.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    ledger.add(f"Task 4: clean.json sha {ledger.sha(lib / 'clean.json')}: {sum(bool(v['degenerate']) for v in out.values())} "
               f"of {len(out)} candidates degenerate, {sum(any(t['removable'] for t in v['term_contrib']) for v in out.values())} "
               f"with a removable term")
    return out


if __name__ == "__main__":
    main()
