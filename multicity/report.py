"""Version-1 result report of the agent loop (plan 4 Task 15).

build(out_md): from results/multicity/loop (state, pool, inner fits, flags, last ranking, outer-cycle tables, laws
write-back), results/multicity/laws (registered hypotheses, lawtest verdicts) and results/multicity/agent (review
insights, stuck diagnoses): candidate table (source, fold-4 score, 8-city share_S0 / share_grav, flags), discovery
parameter table, cross-city regressions, registered hypotheses with pass / fail only, parameterised skeletons, agent
review excerpts, stuck reports. No confirmation-city or lawtest-city id and no lawtest statistic is printed; agent
text is redacted and the whole document is scanned before it is written.
Usage: ODSR_ROOT=... python -I report.py [OUT_MD]
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import config  # noqa: E402
import scoring  # noqa: E402


def _lawtest_ids():
    return set(config.cities_with_role("lawtest"))


def _confirm_ids():
    return set(config.CONFIRM) | set(config.cities_with_role("confirm"))


def _pat(ids):
    return re.compile("(?<![0-9A-Za-z])(?<![0-9]\\.)(" + "|".join(re.escape(i) for i in sorted(ids)) + ")(?![0-9A-Za-z])(?!\\.[0-9])") if ids else None


def _scan(text, lawtest=None, allow_confirm=False):
    """allow_confirm (plan 6): confirmation-city ids pass, only after the freeze; lawtest ids as before."""
    conf = _confirm_ids()
    if allow_confirm:
        import frozen
        frozen.require()
        conf = set()
    for name, ids in (("confirmation", conf), ("lawtest", _lawtest_ids() if lawtest is None else lawtest)):
        p = _pat(ids)
        m = p.search(text) if p else None
        if m:
            raise ValueError(f"report mentions {name} city {m.group(1)}")


def _redact(text, ids):
    p = _pat(ids)
    return p.sub("[sealed]", text) if p else text


def _f(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _read(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def coverage_curve(cv, refs: dict, complexity: dict) -> list:
    """Greedy library coverage (the library size follows coverage, not a fixed cap; design decision of 2026-11-05): at each step add the
    skeleton covering most still-uncovered cities, covered = near-best in the city and above its floor; ties go to the
    lower complexity, then the cid; stops when no skeleton adds a city. Rows: step, cid, new, covered, n_cities,
    primary_in (cities where it is the primary skeleton)."""
    if not len(cv):
        return []
    ok = cv[cv["finite"].astype(bool) & cv["near_best"].astype(bool)]
    cover = {}
    for r in ok.itertuples():
        if scoring.passes_floor(r.val_nll, refs.get(r.city)):
            cover.setdefault(r.cid, set()).add(r.city)
    prim = cv[cv["primary"].astype(bool)].groupby("cid")["city"].nunique().to_dict() if "primary" in cv else {}
    left, done, out = set(cv["city"]), set(), []
    n = len(left)
    while True:
        cand = [(len(cs & left), c) for c, cs in cover.items() if c not in done]
        cand = [t for t in cand if t[0] > 0]
        if not cand:
            return out
        gain, c = min(cand, key=lambda t: (-t[0], complexity.get(t[1], 10 ** 6), t[1]))
        left -= cover[c]
        done.add(c)
        out.append({"step": len(out) + 1, "cid": c, "new": gain, "covered": n - len(left), "n_cities": n,
                    "primary_in": int(prim.get(c, 0))})


def build(out_md, results=None) -> dict:
    R = Path(results) if results else config.RESULTS
    L = R / "loop"
    sealed = _confirm_ids() | _lawtest_ids()
    st = _read(L / "state.json", {})
    pool = {x["cid"]: x for x in _read(L / "pool.json", [])}
    fits = pd.read_parquet(L / "inner_fits.parquet") if (L / "inner_fits.parquet").exists() else pd.DataFrame()
    flags = _read(L / "flags.json", {})
    ranks = sorted(L.glob("ranked_r*.json"), key=lambda p: int(p.stem.split("_r")[1]))
    ranked = _read(ranks[-1], []) if ranks else []
    cities = sorted(set(fits["city"])) if len(fits) else []
    refs = {c: (_read(R / f"refs/{c}.json", {}) or {}).get("inner", {}) for c in cities}
    out = [f"# Agent loop: version-1 results ({time.strftime('%Y-%m-%d')})", "",
           f"Status: last inner-loop round {st.get('round')}, {st.get('cycle')} outer cycles, {st.get('llm_calls')} LLM calls, "
           f"phase {st.get('phase')}. Scores: search cities, training-period fold 4 (constants fitted on folds 2 and 3); share_S0 "
           "has the modified gravity model as zero and M2 as the upper end, share_grav has the classic power-law gravity model "
           "as zero. Confirmation cities and the test period are excluded; law-test cities are reported as pass / fail only.", ""]
    # 1. candidates
    show = [r["cid"] for r in ranked[:20]]
    out += ["## 1. Candidate skeletons (top 20 of the last ranking, including merged cleaned versions)", "",
            "| rank | cid | top 8 | source | complexity | n flags | flags | near-best cities | fold-4 mean gap (milli-nats) | share_S0 median | share_grav median |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    best = fits[fits["finite"].astype(bool)].groupby("city")["val_nll"].min() if len(fits) else {}
    for k, r in enumerate(ranked[:20], 1):
        x = pool.get(r["cid"], {})
        f = fits[(fits["cid"] == r["cid"]) & fits["finite"].astype(bool)] if len(fits) else pd.DataFrame()
        gap = [1000 * (v - best[c]) for c, v in zip(f.get("city", []), f.get("val_nll", []))]
        s0 = [scoring.share(v, refs.get(c), "S0") for c, v in zip(f.get("city", []), f.get("val_nll", []))]
        sg = [scoring.share(v, refs.get(c), "grav_power") for c, v in zip(f.get("city", []), f.get("val_nll", []))]
        fl = flags.get(r["cid"], {})
        names = [n for n in ("origin_only", "unit_mixing", "disguised_linear") if fl.get(n)] + list(fl.get("degenerate", []))
        src = ", ".join(x.get("source", [])) + (f" (cleaned from {x['parent']})" if x.get("parent") else "")
        med = lambda v: float(np.median([a for a in v if a is not None])) if any(a is not None for a in v) else None  # noqa: E731
        out.append(f"| {k} | {r['cid']} | {'✔' if r.get('gate1') else ''} | {src} r{x.get('round', 0)} | "
                   f"{r.get('complexity', x.get('complexity'))} | {r.get('n_flags', 0)} | {', '.join(names) or '—'} | "
                   f"{r.get('n_near', 0)} | {_f(float(np.mean(gap)) if gap else None, 1)} | {_f(med(s0))} | {_f(med(sg))} |")
    out += [""] + [f"- {c}: `{pool[c]['equation']}`" + (f" (merged from {', '.join(r.get('merged_from', []))})"
                                                       if r.get("merged_from") else "")
                   for r in ranked[:20] for c in [r["cid"]] if c in pool]
    # 2. outer cycles
    for d in sorted(L.glob("c*_discovery")):
        k = d.name.split("_")[0]
        out += ["", f"## 2. Outer cycle {k}: discovery cities (including the validation period of the 8 search cities)", ""]
        summ = pd.read_csv(d / "cities_val_summary.csv", dtype={"city": str}) if (d / "cities_val_summary.csv").exists() else None
        if summ is not None and len(summ):
            out += [f"- Primary skeletons: {summ['primary'].value_counts().to_dict()}; cities above the floor {int(summ['covered'].sum())} / "
                    f"{len(summ)}; share_S0 median {_f(float(summ['share_s0'].median()))}, share_grav median "
                    f"{_f(float(summ['share_grav'].median()))}"]
        if (d / "cities_val.parquet").exists():
            cvd = pd.read_parquet(d / "cities_val.parquet")
            if "city" in cvd:
                cvd = cvd[~cvd["city"].isin(sealed)]
            vr = {c: (_read(R / f"refs/{c}.json", {}) or {}).get("models", {}) for c in set(cvd.get("city", []))}
            curve = coverage_curve(cvd, vr, {c: x.get("complexity", 0) for c, x in pool.items()})
            n = curve[0]["n_cities"] if curve else 0
            out += ["", f"Coverage curve (greedy: add the skeleton that covers most uncovered cities; covered = near-best and above "
                    f"the floor; {n} cities; the library size is set from this at the freeze gate, 6 is for reference only):", "",
                    "| step | cid | new cities | covered (cumulative) | cities as primary skeleton |", "|---|---|---|---|---|"]
            out += [f"| {s_['step']} | {s_['cid']} | {s_['new']} | {s_['covered']} / {s_['n_cities']} | {s_['primary_in']} |"
                    for s_ in curve]
            if curve and curve[-1]["covered"] < n:
                out.append(f"| — | not covered | | {n - curve[-1]['covered']} cities | |")
        if (d / "param_table.parquet").exists():
            pt = pd.read_parquet(d / "param_table.parquet")
            out += ["", "| cid | parameter | identifiable cities / cities | median | range |", "|---|---|---|---|---|"]
            for (cid, param), g in pt.groupby(["cid", "param"]):
                gi = g[g["identifiable"]]
                out.append(f"| {cid} | {param} | {len(gi)} / {len(g)} | {_f(float(g['value'].median()))} | "
                           f"{_f(float(g['value'].min()))} – {_f(float(g['value'].max()))} |")
        cc = _read(d / "cross_city.json", [])
        sig = [s for s in cc if s["ci_lo"] > 0 or s["ci_hi"] < 0]
        out += ["", f"Cross-city univariate regressions: {len(cc)} (parameter x covariate), {len(sig)} with a 95% CI excluding 0:", ""]
        out += [f"- {s['cid']} {s['param']} ~ {s['covariate']}: slope {_f(s['slope'])} ({_f(s['ci_lo'])}, {_f(s['ci_hi'])}), n = {s['n']}"
                for s in sig[:30]]
    # 3. hypotheses
    reg = _read(R / "laws/registered.json", [])
    res = {r["id"]: r for r in _read(R / "laws/results.json", [])}
    out += ["", "## 3. Registered law hypotheses and law-test results (Holm correction over all hypotheses registered so far)", "",
            "| id | statement | skeleton | parameter | city covariate | direction | test | result |", "|---|---|---|---|---|---|---|---|"]
    for h in reg:
        r = res.get(h["id"])
        verdict = "not tested" if r is None else ("pass" if r.get("pass") else "fail")
        out.append(f"| {h['id']} | {_redact(h['statement'], sealed)} | {h['skeleton']} | {h['parameter']} | "
                   f"{h['covariate']} | {h['direction']} | {h['test']} | {verdict} |")
    # 4. parameterised skeletons
    laws_out = [x for p in sorted(L.glob("c*_laws.json")) for x in _read(p, [])]
    out += ["", "## 4. Parameterised skeletons (joint cross-city refit on the discovery cities)", ""]
    for x in laws_out:
        j = x["joint"]
        sh = [v["share_s0"] for v in j["per_city"].values() if v.get("share_s0") is not None]
        out.append(f"- {x['id']}: {j['param']} of {j['cid']} = b0 + b1·z({j['covariate']}), b0 {_f(j['b0'])}, b1 {_f(j['b1'])}, "
                   f"{j['n_params']} parameters, discovery-city share_S0 median {_f(float(np.median(sh)) if sh else None)}")
    if not laws_out:
        out.append("- None (no law passed its test)")
    # 5. agent review excerpts
    out += ["", "## 5. Agent review excerpts (top 20 candidates)", ""]
    for c in show:
        for p in sorted((R / "agent").glob(f"r*/review_{c}.json")):
            rec = _read(p, {})
            r = rec.get("result") or {}
            v = rec.get("verify") or {}
            if r.get("insight"):
                out.append(f"- {c} ({p.parent.name}, {len(r.get('problems', []))} problems, cleaned version "
                           f"{'accepted' if v.get('accepted') else 'not accepted' if v else 'none'}): {_redact(r['insight'], sealed)}")
    cleaned = [x for x in pool.values() if "cleaned_by_agent" in x.get("source", [])]
    out += ["", f"Cleaned versions accepted by code verification ({len(cleaned)}):", ""]
    out += [f"- {x['cid']} (cleaned from {x.get('parent')}, round {x.get('round', 0)}): `{x['equation']}`"
            + (f"; agent note: {_redact(x['insight'], sealed)}" if x.get("insight") else "") for x in cleaned]
    # 6. stuck
    stuck = sorted((R / "agent").glob("stuck_*.json"))
    stuck = [p for p in stuck if not p.name.endswith((".in.json", ".call.json"))]
    out += ["", "## 6. Stuck reports", ""]
    for p in stuck:
        s = _read(p, {})
        r = s.get("result") or {}
        out.append(f"- {p.stem} ({s.get('reason')}): " + _redact("; ".join(f"{k}: {r.get(k, '')}" for k in
                                                                     ("tried", "evidence", "why_framework", "proposal")), sealed))
    if not stuck:
        out.append("- None")
    text = "\n".join(out) + "\n"
    _scan(text)
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(text)
    return {"n_candidates": len(pool), "n_ranked": len(ranked), "n_laws": len(reg),
            "n_pass": sum(bool(r.get("pass")) for r in res.values()), "path": str(out_md)}


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else config.RESULTS / f"report/报告_agent在环_v1_{time.strftime('%Y-%m-%d')}.md"
    print(json.dumps(build(out), ensure_ascii=False))
