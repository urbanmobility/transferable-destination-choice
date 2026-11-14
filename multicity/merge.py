"""Equivalence merging, ranking and the GATE 1 package (plan 2 Task 6).

equivalent(i, j, ...): two candidates are one skeleton when, in every search city, both are fittable, their
  validation losses differ by at most the device-group bootstrap SD of the paired difference, and their etas on the
  city's validation rows have a row-weighted within-origin Spearman rank correlation > 0.99 (a constant eta in any
  city breaks the tie: not equivalent).
merge(): clusters of the pool; the representative of a cluster is its lowest-complexity member (ties: lowest cid),
  merged_from lists the others. linkage (default scoring.MERGE_LINKAGE): "leader" (design decision of 2026-11-10, V3: no
  chain — candidates in (complexity, cid) order, each joins the first representative it is directly equivalent to,
  else becomes one) or "chain" (union-find, plans 2-5). Loss tolerance at scoring.MERGE_MARGIN (0.03); LossTable =
  loss_close with O(1) lookups (identical results, tested).
rank(), near_best(): scoring.rank / scoring.near_best (plan 4 Task 6): near-best =
  in the city's 1-SE set or at most max(1 SE, delta_city) above the city's best, delta_city = m x (NLL_grav - NLL_M2),
  m = scoring.GATE_MARGIN (0.05, V3) for the gate (plans 2-5: 0.03, design decision of 2026-11-05); candidates below the
  floor are dropped; order: near-best in >= min_cities cities, cleanliness flags, complexity, n_near. n_1se is still
  reported. With refs, loss_close uses max(SD, delta_city at MERGE_MARGIN).
gate1_package(): candidates_gate1.json (per candidate: equation, canonical / simplified forms, readability, degeneracy
  and removable-term flags, per-city validation NLL, gain share, near-best and 1-SE membership, B-variant gap) and
  gate1.md.
load_llm(): the simplify / readability results by cid from the LLM stage directory (call inputs *.in.json and failed
  calls, result null, skipped).

Usage (orchestrator): ODSR_ROOT=... ODSR_LINE=M ODSR_CITIES=... python -I merge.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import sr_data  # noqa: E402
from library import candidate, lib_dir  # noqa: E402
from refit_select import PENALTY, eta  # noqa: E402
import scoring  # noqa: E402
from scoring import MARGIN, MIN_CITIES, TOP, delta_city, near_best, rank  # noqa: E402,F401

RHO_MIN = 0.99


def _row(fits, cid, city):
    r = fits[(fits["cid"] == cid) & (fits["city"] == city)]
    return r.iloc[0] if len(r) else None


def _rank_corr(ea, eb, offsets):
    """Row-weighted mean of the per-origin Spearman correlations; None when no origin group is usable."""
    num = den = 0.0
    for a, b in zip(offsets[:-1], offsets[1:]):
        x, y = ea[a:b], eb[a:b]
        if b - a < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        r = spearmanr(x, y)[0]
        if np.isfinite(r):
            num, den = num + r * (b - a), den + (b - a)
    return num / den if den > 0 else None


def loss_close(i, j, fits, boot, cities, refs=None, margin=None):
    """Both fittable in every city and losses within one bootstrap SD of the paired difference — or, with refs
    ({city: refs_city}), within max(SD, delta_city at margin; None = scoring.MERGE_MARGIN) (plan 4)."""
    m = scoring.MERGE_MARGIN if margin is None else margin
    for city in cities:
        ri, rj = _row(fits, i, city), _row(fits, j, city)
        if ri is None or rj is None or not (bool(ri["finite"]) and bool(rj["finite"])):
            return False
        order = list(fits[fits["city"] == city]["cid"])          # bootstrap columns follow the pool order
        d = boot[city][:, order.index(i)] - boot[city][:, order.index(j)]
        d = d[np.isfinite(d)]
        sd = float(d.std()) if len(d) else 0.0
        tol = max(sd, delta_city(refs[city], m)) if refs is not None else sd
        if abs(float(ri["val_nll"]) - float(rj["val_nll"])) > tol:
            return False
    return True


def rank_close(i, j, etas, cities):
    """Row-weighted within-origin Spearman correlation of the two etas > RHO_MIN in every city."""
    for city in cities:
        e = etas[city]
        ea, eb = e["eta"].get(i), e["eta"].get(j)
        if ea is None or eb is None:
            return False
        rho = _rank_corr(np.asarray(ea, float), np.asarray(eb, float), np.asarray(e["offsets"]))
        if rho is None or not rho > RHO_MIN:
            return False
    return True


def equivalent(i, j, fits, boot, etas, cities, refs=None, margin=None):
    return loss_close(i, j, fits, boot, cities, refs, margin) and rank_close(i, j, etas, cities)


class LossTable:
    """loss_close(i, j, fits, boot, cities, refs) with the margin explicit and O(1) row lookups: in every city both rows
    exist and are finite and |val_i - val_j| <= max(SD of the paired bootstrap difference, margin x max(NLL_grav -
    NLL_M2, 0)) (refs None: the SD alone). First row / first column of a cid, as _row and list.index."""

    def __init__(self, fits, boot, cities, refs=None):
        self.cities, self.boot, self.refs = list(cities), boot, refs
        self.rows = {}
        for c in self.cities:
            sub = fits[fits["city"] == c]
            d = {}
            for k, (cid, v, f) in enumerate(zip(sub["cid"], sub["val_nll"], sub["finite"])):
                d.setdefault(cid, (k, v, bool(f)))
            self.rows[c] = d

    def _gap(self, city):
        r = None if self.refs is None else self.refs.get(city)
        g, m2 = scoring._nll(r, "grav_power"), scoring._nll(r, "M2_glm")
        return 0.0 if g is None or m2 is None else max(g - m2, 0.0)

    def close(self, i, j, margin: float) -> bool:
        for c in self.cities:
            a, b = self.rows[c].get(i), self.rows[c].get(j)
            if a is None or b is None or not (a[2] and b[2]):
                return False
            diff = abs(float(a[1]) - float(b[1]))
            dl = margin * self._gap(c) if self.refs is not None else 0.0
            if diff <= dl:
                continue
            d = self.boot[c][:, a[0]] - self.boot[c][:, b[0]]
            d = d[np.isfinite(d)]
            if diff > max(float(d.std()) if len(d) else 0.0, dl):
                return False
        return True

    def pairs(self, cids, margin: float) -> list:
        """Loss-close pairs (a, b), a before b in cids."""
        return [(a, b) for k, a in enumerate(cids) for b in cids[k + 1:] if self.close(a, b, margin)]


def member_of(pairs: set):
    return lambda i, j: (i, j) in pairs or (j, i) in pairs


def _reps(pool, groups: dict) -> list:
    """groups {key: [pool entries]} -> representative = lowest (complexity, cid), merged_from = the other members
    (sorted); list sorted by cid."""
    reps = []
    for members in groups.values():
        rep = min(members, key=lambda x: (x["complexity"], x["cid"]))
        reps.append({**rep, "merged_from": sorted(m["cid"] for m in members if m["cid"] != rep["cid"])})
    return sorted(reps, key=lambda r: r["cid"])


def cluster_chain(pool, pairs, rclose) -> list:
    """Union-find over the loss-close pairs (in order) that are rank-close: chains (A ~ B, B ~ C -> one cluster)."""
    parent = {x["cid"]: x["cid"] for x in pool}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    for i, j in pairs:
        if find(i) != find(j) and rclose(i, j):
            parent[find(j)] = find(i)
    groups = {}
    for x in pool:
        groups.setdefault(find(x["cid"]), []).append(x)
    return _reps(pool, groups)


def cluster_leader(pool, lclose, rclose) -> list:
    """No chain: candidates in (complexity, cid) order; each joins the first representative (creation order) it is
    directly equivalent to (lclose and rclose), else becomes a representative."""
    reps, groups = [], {}
    for x in sorted(pool, key=lambda x: (x["complexity"], x["cid"])):
        for r in reps:
            if lclose(x["cid"], r) and rclose(x["cid"], r):
                groups[r].append(x)
                break
        else:
            reps.append(x["cid"])
            groups[x["cid"]] = [x]
    return _reps(pool, groups)


def rep_of(reps) -> dict:
    out = {}
    for r in reps:
        out[r["cid"]] = r["cid"]
        out.update({m: r["cid"] for m in r.get("merged_from", [])})
    return out


def merge(pool, fits, boot, etas, cities, refs=None, linkage=None, margin=None):
    """etas: {city: {"offsets", "eta": {cid: array}}} or a callable cids -> that dict, called once with the
    candidates of the loss-close pairs only (the rank test is the expensive part). linkage None = scoring.MERGE_LINKAGE,
    margin None = scoring.MERGE_MARGIN."""
    link = scoring.MERGE_LINKAGE if linkage is None else linkage
    if link not in ("chain", "leader"):
        raise ValueError(f"unknown linkage {link!r}")
    m = scoring.MERGE_MARGIN if margin is None else margin
    pairs = LossTable(fits, boot, cities, refs).pairs([x["cid"] for x in pool], m)
    if callable(etas):
        etas = etas(sorted({c for p in pairs for c in p})) if pairs else {}
    rclose = lambda i, j: rank_close(i, j, etas, cities)  # noqa: E731
    if link == "chain":
        return cluster_chain(pool, pairs, rclose)
    return cluster_leader(pool, member_of(set(pairs)), rclose)


def _share(v, g, m2):
    return (g - v) / (g - m2) if v is not None and g > m2 else None


def gate1_package(ranked, pool, fits, refs, clean_out=None, llm_out=None, b_best=None, out_dir=None, cities=None):
    out_dir = Path(out_dir) if out_dir else lib_dir()
    cities = list(cities) if cities else sorted(fits["city"].unique())
    eq = {x["cid"]: x["equation"] for x in pool}
    sims = (llm_out or {}).get("simplify", {})
    reads = (llm_out or {}).get("readability", {})
    cands = []
    for r in ranked:
        cid = r["cid"]
        co = (clean_out or {}).get(cid, {})
        sim, rd = sims.get(cid), reads.get(cid)
        per = {}
        for city in cities:
            row = _row(fits, cid, city)
            g, m2 = refs[city]["grav_power"]["nll_per_visit"], refs[city]["M2_glm"]["nll_per_visit"]
            v = float(row["val_nll"]) if row is not None and bool(row["finite"]) and np.isfinite(row["val_nll"]) else None
            per[city] = {"val_nll": v, "share": _share(v, g, m2),
                         "near_best": city in r.get("near_cities", []),
                         "within_1se": bool(row["within_1se"]) if row is not None else False,
                         "b_gap": (v - b_best[city]) if v is not None and b_best and city in b_best else None}
        cands.append({"cid": cid, "rank": len(cands) + 1, "gate1": bool(r["gate1"]), "n_near": int(r["n_near"]),
                      "n_1se": int(r["n_1se"]), "n_flags": int(r.get("n_flags", 0)),
                      "complexity": int(r["complexity"]), "equation": eq[cid], "canonical": co.get("canonical"),
                      "simplified": sim["sympy"] if sim and sim.get("accepted") else None,
                      "simplify_reason": sim.get("reason") if sim else None,
                      "simplify_rejected": (not sim.get("accepted")) if sim else None,
                      "readability": rd, "degenerate": list(co.get("degenerate", [])),
                      "removable_terms": [t["term"] for t in co.get("term_contrib", []) if t.get("removable")],
                      "term_contrib": co.get("term_contrib", []), "merged_from": list(r.get("merged_from", [])),
                      "source": list(r.get("source", [])), "cities": per})
    pkg = {"cities": cities, "n_candidates": len(cands), "n_gate1": sum(c["gate1"] for c in cands), "candidates": cands}
    jp, mp = out_dir / "candidates_gate1.json", out_dir / "gate1.md"
    jp.write_text(json.dumps(pkg, ensure_ascii=False, indent=1))
    mp.write_text(_markdown(pkg))
    return jp, mp


def _fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def _markdown(pkg):
    cities = pkg["cities"]
    L = ["# GATE 1 candidate skeletons", "", f"{pkg['n_candidates']} candidates (after merging), {pkg['n_gate1']} "
         f"enter GATE 1 "
         f"(the top {TOP} of those near-best in >= {MIN_CITIES} search cities; near-best = in the city's 1-SE set, or "
         f"validation NLL at most "
         f"{scoring.GATE_MARGIN:.0%} of the gravity-to-M2 gap above the city's best (V3 rule of 2026-11-10; 3% in plans "
         f"2–5); lexicographic order as in rank()). Gain share share = (NLL_grav − NLL_skel) / (NLL_grav − NLL_M2), "
         "validation set.", "",
         "| Rank | cid | GATE 1 | Near-best cities | 1-SE cities | Flags | Complexity | Degenerate | Removable terms "
         "| Readability | "
         + " | ".join(f"share {c}" for c in cities) + " |",
         "|" + "---|" * (10 + len(cities))]
    for c in pkg["candidates"]:
        rd = c["readability"]
        L.append(f"| {c['rank']} | {c['cid']} | {'✔' if c['gate1'] else ''} | {c['n_near']} | {c['n_1se']} | {c['n_flags']} | {c['complexity']} | "
                 f"{','.join(c['degenerate']) or '—'} | {len(c['removable_terms'])} | "
                 f"{rd['score'] if rd else '—'} | " + " | ".join(_fmt(c["cities"][x]["share"], 2) for x in cities) + " |")
    for c in pkg["candidates"]:
        L += ["", f"## {c['rank']}. {c['cid']}" + (" (GATE 1)" if c["gate1"] else ""), "",
              f"- Original: `{c['equation']}`"]
        if c["canonical"]:
            L.append(f"- Canonical: `{c['canonical']}`")
        L.append(f"- Simplified: `{c['simplified']}`" if c["simplified"] else
                 f"- Simplified: not accepted ({c['simplify_reason'] or 'no proposal'})" if c["simplify_reason"] is not None
                 else "- Simplified: none")
        if c["readability"]:
            L.append(f"- Readability {c['readability']['score']}/5" + (" (invalid variable list)" if c["readability"].get("invalid") else "") + ":")
            L += [f"  - `{t['term']}`: {t['meaning']}" for t in c["readability"].get("terms", [])]
        if c["removable_terms"]:
            L.append(f"- Removable terms (contribution < 1% of the gain gap, flagged only): {', '.join(f'`{t}`' for t in c['removable_terms'])}")
        if c["merged_from"]:
            L.append(f"- Merged from: {', '.join(c['merged_from'])}")
        L += ["", "| City | Validation NLL | share | Near-best | 1-SE | Gap to B best |", "|---|---|---|---|---|---|"]
        for x in cities:
            p = c["cities"][x]
            L.append(f"| {x} | {_fmt(p['val_nll'], 4)} | {_fmt(p['share'], 3)} | {'✔' if p['near_best'] else ''} | "
                     f"{'✔' if p['within_1se'] else ''} | {_fmt(p['b_gap'], 4)} |")
    return "\n".join(L) + "\n"


def load_llm(d):
    out = {"simplify": {}, "readability": {}}
    d = Path(d)
    if not d.exists():
        return out
    for point in out:
        for p in sorted(d.glob(f"{point}_*.json")):
            if p.name.endswith(".in.json"):
                continue
            r = json.loads(p.read_text()).get("result")
            if r:
                out[point][p.stem.split("_", 1)[1]] = r
    return out


def val_etas(pool, fits, cities):
    """eta of every fittable (cid, city) of `pool` on the city's validation rows, from the per-city refitted
    constants (float32: 100+ candidates x 4M New York rows)."""
    out = {}
    for city in cities:
        va = sr_data.build([city], "tract", "val", "val", None, 0)
        es = {}
        for x in pool:
            row = _row(fits, x["cid"], city)
            if row is None or not bool(row["finite"]):
                continue
            c = candidate({**x, "consts_refit": None})
            consts = np.asarray(json.loads(row["consts"]), float)
            if len(consts) != len(c.consts):
                continue
            e = eta(c, va, consts=consts)
            if np.isfinite(e).all():
                es[x["cid"]] = e.astype(np.float32)
        out[city] = {"offsets": va.offsets, "eta": es}
    return out


def main():
    from config import CITIES_CSV, RESULTS  # noqa: E402
    import ledger  # noqa: E402
    lib = lib_dir()
    cities = list(pd.read_csv(CITIES_CSV, dtype=str).query("role == 'search'")["city"])
    pool = json.loads((lib / "pool.json").read_text())
    fits = pd.read_parquet(lib / "search_city_fits.parquet")
    boot = {c: np.load(lib / f"boot_{c}.npy") for c in cities}
    refs = {c: json.loads((RESULTS / f"refs/{c}.json").read_text())["models"] for c in cities}
    clean_out = json.loads((lib / "clean.json").read_text()) if (lib / "clean.json").exists() else {}
    llm_out = load_llm(lib / "llm")
    bb = lib / "b_best_by_city.json"
    b_best = json.loads(bb.read_text()) if bb.exists() else None
    etas = lambda cids: val_etas([x for x in pool if x["cid"] in set(cids)], fits, cities)  # noqa: E731
    flags = {cid: {"n": 0, "degenerate": v.get("degenerate", [])} for cid, v in clean_out.items()}
    reps = rank(merge(pool, fits, boot, etas, cities, refs), fits, refs, flags)
    jp, mp = gate1_package(reps, pool, fits, refs, clean_out, llm_out, b_best, cities=cities)
    ledger.add(f"Task 6: merge {len(pool)} -> {len(reps)} skeletons, {sum(r['gate1'] for r in reps)} GATE 1 candidates "
               f"(near-best in >= {MIN_CITIES} search cities, margin {scoring.GATE_MARGIN:.0%} of the grav->M2 gap); "
               f"candidates_gate1.json sha {ledger.sha(jp)}, gate1.md sha {ledger.sha(mp)}")
    print(len(pool), "->", len(reps), "skeletons;", sum(r["gate1"] for r in reps), "GATE 1 candidates")


if __name__ == "__main__":
    main()
