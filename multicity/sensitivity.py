"""Tolerance sensitivity test (design decision of 2026-11-09: test the tolerance at 0.05). The loop's frozen rules are re-run with
the multiplier of delta_city = m x (NLL_grav - NLL_M2) set to 0.05 instead of the registered 0.03 — at runtime only
(scoring.MARGIN inside with_margin); scoring.py and every result of the loop / plan 5 stay as they are. Stages:
  rank       copy of the plan-5 loop state (prefreeze/loop: the loop's pool + the 6 reforms), equivalence merge and
             lexicographic rank on the 8 search cities (fold 4) at 0.05 -> every representative near-best in >= 4
             search cities (no top-8 cap: this is the candidate list the 0.05 gate would pass)
  discovery  those candidates + plan 5's 12 on the 35 discovery cities (fit_library defaults; near-best at 0.05),
             reproduction check against prefreeze/discovery, coverage at 0.03 and 0.05 (all / clean)
  report     markdown tables (redaction-checked)
Outputs: results/multicity/prefreeze_m005/. Search + discovery cities only; no lawtest / confirmation city, no test
period, no LLM call.
Usage (compute server, loop env): python -I sensitivity.py rank | discovery | report
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import config  # noqa: E402
import prefreeze  # noqa: E402
import scoring  # noqa: E402

MARGIN = 0.05
ROOT = config.RESULTS / "prefreeze_m005"
PF = prefreeze.ROOT


def with_margin(m: float):
    """The registered rule (one multiplier for conclusions, gate and merge; chain merge) at multiplier m — the rule
    this test varied (11-09). Since the V3 rule decision (11-10) this is scoring.registered_rule(m)."""
    return scoring.registered_rule(m)


def gate_cids(ranked: list, min_cities: int = scoring.MIN_CITIES) -> list:
    """Every representative of the ranking near-best in >= min_cities search cities, in ranking order."""
    return [r["cid"] for r in ranked if r["n_near"] >= min_cities]


def recompute_near(cv, refs: dict, margin: float):
    """cv with near_best (within_1se, or at most max(se_vs_best, margin x gap) above the city's best finite row) and
    primary (the simplest near-best row: complexity, then val_nll, then cid) recomputed for the given multiplier."""
    df = cv.copy()
    fin = df["finite"].astype(bool) & df["val_nll"].notna()
    best = df[fin].groupby("city")["val_nll"].min()
    near = []
    for r, f in zip(df.itertuples(), fin):
        if not f:
            near.append(False)
            continue
        a, b = scoring._nll(refs.get(r.city), "grav_power"), scoring._nll(refs.get(r.city), "M2_glm")
        g = a - b if a is not None and b is not None else 0.0          # no references: the SE alone
        se = r.se_vs_best if r.se_vs_best is not None and r.se_vs_best == r.se_vs_best else 0.0
        near.append(bool(r.within_1se) or (r.val_nll - best[r.city] <= max(se, margin * max(g, 0.0))))
    df["near_best"] = near
    df["primary"] = False
    for _, g in df[df["near_best"]].groupby("city"):
        k = g.sort_values(["complexity", "val_nll", "cid"]).index[0]
        df.loc[k, "primary"] = True
    return df


# ================================================================================================ CLI stages
def stage_rank():
    import ledger
    import loop
    prefreeze.prepare_copy(PF / "loop", ROOT / "loop")
    steps = loop.Steps(json.loads(loop.CONFIG.read_text()), root=ROOT / "loop")
    steps.log = ledger.add
    with with_margin(MARGIN):
        ranked = steps.rank(prefreeze.ROUND)
    gate = gate_cids(ranked)
    by = {r["cid"]: r for r in ranked}
    old = json.loads((PF / "loop" / f"ranked_r{prefreeze.ROUND}.json").read_text())
    p = ROOT / "gate.json"
    p.write_text(json.dumps({"margin": MARGIN, "gate": [{k: by[c].get(k) for k in ("cid", "n_near", "n_flags", "complexity",
                                                                               "merged_from")} for c in gate],
                             "gate_0.03": gate_cids(old)}, ensure_ascii=False, indent=1))
    ledger.add(f"Tolerance test (11-09): search-city rank at multiplier {MARGIN} (copy of prefreeze/loop; "
               f"runtime only, scoring.py unchanged): {len(gate)} representatives near-best in >= 4 search cities "
               f"(at 0.03: {len(gate_cids(old))}): {gate}; gate.json sha {ledger.sha(p)}")
    return 0


def stage_discovery():
    import os
    import pandas as pd
    import fit_library
    import ledger
    gate = [g["cid"] for g in json.loads((ROOT / "gate.json").read_text())["gate"]]
    pool = {x["cid"]: x for x in json.loads((ROOT / "loop" / "pool.json").read_text())}
    pf_cv = pd.read_parquet(PF / "discovery" / "cities_val.parquet")
    cids = list(dict.fromkeys(list(dict.fromkeys(pf_cv["cid"])) + gate))
    out = ROOT / "discovery"
    with with_margin(MARGIN):
        fit_library.main(prefreeze.cities(), procs=int(os.environ.get("LOOP_OUTER_PROCS", "24")),
                         cands=[pool[c] for c in cids], out_dir=out)
    cv = pd.read_parquet(out / "cities_val.parquet")
    rep = prefreeze.check_reproduction(cv, pf_cv, list(dict.fromkeys(pf_cv["cid"])))
    why = prefreeze.reproduction_gate(rep, pd.read_csv(out / "cities_val_summary.csv", dtype={"city": str}))
    if why:
        ledger.add(f"Tolerance test (11-09): STOP — {'; '.join(why)[:1500]}")
        return 3
    flags = json.loads((ROOT / "loop" / "flags.json").read_text())
    refs = prefreeze._val_refs(sorted(set(cv["city"])))
    cov = {f"{m:g}": prefreeze.coverage_tables(recompute_near(cv, refs, m), refs, pool, flags) for m in (0.03, MARGIN)}
    p = out / "coverage.json"
    p.write_text(json.dumps({**cov, "reproduction": rep}, ensure_ascii=False, indent=1))
    last = lambda t: (t[-1]["covered"], t[-1]["n_cities"]) if t else None  # noqa: E731
    ledger.add(f"Tolerance test (11-09): {len(cids)} candidates ({len(gate)} from the 0.05 gate + plan 5's) on "
               f"{len(set(cv['city']))} discovery cities; reproduction of plan 5 max diff {max(rep.values()):.2e}; "
               + "; ".join(f"m {m}: all {[r['cid'] for r in t['all']]} -> {last(t['all'])}, clean "
                           f"{[r['cid'] for r in t['clean']]} -> {last(t['clean'])}" for m, t in cov.items())
               + f"; cities_val.parquet sha {ledger.sha(out / 'cities_val.parquet')}, coverage.json sha {ledger.sha(p)}")
    return 0


def build_report(out_md: Path, root: Path = None) -> dict:
    import pandas as pd
    import report
    root = Path(root) if root else ROOT
    f = report._f
    gate = json.loads((root / "gate.json").read_text())
    cv = pd.read_parquet(root / "discovery/cities_val.parquet")
    cov = json.loads((root / "discovery/coverage.json").read_text())
    pool = {x["cid"]: x for x in json.loads((root / "loop/pool.json").read_text())}
    flags = json.loads((root / "loop/flags.json").read_text())
    refs = {c: json.loads(p.read_text())["models"] for c in sorted(set(cv["city"]))
            if (p := config.RESULTS / f"refs/{c}.json").exists()}
    n3, n5 = recompute_near(cv, refs, 0.03), recompute_near(cv, refs, MARGIN)
    bd = {p.stem: json.loads(p.read_text()) for p in sorted((PF / "banded").glob("*.json"))}
    refs_b = {c: prefreeze._intkeys(b["refs"]) for c, b in bd.items()}
    flex = {c: prefreeze._intkeys(b["profile"]) for c, b in bd.items()}
    visits = {c: prefreeze._intkeys(b["visits"]) for c, b in bd.items()}
    dref = {c: {k: r["d"] for k, r in rb.items() if r} for c, rb in refs_b.items()}
    skel, n_failed = prefreeze.skeleton_profiles(cv, pool, refs_b)
    df = prefreeze.compare(skel, flex, visits, dref)
    cl = prefreeze.closest(df)
    cids = list(dict.fromkeys(cv["cid"]))
    L = [f"# Sensitivity test at tolerance multiplier 0.05 ({len(set(cv['city']))} discovery cities)", "",
         "The registered multiplier is 0.03; this test changes the multiplier of δ_city to 0.05 at runtime only, all "
         "other rules unchanged. Search and discovery cities only.", "",
         "## 1. Search-city gate (near-best in ≥ 4 cities, after equivalence merging)", "",
         f"- 0.03: {len(gate['gate_0.03'])}: {', '.join(gate['gate_0.03'])}",
         f"- 0.05: {len(gate['gate'])}: {', '.join(g['cid'] for g in gate['gate'])}", ""]
    L += prefreeze._md_table(["cid", "Near-best search cities (0.05)", "Flags", "Complexity", "Merged candidates"],
                             [[g["cid"], g["n_near"], g["n_flags"], g["complexity"], len(g.get("merged_from") or [])]
                              for g in gate["gate"]])
    L += ["", "## 2. Discovery cities: number of cities where each candidate is near-best", ""]
    rows = []
    for c in cids:
        a, b = n3[n3.cid == c], n5[n5.cid == c]
        rows.append([c, pool[c]["complexity"], scoring.n_flags(flags.get(c)), int(a.finite.astype(bool).sum()),
                     int(a.near_best.sum()), int(b.near_best.sum()), int(b.primary.sum()),
                     f(prefreeze._median(a.share_s0.tolist()))])
    rows.sort(key=lambda r: -r[5])
    L += prefreeze._md_table(["cid", "Complexity", "Flags", "Fittable cities", "Near-best (0.03)", "Near-best (0.05)",
                              "Primary-skeleton cities (0.05)", "Median share_S0"], rows)
    L += ["", "## 3. Coverage curves (greedy; covered = near-best and above the floor)", ""]
    for m in ("0.03", f"{MARGIN:g}"):
        for k, nm in (("all", "all candidates"), ("clean", "clean candidates only")):
            t = cov[m][k]
            L.append(f"- Multiplier {m}, {nm}: " + (" → ".join(f"{r['cid']} (+{r['new']}, {r['covered']}/{r['n_cities']})"
                                                       for r in t) if t else "none"))
    for m, nn in (("0.03", n3), (f"{MARGIN:g}", n5)):
        ok = {r.city for r in nn[nn.near_best].itertuples() if scoring.n_flags(flags.get(r.cid)) == 0
              and scoring.passes_floor(r.val_nll, refs.get(r.city))}
        L.append(f"- Multiplier {m}, cities not covered with clean candidates only: {', '.join(c for c in sorted(set(cv['city'])) if c not in ok) or 'none'}")
    L += ["", "## 4. Comparison of trends with the flexible banded model (shape)", ""]
    rows = []
    for c in cids:
        g = df[(df.cid == c) & np.isfinite(df.rmse_N.astype(float))]
        rows.append([c, len(g), f(prefreeze._median(g.rmse_d.tolist())), f(prefreeze._median(g.rmse_N_shape.tolist())),
                     f(prefreeze._median(g.slope_N_skel.tolist())), cl["rmse_d"].get(c, 0), cl["rmse_N_shape"].get(c, 0)])
    rows.sort(key=lambda r: -(r[5] + r[6]))
    L += prefreeze._md_table(["cid", "Comparable cities", "Median rmse_d", "Median rmse_N shape", "Median slope of e_N on ln d",
                              "Cities where e_d is closest", "Cities where the e_N shape is closest"], rows)
    L += ["", f"(Candidate, city) pairs left out because the elasticity profile could not be computed: {n_failed}."]
    text = "\n".join(L) + "\n"
    try:
        report._scan(text)
    except ValueError as e:
        raise RuntimeError(f"tolerance report refused: {e}") from e
    Path(out_md).write_text(text)
    return {"gate": len(gate["gate"]), "candidates": len(cids)}


def stage_report():
    import ledger
    p = ROOT / "report.md"
    s = build_report(p)
    ledger.add(f"Tolerance test (11-09) report: {s}; report.md sha {ledger.sha(p)}")
    return 0


STAGES = {"rank": stage_rank, "discovery": stage_discovery, "report": stage_report}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: sensitivity.py {{{'|'.join(STAGES)}}}")
    sys.exit(STAGES[sys.argv[1]]())
