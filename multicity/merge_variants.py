"""Merge-rule variants of the tolerance test (design decision of 2026-11-09: examine the variants side by side).
The 0.05 test (sensitivity.py) showed the equivalence merge at 0.05 joining families that differ on the discovery
cities, partly through chains (union-find: A ~ B and B ~ C put A, B, C together although A !~ C). Three variants,
near-best always at 0.05:

  V1  merge at 0.03, chain (union-find, = plan 5's merge)
  V2  merge at 0.05, no chain (leader: candidates in (complexity, cid) order; each joins the first representative it
      is directly equivalent to, else becomes a representative)
  V3  merge at 0.03, no chain

Equivalence is merge.equivalent's (loss_close with max(SD, delta_city) in every search city, and rank_close > 0.99),
computed once: LossTable = merge.loss_close with O(1) lookups (checked against it on a sample of real pairs);
rank_close precomputed in parallel for every pair loss-close at 0.05 (a superset of 0.03's). Checks before any
variant: the union-find on these tables must reproduce plan 5's clusters (0.03) and the 0.05 test's (0.05).
Stages:
  rank       copy of prefreeze/loop; tables; checks; V1-V3 reps; rank at 0.05 -> gates.json (+ rank_close.parquet)
  discovery  plan 5's 12 + every variant's gate on the 35 discovery cities (one fit_library run, near-best at 0.05);
             reproduction against prefreeze_m005/discovery; coverage per variant x {gate, plan5+gate} x {0.03, 0.05}
  report     markdown (redaction-checked)
Outputs: results/multicity/prefreeze_mv/. Search + discovery cities only; no lawtest / confirmation city, no test
period, no LLM call. These are post-hoc rule variants: a sensitivity analysis, not a change of the loop's rules.
Usage (compute server, loop env): python -I merge_variants.py rank | discovery | report
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import config  # noqa: E402
import prefreeze  # noqa: E402
import scoring  # noqa: E402
import sensitivity as sv  # noqa: E402

NEAR_MARGIN = 0.05
VARIANTS = {"V1": {"merge_margin": 0.03, "linkage": "chain"},
            "V2": {"merge_margin": 0.05, "linkage": "leader"},
            "V3": {"merge_margin": 0.03, "linkage": "leader"}}
ROOT = config.RESULTS / "prefreeze_mv"
PF, M005 = prefreeze.ROOT, sv.ROOT
SEED, N_SAMPLE = 20261109, 3000
KEY_CIDS = ["c0079c1", "c0353c1", "c0353c1r1", "c0355", "c0355r1", "c0354", "c0413c1", "c0413c1r2", "c0413c1r3",
            "c0262"]


# ================================================================================================ equivalence tables
from merge import LossTable, cluster_chain, cluster_leader, member_of, rep_of  # noqa: E402,F401  (defined in merge.py, re-exported here)

_G = {}


def _rc(pair):
    import merge
    return pair, bool(merge.rank_close(pair[0], pair[1], _G["etas"], _G["cities"]))


def rank_table(pairs, etas, cities, procs: int = 1) -> dict:
    """{(a, b): merge.rank_close(a, b, etas, cities)} for the given pairs (fork pool: etas shared, not copied)."""
    _G.update(etas=etas, cities=list(cities))
    if procs <= 1 or len(pairs) < 2:
        return dict(map(_rc, pairs))
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(procs, len(pairs))) as P:
        return dict(P.imap_unordered(_rc, pairs, chunksize=64))


def lookup(table: dict):
    """(i, j) -> table[(i, j)] or table[(j, i)]; KeyError for a pair the table does not hold (never guessed)."""
    def f(i, j):
        return table[(i, j)] if (i, j) in table else table[(j, i)]
    return f


def cluster_mismatches(reps, recorded) -> list:
    """Each recorded representative's member set ({cid} + merged_from) must be one computed cluster."""
    comp = {r["cid"]: {r["cid"], *r.get("merged_from", [])} for r in reps}
    owner = rep_of(reps)
    out = []
    for r in recorded:
        want = {r["cid"], *r.get("merged_from", [])}
        got = comp.get(owner.get(r["cid"]), set())
        if want != got:
            out.append(f"{r['cid']}: recorded {sorted(want)} computed {sorted(got)}")
    return out


def gate_for(reps, fits, refs, flags, near_margin: float = NEAR_MARGIN):
    """scoring.rank with near-best at near_margin (runtime only) -> (ranked, every representative near-best in >=
    MIN_CITIES search cities)."""
    ranked = scoring.rank(reps, fits, refs, flags, margin=near_margin)
    return ranked, sv.gate_cids(ranked)


# ================================================================================================ CLI stages
def stage_rank():
    import pandas as pd
    import ledger
    import loop
    import merge
    prefreeze.prepare_copy(PF / "loop", ROOT / "loop")
    steps = loop.Steps(json.loads(loop.CONFIG.read_text()), root=ROOT / "loop")
    pool, (fits, boot), refs, flags = steps.pool(), steps._fits(), steps.refs(), steps.flags()
    cities, cids = steps.cities("search"), [x["cid"] for x in pool]
    lt = LossTable(fits, boot, cities, refs)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(cids), (N_SAMPLE, 2))
    sample = [(cids[a], cids[b]) for a, b in idx if a != b]
    bad = []
    for m in (0.03, 0.05):
        with sv.with_margin(m):
            bad += [(m, i, j) for i, j in sample if lt.close(i, j, m) != merge.loss_close(i, j, fits, boot, cities, refs)]
    pairs = {m: lt.pairs(cids, m) for m in (0.03, 0.05)}
    for m in (0.03, 0.05):           # every pair the table calls close must be close for merge.loss_close too
        with sv.with_margin(m):
            chk = [pairs[m][k] for k in rng.permutation(len(pairs[m]))[:N_SAMPLE]]
            bad += [(m, i, j) for i, j in chk if not merge.loss_close(i, j, fits, boot, cities, refs)]
    if bad:
        ledger.add(f"Merge-rule variants (2026-11-09): STOP — LossTable disagrees with merge.loss_close on {len(bad)} "
                   f"pairs, e.g. {bad[:5]}")
        return 3
    need = sorted({c for p in pairs[0.05] for c in p})
    etas = loop._inner_etas([x for x in pool if x["cid"] in set(need)], fits, steps.datas())
    table = rank_table(pairs[0.05], etas, cities, procs=int(os.environ.get("RANK_PROCS", "16")))
    rclose = lookup(table)
    s03 = set(pairs[0.03])
    pd.DataFrame([{"a": a, "b": b, "loss_close_003": (a, b) in s03, "rank_close": table[(a, b)]} for a, b in pairs[0.05]]
                 ).to_parquet(ROOT / "rank_close.parquet", index=False)
    chain = {0.03: cluster_chain(pool, pairs[0.03], rclose), 0.05: cluster_chain(pool, pairs[0.05], rclose)}
    mism = {0.03: cluster_mismatches(chain[0.03], json.loads((PF / "loop" / f"ranked_r{prefreeze.ROUND}.json").read_text())),
            0.05: cluster_mismatches(chain[0.05], json.loads((M005 / "loop" / f"ranked_r{prefreeze.ROUND}.json").read_text()))}
    if mism[0.03] or mism[0.05]:
        ledger.add(f"Merge-rule variants (2026-11-09): STOP — the union-find on the precomputed tables does not reproduce "
                   f"the recorded clusters: 0.03 {mism[0.03][:3]} ({len(mism[0.03])}); 0.05 {mism[0.05][:3]} ({len(mism[0.05])})")
        return 3
    reps = {"V1": chain[0.03], "V2": cluster_leader(pool, member_of(set(pairs[0.05])), rclose),
            "V3": cluster_leader(pool, member_of(s03), rclose)}
    out = {}
    for v, rp in reps.items():
        ranked, gate = gate_for(rp, fits, refs, flags)
        by, ro = {r["cid"]: r for r in ranked}, rep_of(rp)
        out[v] = {**VARIANTS[v], "near_margin": NEAR_MARGIN, "n_reps": len(rp),
                  "gate": [{k: by[c].get(k) for k in ("cid", "n_near", "n_flags", "complexity", "merged_from")} for c in gate],
                  "rep_of": {c: ro[c] for c in KEY_CIDS if c in ro}}
    ref = {"chain_0.05": {c: rep_of(chain[0.05])[c] for c in KEY_CIDS if c in rep_of(chain[0.05])}}
    checks = {"loss_sample": len(sample), "loss_close_pairs": {f"{m:g}": len(pairs[m]) for m in pairs},
              "rank_close_true": int(sum(table.values())), "etas_candidates": len(need),
              "reproduces": {"plan5_0.03": True, "m005_0.05": True}}
    p = ROOT / "gates.json"
    p.write_text(json.dumps({"variants": out, "reference": ref, "checks": checks}, ensure_ascii=False, indent=1,
                            default=str))
    ledger.add(f"Merge-rule variants (2026-11-09; near-best 0.05; post-hoc sensitivity, loop rules unchanged): "
               f"LossTable = merge.loss_close on {len(sample)} sampled pairs x 2 margins; loss-close pairs "
               f"{checks['loss_close_pairs']}, rank-close {checks['rank_close_true']}; union-find on the tables reproduces "
               f"plan 5's clusters (0.03) and the 0.05 test's (0.05); gates: "
               + "; ".join(f"{v} ({VARIANTS[v]['merge_margin']}, {VARIANTS[v]['linkage']}) {len(out[v]['gate'])} of "
                           f"{out[v]['n_reps']} reps {[g['cid'] for g in out[v]['gate']]}" for v in out)
               + f"; gates.json sha {ledger.sha(p)}")
    return 0


def _sets(gates, p5):
    return {v: {"gate": [g["cid"] for g in gates[v]["gate"]],
                "plan5+gate": list(dict.fromkeys(p5 + [g["cid"] for g in gates[v]["gate"]]))} for v in VARIANTS}


def stage_discovery():
    import pandas as pd
    import fit_library
    import ledger
    gates = json.loads((ROOT / "gates.json").read_text())["variants"]
    pool = {x["cid"]: x for x in json.loads((ROOT / "loop" / "pool.json").read_text())}
    p5 = list(dict.fromkeys(pd.read_parquet(PF / "discovery" / "cities_val.parquet")["cid"]))
    sets = _sets(gates, p5)
    cids = list(dict.fromkeys(c for v in VARIANTS for c in sets[v]["plan5+gate"]))
    out = ROOT / "discovery"
    with sv.with_margin(NEAR_MARGIN):
        fit_library.main(prefreeze.cities(), procs=int(os.environ.get("LOOP_OUTER_PROCS", "10")),
                         cands=[pool[c] for c in cids], out_dir=out)
    cv = pd.read_parquet(out / "cities_val.parquet")
    old = pd.read_parquet(M005 / "discovery" / "cities_val.parquet")
    both = [c for c in dict.fromkeys(old["cid"]) if c in set(cids)]
    rep = prefreeze.check_reproduction(cv, old, both)
    why = prefreeze.reproduction_gate(rep, pd.read_csv(out / "cities_val_summary.csv", dtype={"city": str}))
    if why:
        ledger.add(f"Merge-rule variants (2026-11-09): STOP — {'; '.join(why)[:1500]}")
        return 3
    flags = json.loads((ROOT / "loop" / "flags.json").read_text())
    refs = prefreeze._val_refs(sorted(set(cv["city"])))
    cov = {v: {s: {f"{m:g}": prefreeze.coverage_tables(sv.recompute_near(cv[cv["cid"].isin(ids)], refs, m), refs, pool, flags)
                   for m in (0.03, NEAR_MARGIN)} for s, ids in sets[v].items()} for v in VARIANTS}
    p = out / "coverage.json"
    p.write_text(json.dumps({"variants": cov, "plan5": p5, "reproduction": rep}, ensure_ascii=False, indent=1))
    last = lambda t: f"{t[-1]['covered']}/{t[-1]['n_cities']} ({len(t)})" if t else "0"  # noqa: E731
    ledger.add(f"Merge-rule variants (2026-11-09): {len(cids)} candidates on {len(set(cv['city']))} discovery cities; "
               f"reproduction of the 0.05 test max diff {max(rep.values()) if rep else 0:.2e}; clean coverage "
               + "; ".join(f"{v} {s}: 0.03 {last(cov[v][s]['0.03']['clean'])}, 0.05 {last(cov[v][s]['0.05']['clean'])}"
                           for v in VARIANTS for s in cov[v])
               + f"; cities_val.parquet sha {ledger.sha(out / 'cities_val.parquet')}, coverage.json sha {ledger.sha(p)}")
    return 0


def build_report(out_md: Path, root: Path = None) -> dict:
    import pandas as pd
    import report
    root = Path(root) if root else ROOT
    g = json.loads((root / "gates.json").read_text())
    gates, ref = g["variants"], g.get("reference", {})
    cv = pd.read_parquet(root / "discovery/cities_val.parquet")
    cj = json.loads((root / "discovery/coverage.json").read_text())
    cov, p5 = cj["variants"], cj["plan5"]
    flags = json.loads((root / "loop/flags.json").read_text())
    refs = {c: json.loads(p.read_text())["models"] for c in sorted(set(cv["city"]))
            if (p := config.RESULTS / f"refs/{c}.json").exists()}
    sets = _sets(gates, p5)
    name = {"chain": "chain (union-find)", "leader": "no chain (direct comparison with the representative)"}
    L = [f"# Three variants of the merge rule (near-best margin 0.05; {len(set(cv['city']))} discovery cities)", "",
         "The registered rule uses 0.03 for both near-best and merging, with chained merging. These are post-hoc rule variants, "
         "a sensitivity analysis only; only search and discovery cities are used.", "",
         "## 1. Search-city gate (near-best in >= 4 cities)", ""]
    rows = [[v, f"{gates[v]['near_margin']:g}", f"{gates[v]['merge_margin']:g}", name[gates[v]["linkage"]], gates[v]["n_reps"],
             len(gates[v]["gate"]), sum(1 for x in gates[v]["gate"] if not x["n_flags"])] for v in VARIANTS]
    L += prefreeze._md_table(["Variant", "Near-best margin", "Merge margin", "Linkage", "Representatives", "Pass gate", "Of which clean"], rows)
    def _g(x):
        k = len(x.get("merged_from") or [])
        return x["cid"] + (f" (merged {k})" if k else "")
    L += [""] + [f"- {v}: " + ", ".join(_g(x) for x in gates[v]["gate"]) for v in VARIANTS]
    L += ["", "## 2. Representative of each key candidate", ""]
    keys = [c for c in KEY_CIDS if any(c in gates[v]["rep_of"] for v in VARIANTS)] or sorted(
        {c for v in VARIANTS for c in gates[v]["rep_of"]})
    rows = [[c, gates["V1"]["rep_of"].get(c, "—"), ref.get("chain_0.05", {}).get(c, "—")]
            + [gates[v]["rep_of"].get(c, "—") for v in ("V2", "V3")] for c in keys]
    L += prefreeze._md_table(["Candidate", "0.03 chain (plan 5 = V1)", "0.05 chain (previous test)", "V2: 0.05 no chain", "V3: 0.03 no chain"], rows)
    L += ["", "## 3. Discovery-city coverage (greedy; covered = near-best and above the floor)", ""]
    last = lambda t: f"{t[-1]['covered']}/{t[-1]['n_cities']} ({len(t)} candidates)" if t else "0"  # noqa: E731
    rows = []
    for v in VARIANTS:
        for s in ("gate", "plan5+gate"):
            for m in ("0.03", f"{NEAR_MARGIN:g}"):
                t = cov[v][s][m]
                rows.append([v, "gate of this variant only" if s == "gate" else "plan 5's 12 + gate of this variant", m, last(t["all"]),
                             last(t["clean"]), " → ".join(f"{r['cid']} (+{r['new']})" for r in t["clean"]) or "none"])
    L += prefreeze._md_table(["Variant", "Candidate set", "Margin", "All candidates", "Clean candidates only", "Greedy picks among clean candidates"], rows)
    L += ["", f"## 4. Cities not covered by clean candidates alone (margin {NEAR_MARGIN:g})", ""]
    allc = sorted(set(cv["city"]))
    for v in VARIANTS:
        for s, ids in sets[v].items():
            nn = sv.recompute_near(cv[cv["cid"].isin(ids)], refs, NEAR_MARGIN)
            ok = {r.city for r in nn[nn.near_best].itertuples() if scoring.n_flags(flags.get(r.cid)) == 0
                  and scoring.passes_floor(r.val_nll, refs.get(r.city))}
            miss = [c for c in allc if c not in ok]
            L.append(f"- {v}, {'gate of this variant only' if s == 'gate' else 'plan 5 + gate of this variant'}: {len(miss)}: "
                     f"{', '.join(miss) or 'none'}")
    text = "\n".join(L) + "\n"
    try:
        report._scan(text)
    except ValueError as e:
        raise RuntimeError(f"merge-variant report refused: {e}") from e
    Path(out_md).write_text(text)
    return {v: len(gates[v]["gate"]) for v in VARIANTS}


def stage_report():
    import ledger
    p = ROOT / "report.md"
    s = build_report(p)
    ledger.add(f"Merge-rule variants (2026-11-09) report: gates {s}; report.md sha {ledger.sha(p)}")
    return 0


STAGES = {"rank": stage_rank, "discovery": stage_discovery, "report": stage_report}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: merge_variants.py {{{'|'.join(STAGES)}}}")
    sys.exit(STAGES[sys.argv[1]]())
