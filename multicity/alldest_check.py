"""Supplementary check (added 2026-11-12): the comparison with EVERY destination in the choice set.

Why: all reported scores are within a destination block — the constants are fitted on choices among the search-block
destinations and scored on choices among the held-out block's destinations. That tests whether a formula ranks
destinations it has never seen; it does not test the allocation over all destinations of a city at once. This check
scores the same constants on the test period with all destinations in the choice set.

Fixed here before anything is computed:
  data        test period x all destinations (block "all"), every origin, the 67 cities. No constant is refitted:
              the skeletons use the frozen fits (confirm/test_dest/cities_val.parquet); the earlier models use plan 9's
              local fits (p9/fits_source.parquet, p9/fits_target.parquet); the uniform constants are plan 9's (target
              cities: p9/target.parquet; source cities, leave-one-out: p9/explore/source_test.parquet).
  references  gravity and M2 as before (fitted on train x search), scored on the same all-destination block.
  reported    per city NLL, CPC and share = (gravity - NLL) / (gravity - M2); then
              (1) median share and CPC with all destinations next to the block values;
              (2) scaled kernel minus the strongest earlier model, local constants and uniform constants: paired share
                  difference on the 32 target cities (as C1 and C2) and on all 67, city-bootstrap interval, wins;
              (3) per city, block share against all-destination share (Spearman).
  not a test  3 of the 5 destination blocks were used to fit the constants, so for those destinations only the period
              is held out. Post hoc, supplementary; the plan-9 contrasts stay as they are.

The pinned plan-9 modules and outputs are only read. Output: results/multicity/p9/alldest/.
Usage (run_stage.sh, M line): alldest_check.py score | report
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import p9  # noqa: E402
import p9_stats as st  # noqa: E402
import prior_models as pm  # noqa: E402

PRIOR = ("strexp3_N", "strexp1_N", "S0_N", "grav_power_N", "grav_power_P")
OPPONENT = "strexp3_N"
B, SEED = 5000, 20261112
_G = {}


def out_dir() -> Path:
    return p9.OUT / "alldest"


# --------------------------------------------------------------------------------------------- pure parts
def add_share(t: pd.DataFrame) -> pd.DataFrame:
    """share = (gravity - NLL) / (gravity - M2) with the references of the same block (rows grav_ref, M2_ref)."""
    ref = t[t["model"].isin(("grav_ref", "M2_ref"))].pivot(index="city", columns="model", values="nll")
    gap = ref["grav_ref"] - ref["M2_ref"]
    bad = sorted(gap.index[~(np.isfinite(gap) & (gap > 0))])
    if bad:
        raise RuntimeError(f"the gravity -> M2 gap is not positive and finite in {bad[:5]}")
    t = t.copy()
    t["share"] = (t["city"].map(ref["grav_ref"]) - t["nll"]) / t["city"].map(gap)
    return t


def _col(t, key) -> pd.Series:
    g = t[(t["model"] == key[0]) & (t["level"] == key[1])]
    return g.set_index("city")["share"]


def paired(t: pd.DataFrame, a: tuple, b: tuple, cities: list, B: int = B, seed: int = SEED) -> dict:
    """Share of a minus share of b over the given cities: median, city-bootstrap interval, wins."""
    da, db = _col(t, a), _col(t, b)
    miss = sorted(set(cities) - (set(da.index) & set(db.index)))
    if miss:
        raise ValueError(f"{a} / {b}: no score for {miss[:5]}")
    return st.contrast((da.loc[cities] - db.loc[cities]), B, seed)


def agreement(block: pd.Series, alld: pd.Series) -> dict:
    """The same cities on the two scorings: Spearman and the median of (all-destination - block)."""
    from scipy.stats import spearmanr
    cs = [c for c in block.index if c in alld.index and np.isfinite(block[c]) and np.isfinite(alld[c])]
    a, b = block.loc[cs].to_numpy(float), alld.loc[cs].to_numpy(float)
    return {"n": len(cs), "spearman": float(spearmanr(a, b)[0]) if len(cs) > 2 else None, "median_diff": float(np.median(b - a))}


# --------------------------------------------------------------------------------------------- stage: score
def _jobs(cfg: dict) -> dict:
    """{city: [(model, level, kind, constants)]} — every constant comes from an existing file."""
    cs = p9.cities("source") + p9.cities("target")
    jobs = {c: [] for c in cs}
    ours = p9.ours_fits("test_dest")
    for cid in cfg["ours"]["reported"]:
        for c in cs:
            f = ours.get(cid, {}).get(c)
            if f:
                jobs[c].append((cid, "local", "ours", f["consts"]))
    fits = pd.concat([pd.read_parquet(p9.OUT / "fits_source.parquet"), pd.read_parquet(p9.OUT / "fits_target.parquet")])
    for r in fits[fits["model"].isin(PRIOR)].itertuples():
        if r.consts and r.city in jobs:
            jobs[r.city].append((r.model, "local", "eq", json.loads(r.consts)))
    uni = pd.concat([pd.read_parquet(p9.OUT / "target.parquet"), pd.read_parquet(p9.OUT / "explore" / "source_test.parquet")])
    uni = uni[(uni["level"] == "uniform") & uni["model"].isin((cfg["ours"]["primary"], OPPONENT))]
    for r in uni.itertuples():
        if r.consts and r.city in jobs:
            jobs[r.city].append((r.model, "uniform", "ours" if r.model == cfg["ours"]["primary"] else "eq", json.loads(r.consts)))
    return jobs


def score_city(city: str) -> list:
    import frozen
    import refs
    import sr_data
    frozen.require()
    ev = sr_data.build([city], "tract", "test", "all", None, 0)
    rows = []
    for model, level, kind, consts in _G["jobs"][city]:
        s = pm.score_cand(_G["cands"][model], consts, ev) if kind == "ours" else pm.score_eq(pm.EQ_MODELS[model].equation, consts, ev)
        rows.append({"city": city, "model": model, "level": level, **s})
    del ev
    g = refs.glm_refs(city, "tract", transit_col=refs.TRANSIT_COL, train_block="search", eval_split="test", eval_block="all",
                      models=("grav_power", "M2_glm"))
    rows.append({"city": city, "model": "grav_ref", "level": "local", "nll": g["grav_power"]["nll_per_visit"], "cpc": g["grav_power"]["cpc"]})
    rows.append({"city": city, "model": "M2_ref", "level": "local", "nll": g["M2_glm"]["nll_per_visit"], "cpc": g["M2_glm"]["cpc"]})
    print(city, "all destinations scored", flush=True)
    return rows


def stage_score():
    import ledger
    cfg = p9.require_fixed()
    if not (p9.OUT / "contrasts.json").exists():
        raise RuntimeError("the plan-9 contrasts are not written yet: this check comes after them")
    _G.update(jobs=_jobs(cfg), cands=p9.ours_candidates())
    cs = list(_G["jobs"])
    rows = [r for rs in p9._pool_map(score_city, cs) for r in rs]
    t = pd.DataFrame(rows).sort_values(["model", "level", "city"]).reset_index(drop=True)
    out_dir().mkdir(parents=True, exist_ok=True)
    t.to_parquet(out_dir() / "scores.parquet", index=False)
    ledger.add(f"Supplementary all-destination check (2026-11-12; post hoc, nothing refitted): {t['city'].nunique()} cities scored on the "
               f"test period x all destinations; alldest/scores.parquet sha {ledger.sha(out_dir() / 'scores.parquet')}")


# --------------------------------------------------------------------------------------------- stage: report
def _f(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def stage_report():
    cfg = p9.require_fixed()
    P = cfg["ours"]["primary"]
    t = add_share(pd.read_parquet(out_dir() / "scores.parquet"))
    blk = pd.concat([pd.read_parquet(p9.OUT / "target.parquet"), pd.read_parquet(p9.OUT / "explore" / "source_test.parquet")])
    target, source = p9.cities("target"), p9.cities("source")
    everyone = source + target
    res = {"n_cities": int(t["city"].nunique()), "models": {}, "paired": {}, "agreement": {}}
    for (m, lv), g in t.groupby(["model", "level"]):
        b = blk[(blk["model"] == m) & (blk["level"] == lv)]
        res["models"][f"{m}|{lv}"] = {
            "n": int(len(g)), "share_all": float(g["share"].median()), "cpc_all": float(g["cpc"].median()), "nll_all": float(g["nll"].median()),
            "share_block": float(b["share"].median()) if len(b) else None, "cpc_block": float(b["cpc"].median()) if len(b) else None}
        if len(b):
            res["agreement"][f"{m}|{lv}"] = agreement(b.set_index("city")["share"], g.set_index("city")["share"])
    for lv in ("local", "uniform"):
        for name, cs in (("target", target), ("all", everyone)):
            res["paired"][f"{lv}|{name}"] = {"all_destinations": paired(t, (P, lv), (OPPONENT, lv), cs),
                                            "block": st.contrast(st.paired(blk[blk["city"].isin(cs)], (P, lv), (OPPONENT, lv)), B, SEED)}
    (out_dir() / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    L = ["# Supplementary check: all destinations in the choice set (test period)", "",
         "Post hoc, not a test. No constant is refitted; 3 of the 5 destination blocks were used to fit the constants, so for "
         "those destinations only the period is held out.", "",
         f"Cities: {res['n_cities']}. Share = (gravity − model) / (gravity − M2); the reference models are rescored on the same basis.", "",
         "## 1. Median per model: all destinations against the held-out block", "",
         "| model | constants | cities | share (all destinations) | share (held-out block) | CPC (all destinations) | CPC (held-out block) | Spearman between the two, per city |", "|---|---|---|---|---|---|---|---|"]
    order = [(P, "local"), (P, "uniform")] + [(c, "local") for c in cfg["ours"]["reported"] if c != P] + \
            [(OPPONENT, "local"), (OPPONENT, "uniform")] + [(m, "local") for m in PRIOR if m != OPPONENT] + [("M2_ref", "local"), ("grav_ref", "local")]
    for m, lv in order:
        r = res["models"].get(f"{m}|{lv}")
        if r:
            a = res["agreement"].get(f"{m}|{lv}", {})
            L.append(f"| {m} | {lv} | {r['n']} | {_f(r['share_all'])} | {_f(r['share_block'])} | {_f(r['cpc_all'])} | {_f(r['cpc_block'])} | {_f(a.get('spearman'), 2)} |")
    L += ["", f"## 2. {P} minus {OPPONENT} (paired share difference per city)", "",
          "| constants | cities | basis | median share difference [95% interval] | cities won |", "|---|---|---|---|---|"]
    for k, r in res["paired"].items():
        lv, name = k.split("|")
        for lab, key in (("all destinations", "all_destinations"), ("held-out block", "block")):
            q = r[key]
            L.append(f"| {lv} | {'32 target cities' if name == 'target' else '67 cities'} | {lab} | {_f(q['median'])} [{_f(q['ci_lo'])}, {_f(q['ci_hi'])}] | {q['wins']}/{q['n']} |")
    (out_dir() / "report_alldest.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


STAGES = {"score": stage_score, "report": stage_report}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        sys.exit(__doc__)
    STAGES[sys.argv[1]]()
