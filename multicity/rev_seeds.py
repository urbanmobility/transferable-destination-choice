"""Revision 2026-11-22 (response to review; post hoc, supplementary): does the Reach model's advantage over
the strongest benchmark depend on the one random assignment of destination blocks, or on the one sample of origins?

For each of the 32 target cities the Reach model and the relaxed stretched exponential (POI count) are refitted and
rescored under
  base      the paper's design (block seed sr_data.FOLD_SEED, origin-sample seed 0): must reproduce contrast C1;
  fold k    k = 1..5: another assignment of the 20 destination clusters to the five blocks (seed FOLD_SEED + k;
            the k-means start changes with it), origin-sample seed 0;
  orig k    k = 1..5: the paper's blocks, another sample of 300 origins (seed k).
Fit: training period x the three fitting blocks, 300 sampled origins, five starts (library._multistart for the Reach
model, prior_models.fit_eq for the benchmark, as in the main analysis). Score: test period x held-out block (fold 0),
every origin. Only locally fitted parameters: uniform parameters would need the discovery cities refitted as well.

Output: results/multicity/rev/seeds.parquet (config, city, nll_reach, nll_bench), one ledger line.
Usage (run_stage.sh, M line): rev_seeds.py run | summary
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401
import rev_banded  # noqa: E402

REACH, BENCH = "c0355r1", "strexp3_N"
CONFIGS = [("base", 0, 0)] + [(f"fold {k}", k, 0) for k in range(1, 6)] + [(f"orig {k}", 0, k) for k in range(1, 6)]
_G = {}


def job(args) -> dict:
    import frozen
    import library
    import prior_models as pm
    import sr_data
    name, dfold, oseed, city = args
    frozen.require()
    base_seed = _G["fold_seed"]
    sr_data.FOLD_SEED = base_seed + dfold
    try:
        tr = sr_data.build([city], "tract", "train", "search", library.LIB_N_ORIG, oseed)
        ev = sr_data.build([city], "tract", "test", "fold:0", None, 0)
    finally:
        sr_data.FOLD_SEED = base_seed
    c = library.to_single(_G["cand"])
    out = {"config": name, "city": city, "nll_reach": np.nan, "nll_bench": np.nan}
    best = library._multistart(c, tr, _G["starts"], 0)
    if best is not None:
        out["nll_reach"] = float(pm.score_cand(c, best[1].consts, ev)["nll"])
    s = pm.EQ_MODELS[BENCH]
    b = pm.fit_eq(s.equation, s.free, tr, _G["starts"], _G["fit_seed"])
    if b is not None:
        out["nll_bench"] = float(pm.score_eq(s.equation, b[1], ev)["nll"])
    print(name, city, round(out["nll_bench"] - out["nll_reach"], 5), flush=True)
    return out


def stage_run():
    import ledger
    import p9
    import sr_data
    cfg = p9.require_fixed()
    _G.update(cand=p9.ours_candidates()[REACH], starts=int(cfg["starts"]), fit_seed=int(cfg["fit_seed"]), fold_seed=int(sr_data.FOLD_SEED))
    cities = p9.cities("target")
    jobs = [(name, df, os_, c) for name, df, os_ in CONFIGS for c in cities]
    t = pd.DataFrame(p9._pool_map(job, jobs)).sort_values(["config", "city"]).reset_index(drop=True)
    p = rev_banded.out_dir() / "seeds.parquet"
    t.to_parquet(p, index=False)
    ledger.add(f"Revision 2026-11-22 (post hoc, supplementary): rev_seeds run: Reach model vs relaxed stretched exponential, local "
               f"fits, {len(CONFIGS)} designs (base, 5 block assignments, 5 origin samples) x {len(cities)} target cities; "
               f"rev/{p.name} sha {ledger.sha(p)}")
    stage_summary()


def stage_summary():
    t = pd.read_parquet(rev_banded.out_dir() / "seeds.parquet")
    t["d"] = t["nll_bench"] - t["nll_reach"]
    for name, _, _ in CONFIGS:
        g = t[t["config"] == name]
        d = g["d"].dropna()
        print(f"{name}: median dNLL {d.median():.4f}, Reach better in {int((d > 0).sum())}/{len(g)} (not finite {int(g['d'].isna().sum())})", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "summary"):
        sys.exit(__doc__)
    {"run": stage_run, "summary": stage_summary}[sys.argv[1]]()
