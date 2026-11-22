"""Revision 2026-11-22 (response to review; post hoc, supplementary): the banded flexible model with the
Reach model's information set.

The paper's banded model holds 15 other destination attributes fixed, so its size elasticity e_N is a partial effect,
while the Reach model has only distance, POI count and area. Two reduced banded models make the comparison like for
like (same fit as rev_banded.variant_city: train x search blocks, all origins, the distance spline, six band terms in
ln(1 + N_j)):

  poiarea   destination covariates: log area only
  poionly   no destination covariate besides the banded POI terms

Output: results/multicity/rev/banded_<variant>.parquet, one ledger line each.
Usage (run_stage.sh, M line): rev_matched.py run poiarea|poionly | summary
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401
import rev_banded  # noqa: E402

VARIANTS = {"poiarea": ("log_area",), "poionly": ()}


def stage_run(variant: str):
    import ledger
    import p9
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}: one of {tuple(VARIANTS)}")
    keep = VARIANTS[variant]

    def design_cols(cols: dict, _variant: str, u: pd.DataFrame) -> tuple:
        missing = [k for k in keep if k not in cols]
        if missing:
            raise KeyError(f"destination covariates {missing} not among {sorted(cols)}")
        return {k: cols[k] for k in keep}, np.log1p(u["N"].to_numpy(float))

    cs = rev_banded._cities()
    rev_banded._G.update(variant=variant, F=rev_banded.fixed_refs(rev_banded.pooled_reference(cs)))
    orig = rev_banded.design_cols
    rev_banded.design_cols = design_cols                    # variant_city looks the name up at call time (fork pool)
    try:
        rows = [r for rs in p9._pool_map(rev_banded.variant_city, cs) for r in rs]
    finally:
        rev_banded.design_cols = orig
    t = pd.DataFrame(rows)
    t["variant"] = variant
    p = rev_banded.out_dir() / f"banded_{variant}.parquet"
    t.to_parquet(p, index=False)
    ledger.add(f"Revision 2026-11-22 (post hoc, supplementary): rev_matched run {variant}: banded flexible model with destination "
               f"covariates {list(keep) or 'none'}, {t['city'].nunique()} cities; rev/{p.name} sha {ledger.sha(p)}")
    print(f"wrote {p} ({t['city'].nunique()} cities)", flush=True)


def stage_summary():
    import prefreeze
    import rev_weekend
    mv = prefreeze.MIN_BAND_VISITS
    for name in ("banded_fixed.parquet", "banded_poiarea.parquet", "banded_poionly.parquet"):
        p = rev_banded.out_dir() / name
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        for model in sorted(t["model"].unique()):
            for ref in sorted(t["ref"].unique()):
                g = t[(t["model"] == model) & (t["ref"] == ref)]
                med = g[g["visits"] >= mv].groupby("band")[["eN", "ed"]].median().round(3)
                r = rev_weekend.rising(g, mv)
                print(f"{name} {model} ref={ref}: e_N {med['eN'].tolist()}; e_d {med['ed'].tolist()}; "
                      f"rises in {int((r['slope'] > 0).sum())} of {len(r)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        stage_run(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "summary":
        stage_summary()
    else:
        sys.exit(__doc__)
