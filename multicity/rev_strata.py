"""Revision 2026-11-22 (response to review; post hoc, supplementary): the banded flexible model within
strata of origins.

Each band coefficient of the banded model is identified by different origin-destination pairs: short bands by origins
near the centre, long bands by peripheral origins. A mix of origin types with flat but different size elasticities
could then look like an elasticity that rises with distance. The check: split each city's origins into thirds by their
distance to the CBD (inner, middle, outer) and refit the paper's banded model on each third alone (same destinations,
same covariates, train x search blocks). The inner third also keeps origins away from the metropolitan boundary.

Output: results/multicity/rev/banded_strata.parquet (stratum, city, band, ref, ed, eN, visits), one ledger line.
Usage (run_stage.sh, M line): rev_strata.py run | summary
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401
import rev_banded  # noqa: E402

STRATA = ("inner", "middle", "outer")


def stratum_mask(d: dict, k: int) -> np.ndarray:
    """Origins of the city in the k-th third (0 inner .. 2 outer) of the distance to the CBD among included origins."""
    form = pd.read_parquet(d["base"] / "form.parquet").set_index("unit").reindex(d["u"]["unit"])
    r = form["r_cbd"].to_numpy(float)
    o = d["orig"] & np.isfinite(r)
    q = np.quantile(r[o], [1 / 3, 2 / 3])
    lab = np.digitize(r, q)                       # 0, 1, 2
    return o & (lab == k)


def stage_run():
    import ledger
    import p9
    import rev_models
    cs = rev_banded._cities()
    rev_banded._G.update(variant="strata", F=rev_banded.fixed_refs(rev_banded.pooled_reference(cs)))
    out = []
    for k, name in enumerate(STRATA):
        def tf(d, k=k):
            d["orig"] = stratum_mask(d, k)
            return d
        orig, mods, _ = rev_models._patch_load(tf)
        try:
            rows = [r for rs in p9._pool_map(rev_banded.variant_city, cs) for r in rs]
        finally:
            for m in mods:
                m.load = orig
        t = pd.DataFrame(rows)
        t["stratum"] = name
        out.append(t)
        print(f"stratum {name}: {t['city'].nunique()} cities", flush=True)
    t = pd.concat(out, ignore_index=True)
    t["variant"] = "strata"
    p = rev_banded.out_dir() / "banded_strata.parquet"
    t.to_parquet(p, index=False)
    ledger.add(f"Revision 2026-11-22 (post hoc, supplementary): rev_strata run: banded flexible model refitted on the inner, "
               f"middle and outer thirds of origins by distance to the CBD, {t['city'].nunique()} cities; rev/{p.name} sha {ledger.sha(p)}")
    stage_summary()


def stage_summary():
    import prefreeze
    import rev_weekend
    mv = prefreeze.MIN_BAND_VISITS
    t = pd.read_parquet(rev_banded.out_dir() / "banded_strata.parquet")
    for name in STRATA:
        for ref in sorted(t["ref"].unique()):
            g = t[(t["stratum"] == name) & (t["ref"] == ref)]
            ok = g[g["visits"] >= mv]
            med = ok.groupby("band")[["eN", "ed"]].median().round(3)
            r = rev_weekend.rising(g, mv)
            print(f"{name} ref={ref}: e_N {med['eN'].tolist()}; e_d {med['ed'].tolist()}; cities per band "
                  f"{ok.groupby('band')['city'].nunique().tolist()}; e_N rises in {int((r['slope'] > 0).sum())} of {len(r)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "summary"):
        sys.exit(__doc__)
    {"run": stage_run, "summary": stage_summary}[sys.argv[1]]()
