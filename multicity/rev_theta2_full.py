"""Revision 2026-11-22 (response to review; post hoc): everything downstream of the Reach model's local fit
in the cities where theta_2 >= 1 binds, recomputed with the bounded fit of rev_theta2.py. Nothing of the main analysis
is written here; paper/amend_theta2.py writes these results into the exported tables.

Per city (the cities of rev/theta2_bound.json), for the main-analysis constants and for the bounded ones:
  block   test period x held-out block: NLL, CPC, share (references of p9.refs)
  all     test period x all destinations: NLL, CPC
  bands   e_d, e_N per distance band at the city's band reference points and at the common reference point
Output: results/multicity/rev/theta2_full.json, one ledger line.
Usage (run_stage.sh, M line): rev_theta2_full.py run
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401
import rev_banded  # noqa: E402

CID, BOUND = "c0355r1", 1.0


def city_block(city: str, fits: dict) -> dict:
    import banded
    import city_outcomes as co
    import library
    import p9
    import prefreeze
    import prior_models as pm
    import sr_data
    c = library.to_single(p9.ours_candidates()[CID])
    D = library.city_data(city, None, "test_dest")
    ev_all = sr_data.build([city], "tract", "test", "all", None, 0)
    ref = p9.refs(city, "test_dest")
    b = co._banded([city])[city]
    refs_band = prefreeze._intkeys(b["refs"])
    refs_fixed = rev_banded.fixed_refs(rev_banded.pooled_reference(rev_banded._cities()))
    pool = co._pool()
    out = {"city": city}
    for name, consts in fits.items():
        cc = dataclasses.replace(c, consts=np.asarray(consts, float))
        sb, sa = pm.score_cand(cc, consts, D["va"]), pm.score_cand(cc, consts, ev_all)
        prof = {}
        for kind, rb in (("band", refs_band), ("fixed", refs_fixed)):
            p = banded.skeleton_profile(pool[CID], consts, rb)
            prof[kind] = {int(k): (None if v is None else [float(v[0]), float(v[1])]) for k, v in p.items()}
        out[name] = {"consts": [float(v) for v in consts], "block_nll": sb["nll"], "block_cpc": sb["cpc"],
                     "block_share": (ref["grav"] - sb["nll"]) / (ref["grav"] - ref["m2"]),
                     "all_nll": sa["nll"], "all_cpc": sa["cpc"], "profiles": prof}
    print(city, "done", flush=True)
    return out


def stage_run():
    import ledger
    rows = json.loads((rev_banded.out_dir() / "theta2_bound.json").read_text())
    cities = sorted({r["city"] for r in rows})
    res = []
    for city in cities:
        fits = {("main" if r["bound"] is None else "bounded"): r["consts"] for r in rows
                if r["city"] == city and (r["bound"] is None or r["bound"] == BOUND)}
        assert set(fits) == {"main", "bounded"}, (city, list(fits))
        res.append(city_block(city, fits))
    p = rev_banded.out_dir() / "theta2_full.json"
    p.write_text(json.dumps(res, indent=1, default=lambda v: None if (isinstance(v, float) and not np.isfinite(v)) else v))
    ledger.add(f"Revision 2026-11-22 (post hoc): rev_theta2_full run: block and all-destination scores and band profiles of the Reach "
               f"model with theta_2 >= {BOUND} in {len(cities)} cities ({', '.join(cities)}); rev/{p.name} sha {ledger.sha(p)}")
    for r in res:
        m, b = r["main"], r["bounded"]
        print(r["city"], "block share", round(m["block_share"], 4), "->", round(b["block_share"], 4), "| block cpc", round(m["block_cpc"], 4), "->",
              round(b["block_cpc"], 4), "| all nll", m["all_nll"], "->", round(b["all_nll"], 5), "| ed 5-10 (band)",
              m["profiles"]["band"].get(3), "->", b["profiles"]["band"].get(3), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "run":
        sys.exit(__doc__)
    stage_run()
