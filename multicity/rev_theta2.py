"""Revision 2026-11-22 (response to review; post hoc, supplementary): the Reach model with a bounded shift.

In four cities the locally fitted shift theta_2 of N_j + theta_2 is negative, which leaves the utility undefined for
tracts with fewer POIs than -theta_2 (Tucson on all destinations); in a fifth it lies between 0 and 1. Here the Reach
model is refitted in every city whose shift is below the larger bound, with theta_2 >= LB for each LB in BOUNDS
(theta_2 >= 1 became part of the model's specification), everything else as in the main-analysis fits (library.city_data mode "test_dest": train x search
blocks, 300 sampled origins, five starts, L-BFGS-B), and scored on the test period x held-out block and on the test
period x all destinations.

Output: results/multicity/rev/theta2_bound.json, one ledger line. Nothing of the main analysis is written.
Usage (run_stage.sh, M line): rev_theta2.py run
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import rev_banded  # noqa: E402

CID, I2 = "c0355r1", 2                    # f = -c0*d**c1/(c2 + N)**c3 + a**c4: the shift is constant 2
BOUNDS = (1.0, 0.01)
STARTS, SEED = 5, 0


def fit_bounded(c, tr, lb: float) -> tuple:
    """(training NLL, constants) of the best of STARTS starts with constant I2 >= lb (as library._multistart)."""
    import library
    from refit_select import PENALTY, _obj
    rng = np.random.default_rng(SEED)
    bounds = [(None, None)] * len(c.consts)
    bounds[I2] = (lb, None)
    best = None
    for k in range(STARTS):
        s0 = np.asarray(c.consts, float) if k == 0 else np.asarray(c.consts, float) * np.exp(rng.normal(0.0, library.PERTURB_SD, len(c.consts)))
        s0[I2] = max(s0[I2], lb)
        if _obj(s0, c, tr)[0] >= PENALTY:
            continue
        r = minimize(lambda th: _obj(th, c, tr), s0, jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 500, "gtol": 1e-9, "ftol": 1e-13})
        f = float(_obj(r.x, c, tr)[0])
        if f < PENALTY and (best is None or f < best[0]):
            best = (f, [float(v) for v in r.x])
    if best is None:
        raise RuntimeError("no finite start")
    return best


def city_rows(city: str) -> list:
    import library
    import p9
    import prior_models as pm
    import sr_data
    from refit_select import loss
    import dataclasses
    c = library.to_single(p9.ours_candidates()[CID])
    D = library.city_data(city, None, "test_dest")
    ev = sr_data.build([city], "tract", "test", "all", None, 0)
    ref = p9.refs(city, "test_dest")
    cv = pd.read_parquet(config.RESULTS / "confirm/test_dest/cities_val.parquet")
    main = cv[(cv["cid"] == CID) & (cv["city"].astype(str) == city)].iloc[0]
    fits = [("main analysis", None, float(main["train_nll"]), [float(v) for v in json.loads(main["consts"])] if isinstance(main["consts"], str) else [float(v) for v in main["consts"]])]
    fits += [(f"theta2 >= {lb}", lb) + fit_bounded(c, D["tr"], lb) for lb in BOUNDS]
    rows = []
    for name, lb, tr_nll, consts in fits:
        cc = dataclasses.replace(c, consts=np.asarray(consts, float))
        nll = float(loss(cc, D["va"]))
        a = pm.score_cand(cc, consts, ev)
        rows.append({"city": city, "fit": name, "bound": lb, "consts": consts, "train_nll": tr_nll, "block_nll": nll,
                     "block_share": (ref["grav"] - nll) / (ref["grav"] - ref["m2"]), "all_nll": a["nll"], "all_cpc": a["cpc"]})
    print(json.dumps(rows), flush=True)
    return rows


def stage_run():
    import ledger
    cv = pd.read_parquet(config.RESULTS / "confirm/test_dest/cities_val.parquet")
    g = cv[cv["cid"] == CID]
    th2 = g["consts"].map(lambda v: float((json.loads(v) if isinstance(v, str) else list(v))[I2]))
    cities = sorted(g["city"].astype(str)[th2 < max(BOUNDS)])           # every city whose fit violates the larger bound
    rows = [r for c in cities for r in city_rows(c)]
    p = rev_banded.out_dir() / "theta2_bound.json"
    p.write_text(json.dumps(rows, indent=1))
    ledger.add(f"Revision 2026-11-22 (post hoc, supplementary): rev_theta2 run: Reach model refitted with theta_2 >= {list(BOUNDS)} in "
               f"the {len(cities)} cities with a shift below {max(BOUNDS)} ({', '.join(cities)}); rev/{p.name} sha {ledger.sha(p)}")
    for r in rows:
        print(r["city"], r["fit"], "theta2", round(r["consts"][I2], 3), "block share", round(r["block_share"], 4),
              "block nll", round(r["block_nll"], 5), "all nll", None if not np.isfinite(r["all_nll"]) else round(r["all_nll"], 5), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "run":
        sys.exit(__doc__)
    stage_run()
