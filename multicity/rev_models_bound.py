"""Revision 2026-11-23 (response to review; post hoc, supplementary): rev_models with theta_2 >= 1.

The shift of the Reach model is constrained to theta_2 >= 1 (part of its specification). rev_models.py fitted its Reach-model forms
with a free shift. Here every form that contains the Reach model's distance term with a free shift (MODELS) is refitted
with the bound in the cities where the fitted shift is below it or the fit could not be scored; every other fit is
kept. The refit is the one of the amended main analysis (rev_theta2.fit_bounded: the equation's numerals and four
perturbations of them as starts, L-BFGS-B), with the numerals that rev_models holds fixed kept fixed. Uniform
constants, scores and paired differences are then recomputed by rev_models' own transfer and report stages.
The frozen fits (sk_frozen, tp_frozen, tr_frozen) are records of the freeze and stay as they are; the separable twins
are not the Reach model and keep their free shift.

Output: results/multicity/rev/bound/models_<line>_{local,uniform,scored}.parquet, summary_models_<line>.json,
report_models_<line>.md (the files of rev_models, in a directory of their own; nothing in rev/ is overwritten).
A later call for the same line refits only what is still below the bound and rewrites the files of rev/bound/.
Usage (run_stage.sh, M line): rev_models_bound.py run all|all2|nonwork|poitrain
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401
import rev_models as rm  # noqa: E402
import rev_theta2  # noqa: E402

BOUND, I2 = 1.0, rev_theta2.I2            # the Reach shift is the third numeral of every form below
MODELS = {"all": ["sk", "sk_E", "sk_lna", "sk_plus"], "all2": ["sk_plus1"], "nonwork": ["sk"], "poitrain": ["sk"]}


def fit_bounded(c, tr, free) -> tuple:
    """rev_theta2.fit_bounded with a mask: the numerals outside `free` stay at the equation's values."""
    import library
    from refit_select import PENALTY, _obj
    rng = np.random.default_rng(rev_theta2.SEED)
    free = np.asarray(free, bool)
    c0 = np.asarray(c.consts, float)
    bounds = [(None, None) if f else (v, v) for f, v in zip(free, c0)]
    bounds[I2] = (BOUND, None)
    best = None
    for k in range(rev_theta2.STARTS):
        s0 = c0.copy() if k == 0 else np.where(free, c0 * np.exp(rng.normal(0.0, library.PERTURB_SD, len(c0))), c0)
        s0[I2] = max(s0[I2], BOUND)
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


def refit_job(a) -> dict:
    import prior_models as pm
    from refit_select import parse_equation
    city, m = a
    tr, ev = rm._data(city)
    eq, free = rm.REV_EQ[m]
    assert free[I2], m
    r = {"city": city, "model": m, "level": "local", "consts": None, "train_nll": np.nan, "nll": np.nan, "cpc": np.nan}
    try:
        c = parse_equation(eq, None, "A", 9)
        f, k = rev_theta2.fit_bounded(c, tr, BOUND) if all(free) else fit_bounded(c, tr, free)
        r.update(consts=json.dumps(k), train_nll=f, **pm.score_eq(eq, k, ev))
    except RuntimeError:
        pass
    print(city, m, "bounded refit", r["nll"], flush=True)
    return r


def stage_run(line: str):
    import ledger
    import p9
    if line not in MODELS:
        raise ValueError(f"line {line!r}: one of {list(MODELS)}")
    src, tgt, role = rm._setup(line)
    base = rm.out_dir()
    out = base / "bound"
    out.mkdir(parents=True, exist_ok=True)
    done = out / f"models_{line}_local.parquet"
    loc = pd.read_parquet(done if done.exists() else base / f"models_{line}_local.parquet")
    loc["city"] = loc["city"].astype(str)
    jobs = []
    for m in MODELS[line]:
        g = loc[loc["model"] == m]
        sh = g["consts"].map(lambda v: float(json.loads(v)[I2]) if isinstance(v, str) and v else np.nan)
        jobs += [(c, m) for c in g["city"][(sh < BOUND) | sh.isna() | ~np.isfinite(g["nll"].astype(float))]]
    new = loc
    if jobs:
        with rm.use_line(line):
            rows = pd.DataFrame(p9._pool_map(refit_job, jobs))
        rows["set"] = rows["city"].map(role)
        key = set(zip(rows["city"], rows["model"]))
        keep = loc[[(c, m) not in key for c, m in zip(loc["city"], loc["model"])]]
        new = pd.concat([keep, rows[loc.columns]], ignore_index=True)
        assert len(new) == len(loc)
        new.to_parquet(done, index=False)
        ledger.add(f"Revision 2026-11-23 (post hoc, supplementary): rev_models_bound run {line}: {MODELS[line]} refitted with theta_2 >= {BOUND} "
                   f"in {len(jobs)} city fits ({', '.join(f'{m} {c}' for c, m in sorted(jobs, key=lambda t: (t[1], t[0])))}), all other fits "
                   f"kept; rev/bound/models_{line}_local.parquet sha {ledger.sha(done)}")
    add = ledger.add                      # rev_models' transfer and report stages, writing under rev/bound/
    rm.out_dir = lambda: out
    rm.LINE_MODELS[line] = [m for m in rm.LINE_MODELS[line] if m in set(new["model"])]     # forms added to REV_EQ for line all2 later
    ledger.add = lambda msg: add(f"theta_2 >= {BOUND} variant, files under rev/bound/: {msg}")
    try:
        rm.stage_transfer(line)
        rm.stage_report(line)
    finally:
        ledger.add = add


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        sys.exit(__doc__)
    stage_run(sys.argv[2])
