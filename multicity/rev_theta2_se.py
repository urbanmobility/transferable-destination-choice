"""Revision 2026-11-22 (response to review; post hoc): standard errors of the Reach model's constants in the
cities where theta_2 >= 1 binds.

The main analysis reports cluster-robust standard errors of the local fits (refit_select.const_cov: sandwich over the
20 device groups, on the training rows of library.city_data mode "test_dest"). With the shift at its bound the
estimator has four free constants, so the same sandwich is taken over those four, with theta_2 held at the bound; theta_2
itself has no standard error there. As a check of the procedure, the full five-constant sandwich at the main-analysis
constants is recomputed and compared with the stored consts_se.

Output: results/multicity/rev/theta2_se.json, one ledger line. Nothing of the main analysis is written.
Usage (run_stage.sh, M line): rev_theta2_se.py run
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import rev_banded  # noqa: E402

CID, I2 = "c0355r1", 2


def cov_free(c, data, Yg, free, h=1e-5) -> np.ndarray:
    """refit_select.const_cov restricted to the constants in `free` (the others held fixed): same steps, same
    central differences, same G / (G - 1) factor. With every constant free it is const_cov itself."""
    from refit_select import _obj, eta
    th = np.asarray(c.consts, float)
    idx = np.flatnonzero(free)
    P = len(idx)
    V = float(data.y.sum())
    steps = h * np.maximum(1.0, np.abs(th))
    H = np.zeros((P, P))
    for a, j in enumerate(idx):
        e = np.zeros(len(th))
        e[j] = steps[j]
        H[:, a] = (_obj(th + e, c, data)[1][idx] - _obj(th - e, c, data)[1][idx]) / (2 * steps[j])
    H = 0.5 * (H + H.T) * V
    Hi = np.linalg.inv(H)
    e0 = eta(c, data, consts=th)
    J = np.zeros((len(e0), P))
    for a, j in enumerate(idx):
        e = np.zeros(len(th))
        e[j] = steps[j]
        J[:, a] = (eta(c, data, consts=th + e) - eta(c, data, consts=th - e)) / (2 * steps[j])
    st = data.offsets[:-1]
    rep = np.diff(data.offsets)
    ce = e0 - np.repeat(np.maximum.reduceat(e0, st), rep)
    p = np.exp(ce)
    p /= np.repeat(np.add.reduceat(p, st), rep)
    dl = J - np.repeat(np.add.reduceat(p[:, None] * J, st, axis=0), rep, axis=0)
    S = -(np.asarray(Yg, float) @ dl)
    G = S.shape[0]
    return Hi @ (S.T @ S) @ Hi * (G / max(G - 1, 1))


def city_row(city: str, main: list, bounded: list, stored: list) -> dict:
    import library
    import p9
    from refit_select import _device_group_y, const_cov
    c = library.to_single(p9.ours_candidates()[CID])
    D = library.city_data(city, None, "test_dest")
    Yg = _device_group_y(D["tr"], [city], "tract", "train")
    cm = dataclasses.replace(c, consts=np.asarray(main, float))
    se_ref = np.sqrt(np.diag(const_cov(cm, D["tr"], Yg)))                      # the main analysis' routine
    se_all = np.sqrt(np.diag(cov_free(cm, D["tr"], Yg, np.ones(5, bool))))     # this file's routine, all free
    free = np.ones(5, bool)
    free[I2] = False
    cb = dataclasses.replace(c, consts=np.asarray(bounded, float))
    se_b = np.full(5, np.nan)
    se_b[free] = np.sqrt(np.diag(cov_free(cb, D["tr"], Yg, free)))
    row = {"city": city, "consts_main": [float(v) for v in main], "se_stored": stored,
           "se_main_const_cov": [float(v) for v in se_ref], "se_main_cov_free": [float(v) for v in se_all],
           "consts_bounded": [float(v) for v in bounded], "se_bounded": [None if not np.isfinite(v) else float(v) for v in se_b]}
    print(json.dumps(row), flush=True)
    return row


def stage_run():
    import ledger
    full = json.loads((rev_banded.out_dir() / "theta2_full.json").read_text())
    cv = pd.read_parquet(config.RESULTS / "confirm/test_dest/cities_val.parquet")
    cv = cv[cv["cid"] == CID].assign(city=lambda d: d["city"].astype(str)).set_index("city")
    rows = []
    for r in full:
        st = cv.loc[r["city"], "consts_se"]
        st = json.loads(st) if isinstance(st, str) else [float(v) for v in st]
        rows.append(city_row(r["city"], r["main"]["consts"], r["bounded"]["consts"], st))
    p = rev_banded.out_dir() / "theta2_se.json"
    p.write_text(json.dumps(rows, indent=1))
    worst = max(abs(a / b - 1) for r in rows for a, b in zip(r["se_main_cov_free"], r["se_stored"]))
    ledger.add(f"Revision 2026-11-22 (post hoc): rev_theta2_se run: cluster-robust standard errors of the Reach model's four free "
               f"constants with theta_2 at its bound in {len(rows)} cities ({', '.join(r['city'] for r in rows)}); stored main-analysis "
               f"standard errors reproduced to a relative difference of {worst:.2e}; rev/{p.name} sha {ledger.sha(p)}")
    print("largest relative difference to the stored standard errors", worst, flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "run":
        sys.exit(__doc__)
    stage_run()
