"""Second step of the theta_2 >= 1 amendment (2026-11-22): standard errors of the Reach model's constants in the five
cities where the bound is active. amend_theta2.py replaced their constants and left the standard errors of the fits
without the bound. Here they are replaced by the cluster-robust standard errors of the four free constants with
theta_2 held at the bound (multicity/rev_theta2_se.py, run with the data root -> figdata/rev/theta2_se.json); theta_2 itself
has none. Runs once: the stored values must equal the ones of the fits without the bound.
Usage: python amend_theta2_se.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

FD = Path(__file__).resolve().parents[2] / "paper" / "figdata"
RV, CID = FD / "rev", "c0355r1"

rows = json.loads((RV / "theta2_se.json").read_text())
cv = pd.read_parquet(RV / "cities_val.parquet")
assert len(rows) == 5
for r in rows:
    assert np.allclose(r["se_stored"], r["se_main_cov_free"], rtol=1e-9, atol=0), r["city"]       # procedure reproduces the stored values
    assert r["consts_bounded"][2] == 1.0 and r["se_bounded"][2] is None, r["city"]
    row = (cv["cid"] == CID) & (cv["city"].astype(str) == r["city"])
    assert row.sum() == 1
    assert np.allclose(json.loads(cv.loc[row, "consts"].iloc[0]), r["consts_bounded"], atol=1e-9), r["city"]
    assert np.allclose(json.loads(cv.loc[row, "consts_se"].iloc[0]), r["se_stored"], rtol=1e-9, atol=0), (r["city"], "already amended?")
    cv.loc[row, "consts_se"] = json.dumps(r["se_bounded"])
    cv.loc[row, "consts_cov"] = json.dumps(None)                # the stored covariance belonged to the fit without the bound
    print(r["city"], [round(v, 4) for v in r["se_stored"]], "->", [None if v is None else round(v, 4) for v in r["se_bounded"]])
cv.to_parquet(RV / "cities_val.parquet", index=False)
m = RV / "theta2_bound_applied.json"
d = json.loads(m.read_text())
(d if isinstance(d, dict) else {}).update({"standard_errors": "replaced by amend_theta2_se.py (theta2_se.json)"})
if isinstance(d, dict):
    m.write_text(json.dumps(d, indent=1, ensure_ascii=False))
