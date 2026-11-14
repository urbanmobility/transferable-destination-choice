"""Reference scores on the SR selection split (val x validation destinations), plan Task 4 Step 3.

grav_power, grav_exp, M2_glm and M3_glm fitted on train x search destinations (all origins) and scored
on val x validation-block destinations with the same per-visit NLL as refit_select.loss.
Output: results/sr/L/valrefs_<scale>.json   Usage: python -I val_refs.py SCALE REGION [REGION ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines_l import score_eta  # noqa: E402
from sr_data import ROOT, dest_mask, folds, load  # noqa: E402
from choice_model import ChoiceModel, Design, bspline_basis  # noqa: E402
from run_gate_l import DIST_DF, features, present_classes, standardise  # noqa: E402


def refs(region, scale, transit_col=None, train_block="search", eval_split="val", eval_block="val", models=None):
    """transit_col: passed to run_gate_l.features (multi-city M2 uses the OSM stop count n_transit_osm).
    train_block / eval_split / eval_block: fitted on train x train_block, scored on eval_split x eval_block (default:
    the validation scoring; multi-city plan 4 inner loop: search23 / train / fold:4). models: subset of the four."""
    d = load(region, scale)
    u, D, T, dest, orig, groups, N = d["u"], d["D"], d["T"], d["dest"], d["orig"], d["groups"], d["N"]
    f = folds(d)
    ds, dv = dest_mask(d, f, train_block), dest_mask(d, f, eval_block)
    logd = np.log(np.maximum(D, 0.05))
    off = ~np.eye(N, dtype=bool) & dest[None, :] & dest[:, None]
    kn = np.quantile(logd[off], np.linspace(0, 1, DIST_DF - 1)[1:-1])
    Bd = bspline_basis(logd.ravel(), kn, lo=logd[off].min(), hi=logd[off].max()).reshape(N, N, -1)
    f0, fplus, ftot, fcomp = features(u, transit_col=transit_col)
    Xg, _ = standardise({"log_pop": f0["log_pop"]}, dest)
    Xm2, _ = standardise({k: v for b in (f0, fplus, ftot, fcomp) for k, v in b.items()}, dest)
    spec = {"grav_power": (logd[:, :, None], Xg, False), "grav_exp": (D[:, :, None], Xg, False),
            "M2_glm": (Bd, Xm2, False), "M3_glm": (Bd, Xm2, True)}
    av = (dv[None, :] & ~np.eye(N, dtype=bool))[orig]
    out = {}
    for name, (B, X, gd) in spec.items():
        if models is not None and name not in models:
            continue
        des = Design(B=B, X=X, groups=groups if gd else np.zeros(N, int), n_groups=3 if gd else 1,
                     origin_mask=orig, dest_mask=ds)
        m = ChoiceModel(des, group_distance=gd).fit(T["train"])
        out[name] = {**score_eta(m.eta(m.theta)[0], T[eval_split][orig], av), "n_params": int(len(m.theta))}
        if name in ("M2_glm", "M3_glm"):
            out[name]["classes"] = present_classes(u)      # the class set behind the composition features
    return out


if __name__ == "__main__":
    scale, regions = sys.argv[1], sys.argv[2:]
    res = {r: refs(r, scale) for r in regions}
    tot = {}
    for r in res.values():
        for k, v in r.items():
            t = tot.setdefault(k, [0.0, 0.0])
            t[0] += v["nll_per_visit"] * v["visits"]
            t[1] += v["visits"]
    res["pooled_nll_per_visit"] = {k: a / b for k, (a, b) in tot.items()}
    (ROOT / f"results/sr/L/valrefs_{scale}_{'_'.join(regions)}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["pooled_nll_per_visit"], indent=1))
