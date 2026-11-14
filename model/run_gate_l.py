"""Gate-2 ladder on the L line (plan §4.3, §7.4): gravity baselines, M0, M0+, M1, M2, M3, M4lambda,
GLM and GAM, fitted on the training split, scored on validation and test with each split's
own budgets. Origins: included (>= 30 home devices) with a known SES tertile; choice set: all
units except the origin and units with neither population nor facilities.

Usage: python -I run_gate_l.py REGION SCALE [COUNT_COL] [SNAPSHOT_SUFFIX]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from choice_model import ChoiceModel, Design, bspline_basis  # noqa: E402

ROOT = Path(os.environ.get("ODSR_ROOT", Path(__file__).resolve().parents[2]))
CLASSES = ("food", "retail", "health", "education", "leisure", "service", "office", "transit")
REF_CLASS = "retail"
DIST_DF = 6
GAM_DF = 4
RIDGES = (1e-5, 1e-4, 1e-3)  # per-visit log-loss scale


def present_classes(u, sfx=""):
    """Facility classes with at least one POI in the table. A class with none (Veraset carries no NAICS 62 health
    places) is left out of the composition: its smoothed share would be 0 and its alr -inf. Round 1 (all 8 classes
    present) is unchanged; the multi-city references record the list per city."""
    n_all = u[[f"n_{c}{sfx}" for c in CLASSES]].to_numpy(float)
    return [c for c, tot in zip(CLASSES, n_all.sum(0)) if tot > 0]


def features(u, sfx="", transit_col=None):
    """transit_col: column for the transit-stop count of M2 (multi-city: OSM stops in n_transit_osm, while the
    8 composition classes come from the Veraset POIs); default n_transit<sfx> as in round 1."""
    f = {}
    f["log_pop"] = np.log1p(u["population"])
    f["log_area"] = np.log(u["area_km2"] + 0.01)
    plus = {"log_jobs": np.log1p(u["jobs"]), "log_road_km": np.log1p(u["road_km"]),
            "log_pop_potential": np.log1p(u["pop_potential"]),
            "log_transit": np.log1p(u[transit_col or f"n_transit{sfx}"])}
    total = {"log_N": np.log1p(u[f"N{sfx}"])}
    present = present_classes(u, sfx)
    assert REF_CLASS in present, f"reference class {REF_CLASS} has no POI"
    n = u[[f"n_{c}{sfx}" for c in present]].to_numpy(float)
    pbar = n.sum(0) / n.sum()
    sh = (n + pbar) / (n.sum(1, keepdims=True) + 1.0)          # Dirichlet smoothing, alpha0 = 1
    comp = {}
    ref = present.index(REF_CLASS)
    for k, c in enumerate(present):
        if k != ref:
            comp[f"alr_{c}"] = np.log(sh[:, k] / sh[:, ref])
    comp["richness"] = (n > 0).sum(1).astype(float)
    comp["entropy"] = -(sh * np.log(sh)).sum(1)
    comp["log_park_share"] = np.log(u[f"park_share{sfx}"] + 1e-3)
    return f, plus, total, comp


def standardise(cols, mask):
    M = np.column_stack(list(cols.values())).astype(float)
    mu, sd = M[mask].mean(0), M[mask].std(0)
    sd[sd == 0] = 1
    return (M - mu) / sd, list(cols)


def gam_expand(M, mask):
    out, pen = [], []
    for j in range(M.shape[1]):
        x = M[:, j]
        kn = np.quantile(x[mask], np.linspace(0, 1, GAM_DF)[1:-1])
        B = bspline_basis(x, kn, lo=x[mask].min(), hi=x[mask].max())
        B = B - B[mask].mean(0)
        out.append(np.column_stack([x, B]))
        pen.extend([0.0] + [1.0] * B.shape[1])                  # linear term unpenalised
    return np.column_stack(out), np.array(pen)


def main(region, scale, count_col="n_dev_5", sfx=""):
    t_start = time.time()
    base = ROOT / f"derived/model_inputs/L/{region}/{scale}"
    u = pd.read_parquet(base / "units.parquet")
    D = np.load(base / "D.npy").astype(np.float64)
    T = {sp: np.load(base / f"T_{sp}.npz")[count_col].astype(np.float64) for sp in ("train", "val", "test")}
    N = len(u)
    dest = ~((u["population"] == 0) & (u[f"N{sfx}"] == 0)).to_numpy()
    orig = (u["origin_included"].fillna(False).astype(bool) & (u["ses"] > 0) & dest).to_numpy()
    groups = np.clip(u["ses"].to_numpy() - 1, 0, None)

    logd = np.log(np.maximum(D, 0.05))
    off = ~np.eye(N, dtype=bool) & dest[None, :] & dest[:, None]
    kn = np.quantile(logd[off], np.linspace(0, 1, DIST_DF - 1)[1:-1])
    Bd = bspline_basis(logd.ravel(), kn, lo=logd[off].min(), hi=logd[off].max()).reshape(N, N, -1)

    f0, fplus, ftot, fcomp = features(u, sfx)
    ladders = {"M0": [f0], "M0+": [f0, fplus], "M1": [f0, fplus, ftot], "M2": [f0, fplus, ftot, fcomp]}
    results, fitted = {}, {}

    def run(name, B, X, cols, n_groups=1, gd=False, gs=False, ridge=0.0, penalty=None, th0=None):
        des = Design(B=B, X=X, groups=groups if n_groups > 1 else np.zeros(N, int), n_groups=n_groups,
                     origin_mask=orig, dest_mask=dest, penalty=penalty)
        m = ChoiceModel(des, group_distance=gd, group_scale=gs, ridge=ridge).fit(T["train"], th0=th0)
        r = {"converged": bool(m.result.success), "iters": int(m.result.nit), "n_params": int(len(m.theta)),
             "train_objective_per_visit": float(m.result.fun)}
        for sp in ("val", "test"):
            r[sp] = m.evaluate(T[sp])
        if gs:
            r["lambda"] = m.unpack(m.theta)[2].tolist()
        r["coef"] = dict(zip(cols, np.round(m.unpack(m.theta)[1][:len(cols)], 4).tolist())) if len(cols) == X.shape[1] else None
        results[name] = r
        fitted[name] = m
        print(f"{name:14s} val nll {r['val']['nll_per_visit']:.5f} dev {r['val']['deviance']:.1f} "
              f"test dev {r['test']['deviance']:.1f} cpc {r['test']['cpc']:.4f} it {r['iters']} ok {r['converged']}", flush=True)
        return m

    # gravity baselines: mass = log population
    Xg, cg = standardise({"log_pop": f0["log_pop"]}, dest)
    run("grav_power", logd[:, :, None], Xg, cg)
    run("grav_exp", D[:, :, None], Xg, cg)

    # GLM ladder
    for name, blocks in ladders.items():
        cols = {k: v for b in blocks for k, v in b.items()}
        X, cn = standardise(cols, dest)
        run(f"{name}_glm", Bd, X, cn)
    Xm2, cm2 = standardise({k: v for b in ladders["M2"] for k, v in b.items()}, dest)
    nb = Bd.shape[2]
    warm3 = lambda th: np.r_[np.tile(th[:nb], 3), th[nb:]]   # M2 solution -> M3 start (nested)
    run("M3_glm", Bd, Xm2, cm2, n_groups=3, gd=True, th0=warm3(fitted["M2_glm"].theta))
    run("M4lam_glm", Bd, Xm2, cm2, n_groups=3, gd=True, gs=True,
        th0=np.r_[fitted["M3_glm"].theta, np.zeros(2)])

    # GAM ladder: ridge chosen on validation NLL per model
    for name, blocks in list(ladders.items()) + [("M3", ladders["M2"]), ("M4lam", ladders["M2"])]:
        cols = {k: v for b in blocks for k, v in b.items()}
        X, cn = standardise(cols, dest)
        Xe, pen = gam_expand(X, dest)
        best = None
        for lam in RIDGES:
            gd = name in ("M3", "M4lam")
            gsc = name == "M4lam"
            key = f"{name}_gam_r{lam}"
            th0 = None
            if name == "M3":
                th0 = warm3(fitted[f"M2_gam_r{lam}"].theta)
            elif name == "M4lam":
                th0 = np.r_[fitted[f"M3_gam_r{lam}"].theta, np.zeros(2)]
            run(key, Bd, Xe, [f"{c}" for c in cn], n_groups=3 if gd else 1, gd=gd, gs=gsc, ridge=lam, penalty=pen, th0=th0)
            if best is None or results[key]["val"]["nll_per_visit"] < results[best]["val"]["nll_per_visit"]:
                best = key
        results[f"{name}_gam"] = {**results[best], "ridge_selected": best}

    def gain(a, b, sp):
        da, db = results[a][sp]["deviance"], results[b][sp]["deviance"]
        return {"rel_dev_gain": (da - db) / da, "logscore_gain_nats_per_visit": (da - db) / (2 * results[a][sp]["visits"])}

    gates = {}
    for impl in ("glm", "gam"):
        for sp in ("val", "test"):
            gates[f"{impl}_{sp}"] = {
                "M1_vs_M0+ (total)": gain(f"M0+_{impl}", f"M1_{impl}", sp),
                "M2_vs_M1 (composition; threshold 2%)": gain(f"M1_{impl}", f"M2_{impl}", sp),
                "M3_vs_M2 (group distance)": gain(f"M2_{impl}", f"M3_{impl}", sp),
                "M4lam_vs_M3 (scale)": gain(f"M3_{impl}", f"M4lam_{impl}", sp),
            }
    summary = {"region": region, "scale": scale, "count": count_col, "snapshot": sfx or "ca190101",
               "units": int(N), "origins": int(orig.sum()), "choice_set": int(dest.sum()),
               "visits": {sp: float(T[sp][orig].sum()) for sp in T},
               "share_visits_outside_choice_set": {sp: float(T[sp][orig][:, ~dest].sum() / T[sp][orig].sum()) for sp in T},
               "dist_df": DIST_DF, "gam_df": GAM_DF,
               "gates": gates, "models": results, "seconds": round(time.time() - t_start)}
    out = ROOT / f"results/gate2/L/{region}_{scale}_{count_col}{sfx}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(gates, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])
