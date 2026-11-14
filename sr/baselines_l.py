"""Traditional baselines and reference ladders under the SR evaluation (design §5.4, plan Task 1).

All models sit in the same fixed-budget conditional logit and are scored two ways:
  temporal : fitted on train x all destinations, scored on test x all destinations;
  dest-cv  : 5 outer destination folds (sr_data.folds); fold k fitted on train x destinations not in
             fold k, scored on test x fold-k destinations (renormalised within the fold); summed.
Models: grav_power (log d + log pop), grav_exp (d + log pop), rad (radiation, m = population + 1),
rad_N (m = facilities N + 1), ext_rad (Yang et al. 2014, alpha fitted), M2_glm, M3_glm (references for
the success bar). The +1 keeps zero-population destinations finite (they hold facilities and visits).
Output: results/sr/L/baselines_<R>_<scale>.json
Usage: OMP_NUM_THREADS=8 python -I baselines_l.py REGION SCALE
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sr_data import ROOT, folds, load  # noqa: E402
from choice_model import ChoiceModel, Design, bspline_basis  # noqa: E402
from run_gate_l import DIST_DF, features, standardise  # noqa: E402

OUT = ROOT / "results/sr/L"


def radiation_eta(m_i, m_j, S):
    return np.log(m_j)[None, :] - np.log(m_i[:, None] + S) - np.log(m_i[:, None] + m_j[None, :] + S)


def ext_radiation_eta(m_i, m_j, S, alpha):
    a = m_i[:, None] + S
    b = a + m_j[None, :]
    num = np.log(np.maximum(b ** alpha - a ** alpha, 1e-300)) + np.log(m_i ** alpha + 1)[:, None]
    return num - np.log(a ** alpha + 1) - np.log(b ** alpha + 1)


def score_eta(eta, Tm, avail):
    """Deviance, nats and CPC of predictions exp(eta) normalised over avail, with Tm's budgets."""
    Tm = np.where(avail, Tm, 0.0)
    e = np.where(avail, eta, -np.inf)
    m = np.max(np.where(avail, eta, -1e300), 1, keepdims=True)
    lp = np.where(avail, e - m - np.log(np.exp(e - m).sum(1, keepdims=True)), -np.inf)
    O = Tm.sum(1, keepdims=True)
    mu = O * np.exp(lp)
    pos = Tm > 0
    V = Tm.sum()
    nll = -np.sum(Tm[pos] * lp[pos])
    return {"visits": float(V), "nll_per_visit": float(nll / V),
            "deviance": float(2 * np.sum(Tm[pos] * np.log(Tm[pos] / mu[pos]))),
            "cpc": float(2 * np.minimum(Tm, mu).sum() / (Tm.sum() + mu.sum()))}


def main(region, scale):
    d = load(region, scale)
    u, D, T, dest, orig, groups, N = d["u"], d["D"], d["T"], d["dest"], d["orig"], d["groups"], d["N"]
    fold = folds(d)
    S = np.load(d["base"] / "S.npy").astype(np.float64)[orig]
    logd = np.log(np.maximum(D, 0.05))
    off = ~np.eye(N, dtype=bool) & dest[None, :] & dest[:, None]
    kn = np.quantile(logd[off], np.linspace(0, 1, DIST_DF - 1)[1:-1])
    Bd = bspline_basis(logd.ravel(), kn, lo=logd[off].min(), hi=logd[off].max()).reshape(N, N, -1)
    f0, fplus, ftot, fcomp = features(u)
    Xg, _ = standardise({"log_pop": f0["log_pop"]}, dest)
    Xm2, _ = standardise({k: v for b in (f0, fplus, ftot, fcomp) for k, v in b.items()}, dest)
    glm = {"grav_power": (logd[:, :, None], Xg, False), "grav_exp": (D[:, :, None], Xg, False),
           "M2_glm": (Bd, Xm2, False), "M3_glm": (Bd, Xm2, True)}
    pop1 = u["population"].to_numpy(float) + 1.0
    N1 = u["N"].to_numpy(float) + 1.0

    def avail_of(dm):
        return (dm[None, :] & ~np.eye(N, dtype=bool))[orig]

    def fit_glm(name, dm):
        B, X, gd = glm[name]
        des = Design(B=B, X=X, groups=groups if gd else np.zeros(N, int), n_groups=3 if gd else 1,
                     origin_mask=orig, dest_mask=dm)
        return ChoiceModel(des, group_distance=gd).fit(T["train"])

    def glm_eta(m):
        return m.eta(m.theta)[0]

    def fit_alpha(av):
        Ttr = T["train"][orig]
        f = lambda a: score_eta(ext_radiation_eta(pop1[orig], pop1, S, a), Ttr, av)["nll_per_visit"]  # noqa: E731
        r = minimize_scalar(f, bounds=(0.05, 3.0), method="bounded", options={"xatol": 1e-4})
        return float(r.x)

    res = {"region": region, "scale": scale, "temporal": {}, "dest_cv": {}, "fold_sizes": np.bincount(fold[dest]).tolist()}
    # temporal
    av = avail_of(dest)
    for name in glm:
        m = fit_glm(name, dest)
        res["temporal"][name] = {**score_eta(glm_eta(m), T["test"][orig], av), "n_params": int(len(m.theta))}
    res["temporal"]["rad"] = score_eta(radiation_eta(pop1[orig], pop1, S), T["test"][orig], av)
    res["temporal"]["rad_N"] = score_eta(radiation_eta(N1[orig], N1, S), T["test"][orig], av)
    a = fit_alpha(av)
    res["temporal"]["ext_rad"] = {**score_eta(ext_radiation_eta(pop1[orig], pop1, S, a), T["test"][orig], av), "alpha": a}
    print("temporal", {k: round(v["deviance"]) for k, v in res["temporal"].items()}, flush=True)
    # destination folds
    tot = {}
    for k in range(5):
        dtr, dte = dest & (fold != k), dest & (fold == k)
        av_tr, av_te = avail_of(dtr), avail_of(dte)
        etas = {name: glm_eta(fit_glm(name, dtr)) for name in glm}
        etas["rad"] = radiation_eta(pop1[orig], pop1, S)
        etas["rad_N"] = radiation_eta(N1[orig], N1, S)
        ak = fit_alpha(av_tr)
        etas["ext_rad"] = ext_radiation_eta(pop1[orig], pop1, S, ak)
        for name, e in etas.items():
            sc = score_eta(e, T["test"][orig], av_te)
            t = tot.setdefault(name, {"deviance": 0.0, "visits": 0.0, "folds": []})
            t["deviance"] += sc["deviance"]
            t["visits"] += sc["visits"]
            t["folds"].append(sc["deviance"])
        tot["ext_rad"].setdefault("alpha", []).append(ak)
        print("fold", k, {n: round(tot[n]["folds"][-1]) for n in tot}, flush=True)
    for name, t in tot.items():
        t["nats_vs_grav_power"] = (tot["grav_power"]["deviance"] - t["deviance"]) / (2 * t["visits"])
        t["rel_vs_M2_glm"] = (tot["M2_glm"]["deviance"] - t["deviance"]) / tot["M2_glm"]["deviance"]
    res["dest_cv"] = tot
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"baselines_{region}_{scale}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({n: {"dev": round(t["deviance"]), "rel_vs_M2": round(t["rel_vs_M2_glm"], 4)} for n, t in tot.items()}))


if __name__ == "__main__":
    main(*sys.argv[1:])
