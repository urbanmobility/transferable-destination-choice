"""Plan 9: the earlier models, each fitted by maximum likelihood in the same conditional logit as the skeletons
(plan-9 design: earlier models and transfer, §6.2). No result files are read or written here.

Every model comes "as published" (population as the mass, suffix _P) and with the mass replaced by the POI count
(suffix _N), so that "POI count as the mass" and "the shape of the formula" can be told apart.

Equation path (refit_select, the data rows of the skeleton fits; #1 d, #2 log d, #3 P_i, #4 P_j, #6 N_j):
  grav_power, grav_exp   classical gravity with a free mass exponent;
  S0                     the project's reference: log(N_j + c) - b log(d + d0);
  strexp1                Guo et al. 2026, w_j / exp(d^alpha), one parameter (d in km);
  strexp3                the same with a free mass exponent and scale (a relaxation, not the published form);
  cab_pow, cab_exp       Cabanas-Tirapu et al. 2025, the "power-law" and the "exponential" closed forms (the overall
                         factor of the power-law form cancels in a destination choice); _N swaps the destination mass.
Matrix path (origins x units arrays):
  rad, ext_rad (alpha), io (Schneider, L), pwo (Yan et al. 2014), ops (Liu & Yan 2019): _N uses the POI count at
  both ends and in the intervening opportunities, because the origin mass does not cancel in these models;
  kwon (Kwon et al. 2023): units in 10 equal-count groups by their traffic volume T_i (inflow + outflow of the
  training period), eta = A_kk' + log T_j - gamma_kk' log d, symmetric in the two groups: 55 exponents + 55
  intercepts, fitted here by maximum likelihood instead of the paper's per-pair OLS. It needs the city's own trips.
"""
import dataclasses
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401  (puts scripts/sr on the path)
import sr_data  # noqa: E402
from baselines_l import ext_radiation_eta, radiation_eta, score_eta  # noqa: E402
from refit_select import PENALTY, eta as eq_eta, loss as eq_loss, parse_equation, refit  # noqa: E402

EqSpec = namedtuple("EqSpec", "equation free mass original")
MatSpec = namedtuple("MatSpec", "family mass start bounds original needs_trips")
N_KW, N_PAIR = 10, 55
PERTURB_SD = 0.3
GRID = 25

_T, _F = True, False
_EQ = {     # name -> (equation with the population mass, free mask, published form?, has a _P version?)
    "grav_power": ("f = 1.0*log(#4 + 1.0) - 1.5*#2", (_T, _F, _T), True, True),
    "grav_exp": ("f = 1.0*log(#4 + 1.0) - 0.1*#1", (_T, _F, _T), True, True),
    "S0": ("f = log(#4 + 10.0) - 1.5*log(#1 + 1.0)", (_T, _T, _T), False, False),
    "strexp1": ("f = log(#4 + 1.0) - #1^0.45", (_F, _T), True, True),
    "strexp3": ("f = 1.0*log(#4 + 1.0) - 1.0*#1^0.45", (_T, _F, _T, _T), False, True),
    "cab_pow": ("f = 1.0*log(exp(-14.0)*(#3*#4 + exp(5.0)*#4 + exp(16.0))/#1^1.7 + 1.0)", (_T, _T, _T, _T, _T, _F), True, True),
    "cab_exp": ("f = 4.0*(1.0 + exp(-8.0)*((#4 + exp(5.0))*(#3 + exp(9.0)))^0.54/#1)^0.26",
                (_T, _F, _T, _T, _T, _T, _T), True, True),
}
# POI-count versions that need their own start: the same formula, numerals at the scale of POI counts (source-city
# audit 2026-11-11: from the population-scaled start cab_exp_N stopped 0.02-0.04 nats above plain gravity in 4 cities)
_EQ_N_START = {"cab_exp": "f = 4.0*(1.0 + exp(-5.5)*((#4 + exp(2.0))*(#3 + exp(9.0)))^0.54/#1)^0.26"}
EQ_MODELS = {}
for _k, (_eq, _free, _orig, _has_p) in _EQ.items():
    if _has_p:
        EQ_MODELS[f"{_k}_P"] = EqSpec(_eq, _free, "P", _orig)
    EQ_MODELS[f"{_k}_N"] = EqSpec(_EQ_N_START.get(_k, _eq).replace("#4", "#6"), _free, "N", False)

MAT_MODELS = {}
for _k, _start, _bounds in (("rad", [], None), ("ext_rad", [0.5], [(0.01, 5.0)]),
                            ("io", [float(np.log(1e-5))], [(float(np.log(1e-9)), float(np.log(10.0)))]),
                            ("pwo", [], None), ("ops", [], None)):
    for _m in ("P", "N"):
        MAT_MODELS[f"{_k}_{_m}"] = MatSpec(_k, _m, list(_start), _bounds, _m == "P", False)
MAT_MODELS["kwon"] = MatSpec("kwon", None, [1.0] * N_PAIR + [0.0] * N_PAIR, None, True, True)


# --------------------------------------------------------------------------------------------- scoring
def score_rows(e, data) -> dict:
    """Per-visit NLL and CPC of eta on long-format rows (grouped by origin); nan when eta is not finite."""
    e = np.asarray(e, float)
    if not np.isfinite(e).all():
        return {"nll": np.nan, "cpc": np.nan}
    st, rep = data.offsets[:-1], np.diff(data.offsets)
    ce = e - np.repeat(np.maximum.reduceat(e, st), rep)
    lp = ce - np.repeat(np.log(np.add.reduceat(np.exp(ce), st)), rep)
    mu = np.repeat(np.add.reduceat(data.y, st), rep) * np.exp(lp)
    V = data.y.sum()
    return {"nll": float(-(data.y * lp).sum() / V), "cpc": float(2 * np.minimum(data.y, mu).sum() / (V + mu.sum()))}


def score_cand(c, consts, data) -> dict:
    """score_rows of a parsed equation (a refit_select Candidate) with the given constants."""
    with np.errstate(all="ignore"):
        return score_rows(eq_eta(c, data, consts=np.asarray(consts, float)), data)


def score_eq(equation: str, consts, data) -> dict:
    return score_cand(parse_equation(equation, None, "A", 9), consts, data)


def score_mat(e, T, av) -> dict:
    with np.errstate(all="ignore"):
        if not np.isfinite(e[av]).all():
            return {"nll": np.nan, "cpc": np.nan}
        s = score_eta(e, T, av)
    return {"nll": s["nll_per_visit"], "cpc": s["cpc"]}


# --------------------------------------------------------------------------------------------- equation path
def fit_eq(equation: str, free, tr, starts: int = 5, seed: int = 0):
    """Multi-start refit of the free numerals on tr -> (training NLL, constants) of the best start; None when no
    start gives a finite loss. Start 0 is the equation's own numerals; the others perturb the free ones."""
    c0 = parse_equation(equation, None, "A", 9)
    free = np.asarray(free, bool)
    if len(free) != len(c0.consts):
        raise ValueError(f"free mask of length {len(free)} for {len(c0.consts)} numerals: {equation}")
    rng, best = np.random.default_rng(seed), None
    for k in range(starts):
        s = c0.consts.copy()
        if k:
            s[free] = s[free] * np.exp(rng.normal(0.0, PERTURB_SD, int(free.sum())))
        cs = dataclasses.replace(c0, consts=s)
        with np.errstate(all="ignore"):
            if eq_loss(cs, tr) >= PENALTY:
                continue
            r = refit(cs, tr, free=free)
            lt = eq_loss(r, tr)
        if lt < PENALTY and (best is None or lt < best[0]):
            best = (float(lt), [float(v) for v in r.consts])
    return best


# --------------------------------------------------------------------------------------------- matrix path
def dest_centred(D, mass):
    """SJ[i, j] = mass within distance D[i, j] of j (j and i included): the PWO model's S_ji."""
    N = len(mass)
    SJ = np.empty((N, N))
    for j in range(N):
        order = np.argsort(D[j], kind="stable")
        cum = np.cumsum(mass[order])
        SJ[:, j] = cum[np.searchsorted(D[j][order], D[:, j], side="right") - 1]
    return SJ


def kwon_groups(vol, dest):
    """Group 0..9 per unit: the choice-set units in 10 equal-count groups by ascending volume; 0 outside."""
    g = np.zeros(len(vol), int)
    idx = np.flatnonzero(dest)
    g[idx[np.argsort(vol[idx], kind="stable")]] = (N_KW * np.arange(len(idx))) // len(idx)
    return g


def kwon_pairs(g):
    """Symmetric pair index 0..54 of the groups of the two ends."""
    lo, hi = np.minimum(g[:, None], g[None, :]), np.maximum(g[:, None], g[None, :])
    return lo * N_KW - lo * (lo - 1) // 2 + (hi - lo)


def city_arrays(d: dict, fold, eval_split: str, eval_block: str, S_pop, n_orig: int = 300, seed: int = 0) -> dict:
    """The matrix-path arrays of a loaded city: training block = train x search on n_orig sampled origins, scoring
    block = eval_split x eval_block on every origin. S_pop = the city's S.npy (population strictly closer)."""
    from form_vars import intervening
    N, u, D, T = d["N"], d["u"], d["D"], d["T"]
    eye = np.eye(N, dtype=bool)
    io_tr, io_ev = sr_data.sample_origins(d, n_orig, seed), np.flatnonzero(d["orig"])
    ds, de = sr_data.dest_mask(d, fold, "search"), sr_data.dest_mask(d, fold, eval_block)
    pop, poi = u["population"].to_numpy(float), u["N"].to_numpy(float)
    mass = {"P": pop + 1.0, "N": poi + 1.0}
    vol = T["train"].sum(0) + T["train"].sum(1)                  # Kwon et al.: T_i = inflow + outflow
    return {"D": D, "logd": np.log(np.maximum(D, 0.05)), "mass": mass,
            "S": {"P": np.asarray(S_pop, float), "N": intervening(D, poi)},
            "SJ": {m: dest_centred(D, mass[m]) for m in mass},
            "kw_g": kwon_pairs(kwon_groups(vol, d["dest"])), "logvol": np.log(vol + 1.0),
            "io": {"tr": io_tr, "ev": io_ev},
            "av": {"tr": (ds[None, :] & ~eye)[io_tr], "ev": (de[None, :] & ~eye)[io_ev]},
            "T": {"tr": T["train"][io_tr], "ev": T[eval_split][io_ev]}}


def mat_eta(name: str, th, A: dict, io) -> np.ndarray:
    """Utilities of the origins io (rows) for every unit (columns)."""
    s = MAT_MODELS[name]
    if s.family == "kwon":
        th = np.asarray(th, float)
        G = A["kw_g"][io]
        return A["logvol"][None, :] + th[N_PAIR:][G] - th[:N_PAIR][G] * A["logd"][io]
    m, S = A["mass"][s.mass], A["S"][s.mass][io]
    with np.errstate(all="ignore"):
        if s.family == "rad":
            return radiation_eta(m[io], m, S)
        if s.family == "ext_rad":
            if not th[0] > 0:                                    # only defined for alpha > 0 (a predicted alpha can
                return np.full(S.shape, np.nan)                  # leave the domain: a failure, not a flat model)
            return ext_radiation_eta(m[io], m, S, th[0])
        if s.family == "io":                                     # Schneider: exp(-L s) - exp(-L (s + m_j))
            L = np.exp(th[0])
            return -L * S + np.log(-np.expm1(-L * m))[None, :]
        if s.family == "pwo":                                    # Yan et al. 2014: m_j (1 / S_ji - 1 / M)
            return np.log(m)[None, :] + np.log(np.maximum(1.0 / A["SJ"][s.mass][io] - 1.0 / m.sum(), 1e-300))
        if s.family == "ops":                                    # Liu & Yan 2019: m_j / (m_i + s_ij + m_j)
            return np.log(m)[None, :] - np.log(m[io][:, None] + S + m[None, :])
    raise ValueError(name)


def kwon_objective(th, A: dict) -> tuple:
    """Training NLL per visit of the Kwon model and its gradient (55 exponents, then 55 intercepts)."""
    io, av = A["io"]["tr"], A["av"]["tr"]
    G, ld = A["kw_g"][io][av], A["logd"][io][av]
    Tm = np.where(av, A["T"]["tr"], 0.0)
    O, V = Tm.sum(1, keepdims=True), Tm.sum()
    e = np.where(av, mat_eta("kwon", th, A, io), -np.inf)
    m = e.max(1, keepdims=True)
    lp = e - m - np.log(np.exp(e - m).sum(1, keepdims=True))
    R = (Tm - O * np.exp(lp))[av]
    g = np.r_[np.bincount(G, weights=R * ld, minlength=N_PAIR), -np.bincount(G, weights=R, minlength=N_PAIR)] / V
    return float(-(Tm[av] * lp[av]).sum() / V), g


def fit_mat(name: str, A: dict) -> list:
    """Maximum-likelihood parameters on the training block: [] without parameters; one parameter by a grid of GRID
    points over the bounds refined between the best point's neighbours; Kwon by L-BFGS-B with the analytic gradient."""
    s = MAT_MODELS[name]
    if not s.start:
        return []
    if s.family == "kwon":
        r = minimize(kwon_objective, np.asarray(s.start, float), args=(A,), jac=True, method="L-BFGS-B",
                     options={"maxiter": 1000})
        return [float(v) for v in r.x]

    def obj(t):
        v = score_mat(mat_eta(name, [t], A, A["io"]["tr"]), A["T"]["tr"], A["av"]["tr"])["nll"]
        return v if np.isfinite(v) else PENALTY
    grid = np.linspace(*s.bounds[0], GRID)
    k = int(np.argmin([obj(t) for t in grid]))
    r = minimize_scalar(obj, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, GRID - 1)]), method="bounded")
    return [float(r.x if obj(r.x) <= obj(grid[k]) else grid[k])]
