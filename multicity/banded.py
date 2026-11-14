"""Distance-band diagnostics for the pre-freeze skeleton comparison (plan 5 Tasks 4-5): the
families agree at 5 km and differ in how steep the decay is near the origin and in whether the mass
elasticity rises with distance.

band_refs(): per distance band, the medians of every variable over the band's rows (reference point) and the city's
  log(S_ij + 1)-on-log d slope (as laws.reference_points).
skeleton_profile(): per band, (e_d, e_N) of a skeleton at the band's reference point: e_d = d eta / d log d along
  S_ij(d) (laws._elasticity_expr), e_N = N_j d eta / d N_j.
"""
import math
import sys
from pathlib import Path

import numpy as np
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
from refit_select import XS, parse_equation  # noqa: E402
from sr_data import VARS  # noqa: E402

BANDS = ((0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, math.inf))     # km, [lo, hi)
_N = XS[VARS.index("N_j")]


def band_refs(tr) -> dict:
    """{band: reference dict or None (no row)}; reference = medians of VARS over the band's rows, logd = log of the
    median d, dlogS_dlogd = the city's OLS slope over every row, n_rows."""
    X = np.asarray(tr.X, float)
    d = X[:, VARS.index("d")]
    slope = float(np.polyfit(np.log(np.maximum(d, 0.05)), np.log(X[:, VARS.index("S_ij")] + 1.0), 1)[0])
    out = {}
    for b, (lo, hi) in enumerate(BANDS):
        m = (d >= lo) & (d < hi)
        if not m.any():
            out[b] = None
            continue
        r = {v: float(np.median(X[m, k])) for k, v in enumerate(VARS)}
        r.update(logd=float(np.log(r["d"])), dlogS_dlogd=slope, n_rows=int(m.sum()))
        out[b] = r
    return out


def _profile_fns(x):
    import laws
    from library import to_single
    c = to_single(parse_equation(x["equation"], x.get("template"), x.get("variant", "A"), int(x["complexity"])))
    e = c.comps["f"]
    if "g" in c.comps:
        e = e + sympy.Symbol(f"c{c.n_lit}") * c.comps["g"] + sympy.Symbol(f"c{c.n_lit + 1}") * c.comps["h"]
    args = list(XS) + [sympy.Symbol(f"c{i}") for i in range(len(c.consts))] + [sympy.Symbol("slope")]
    fd = sympy.lambdify(args, laws._elasticity_expr(c), "numpy")
    fn = sympy.lambdify(args, _N * sympy.diff(e, _N), "numpy")
    return fd, fn, len(c.consts)


def skeleton_profile(x: dict, consts, refs_b: dict) -> dict:
    """{band: (e_d, e_N) or None} at each band's reference point under the given constants."""
    fd, fn, n = _profile_fns(x)
    th = [float(v) for v in consts]
    if len(th) != n:
        raise ValueError(f"{x['cid']}: {len(th)} constants for {n}")
    out = {}
    for b, ref in refs_b.items():
        if ref is None:
            out[b] = None
            continue
        xs = [float(ref.get(v, 0.0)) for v in VARS]
        with np.errstate(all="ignore"):
            out[b] = (float(fd(*xs, *th, float(ref["dlogS_dlogd"]))), float(fn(*xs, *th, float(ref["dlogS_dlogd"]))))
    return out


# ------------------------------------------------------------------------------------------ the flexible model (Task 5)
H_LOGD = 1e-4


def distance_spline(D, dest):
    """The M2 distance basis of val_refs.refs: B-spline in log max(d, 0.05), DIST_DF - 1 columns, interior knots at
    quantiles of the off-diagonal destination pairs. Returns (Bd (N, N, nb), (knots, lo, hi))."""
    from choice_model import bspline_basis
    from run_gate_l import DIST_DF
    N = len(D)
    logd = np.log(np.maximum(D, 0.05))
    off = ~np.eye(N, dtype=bool) & dest[None, :] & dest[:, None]
    kn = np.quantile(logd[off], np.linspace(0, 1, DIST_DF - 1)[1:-1])
    lo, hi = float(logd[off].min()), float(logd[off].max())
    return bspline_basis(logd.ravel(), kn, lo=lo, hi=hi).reshape(N, N, -1), (kn, lo, hi)


def fit_banded(Bd, X, D, logN, T_train, orig, dest_train, knots) -> dict:
    """M2-type conditional logit (distance spline Bd, destination features X) plus one pair feature per distance
    band, 1[d_ij in b] * (log1p(N_j) - its mean over dest_train): gamma_b is the band's d eta / d log1p(N_j). Centring
    keeps the band indicator itself out of Z (otherwise each gamma also shifts the band's distance level, which the
    smooth spline cannot follow). Bands with no (origin, available destination) pair are dropped -> None."""
    from choice_model import ChoiceModel, Design
    N = len(D)
    ln = np.asarray(logN, float) - float(np.mean(np.asarray(logN, float)[dest_train]))
    avail = (dest_train[None, :] & ~np.eye(N, dtype=bool))[orig]
    cols, keep = [], []
    for b, (lo, hi) in enumerate(BANDS):
        m = (D >= lo) & (D < hi)
        if (m[orig] & avail).any():
            cols.append(m * ln[None, :])
            keep.append(b)
    Z = np.stack(cols, axis=2) if cols else None
    des = Design(B=Bd, X=X, groups=np.zeros(N, int), n_groups=1, origin_mask=orig, dest_mask=dest_train, Z=Z)
    m = ChoiceModel(des).fit(T_train)
    s, _, _ = m.unpack(m.theta)
    z = m.extras(m.theta)[2]
    gamma = {b: None for b in range(len(BANDS))}
    gamma.update({b: float(v) for b, v in zip(keep, z)})
    return {"gamma": gamma, "spline_coef": [float(v) for v in s], "knots": knots, "train_nll": float(m.result.fun),
            "converged": bool(m.result.success), "model": m}


def flex_profile(fit: dict, refs_b: dict) -> dict:
    """{band: (e_d, e_N) or None}: e_d = derivative of the distance spline at log of the band's reference distance
    (central difference), e_N = gamma_b * N / (N + 1) at the band's reference N_j."""
    from choice_model import bspline_basis
    kn, lo, hi = fit["knots"]
    coef = np.asarray(fit["spline_coef"], float)
    out = {}
    for b, ref in refs_b.items():
        g = fit["gamma"].get(b)
        if ref is None or g is None:
            out[b] = None
            continue
        x = float(np.log(max(ref["d"], 0.05)))
        B = bspline_basis(np.array([x - H_LOGD, x + H_LOGD]), np.asarray(kn, float), lo=lo, hi=hi)
        s = B @ coef
        n = float(ref["N_j"])
        out[b] = (float((s[1] - s[0]) / (2 * H_LOGD)), float(g * n / (n + 1.0)))
    return out


def city_banded(city: str, out_dir=None, allowed=None, allow_confirm: bool = False) -> dict:
    """The flexible model of one search / discovery city: M2's design without log_N (the per-band terms carry it),
    fitted on train x search block; validation NLL on val x validation block (beside M2's from refs/<city>.json);
    band reference points from the training rows of 50 sampled origins; training visits per band. Only T["train"]
    and T["val"] are used. out_dir: also written to <out_dir>/<city>.json. allowed: the cities this call may read
    (default prefreeze.cities(); lawtest cities only for registered hypotheses, lawtest_extra.py); a confirmation
    city is always refused."""
    import p7seal
    p7seal.guard_line(p7seal.current_line())         # plan 7: sealed data lines need the signed registration
    import json
    import config
    import prefreeze
    import sr_data
    sealed = set(config.CONFIRM) | set(config.cities_with_role("confirm"))
    ok = prefreeze.cities() if allowed is None else list(allowed)
    if allow_confirm:                      # plan 6: confirmation cities only after the freeze
        import frozen
        frozen.require()
        sealed = set()
    if city in sealed or city not in ok:
        raise ValueError(f"{city} is outside the allowed cities of this call (confirmation cities never; default: "
                         "search + discovery, plan 5 data boundary)")
    from baselines_l import score_eta
    from run_gate_l import features, standardise
    d = sr_data.load(city, "tract")
    u, D, T, dest, orig, N = d["u"], d["D"], d["T"], d["dest"], d["orig"], d["N"]
    f = sr_data.folds(d)
    ds, dv = sr_data.dest_mask(d, f, "search"), sr_data.dest_mask(d, f, "val")
    Bd, knots = distance_spline(D, dest)
    f0, fplus, _, fcomp = features(u, transit_col="n_transit_osm")
    X, _ = standardise({k: v for blk in (f0, fplus, fcomp) for k, v in blk.items()}, dest)
    fit = fit_banded(Bd, X, D, np.log1p(u["N"].to_numpy(float)), T["train"], orig, ds, knots)
    m = fit.pop("model")
    eye = np.eye(N, dtype=bool)
    val = score_eta(m.eta(m.theta)[0], T["val"][orig], (dv[None, :] & ~eye)[orig])["nll_per_visit"]
    Tt, Do = np.where((ds[None, :] & ~eye)[orig], T["train"][orig], 0.0), D[orig]
    visits = {b: float(Tt[(Do >= lo) & (Do < hi)].sum()) for b, (lo, hi) in enumerate(BANDS)}
    refs = band_refs(sr_data.build([city], "tract", "train", "search", 50, 0))
    m2 = json.loads((config.RESULTS / f"refs/{city}.json").read_text())["models"].get("M2_glm", {})
    kn, lo, hi = knots
    out = {"city": city, "gamma": fit["gamma"], "profile": flex_profile(fit, refs), "refs": refs, "visits": visits,
           "train_nll": fit["train_nll"], "val_nll": float(val), "m2_val_nll": m2.get("nll_per_visit"),
           "converged": fit["converged"], "spline_coef": fit["spline_coef"], "knots": [list(map(float, kn)), lo, hi]}
    if out_dir is not None:
        from pathlib import Path
        p = Path(out_dir) / f"{city}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
    return out
