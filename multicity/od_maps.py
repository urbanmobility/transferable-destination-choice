"""Exploratory illustration (2026-11-12): observed against predicted OD flows in one city.

Why: the paper is about estimating OD flows, and no figure shows a flow. This is a first look, drawn in
several ways (largest flows only, errors by destination, ...), to see whether the difference from the earlier models
can be seen at all. Whether it becomes a figure is decided afterwards.

Fixed here before anything is computed, so that the picture is not chosen after seeing it:
  city     Denver (19740): the example city of Fig. 1b and a confirmation city. No other city is looked at.
  data     test period x all destinations, every origin — the rows the all-destination check has already scored.
  flows    T_i * P(j|i), with T_i the origin's observed trips of the test period (as in the CPC of every score).
  display  the uniform constants (no Denver trip enters them). The local fits are exported beside them, for reference.
  models   c0355r1 (ours), strexp3_N (the strongest earlier model), grav_power_N, grav_power_P (classic gravity).
Nothing is refitted and no test is run: this is an illustration, not a result.

Second export (2026-11-12, after the first drafts): the comparison should be with the earlier models as they
were published, not with our relaxed stretched exponential, and each of them also with the POI count as the mass.
Fixed before the second run: every earlier model of plan 9 that needs no local trips is exported in both versions
(_P = population, the published setting; _N = POI count); the draft shows the three best known — power gravity,
radiation, and the stretched exponential of Guo et al. — and a sheet with all of them. Kwon's model is left out: its
mass is the city's own trip volume, so it has no version without local trips.

Third export (2026-11-12, after Denver became Fig. 6): the same for the other five confirmation cities, for
the supplement. Fixed before the run: all six confirmation cities and no other; the same data, constants, models and
classes; every city is drawn in the same way and all of them are shown, whatever they look like. The outputs now sit
in one folder per city, with the simplified tract outlines (public Census geometry) beside them.

Only classes and aggregates leave the server (the raw data are licensed), as for Fig. 1b:
  pairs.csv    OD pairs among the 2000 largest of any source: nested rank tiers (top 100 / 200 / 500 / 1000 / 2000)
  dest.csv     per destination: quintile of the observed arrivals, class of log2(predicted / observed arrivals)
  home.csv     the home tract of Fig. 1b: rank thirds of its 12 largest destinations, observed and predicted
  hist.csv     two-dimensional histogram of log10 observed against log10 predicted flow (pair counts)
  calib.csv    by class of predicted flow: mean and quantiles of the observed flow (classes with >= 20 pairs)
  dist.csv     share of trips by distance class, observed and predicted
  overlap.csv  for K = 100 ... 5000: how many of the K largest observed flows are among the K largest predicted,
               and the share of the observed trips carried by the K largest predicted pairs
  summary.json city-level numbers (NLL, CPC, pairs, origins) and the check against the all-destination scores

Output: results/multicity/paper/od_maps/<city>/. Usage (run_stage.sh, M line): od_maps.py export [city ...]
(default: the six confirmation cities)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DENVER = "19740"
OURS = ("c0355r1", "c0262", "c0514c1")
FAMILIES = ("grav_power", "grav_exp", "strexp1", "cab_pow", "cab_exp", "rad", "ext_rad", "io", "pwo", "ops")
MODELS = OURS + ("strexp3_N",) + tuple(f"{f}_{m}" for f in FAMILIES for m in "PN")
LEVELS = ("uniform", "local")
TIERS = (100, 200, 500, 1000, 2000)
TOP_K = (100, 200, 500, 1000, 2000, 5000)
RATIO_CUTS = (-1.0, -0.5, -0.2, 0.2, 0.5, 1.0)           # log2(predicted / observed arrivals): classes -3 ... 3
LOG_EDGES = np.round(np.arange(-2.0, 4.0001, 0.25), 2)   # log10 flow; below the first edge (zeros included): class -1
DIST_EDGES = np.round(np.geomspace(0.5, 120.0, 25), 3)   # km
MIN_PAIRS = 20
OUT = config.results_dir("M") / "paper" / "od_maps"


# --------------------------------------------------------------------------------------------- pure parts
def flows(e, y, offsets) -> np.ndarray:
    """Predicted flow of every row: the origin's total times the conditional-logit probability (as prior_models.score_rows)."""
    e, y = np.asarray(e, float), np.asarray(y, float)
    st, rep = offsets[:-1], np.diff(offsets)
    ce = e - np.repeat(np.maximum.reduceat(e, st), rep)
    p = np.exp(ce) / np.repeat(np.add.reduceat(np.exp(ce), st), rep)
    return np.repeat(np.add.reduceat(y, st), rep) * p


def cpc(y, mu) -> float:
    return float(2 * np.minimum(y, mu).sum() / (y.sum() + mu.sum()))


def nll(y, mu, offsets) -> float:
    tot = np.repeat(np.add.reduceat(y, offsets[:-1]), np.diff(offsets))
    ok = y > 0
    return float(-(y[ok] * np.log(mu[ok] / tot[ok])).sum() / y.sum())


def top(v, k: int) -> np.ndarray:
    """Indices of the k largest positive values, largest first (ties by position)."""
    v = np.asarray(v, float)
    order = np.argsort(-v, kind="stable")[:k]
    return order[v[order] > 0]


def tiers(v, cuts=TIERS) -> np.ndarray:
    """0 outside the largest cuts[-1] values; else 1 for the largest cuts[0], 2 for the next up to cuts[1], ..."""
    out = np.zeros(len(v), int)
    order = top(v, cuts[-1])
    out[order] = np.searchsorted(np.asarray(cuts), np.arange(len(order)), side="right") + 1
    return out


def ratio_class(pred, obs, cuts=RATIO_CUTS) -> np.ndarray:
    """Class -3 ... 3 of log2(pred / obs); 3 where nothing was observed but something is predicted, 0 where both are 0."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log2(pred / obs)
    r = np.where((obs == 0) & (pred == 0), 0.0, r)
    return np.searchsorted(np.asarray(cuts), r, side="right") - 3


def log_class(v, edges=LOG_EDGES) -> np.ndarray:
    """Class of log10(v): -1 below the first edge (zeros included), else the index of the bin, the last one open."""
    v = np.asarray(v, float)
    with np.errstate(divide="ignore"):
        lv = np.where(v > 0, np.log10(np.maximum(v, 1e-300)), -np.inf)
    return np.minimum(np.searchsorted(edges, lv, side="right") - 1, len(edges) - 2)


def histogram(y, mu) -> pd.DataFrame:
    t = pd.DataFrame({"obs_class": log_class(y), "pred_class": log_class(mu)})
    return t.value_counts().rename("n_pairs").reset_index().sort_values(["obs_class", "pred_class"]).reset_index(drop=True)


def calibration(y, mu, min_pairs: int = MIN_PAIRS) -> pd.DataFrame:
    """By class of predicted flow: pairs, mean predicted, mean and quantiles of the observed flow."""
    t = pd.DataFrame({"pred_class": log_class(mu), "y": np.asarray(y, float), "mu": np.asarray(mu, float)})
    g = t[t["pred_class"] >= 0].groupby("pred_class")
    out = g.agg(n_pairs=("y", "size"), pred_mean=("mu", "mean"), obs_mean=("y", "mean"),
                obs_q10=("y", lambda s: s.quantile(0.1)), obs_q50=("y", "median"), obs_q90=("y", lambda s: s.quantile(0.9))).reset_index()
    return out[out["n_pairs"] >= min_pairs].reset_index(drop=True)


def distance_shares(dist, w, edges=DIST_EDGES) -> np.ndarray:
    """Share of the trips w in every distance class (the first and the last class are open)."""
    k = np.clip(np.searchsorted(edges, np.asarray(dist, float), side="right") - 1, 0, len(edges) - 2)
    s = np.bincount(k, weights=np.asarray(w, float), minlength=len(edges) - 1)
    return s / s.sum()


def overlap(y, mu, ks=TOP_K) -> list:
    """For every K: the share of the K largest observed flows that are among the K largest predicted ones, the share of
    the observed trips on the K largest predicted pairs, and the most that K pairs can carry."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    rows = []
    for k in ks:
        a, b = top(y, k), top(mu, k)
        rows.append({"K": int(k), "overlap": len(np.intersect1d(a, b)) / max(len(a), 1),
                     "trips_on_predicted_top": float(y[b].sum() / y.sum()), "trips_on_observed_top": float(y[a].sum() / y.sum())})
    return rows


def rank_thirds(v, n: int = 12) -> np.ndarray:
    """0 outside the n largest positive values; 3 / 2 / 1 for the first / second / last third of them."""
    out = np.zeros(len(v), int)
    order = top(v, n)
    out[order] = 3 - np.minimum(2, (3 * np.arange(len(order))) // max(len(order), 1))
    return out


# --------------------------------------------------------------------------------------------- stage
def constants(cfg: dict, CITY: str) -> dict:
    """{(model, level): (kind, constants)} from the existing files only: the frozen fits and plan 9's fits and uniform constants."""
    import p9
    out = {}
    ours = p9.ours_fits("test_dest")
    for m in MODELS:
        if m in ours and CITY in ours[m]:
            out[(m, "local")] = ("ours", ours[m][CITY]["consts"])
    fits = pd.read_parquet(p9.OUT / "fits_target.parquet")
    for r in fits[(fits["city"].astype(str) == CITY) & fits["model"].isin(MODELS)].itertuples():
        if r.consts and (r.model, "local") not in out:
            out[(r.model, "local")] = (model_kind(r.model, cfg["ours"]["reported"]), json.loads(r.consts))
    uni = pd.read_parquet(p9.OUT / "target.parquet")
    for r in uni[(uni["city"].astype(str) == CITY) & (uni["level"] == "uniform") & uni["model"].isin(MODELS)].itertuples():
        if r.consts:
            out[(r.model, "uniform")] = (model_kind(r.model, cfg["ours"]["reported"]), json.loads(r.consts))
    import prior_models as pm
    for m in MODELS:                              # a model without constants is the same at every level
        if m in pm.MAT_MODELS and not pm.MAT_MODELS[m].start:
            out[(m, "uniform")] = ("mat", [])
    return {k: v for k, v in out.items() if v[0] != "mat" or k[1] == "uniform"}     # matrix path: the display level only


def model_kind(model: str, ours) -> str:
    import prior_models as pm
    return "ours" if model in ours else "mat" if model in pm.MAT_MODELS else "eq"


def matrix_flows(model: str, consts, A: dict, o, j, y) -> np.ndarray:
    """Predicted flow of every long-format row from a matrix-path model (origins x units), with the same origin totals."""
    import prior_models as pm
    io, av, T = A["io"]["ev"], A["av"]["ev"], np.where(A["av"]["ev"], A["T"]["ev"], 0.0)
    e = pm.mat_eta(model, list(consts), A, io)
    if not np.isfinite(e[av]).all():
        raise RuntimeError(f"{model}: eta is not finite")
    e = np.where(av, e, -np.inf)
    w = np.exp(e - e.max(1, keepdims=True))
    MU = T.sum(1, keepdims=True) * w / w.sum(1, keepdims=True)
    row = np.full(A["D"].shape[0], -1)
    row[io] = np.arange(len(io))
    if (row[o] < 0).any() or not av[row[o], j].all() or int(av.sum()) != len(y) or not np.array_equal(T[row[o], j], y):
        raise RuntimeError(f"{model}: the matrix-path rows are not the long-format rows")
    s = pm.score_mat(pm.mat_eta(model, list(consts), A, io), A["T"]["ev"], av)
    mu = MU[row[o], j]
    if abs(s["cpc"] - cpc(y, mu)) > 1e-9:
        raise RuntimeError(f"{model}: CPC {cpc(y, mu)} differs from score_mat ({s['cpc']})")
    return mu


def confirmation_cities() -> pd.DataFrame:
    c = pd.read_csv(Path(__file__).resolve().parent / "config" / "cities.csv", dtype=str)
    return c[c["role"] == "confirm"].set_index("city")


def stage_export(cities=None):
    conf = confirmation_cities()
    cities = list(cities or conf.index)
    other = [c for c in cities if c not in conf.index]
    if other:
        raise RuntimeError(f"{other}: only the confirmation cities are looked at")
    for city in cities:
        export_city(city, OUT / city, conf.loc[city, "name"])


def export_city(CITY: str, OUT: Path, name: str):
    import frozen
    import geopandas as gpd
    import ledger
    import p9
    import paper_figdata as pf
    import prior_models as pm
    import sr_data
    from refit_select import eta as eq_eta, parse_equation
    frozen.require()
    cfg = p9.require_fixed()
    if CITY not in p9.cities("target"):
        raise RuntimeError(f"{CITY} is not a target city of plan 9")
    scores = pd.read_parquet(p9.OUT / "alldest" / "scores.parquet")      # the same rows were scored there: this must agree
    scores = scores[scores["city"].astype(str) == CITY].set_index(["model", "level"])
    K, cands = constants(cfg, CITY), p9.ours_candidates()
    missing = [m for m in MODELS if (m, "uniform") not in K]
    print("constants:", sorted(K), "| no uniform constants:", missing, flush=True)
    if missing:
        raise RuntimeError(f"no uniform constants for {missing}: the display level must be complete")

    d = sr_data.load(CITY, "tract")
    ev = sr_data.build([CITY], "tract", "test", "all", None, 0)
    y, off = np.asarray(ev.y, float), ev.offsets
    geoid = d["u"]["unit"].astype(str).to_numpy()
    o, j = ev.origin_unit, ev.dest_unit
    dist = d["D"][o, j]
    A = pm.city_arrays(d, sr_data.folds(d), "test", "all", np.load(d["base"] / "S.npy").astype(np.float64), n_orig=cfg["n_orig"], seed=0)
    pred, summ = {}, {"city": CITY, "name": name, "n_pairs": int(len(y)), "n_origins": int(len(off) - 1), "n_pairs_observed": int((y > 0).sum()), "models": {}}
    for (m, lv), (kind, consts) in sorted(K.items()):
        if kind == "mat":
            mu = matrix_flows(m, consts, A, o, j, y)
        else:
            c = cands[m] if kind == "ours" else parse_equation(pm.EQ_MODELS[m].equation, None, "A", 9)
            with np.errstate(all="ignore"):
                e = eq_eta(c, ev, consts=np.asarray(consts, float))
            if not np.isfinite(e).all():
                raise RuntimeError(f"{m} {lv}: eta is not finite")
            mu = flows(e, y, off)
        s = {"kind": kind, "nll": nll(y, mu, off), "cpc": cpc(y, mu)}
        if (m, lv) in scores.index:                                   # the all-destination check scored these very rows
            ref = scores.loc[(m, lv)]
            if abs(ref["nll"] - s["nll"]) > 1e-6 or abs(ref["cpc"] - s["cpc"]) > 1e-6:
                raise RuntimeError(f"{m} {lv}: NLL/CPC {s} differ from the all-destination check ({ref['nll']}, {ref['cpc']})")
            s["matches_alldest_check"] = True
        pred[f"{m}|{lv}"] = mu
        summ["models"][f"{m}|{lv}"] = s
        print(m, lv, s, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    src = {"observed": y, **pred}
    T = pd.DataFrame({"o": geoid[o], "d": geoid[j], "dist_km": np.round(dist, 2), **{f"tier_{k}": tiers(v) for k, v in src.items()}})
    T[(T.filter(like="tier_") > 0).any(axis=1)].to_csv(OUT / "pairs.csv", index=False)

    arr = {k: np.bincount(j, weights=v, minlength=d["N"]) for k, v in src.items()}
    dm = np.flatnonzero(d["dest"])
    D = pd.DataFrame({"GEOID": geoid[dm], "obs_class": pf.quantile_class(arr["observed"][dm], 5), "obs_zero": arr["observed"][dm] == 0,
                      **{f"ratio_{k}": ratio_class(arr[k][dm], arr["observed"][dm]) for k in pred}})
    D.to_csv(OUT / "dest.csv", index=False)

    u = d["u"]
    orig = np.flatnonzero(d["orig"])
    home = pf.central_busy_origin(u["x"].to_numpy(float), u["y"].to_numpy(float), u["population"].fillna(0).to_numpy(float),
                                  d["T"]["train"][orig].sum(1), orig)             # the home tract of Fig. 1b
    rows = np.flatnonzero(o == home)
    H = pd.DataFrame({"home": geoid[home], "d": geoid[j[rows]], **{f"rank_{k}": rank_thirds(v[rows]) for k, v in src.items()}})
    H[(H.filter(like="rank_") > 0).any(axis=1)].to_csv(OUT / "home.csv", index=False)

    pd.concat([histogram(y, mu).assign(source=k) for k, mu in pred.items()]).to_csv(OUT / "hist.csv", index=False)
    pd.concat([calibration(y, mu).assign(source=k) for k, mu in pred.items()]).to_csv(OUT / "calib.csv", index=False)
    pd.DataFrame({"dist_from": DIST_EDGES[:-1], "dist_to": DIST_EDGES[1:], **{k: distance_shares(dist, v) for k, v in src.items()}}) \
        .to_csv(OUT / "dist.csv", index=False)
    pd.DataFrame([{"source": k, **r} for k, mu in pred.items() for r in overlap(y, mu)]).to_csv(OUT / "overlap.csv", index=False)
    summ.update(log_edges=LOG_EDGES.tolist(), ratio_cuts=list(RATIO_CUTS), tiers=list(TIERS), home=str(geoid[home]))
    g = gpd.read_file(sr_data.ROOT / f"units/M/{CITY}_tracts.geojson")[["GEOID", "geometry"]]      # public Census outlines
    g = g.assign(GEOID=g["GEOID"].astype(str))
    g = g[g["GEOID"].isin(set(geoid))]
    g["geometry"] = g["geometry"].simplify(0.0005)
    g.to_file(OUT / "tracts.geojson", driver="GeoJSON")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    ledger.add(f"Exploratory illustration (2026-11-12; nothing refitted, no test): observed against predicted OD flows of "
               f"{name} ({CITY}), test period x all destinations, {len(MODELS)} models (ours, the relaxed stretched exp., and every "
               f"earlier model without local trips as published and with the POI count) at the uniform constants, local where fitted; "
               f"classes and aggregates only in results/multicity/paper/od_maps/{CITY}; summary.json sha {ledger.sha(OUT / 'summary.json')}")
    print("written:", sorted(p.name for p in OUT.iterdir()), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] != ["export"]:
        sys.exit(__doc__)
    stage_export(sys.argv[2:])
