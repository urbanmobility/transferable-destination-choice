"""Revision 2026-11-23 (response to review; post hoc, supplementary): four checks.

decomp   Where does the Reach model's likelihood gain over the strongest benchmark (relaxed stretched exponential,
         POI count) come from? No refit: the stored constants of plan 9 (local and uniform; the Reach model's bounded
         local constants where theta_2 >= 1 binds, rev/theta2_full.json) are scored on the test period, on the held-out
         block and on all destinations, and the log-likelihood of each model is summed within groups of visits: the six
         distance bands, quartiles of the destination's POI count and quartiles of its area (quartiles over the
         destination tracts of the evaluation set). Output: rev/decomp.parquet
         (city, level, set, by, key, visits, ll_reach, ll_bench).
knots    The flexible model places the interior knots of its distance spline at quantiles of each city's own
         tract-to-tract distances. Here the banded flexible model is refitted in every city with the same interior
         knots in all cities (the medians of the cities' own knots), everything else as in the paper's banded model.
         Output: rev/banded_knots.parquet (the columns of the other banded variants).
uncond   The main score conditions on the visit falling in the held-out block: P(j | i, j in B). Here the visits
         of the test period that fall in the held-out block are scored with every destination of the city in the
         denominator, and the unconditional NLL is split into the conditional part (the main score) and the part that
         comes from the probability the model gives to the block, -log P(B | i). Stored constants as in decomp. Also:
         predicted and observed share of visits in the block, and the conditional NLL with each origin's visits weighted
         by residents per device (population / home_devices). Output: rev/uncond.parquet.
rstar    A turning distance that does not depend on the Two-regime power law: the distance at which the
         distance elasticity of the flexible model (the spline of the stored banded fits, no refit) crosses -1.25, -1.5
         or -1.75 for the last time, so that it stays steeper than the threshold beyond it; grid of 600 log-spaced
         distances from 0.3 to 60 km, linear interpolation; missing when the curve is never above the threshold or is
         still above it at the end of the city's distance range. (A first version took the first crossing from 0.3 km
         outwards; most curves are not monotone below 1 km, which left that distance undefined in some cities.)
         Output: rev/rstar.parquet.
Usage (run_stage.sh, M line): rev_round5.py decomp | knots | uncond | rstar
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

REACH, BENCH = "c0355r1", "strexp3_N"
_G = {}


def _logp(e, data) -> np.ndarray:
    st, rep = data.offsets[:-1], np.diff(data.offsets)
    ce = e - np.repeat(np.maximum.reduceat(e, st), rep)
    return ce - np.repeat(np.log(np.add.reduceat(np.exp(ce), st)), rep)


def _quartile(v, dest_unit) -> np.ndarray:
    """Quartile 0..3 of each row's destination value among the destination tracts of the evaluation set."""
    _, first = np.unique(dest_unit, return_index=True)
    return np.digitize(v, np.quantile(v[first], [0.25, 0.5, 0.75]))


def decomp_city(city: str) -> list:
    import banded
    import frozen
    import library
    import p9
    import prior_models as pm
    import sr_data
    from refit_select import parse_equation
    frozen.require()
    cands = {REACH: library.to_single(p9.ours_candidates()[REACH]), BENCH: parse_equation(pm.EQ_MODELS[BENCH].equation, None, "A", 9)}
    K, nll = _G["K"], _G["nll"]
    rows = []
    for name, block in (("block", "fold:0"), ("all", "all")):
        ev = sr_data.build([city], "tract", "test", block, None, 0)
        d, n, a = (ev.X[:, ev.names.index(k)].astype(float) for k in ("d", "N_j", "a_j"))
        groups = {"distance band": np.array([next(b for b, (lo, hi) in enumerate(banded.BANDS) if lo <= x < hi) for x in d]),
                  "POI-count quartile": _quartile(n, ev.dest_unit), "area quartile": _quartile(a, ev.dest_unit)}
        V = float(ev.y.sum())
        for level in ("local", "uniform"):
            ll = {}
            for m in (REACH, BENCH):
                with np.errstate(all="ignore"):
                    lp = _logp(pm.eq_eta(cands[m], ev, consts=np.asarray(K[(city, m, level)], float)), ev)
                ll[m] = ev.y * lp
                if name == "block":          # the stored block score must be reproduced
                    assert abs(-ll[m].sum() / V - nll[(city, m, level)]) < 1e-6, (city, m, level, -ll[m].sum() / V, nll[(city, m, level)])
            for by, g in groups.items():
                for k in np.unique(g):
                    s = g == k
                    rows.append({"city": city, "level": level, "set": name, "by": by, "key": int(k), "visits": float(ev.y[s].sum()),
                                 "ll_reach": float(ll[REACH][s].sum()), "ll_bench": float(ll[BENCH][s].sum())})
    print(city, "decomposition done", flush=True)
    return rows


def uncond_city(city: str) -> list:
    import frozen
    import library
    import p9
    import prior_models as pm
    import sr_data
    from refit_select import parse_equation
    frozen.require()
    cands = {REACH: library.to_single(p9.ours_candidates()[REACH]), BENCH: parse_equation(pm.EQ_MODELS[BENCH].equation, None, "A", 9)}
    K, nll = _G["K"], _G["nll"]
    d = sr_data.load(city, "tract")
    in_b = sr_data.dest_mask(d, sr_data.folds(d), "fold:0")
    ev = sr_data.build([city], "tract", "test", "all", None, 0)
    st, rep = ev.offsets[:-1], np.diff(ev.offsets)
    rb = in_b[ev.dest_unit]
    yb = ev.y * rb
    V, VB = float(ev.y.sum()), float(yb.sum())
    Ti = np.add.reduceat(ev.y, st)
    obs_i = np.add.reduceat(yb, st) / np.maximum(Ti, 1e-300)
    u = d["u"]
    w = (u["population"].to_numpy(float) / u["home_devices"].to_numpy(float))[ev.origin_unit]
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    rows = []
    for level in ("local", "uniform"):
        for m in (REACH, BENCH):
            with np.errstate(all="ignore"):
                lp = _logp(pm.eq_eta(cands[m], ev, consts=np.asarray(K[(city, m, level)], float)), ev)
            PB = np.add.reduceat(np.exp(lp) * rb, st)
            lpb = np.repeat(np.log(PB), rep)
            cond = -float((yb * (lp - lpb)).sum()) / VB
            assert abs(cond - nll[(city, m, level)]) < 1e-6, (city, m, level, cond, nll[(city, m, level)])      # the main score
            rows.append({"city": city, "level": level, "model": m, "visits": V, "visits_block": VB, "nll_cond": cond,
                         "nll_block": -float((yb * lpb).sum()) / VB, "nll_uncond": -float((yb * lp).sum()) / VB,
                         "share_obs": VB / V, "share_pred": float((Ti * PB).sum() / V),
                         "mae_origin": float((Ti * np.abs(PB - obs_i)).sum() / V),
                         "nll_cond_weighted": -float((w * yb * (lp - lpb)).sum() / (w * yb).sum())})
    print(city, "unconditional scores done", flush=True)
    return rows


def _stored():
    import p9
    p9.require_fixed()
    t = pd.read_parquet(p9.OUT / "target.parquet")
    t = t[t["model"].isin([REACH, BENCH]) & t["level"].isin(["local", "uniform"])].assign(city=lambda x: x["city"].astype(str))
    K = {(r.city, r.model, r.level): [float(v) for v in (json.loads(r.consts) if isinstance(r.consts, str) else r.consts)] for r in t.itertuples()}
    nll = {(r.city, r.model, r.level): float(r.nll) for r in t.itertuples()}
    for r in json.loads((rev_banded.out_dir() / "theta2_full.json").read_text()):
        if (r["city"], REACH, "local") in K:
            K[(r["city"], REACH, "local")] = r["bounded"]["consts"]
            nll[(r["city"], REACH, "local")] = r["bounded"]["block_nll"]
    _G.update(K=K, nll=nll)
    return p9.cities("target")


def stage_uncond():
    import ledger
    import p9
    cities = _stored()
    rows = [r for rs in p9._pool_map(uncond_city, cities) for r in rs]
    p = rev_banded.out_dir() / "uncond.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    ledger.add(f"Revision 2026-11-23 (post hoc, supplementary): rev_round5 uncond: test-period visits in the held-out block scored with all "
               f"destinations of the city in the denominator, split into the conditional score and the block-probability part; "
               f"conditional score also with origins weighted by residents per device; Reach model and strongest benchmark, local and "
               f"uniform constants, {len(cities)} target cities; rev/{p.name} sha {ledger.sha(p)}")


THRESHOLDS = (-1.25, -1.5, -1.75)


def stage_rstar():
    import banded
    import city_outcomes as co
    import ledger
    from choice_model import bspline_basis
    cs = rev_banded._cities()
    bd = co._banded(cs)
    grid, h = np.log(np.geomspace(0.3, 60.0, 600)), banded.H_LOGD
    rows = []
    for c in cs:
        kn, lo, hi = bd[c]["knots"]
        coef = np.asarray(bd[c]["spline_coef"], float)
        x = grid[(grid > lo) & (grid < hi)]
        e = (bspline_basis(x + h, np.asarray(kn, float), lo=lo, hi=hi) @ coef
             - bspline_basis(x - h, np.asarray(kn, float), lo=lo, hi=hi) @ coef) / (2 * h)
        row = {"city": c}
        for thr in THRESHOLDS:
            above = np.flatnonzero(e > thr)
            if not len(above) or above[-1] == len(e) - 1:
                row[f"r_{abs(thr):.2f}"] = np.nan
                continue
            i = above[-1]
            t = (thr - e[i]) / (e[i + 1] - e[i])
            row[f"r_{abs(thr):.2f}"] = float(np.exp(x[i] + t * (x[i + 1] - x[i])))
        rows.append(row)
    t = pd.DataFrame(rows)
    p = rev_banded.out_dir() / "rstar.parquet"
    t.to_parquet(p, index=False)
    ledger.add(f"Revision 2026-11-23 (post hoc, supplementary): rev_round5 rstar: distance beyond which the flexible model's distance elasticity "
               f"stays below {list(THRESHOLDS)} (last crossing; replaces the first-crossing version of the same day), {len(t)} cities (defined in {[int(t[k].notna().sum()) for k in t.columns[1:]]}); "
               f"rev/{p.name} sha {ledger.sha(p)}")
    print(t.describe().round(2).to_string(), flush=True)


def stage_decomp():
    import ledger
    import p9
    p9.require_fixed()
    t = pd.read_parquet(p9.OUT / "target.parquet")
    t = t[t["model"].isin([REACH, BENCH]) & t["level"].isin(["local", "uniform"])].assign(city=lambda x: x["city"].astype(str))
    K = {(r.city, r.model, r.level): [float(v) for v in (json.loads(r.consts) if isinstance(r.consts, str) else r.consts)] for r in t.itertuples()}
    nll = {(r.city, r.model, r.level): float(r.nll) for r in t.itertuples()}
    for r in json.loads((rev_banded.out_dir() / "theta2_full.json").read_text()):       # theta_2 >= 1: the amended local fits
        if (r["city"], REACH, "local") in K:
            K[(r["city"], REACH, "local")] = r["bounded"]["consts"]
            nll[(r["city"], REACH, "local")] = r["bounded"]["block_nll"]
    _G.update(K=K, nll=nll)
    cities = p9.cities("target")
    rows = [r for rs in p9._pool_map(decomp_city, cities) for r in rs]
    p = rev_banded.out_dir() / "decomp.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    ledger.add(f"Revision 2026-11-23 (post hoc, supplementary): rev_round5 decomp: log-likelihood of the Reach model and of the strongest "
               f"benchmark by distance band, POI-count quartile and area quartile of the destination, {len(cities)} target cities, local "
               f"and uniform constants, held-out block and all destinations; rev/{p.name} sha {ledger.sha(p)}")


def common_knots(cities: list) -> np.ndarray:
    import city_outcomes as co
    bd = co._banded(cities)
    assert len(bd) == len(cities)
    return np.median(np.array([bd[c]["knots"][0] for c in cities], float), axis=0)


def stage_knots():
    import banded
    import ledger
    import p9
    cs = rev_banded._cities()
    kn = common_knots(cs)
    orig = banded.distance_spline

    def spline(D, dest):
        from choice_model import bspline_basis
        N = len(D)
        logd = np.log(np.maximum(D, 0.05))
        off = ~np.eye(N, dtype=bool) & dest[None, :] & dest[:, None]
        lo, hi = float(logd[off].min()), float(logd[off].max())
        assert lo < kn.min() and kn.max() < hi, (lo, hi, kn)
        return bspline_basis(logd.ravel(), kn, lo=lo, hi=hi).reshape(N, N, -1), (kn, lo, hi)

    rev_banded._G.update(variant="knots", F=rev_banded.fixed_refs(rev_banded.pooled_reference(cs)))
    banded.distance_spline = spline
    try:
        rows = [r for rs in p9._pool_map(rev_banded.variant_city, cs) for r in rs]
    finally:
        banded.distance_spline = orig
    t = pd.DataFrame(rows)
    t["variant"] = "knots"
    p = rev_banded.out_dir() / "banded_knots.parquet"
    t.to_parquet(p, index=False)
    ledger.add(f"Revision 2026-11-23 (post hoc, supplementary): rev_round5 knots: banded flexible model refitted with the same interior "
               f"spline knots in every city (exp: {[round(float(np.exp(v)), 2) for v in kn]} km, the medians of the cities' own knots), "
               f"{t['city'].nunique()} cities; rev/{p.name} sha {ledger.sha(p)}")
    print("knots (km)", [round(float(np.exp(v)), 2) for v in kn], flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("decomp", "knots", "uncond", "rstar"):
        sys.exit(__doc__)
    {"decomp": stage_decomp, "knots": stage_knots, "uncond": stage_uncond, "rstar": stage_rstar}[sys.argv[1]]()
