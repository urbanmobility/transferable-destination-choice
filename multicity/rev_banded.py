"""Revision 2026-11-13 (response to review, Task R2): the banded flexible model, reviewer variants.

Post-freeze, supplementary; outputs in results/multicity/rev/.

  fixed    no refit: the existing banded fits (prefreeze/banded, prefreeze/lawtest_banded, confirm/banded) and the
           frozen formula fits, evaluated at ONE reference point shared by every city and band: d at the band's
           geometric midpoint (0.5, 1.41, 3.16, 7.07, 14.1, 28.3 km), N_j, a_j, E_j, s_* at the pooled median over all
           destination tracts of the 67 cities. (The paper's profiles use each city's band medians.)
  emp      refit with LODES employment as the banded mass: 1[d in b] (ln(1+E_j) - mean); log_jobs leaves the
           destination covariates (it would be collinear with the band terms); log N stays out, as in the paper.
  empN     as emp, with log(1+N_j) added back as a destination covariate (e_E net of the POI count).
  nonwork  the paper's banded model refitted on non-work counts (n_nonwork_5).
For the refitted variants: train x search blocks, all origins, as banded.city_banded; e_N = gamma_b M/(M+1) at the
reference mass M; profiles at the band reference points and at the fixed point.
Usage (run_stage.sh, M line): rev_banded.py run fixed|emp|empN|nonwork
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401
import banded  # noqa: E402

FIXED_D = tuple(math.sqrt(lo * hi) if lo > 0 else 0.5 for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 40)))
VARIANTS = ("fixed", "emp", "empN", "nonwork")
UNIT_VARS = {"N_j": "N", "a_j": "area_km2", "E_j": "jobs", "P_j": "population"}
_G = {}


def out_dir() -> Path:
    return config.results_dir("M") / "rev"


# --------------------------------------------------------------------------------------------- pure parts
def fixed_refs(pooled: dict) -> dict:
    """{band: reference dict}: the pooled values for every variable, d at the band's geometric midpoint."""
    from sr_data import VARS
    out = {}
    for b, d in enumerate(FIXED_D):
        r = {v: float(pooled.get(v, 0.0)) for v in VARS}
        r.update({k: float(v) for k, v in pooled.items()})
        r.update(d=float(d), logd=float(math.log(d)), dlogS_dlogd=0.0, n_rows=0)
        out[b] = r
    return out


def with_mass(refs_b: dict, key: str) -> dict:
    """A copy of the band references with N_j set to the reference value of `key` (banded.flex_profile reads N_j)."""
    return {b: (None if r is None else {**r, "N_j": float(r[key])}) for b, r in refs_b.items()}


def design_cols(cols: dict, variant: str, u: pd.DataFrame) -> tuple:
    """(destination covariates, banded log-mass) of the variant; the input dict is left as it is."""
    c = dict(cols)
    if variant in ("emp", "empN"):
        c.pop("log_jobs", None)
        if variant == "empN":
            c["log_N"] = np.log1p(u["N"].to_numpy(float))
        return c, np.log1p(u["jobs"].to_numpy(float))
    return c, np.log1p(u["N"].to_numpy(float))


def profile_rows(city: str, model: str, fit: dict, refs_b: dict, ref_kind: str, visits: dict) -> list:
    prof = banded.flex_profile(fit, refs_b)
    return [{"city": city, "model": model, "band": int(b), "ref": ref_kind, "d_ref": float(refs_b[b]["d"]),
             "ed": float(v[0]), "eN": float(v[1]), "visits": float(visits.get(b) or 0.0)}
            for b, v in prof.items() if v is not None and refs_b.get(b) is not None]


# --------------------------------------------------------------------------------------------- inputs
def pooled_reference(cities: list) -> dict:
    """Median over every destination tract of the cities of N_j, a_j, E_j, P_j and the class shares s_*."""
    import sr_data
    vals = {k: [] for k in list(UNIT_VARS) + [f"s_{c}" for c in sr_data.CLASSES]}
    for c in cities:
        u = pd.read_parquet(sr_data.ROOT / f"derived/model_inputs/{sr_data.LINE}/{c}/tract/units.parquet")
        dest = ~((u["population"] == 0) & (u["N"] == 0))
        u = u[dest]
        for k, col in UNIT_VARS.items():
            vals[k].append(u[col].to_numpy(float))
        n = u["N"].to_numpy(float)
        for cl in sr_data.CLASSES:
            col = f"n_{cl}"
            if col in u:
                with np.errstate(invalid="ignore", divide="ignore"):
                    vals[f"s_{cl}"].append(np.where(n > 0, u[col].to_numpy(float) / n, np.nan))
    return {k: float(np.nanmedian(np.concatenate(v))) for k, v in vals.items() if v}


def _cities() -> list:
    import p9
    return p9.cities("source") + p9.cities("target")


# --------------------------------------------------------------------------------------------- variants
def fixed_rows(cities: list, F: dict) -> list:
    import city_outcomes as co
    import paper_figdata as pf
    import prefreeze
    rows = []
    bd = co._banded(cities)
    for c, b in bd.items():
        fit = {"gamma": prefreeze._intkeys(b["gamma"]), "spline_coef": b["spline_coef"], "knots": tuple(b["knots"])}
        rows += profile_rows(c, "Flexible", fit, F, "fixed", prefreeze._intkeys(b.get("visits")))
    cv = pf._frozen_cv()
    prof, failed = prefreeze.skeleton_profiles(cv[cv["cid"].isin(pf.SKELETONS) & cv["city"].isin(cities)], co._pool(),
                                               {c: F for c in cities})
    vis = {c: prefreeze._intkeys(b.get("visits")) for c, b in bd.items()}
    for (cid, c), p in prof.items():
        for i, v in p.items():
            if v is not None and np.isfinite(v).all():
                rows.append({"city": c, "model": pf.SKELETONS[cid], "band": int(i), "ref": "fixed", "d_ref": F[i]["d"],
                             "ed": float(v[0]), "eN": float(v[1]), "visits": float((vis.get(c) or {}).get(i) or 0.0)})
    print(f"fixed: {len(bd)} banded files, {failed} formula profiles not evaluable", flush=True)
    return rows


def variant_city(city: str) -> list:
    import frozen
    import sr_data
    from run_gate_l import features, standardise
    frozen.require()
    variant, F = _G["variant"], _G["F"]
    d = sr_data.load(city, "tract")
    u, D, T, dest, orig, N = d["u"], d["D"], d["T"], d["dest"], d["orig"], d["N"]
    f = sr_data.folds(d)
    ds = sr_data.dest_mask(d, f, "search")
    Bd, knots = banded.distance_spline(D, dest)
    f0, fplus, _, fcomp = features(u, transit_col="n_transit_osm")
    cols, mass = design_cols({k: v for blk in (f0, fplus, fcomp) for k, v in blk.items()}, variant, u)
    X, _ = standardise(cols, dest)
    fit = banded.fit_banded(Bd, X, D, mass, T["train"], orig, ds, knots)
    fit.pop("model")
    eye = np.eye(N, dtype=bool)
    Tt, Do = np.where((ds[None, :] & ~eye)[orig], T["train"][orig], 0.0), D[orig]
    visits = {b: float(Tt[(Do >= lo) & (Do < hi)].sum()) for b, (lo, hi) in enumerate(banded.BANDS)}
    refs = banded.band_refs(sr_data.build([city], "tract", "train", "search", 50, 0))
    key = "E_j" if variant in ("emp", "empN") else "N_j"
    rows = (profile_rows(city, "Flexible", fit, with_mass(refs, key), "band", visits)
            + profile_rows(city, "Flexible", fit, with_mass(F, key), "fixed", visits))
    for r in rows:
        r["converged"] = bool(fit["converged"])
        r["gamma"] = fit["gamma"].get(r["band"])
    print(city, variant, "banded done", flush=True)
    return rows


def stage_run(variant: str):
    import ledger
    import p9
    import rev_models
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}: one of {VARIANTS}")
    cs = _cities()
    pooled = pooled_reference(cs)
    F = fixed_refs(pooled)
    _G.update(variant=variant, F=F)
    if variant == "fixed":
        rows = fixed_rows(cs, F)
    else:
        with rev_models.use_counts("n_nonwork_5" if variant == "nonwork" else "n_dev_5"):
            rows = [r for rs in p9._pool_map(variant_city, cs) for r in rs]
    t = pd.DataFrame(rows)
    t["variant"] = variant
    out_dir().mkdir(parents=True, exist_ok=True)
    p = out_dir() / f"banded_{variant}.parquet"
    t.to_parquet(p, index=False)
    (out_dir() / "banded_fixed_reference.json").write_text(json.dumps({"d": list(FIXED_D), **pooled}, indent=1))
    ledger.add(f"Revision R2 (post hoc, supplementary): rev_banded {variant}: {t['city'].nunique()} cities; "
               f"rev/{p.name} sha {ledger.sha(p)}; pooled reference {json.dumps({k: round(v, 3) for k, v in pooled.items()})}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        sys.exit(__doc__)
    stage_run(sys.argv[2])
