"""Paper figures: export the small tables every panel is drawn from (figure design of 2026-11-11).

Read-only on the existing result files; nothing is refitted. Our formulas always use the frozen library fits
(training-period fit, test block, 67 cities); the earlier models use the plan-9 results. City-level statistics are
exported as they are; the example city of Fig. 1b carries only classes (POI quintile, destination block, three flow
ranks of one home tract) — no tract-level counts leave the server (the raw data are licensed).

Output: results/multicity/paper/figdata/ and MANIFEST.json (sha of every input and output file).
Usage (run_stage.sh, M line): paper_figdata.py export
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

OUT = config.results_dir("M") / "paper" / "figdata"
R = config.RESULTS
SKELETONS = {"c0355r1": "Scaled kernel", "c0262": "Tempered power law", "c0514c1": "Two-regime power law"}
REFERENCE = {"c0079c1": "Log gravity"}
ABBR = {"35620": "NYC", "31080": "LA", "16980": "CHI", "26420": "HOU", "12060": "ATL", "BAY9": "SF", "42660": "SEA",
        "14460": "BOS"}
EXAMPLE_CITY = "19740"          # Denver (a confirmation city)
H101_DISC = -0.110              # H1-01 discovery slope as reported in the agent-in-the-loop report of 2026-11-09; no table kept
_INPUTS = []


# --------------------------------------------------------------------------------------------- pure helpers
def ols_slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 else float("nan")


def weighted_centroid(x, y, w) -> tuple:
    x, y, w = (np.asarray(v, float) for v in (x, y, w))
    if not w.sum() > 0:
        return float(x.mean()), float(y.mean())
    return float((x * w).sum() / w.sum()), float((y * w).sum() / w.sum())


def quantile_class(v, k: int = 5) -> np.ndarray:
    """Class 1..k by rank (equal-count classes, ties broken by position)."""
    r = pd.Series(np.asarray(v, float)).rank(method="first").to_numpy()
    return np.minimum((k * (r - 1) / len(r)).astype(int) + 1, k)


def flow_classes(counts, top: int = 12) -> np.ndarray:
    """0 for a destination outside the `top` most visited (or not visited); 3 / 2 / 1 for the first / second / last
    third of the top list. Classes only: the counts themselves are not exported."""
    c = np.asarray(counts, float)
    out = np.zeros(len(c), int)
    order = [i for i in np.argsort(-c, kind="stable") if c[i] > 0][:top]
    for rank, i in enumerate(order):
        out[i] = 3 - min(2, (3 * rank) // max(len(order), 1))
    return out


def central_busy_origin(x, y, pop, trips, orig) -> int:
    """The origin unit nearest to the population-weighted centre among the origins in the top quarter by trips."""
    cx, cy = weighted_centroid(x, y, pop)
    trips, orig = np.asarray(trips, float), np.asarray(orig, int)
    cand = orig[trips >= np.quantile(trips, 0.75)]
    return int(cand[np.argmin((x[cand] - cx) ** 2 + (y[cand] - cy) ** 2)])


def set_of(role: str) -> str:
    return {"search": "discovery", "discovery": "discovery", "lawtest": "heldout", "confirm": "confirmation"}[role]


# --------------------------------------------------------------------------------------------- inputs
def _p(rel) -> Path:
    p = R / rel
    _INPUTS.append(p)
    return p


def _roles() -> dict:
    return {str(c): r for r in ("search", "discovery", "lawtest", "confirm") for c in config.cities_with_role(r)}


def _write(df: pd.DataFrame, name: str) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"{name}: {df.shape}", flush=True)
    return df


# --------------------------------------------------------------------------------------------- tables
def cities() -> pd.DataFrame:
    import sr_data
    roles = _roles()
    names = pd.read_csv(config.CITIES_CSV, dtype=str).set_index("city")["name"]
    fm = pd.read_csv(_p("form_metrics.csv"), dtype={"city": str}).set_index("city")
    rows = []
    for c, r in roles.items():
        u = pd.read_parquet(sr_data.ROOT / f"derived/model_inputs/M/{c}/tract/units.parquet", columns=["x", "y", "population"])
        x, y = weighted_centroid(u["x"], u["y"], u["population"].fillna(0))
        rows.append({"city": c, "name": names.get(c, c), "abbr": ABBR.get(c, ""), "role": r, "set": set_of(r),
                     "population": float(fm.loc[c, "population"]), "log_population": float(np.log(fm.loc[c, "population"])),
                     "x": x, "y": y, "n_units": int(len(u))})
    return pd.DataFrame(rows)


def _frozen_cv() -> pd.DataFrame:
    cv = pd.read_parquet(_p("confirm/test_dest/cities_val.parquet"))
    return cv.assign(city=cv["city"].astype(str))


def bands(city_list) -> pd.DataFrame:
    """Flexible-model and formula elasticities per distance band at each city's band reference point."""
    import city_outcomes as co
    import prefreeze
    bd = co._banded(city_list)
    refs_b = {c: prefreeze._intkeys(b["refs"]) for c, b in bd.items()}
    rows = []
    usable = {}
    for c, b in bd.items():
        pr, rf, vi = (prefreeze._intkeys(b.get(k)) for k in ("profile", "refs", "visits"))
        for i in range(len(co.BAND_KEYS)):
            ok = prefreeze._usable(pr, i) and (vi.get(i) or 0) >= prefreeze.MIN_BAND_VISITS and bool(rf.get(i))
            usable[(c, i)] = ok
            if ok:
                rows.append({"city": c, "model": "Flexible", "band": i, "band_key": co.BAND_KEYS[i],
                             "d_ref": float(rf[i]["d"]), "ed": float(pr[i][0]), "eN": float(pr[i][1])})
    cv = _frozen_cv()
    names = {**SKELETONS, **REFERENCE}
    prof, failed = prefreeze.skeleton_profiles(cv[cv["cid"].isin(names) & cv["city"].isin(city_list)], co._pool(), refs_b)
    for (cid, c), p in prof.items():
        for i, v in p.items():
            if v is not None and usable.get((c, i)) and np.isfinite(v).all():
                rows.append({"city": c, "model": names[cid], "band": int(i), "band_key": co.BAND_KEYS[int(i)],
                             "d_ref": float(refs_b[c][i]["d"]), "ed": float(v[0]), "eN": float(v[1])})
    print(f"bands: {len(bd)} cities with band files, {failed} formula profiles not evaluable", flush=True)
    return pd.DataFrame(rows)


def shares(city_list) -> pd.DataFrame:
    import p9
    cv = _frozen_cv()
    cv = cv[cv["cid"].isin(SKELETONS) & cv["city"].isin(city_list) & cv["finite"].astype(bool)]
    rows = []
    for r in cv.itertuples():
        ref = p9.refs(r.city, "test_dest")
        rows.append({"city": r.city, "model": SKELETONS[r.cid], "nll": float(r.val_nll),
                     "share": (ref["grav"] - float(r.val_nll)) / (ref["grav"] - ref["m2"])})
    return pd.DataFrame(rows)


def size_laws(disc_val: pd.DataFrame) -> pd.DataFrame:
    """The seven registered size hypotheses: discovery slope (recomputed on the discovery cities' validation-block
    table, the data the hypotheses came from) and the held-out slope with its interval as recorded."""
    res = {r["id"]: r for f in ("laws/results.json", "laws/results_extra.json", "laws/results_v3.json")
           for r in json.loads(_p(f).read_text())}
    p7r = {r["id"]: r for r in json.loads(_p("laws/results_p7.json").read_text())["rows"]}
    p7h = {h["id"]: h for h in json.loads(_p("laws/registered_p7.json").read_text())}
    x = disc_val["log_population"]
    spec = [("H3-01", "Flexible model", ols_slope(x, disc_val["ed_5_10km"]), "validation table"),
            ("H4-03", "Scaled kernel", ols_slope(x, disc_val["ed_5_10km_c0355r1"]), "validation table"),
            ("H4-01", "Tempered power law", ols_slope(x, disc_val["ed_5_10km_c0262"]), "validation table"),
            ("H4-02", "Two-regime power law", ols_slope(x, disc_val["ed_5_10km_c0514c1"]), "validation table"),
            ("H1-01", "Variable-exponent form", H101_DISC, "report v1")]
    rows = [{"id": i, "label": lab, "quantity": "Decay 5-10 km", "disc_slope": d, "disc_source": src,
             "test_slope": res[i]["stat"], "ci_lo": res[i]["ci_lo"], "ci_hi": res[i]["ci_hi"], "passed": bool(res[i]["pass"])}
            for i, lab, d, src in spec]
    for i, lab in (("P7-01", "Turning distance"), ("P7-02", "Transfer loss")):
        rows.append({"id": i, "label": lab, "quantity": lab, "disc_slope": p7h[i]["evidence"]["stat"],
                     "disc_source": "registration", "test_slope": p7r[i]["stat"], "ci_lo": p7r[i]["ci_lo"],
                     "ci_hi": p7r[i]["ci_hi"], "passed": bool(p7r[i]["supported"])})
    return pd.DataFrame(rows)


def p7() -> pd.DataFrame:
    res = {r["id"]: r for r in json.loads(_p("laws/results_p7.json").read_text())["rows"]}
    rows = []
    for h in json.loads(_p("laws/registered_p7.json").read_text()):
        r, e = res[h["id"]], h["evidence"]
        rows.append({"id": h["id"], "group": "size" if h["covariate"] == "log_population" else "form",
                     "outcome": h["outcome"], "covariate": h["covariate"], "direction": h["direction"],
                     "disc_slope": e["stat"], "disc_lo": e["ci"][0], "disc_hi": e["ci"][1], "test_slope": r["stat"],
                     "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"], "holm": bool(r["holm"]),
                     "supported": bool(r["supported"])})
    return pd.DataFrame(rows)


def axis1(cit: pd.DataFrame) -> pd.DataFrame:
    """First form axis (already orthogonal to size) and the long-distance decay left after a line on log population."""
    import explore_form as ef
    out, _, _ = ef.load_frame()
    sc = pd.read_csv(_p("explore/e2_axis_scores.csv"), dtype={"city": str})
    t = sc[["city", "axis1"]].merge(out.assign(city=out["city"].astype(str))[["city", "ed_10_20km", "ed_20km"]], on="city")
    t = t.merge(cit[["city", "set", "log_population"]], on="city")
    for col in ("ed_10_20km", "ed_20km"):
        ok = t[col].notna()
        b, a = np.polyfit(t.loc[ok, "log_population"], t.loc[ok, col].astype(float), 1)
        t[f"{col}_resid"] = t[col].astype(float) - (a + b * t["log_population"])
    return t


def types() -> pd.DataFrame:
    t = json.loads(_p("explore/types.json").read_text())["features"]
    return pd.DataFrame([{"features": f, "n": v["n"], **row} for f, v in t.items() for row in v["table"]])


def p9_tables() -> dict:
    import p9
    import p9_stats as st
    import prior_models as pm
    tg = pd.read_parquet(_p("p9/target.parquet")).drop(columns=["consts"])
    s = st.summary(tg)
    meta = {}
    for k, sp in pm.EQ_MODELS.items():
        meta[k] = {"family": k.rsplit("_", 1)[0], "mass": sp.mass, "kind": "prior", "original": bool(sp.original),
                   "n_free": int(np.sum(sp.free))}
    for k, sp in pm.MAT_MODELS.items():
        meta[k] = {"family": sp.family, "mass": sp.mass or "T", "kind": "prior", "original": bool(sp.original),
                   "n_free": len(sp.start)}
    cfg = json.loads(p9.CONFIG.read_text())
    cands = p9.ours_candidates()
    for cid in cfg["ours"]["reported"]:
        meta[cid] = {"family": {**SKELETONS, **REFERENCE}.get(cid, "Variable-exponent form"), "mass": "N", "kind": "ours",
                     "original": False, "n_free": len(cands[cid].consts)}
    m = pd.DataFrame(meta).T.rename_axis("model").reset_index()
    refs = pd.DataFrame([{"city": c, **p9.refs(c, "test_dest")} for c in sorted(tg["city"].unique())])
    con = json.loads(_p("p9/contrasts.json").read_text())
    return {"p9_target.csv": tg, "p9_summary.csv": s.merge(m, on="model", how="left"), "p9_refs.csv": refs,
            "p9_diffs.csv": pd.read_csv(_p("p9/contrast_diffs.csv"), dtype={"city": str}), "contrasts": con}


def c3_tables(cit: pd.DataFrame) -> dict:
    c3 = json.loads(_p("p9/explore/c3_explore.json").read_text())
    g = pd.DataFrame(c3["cities"]).assign(city=lambda d: d["city"].astype(str))
    cv = _frozen_cv()
    cv = cv[(cv["cid"] == "c0355r1") & cv["finite"].astype(bool)]
    k = pd.DataFrame([[r.city] + [float(v) for v in json.loads(r.consts)] for r in cv.itertuples()])
    k.columns = ["city"] + [f"theta{i}" for i in range(k.shape[1] - 1)]
    k = k.merge(cit[["city", "set", "role", "log_population"]], on="city")
    return {"c3_cities.csv": g, "consts.csv": k, "c3": {a: c3[a] for a in ("block", "dose_all", "dose_source", "dose_target")}}


def example_city(city: str = EXAMPLE_CITY):
    """Tract geometry (EPSG:4326) with classes only: POI quintile, destination block, and — for a busy home tract
    near the centre — three flow ranks of its top destinations."""
    import geopandas as gpd
    import sr_data
    d = sr_data.load(city, "tract")
    u, fold = d["u"], sr_data.folds(d)
    T = d["T"]["train"]
    orig = np.flatnonzero(d["orig"])
    home = central_busy_origin(u["x"].to_numpy(float), u["y"].to_numpy(float), u["population"].fillna(0).to_numpy(float),
                               T[orig].sum(1), orig)
    row = T[home].copy()
    row[home] = 0
    cls = pd.DataFrame({"GEOID": u["unit"].astype(str), "poi_class": quantile_class(u["N"].to_numpy(float), 5),
                        "block": fold, "in_choice_set": d["dest"], "home": np.arange(d["N"]) == home,
                        "flow_class": flow_classes(row, 12)})
    g = gpd.read_file(sr_data.ROOT / f"units/M/{city}_tracts.geojson")[["GEOID", "geometry"]]
    g = g.assign(GEOID=g["GEOID"].astype(str)).merge(cls, on="GEOID", how="inner")
    g["geometry"] = g["geometry"].simplify(0.0005)
    return g


# --------------------------------------------------------------------------------------------- stage
def stage_export():
    import ledger
    OUT.mkdir(parents=True, exist_ok=True)
    cit = _write(cities(), "cities.csv")
    cs = list(cit["city"])
    if len(cs) != 67:
        raise RuntimeError(f"{len(cs)} cities, expected 67")
    _write(bands(cs), "bands.csv")
    sh = _write(shares(cs), "shares.csv")
    p9t = p9_tables()
    tg = p9t["p9_target.csv"]
    chk = tg[(tg["level"] == "local") & tg["model"].isin(SKELETONS)].assign(model=lambda d: d["model"].map(SKELETONS))
    chk = chk.merge(sh, on=["city", "model"], suffixes=("_p9", ""))
    gap = float((chk["share_p9"] - chk["share"]).abs().max())
    if not (len(chk) == 32 * len(SKELETONS) and gap <= 1e-9):
        raise RuntimeError(f"the frozen fits and the plan-9 rows disagree: {len(chk)} rows, largest share difference {gap}")
    disc = pd.read_parquet(_p("p7/city_outcomes.parquet"))
    _write(size_laws(disc[disc["block"] == "val"]), "size_laws.csv")
    _write(p7(), "p7.csv")
    e1 = pd.read_csv(_p("explore/e1_pairs.csv"))
    _write(e1[e1["variant"] == "main"], "e1_pairs.csv")
    (OUT / "e1_summary.json").write_text(json.dumps(json.loads(_p("explore/e1_summary.json").read_text())["main"], indent=1))
    e2 = pd.read_csv(_p("explore/e2_effects.csv"))
    _write(e2[e2["role"] == "form"], "e2_effects.csv")
    _write(axis1(cit), "axis1.csv")
    _write(types(), "types.csv")
    for name in ("p9_target.csv", "p9_summary.csv", "p9_refs.csv", "p9_diffs.csv"):
        _write(p9t[name], name)
    (OUT / "p9_contrasts.json").write_text(json.dumps(p9t["contrasts"], ensure_ascii=False, indent=1))
    c3 = c3_tables(cit)
    _write(c3["c3_cities.csv"], "c3_cities.csv")
    _write(c3["consts.csv"], "consts.csv")
    cv = _frozen_cv()
    tu = cv[(cv["cid"] == "c0514c1") & cv["finite"].astype(bool)]
    _write(pd.DataFrame({"city": tu["city"], "theta_turn": [float(json.loads(v)[1]) for v in tu["consts"]]})
           .merge(cit[["city", "set", "role", "log_population"]], on="city"), "turning.csv")
    (OUT / "c3_summary.json").write_text(json.dumps(c3["c3"], indent=1))
    ad = OUT.parent.parent / "p9" / "alldest"                # supplementary all-destination check (added 2026-11-12), if it has run
    if (ad / "scores.parquet").exists():
        import alldest_check as ac
        _INPUTS.append(ad / "scores.parquet")
        _write(ac.add_share(pd.read_parquet(ad / "scores.parquet")), "alldest_scores.csv")
        (OUT / "alldest_summary.json").write_text((ad / "summary.json").read_text())
    g = example_city()
    g.to_file(OUT / "example_city.geojson", driver="GeoJSON")
    print(f"example_city.geojson: {len(g)} tracts, home {int(g['home'].sum())}, flows {int((g['flow_class'] > 0).sum())}", flush=True)
    man = {"inputs": {str(p.relative_to(R)): ledger.sha(p) for p in sorted(set(_INPUTS))},
           "outputs": {p.name: ledger.sha(p) for p in sorted(OUT.iterdir()) if p.name != "MANIFEST.json"},
           "frozen_vs_plan9_share_gap": gap}
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    ledger.add(f"Paper figure data exported (read-only, nothing refitted): {len(man['outputs'])} tables in "
               f"results/multicity/paper/figdata; frozen fits vs plan-9 rows of our formulas agree to {gap:.1e}; "
               f"MANIFEST.json sha {ledger.sha(OUT / 'MANIFEST.json')}")


if __name__ == "__main__":
    if sys.argv[1:] != ["export"]:
        sys.exit(__doc__)
    stage_export()
