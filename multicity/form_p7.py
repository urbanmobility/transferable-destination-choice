"""Plan 7 Task 5 (plan-7 design, covariate set P2): new city covariates. The definitions (DEFS) are the ledgered
config/p7_covariates.json, fixed before any covariate-outcome relation was computed. Covariates are city
characteristics: they are always computed from the M line's unit tables (one value per city whatever the data line).
Law-test cities are refused until the plan-7 registration is signed.

Output: results/multicity/p7/<COV_FILES[tag]> (written once: an existing file is never overwritten, plan 8)
Usage (run_stage.sh): form_p7.py run TAG   (TAG: discovery = the 35 discovery cities | lawtest = the 26 law-test cities | confirm = the 6
confirmation cities)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DEFS = json.loads((Path(__file__).resolve().parent / "config" / "p7_covariates.json").read_text())
POI_CLASSES = ("food", "retail", "health", "education", "leisure", "service", "office", "transit")
CBD_RADIUS_M = 10_000.0
COV_FILES = {"discovery": "covariates.csv", "lawtest": "covariates_lawtest.csv", "confirm": "covariates_confirm.csv"}
CIRCUITY = {"n_pairs": 500, "dmin_km": 2.0, "dmax_km": 20.0, "min_valid": 100, "seed": 20261110}


def _built(u):
    return u[u["home_devices"] >= config.MIN_ORIGIN_DEVICES]


def transit_per_1k(u) -> float:
    p = float(u["population"].sum())
    return 1000.0 * float(u["n_transit_osm"].sum()) / p if p > 0 else float("nan")


def no_car_share(u) -> float:
    ok = u["hh_no_vehicle_share"].notna() & (u["population"] > 0)
    w = u.loc[ok, "population"].astype(float)
    return float((w * u.loc[ok, "hh_no_vehicle_share"]).sum() / w.sum()) if w.sum() > 0 else float("nan")


def tract_area(u) -> float:
    w = u["population"].astype(float)
    return float((w * u["area_km2"]).sum() / w.sum()) if w.sum() > 0 else float("nan")


def road_density(u) -> float:
    b = _built(u)
    a = float(b["area_km2"].sum())
    return float(b["road_km"].sum()) / a if a > 0 else float("nan")


def poi_mix(u) -> float:
    tot = np.array([float(u[f"n_{c}"].sum()) for c in POI_CLASSES if f"n_{c}" in u])
    tot = tot[tot > 0]
    if not len(tot):
        return float("nan")
    s = tot / tot.sum()
    return float(-(s * np.log(s)).sum())


def jobs_housing(u) -> float:
    J, P = float(u["jobs"].sum()), float(u["population"].sum())
    if J <= 0 or P <= 0:
        return float("nan")
    return float(0.5 * np.abs(u["jobs"] / J - u["population"] / P).sum())


def cbd_xy(u, cbd_units) -> tuple:
    c = u[u["unit"].isin(set(cbd_units))]
    w = c["jobs"].astype(float)
    return float((w * c["x"]).sum() / w.sum()), float((w * c["y"]).sum() / w.sum())


def compactness(u, cbd) -> float:
    d = np.hypot(u["x"] - cbd[0], u["y"] - cbd[1])
    P = float(u["population"].sum())
    return float(u.loc[d <= CBD_RADIUS_M, "population"].sum() / P) if P > 0 else float("nan")


def circuity_from_lines(lines, points, n_pairs=500, dmin_km=2.0, dmax_km=20.0, min_valid=100, seed=20261110) -> float:
    """Median network / straight-line ratio over up to n_pairs random point pairs dmin-dmax km apart, each point
    snapped to the nearest road vertex; graph = consecutive vertices of the lines (coordinates rounded to 1 m);
    pairs without a path are dropped; NaN with fewer than min_valid ratios."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString, MultiLineString
    segs = []
    for g in lines:
        parts = g.geoms if isinstance(g, MultiLineString) else [g] if isinstance(g, LineString) else []
        for p in parts:
            c = np.round(np.asarray(p.coords, float)[:, :2])
            segs.extend(zip(map(tuple, c[:-1]), map(tuple, c[1:])))
    if not segs:
        return float("nan")
    nodes = {}
    for a, b in segs:
        nodes.setdefault(a, len(nodes))
        nodes.setdefault(b, len(nodes))
    xy = np.array(list(nodes), float)
    i = np.array([nodes[a] for a, _ in segs])
    j = np.array([nodes[b] for _, b in segs])
    w = np.hypot(*(xy[i] - xy[j]).T)
    keep = w > 0
    G = coo_matrix((w[keep], (i[keep], j[keep])), shape=(len(xy), len(xy))).tocsr()
    snap = cKDTree(xy).query(np.asarray(points, float))[1]
    P = xy[snap]
    rng = np.random.default_rng(seed)
    a, b = np.triu_indices(len(P), 1)
    dd = np.hypot(*(P[a] - P[b]).T) / 1000.0
    ok = np.flatnonzero((dd >= dmin_km) & (dd <= dmax_km) & (snap[a] != snap[b]))
    if not len(ok):
        return float("nan")
    pick = rng.choice(ok, size=min(n_pairs, len(ok)), replace=False)
    src = np.unique(snap[a[pick]])
    dist = dijkstra(G, directed=False, indices=src)
    row = {s: k for k, s in enumerate(src)}
    r = [dist[row[snap[a[k]]], snap[b[k]]] / (dd[k] * 1000.0) for k in pick]
    r = [x for x in r if np.isfinite(x)]
    return float(np.median(r)) if len(r) >= min_valid else float("nan")


def compute(city: str) -> dict:
    import geopandas as gpd
    L = config.line_constants("M")
    base = L["MI"] / city / "tract"
    u = pd.read_parquet(base / "units.parquet")
    rep = json.loads((L["MI"] / city / "form_report.json").read_text())["tract"]
    fm = pd.read_csv(L["RESULTS"] / "form_metrics.csv", dtype={"city": str}).set_index("city")
    roads = gpd.read_file(base / "roads_builtup.gpkg").geometry.tolist() if (base / "roads_builtup.gpkg").exists() else []
    b = _built(u)
    cbd = cbd_xy(u, rep.get("cbd_units", [])) if rep.get("cbd_units") else (float("nan"), float("nan"))
    return {"city": city, "transit_per_1k": transit_per_1k(u), "no_car_share": no_car_share(u),
            "road_density": road_density(u), "poi_mix": poi_mix(u), "jobs_housing": jobs_housing(u),
            "compactness": compactness(u, cbd) if cbd[0] == cbd[0] else float("nan"),
            "secondary_centres": (float(fm.loc[city, "subcentre_job_share"] - fm.loc[city, "cbd_job_share"])
                                  if city in fm.index else float("nan")),
            "circuity": circuity_from_lines(roads, b[["x", "y"]].to_numpy(float), **CIRCUITY),
            "tract_area": tract_area(u)}


def cities_for(tag: str) -> list:
    if tag == "discovery":
        return config.cities_with_role("search") + config.cities_with_role("discovery")
    if tag in ("lawtest", "confirm"):
        return config.cities_with_role(tag)
    raise ValueError(f"city group {tag!r}: one of {tuple(COV_FILES)}")


def main(cities, name: str = "covariates.csv") -> pd.DataFrame:
    """Covariates of the given cities -> results/multicity/p7/<name>; name must be one of COV_FILES' files and the
    file must not exist yet (plan-7 files are never overwritten)."""
    import p7seal
    cities = [str(c) for c in cities]
    p7seal.guard_cities(cities)
    if name not in COV_FILES.values():
        raise ValueError(f"covariate file {name!r}: one of {sorted(COV_FILES.values())}")
    out = config.line_constants("M")["RESULTS"] / "p7"
    if (out / name).exists():
        raise FileExistsError(f"{out / name} exists: covariate files are written once")
    rows = []
    for c in cities:
        try:
            rows.append(compute(c))
        except Exception as e:  # noqa: BLE001 — reported per city, the others go on
            print(c, "FAILED", repr(e), flush=True)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / name, index=False)
    return df


if __name__ == "__main__":
    if sys.argv[1:2] != ["run"] or len(sys.argv) != 3 or sys.argv[2] not in COV_FILES:
        sys.exit(__doc__)
    import ledger
    tag = sys.argv[2]
    cs = cities_for(tag)
    d = main(cs, name=COV_FILES[tag])
    p = config.line_constants("M")["RESULTS"] / "p7" / COV_FILES[tag]
    miss = {k: int(d[k].isna().sum()) for k in DEFS if k in d}
    ledger.add(f"P2 covariates ({tag}): {len(d)} of {len(cs)} cities; missing per covariate {miss}; {p.name} sha {ledger.sha(p)}")
