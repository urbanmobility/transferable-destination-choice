"""Urban-form metrics per city (registered before the search starts, not changed after the freeze):
1. pop_density_gini   population-weighted Gini of tract population density (low = evenly spread)
2. cbd_job_share, subcentre_job_share   jobs in the CBD subcentre / in all subcentres, over the city's jobs
3. n_subcentres       Giuliano-Small subcentres (scripts/sr/form_vars.py)
4. orientation_entropy, orientation_order   Boeing (2019): 36 x 10-degree bins of main-road bearings (both
                      directions, length-weighted) inside built-up tracts; order = 1 - ((H - ln 4)/(ln 36 - ln 4))^2
5. water_share        TIGER AREAWATER area inside the convex hull of the built-up tracts / hull area
Output: results/multicity/form_metrics/<city>.json ; results/multicity/form_metrics.csv
Usage: ODSR_ROOT=... python -I form_metrics.py CITY [CITY ...]
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CITIES_CSV, CRS_M, MIN_ORIGIN_DEVICES, RESULTS, ROOT  # noqa: E402
from cities import counties_of  # noqa: E402

N_BINS = 36


def gini(values, weights):
    """Weighted Gini coefficient (0 = all equal)."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    o = np.argsort(v)
    v, w = v[o], w[o] / w.sum()
    tot = (v * w).sum()
    if tot <= 0:
        return float("nan")
    s = np.cumsum(v * w) / tot
    s_prev = np.r_[0.0, s[:-1]]
    return float(1 - (w * (s_prev + s)).sum())


def _segments(lines):
    """Bearings (degrees from north, both directions) and lengths of every segment of the lines."""
    b, l = [], []
    for g in lines:
        parts = [p for p in shapely.get_parts(shapely.get_parts(g)) if p.geom_type == "LineString"]   # flattens collections
        for p in parts:
            c = np.asarray(p.coords)
            d = np.diff(c, axis=0)
            ln = np.hypot(d[:, 0], d[:, 1])
            ang = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
            b += [ang, (ang + 180) % 360]
            l += [ln, ln]
    if not b:
        return np.array([]), np.array([])
    return np.concatenate(b), np.concatenate(l)


def orientation_entropy(lines):
    b, l = _segments(lines)
    if len(b) == 0 or l.sum() == 0:
        return float("nan")
    bins = np.floor(np.round(b, 6) / (360 / N_BINS)).astype(int) % N_BINS     # round: 29.9999999 belongs to the 30 bin
    h = np.bincount(bins, weights=l, minlength=N_BINS)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def orientation_order(lines):
    h = orientation_entropy(lines)
    return float(1 - ((h - np.log(4)) / (np.log(N_BINS) - np.log(4))) ** 2)


def water_share(city, root, cities_csv, built_union_m):
    polys = []
    for c in counties_of(city, cities_csv):
        f = Path(root) / f"external/tiger/areawater/tl_2019_{c}_areawater.zip"
        if f.exists():
            polys.append(gpd.read_file("zip://" + str(f))[["geometry"]].to_crs(CRS_M))
    if not polys or built_union_m.is_empty:
        return float("nan")
    hull = built_union_m.convex_hull
    water = pd.concat(polys).union_all()
    return float(shapely.intersection(water, hull).area / hull.area)


def compute(city, root=ROOT, cities_csv=CITIES_CSV):
    root = Path(root)
    base = root / f"derived/model_inputs/M/{city}/tract"
    u = pd.read_parquet(base / "units.parquet")
    rep = json.loads((root / f"derived/model_inputs/M/{city}/form_report.json").read_text())["tract"]
    pop, area, jobs = u["population"].to_numpy(float), u["area_km2"].clip(lower=0.01).to_numpy(float), u["jobs"].to_numpy(float)
    ok = pop > 0
    jt = jobs.sum()
    sj = rep.get("subcentre_jobs", [])
    cbd = rep.get("cbd_subcentre", -1)
    roads = gpd.read_file(base / "roads_builtup.gpkg").geometry.tolist() if (base / "roads_builtup.gpkg").exists() else []
    g = gpd.read_file(root / f"units/M/{city}_tracts.geojson").set_index("GEOID").to_crs(CRS_M)
    built = g.loc[u.loc[u["home_devices"] >= MIN_ORIGIN_DEVICES, "unit"]].union_all() if (u["home_devices"] >= MIN_ORIGIN_DEVICES).any() else shapely.Polygon()
    m = {"city": city, "population": float(pop.sum()), "jobs": float(jt),
         "pop_density_gini": gini(pop[ok] / area[ok], pop[ok]) if ok.any() else float("nan"),
         "cbd_job_share": float(sj[cbd] / jt) if (cbd >= 0 and jt > 0 and sj) else 0.0,
         "subcentre_job_share": float(sum(sj) / jt) if jt > 0 else 0.0,
         "n_subcentres": int(rep.get("n_subcentres", 0)),
         "orientation_entropy": orientation_entropy(roads), "orientation_order": orientation_order(roads),
         "water_share": water_share(city, root, cities_csv, built)}
    out = root / "results/multicity/form_metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{city}.json").write_text(json.dumps(m, indent=1))
    return m


def main(cities):
    rows = []
    for c in cities:
        try:
            rows.append(compute(c))
            print(c, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items() if k != "city"}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(c, "FAILED", repr(e), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "form_metrics.csv", index=False)
    return df


if __name__ == "__main__":
    main(sys.argv[1:])
