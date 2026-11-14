"""Urban-form variables and the intervening-opportunity matrix for the SR round (design §5.1).

Per region and scale (tract, zcta):
  form.parquet  unit, pop_dens, job_dens (per km2 land), sub_id (employment subcentre, -1 = none),
                r_sub (km to the nearest subcentre member unit), r_cbd (km to the CBD subcentre),
                acc_jobs (sum_k E_k exp(-d_ik / 10 km), k != i)
  S.npy         S[i, j] = population of units strictly closer to i than j (excluding i and j)
Subcentres: Giuliano-Small rule, contiguous units with job density >= 10 jobs/acre and >= 10,000
jobs in total; if a region has fewer than 2, the density threshold is lowered to 8 then 6 jobs/acre
(reported). CBD = the subcentre with the most jobs. Adjacency = polygons that touch or overlap.
No flow data are used.
Usage: python -I form_vars.py REGION [REGION ...]
"""
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sr_data import LINE, ROOT  # noqa: E402

MNR = Path(os.environ.get("ODSR_MNR", ROOT / "external/mnr"))


def units_geojson(region):
    """Tract polygons of a region: round 1 keeps units/<R>_tracts.geojson, other lines use units/<LINE>/."""
    return ROOT / (f"units/{region}_tracts.geojson" if LINE == "L" else f"units/{LINE}/{region}_tracts.geojson")
CRS_M = "EPSG:3310"
ACRE_KM2 = 0.00404686
ACC_KM = 10.0


def intervening(D, pop):
    """S[i, j] = sum of pop[k] over k != i, j with D[i, k] < D[i, j] (ties excluded)."""
    N = len(pop)
    S = np.empty((N, N))
    for i in range(N):
        d = D[i].copy()
        d[i] = np.inf                       # i itself never counts
        order = np.argsort(d, kind="stable")
        ds, ps = d[order], pop[order]
        cum = np.concatenate([[0.0], np.cumsum(ps)])
        first = np.searchsorted(ds, ds, side="left")   # units strictly closer than each j
        row = np.empty(N)
        row[order] = cum[first]
        S[i] = row
    S[np.arange(N), np.arange(N)] = 0.0
    return S


def subcenters(units, adj, dens_min=10 / ACRE_KM2, total_min=10000):
    """Contiguous dense-employment clusters with enough total jobs; returns a label per unit."""
    jobs = units["jobs"].to_numpy(float)
    dens = jobs / units["area_km2"].clip(lower=1e-6).to_numpy(float)
    dense = dens >= dens_min
    lab = np.full(len(jobs), -1)
    seen = np.zeros(len(jobs), bool)
    nxt = 0
    for s in np.flatnonzero(dense):
        if seen[s]:
            continue
        comp, stack = [], [s]
        seen[s] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for m in adj.get(k, []):
                if dense[m] and not seen[m]:
                    seen[m] = True
                    stack.append(m)
        if jobs[comp].sum() >= total_min:
            lab[comp] = nxt
            nxt += 1
    return lab


def adjacency(geom):
    g = gpd.GeoDataFrame(geometry=geom.reset_index(drop=True))
    j = gpd.sjoin(g, g, predicate="intersects")
    j = j[j.index != j["index_right"]]
    adj = {k: [] for k in range(len(g))}
    for a, b in zip(j.index, j["index_right"]):
        adj[int(a)].append(int(b))
    return adj


def main(region, scales=("tract", "zcta")):
    rep = {}
    for scale in scales:
        base = ROOT / f"derived/model_inputs/{LINE}/{region}/{scale}"
        u = pd.read_parquet(base / "units.parquet")
        if scale == "tract":
            g = gpd.read_file(units_geojson(region))[["GEOID", "geometry"]]
        else:
            g = gpd.read_file(MNR / f"{region}/raw_data/{region}.geojson")[["GEOID", "geometry"]]
            g["GEOID"] = g["GEOID"].astype(str).str.zfill(5)
        g = g.set_index("GEOID").reindex(u["unit"]).to_crs(CRS_M)
        adj = adjacency(g.geometry)
        D = np.load(base / "D.npy").astype(np.float64)
        area = u["area_km2"].clip(lower=0.01).to_numpy(float)
        thr_used, sc = None, None
        for thr in (10, 8, 6):
            sc = subcenters(u, adj, dens_min=thr / ACRE_KM2)
            thr_used = thr
            if sc.max() + 1 >= 2:
                break
        n_sc = int(sc.max() + 1)
        jobs = u["jobs"].to_numpy(float)
        cbd = int(np.argmax([jobs[sc == k].sum() for k in range(n_sc)])) if n_sc else -1
        r_sub = D[:, sc >= 0].min(1) if n_sc else np.full(len(u), np.nan)
        r_cbd = D[:, sc == cbd].min(1) if n_sc else np.full(len(u), np.nan)
        W = np.exp(-D / ACC_KM)
        np.fill_diagonal(W, 0.0)
        out = pd.DataFrame({"unit": u["unit"], "pop_dens": u["population"].to_numpy(float) / area,
                            "job_dens": jobs / area, "sub_id": sc, "r_sub": r_sub, "r_cbd": r_cbd,
                            "acc_jobs": W @ jobs})
        out.to_parquet(base / "form.parquet", index=False)
        np.save(base / "S.npy", intervening(D, u["population"].to_numpy(float)).astype(np.float32))
        rep[scale] = {"units": len(u), "density_threshold_jobs_per_acre": thr_used, "n_subcentres": n_sc,
                      "cbd_subcentre": cbd,
                      "subcentre_jobs": [float(jobs[sc == k].sum()) for k in range(n_sc)],
                      "subcentre_units": [int((sc == k).sum()) for k in range(n_sc)],
                      "cbd_units": u["unit"][sc == cbd].tolist()[:20]}
    (ROOT / f"derived/model_inputs/{LINE}/{region}/form_report.json").write_text(json.dumps(rep, indent=1))
    print(region, json.dumps({s: {k: v for k, v in r.items() if k in ("n_subcentres", "density_threshold_jobs_per_acre")}
                               for s, r in rep.items()}), flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r)
