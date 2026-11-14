"""Model inputs for the L line (as specified in the L-line study plan and protocol).

Per region and scale (tract main, ZCTA parallel):
  units.parquet    unit, x/y (EPSG:3310 m; TIGER internal point for tracts, representative
                   point for ZCTAs), population, area_km2 (land), income, ses (pop-weighted
                   tertile 1..3, 0 = missing), jobs (LODES 2018 WAC C000), road_km (2019 OSM
                   motorway..tertiary incl. links), pop_potential (sum_k pop_k exp(-d/5 km),
                   k != j; geometric, no OD input), origin_included, facility columns
                   (n_<cls>, N, park_share) for the main (ca190101) and current (ca_latest) snapshots
  T_<split>.npz    dense N x N visit counts (n_dev_5 main; n_dev_10, n_dev_20 also stored),
                   summed over the split's dates with the incomplete raw days removed
  D.npy            C0 distance matrix (km)
Splits: train 2018-12-15..2019-01-28, val 01-29..02-12, test 02-13..02-27; incomplete days
2019-01-07, 02-01, 02-23 removed (as specified in the protocol).
Usage: python -I prepare_l_inputs.py REGION
"""
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(os.environ.get("ODSR_ROOT", Path(__file__).resolve().parents[2]))
MNR = Path(os.environ.get("ODSR_MNR", ROOT / "external/mnr"))
CRS_M = "EPSG:3310"
SPLITS = {"train": ("2018-12-15", "2019-01-28"), "val": ("2019-01-29", "2019-02-12"), "test": ("2019-02-13", "2019-02-27")}
BAD_DAYS = {"2019-01-07", "2019-02-01", "2019-02-23"}
POT_SCALE_KM = 5.0
TIGER = "zip://" + str(ROOT / "external/tiger/raw/tl_2010_06_tract10.zip")
RENAME = {"06037930401": "06037137000"}


def ses_tertile(income, pop):
    s = pd.Series(0, index=income.index)
    ok = income.notna() & (pop > 0)
    order = income[ok].sort_values().index
    c = pop[order].cumsum() / pop[order].sum()
    s[order] = np.where(c <= 1 / 3, 1, np.where(c <= 2 / 3, 2, 3))
    return s


def road_km(units_m, roads_m):
    tree = shapely.STRtree(roads_m)
    out = np.zeros(len(units_m))
    for k, g in enumerate(units_m):
        idx = tree.query(g, predicate="intersects")
        if len(idx):
            out[k] = shapely.length(shapely.intersection(roads_m[idx], g)).sum() / 1000
    return out


def load_roads(bounds):
    import json as _j
    geoms = []
    x0, y0, x1, y1 = bounds
    with open(ROOT / "derived/osm/ca190101_mainroads.geojsonseq") as f:
        for line in f:
            g = _j.loads(line)["geometry"]
            xs = [c[0] for c in g["coordinates"]]
            ys = [c[1] for c in g["coordinates"]]
            if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:
                continue
            geoms.append(shapely.geometry.shape(g))
    return gpd.GeoSeries(geoms, crs="EPSG:4326").to_crs(CRS_M).to_numpy()


def main(region):
    for scale in ("tract", "zcta"):
        out = ROOT / f"derived/model_inputs/L/{region}/{scale}"
        out.mkdir(parents=True, exist_ok=True)
        if scale == "tract":
            u = gpd.read_file(ROOT / f"units/{region}_tracts.geojson")
            t = gpd.read_file(TIGER, columns=["GEOID10", "INTPTLAT10", "INTPTLON10"], ignore_geometry=True)
            t["GEOID"] = t["GEOID10"].replace(RENAME)
            u = u.merge(t[["GEOID", "INTPTLAT10", "INTPTLON10"]], on="GEOID", how="left")
            pt = gpd.GeoSeries(gpd.points_from_xy(u.INTPTLON10.astype(float), u.INTPTLAT10.astype(float)), crs="EPSG:4326")
            acs = pd.read_csv(ROOT / "external/acs/acs5_2018_tract_CA.csv", dtype={"tract": str}).set_index("tract")
        else:
            u = gpd.read_file(MNR / f"{region}/raw_data/{region}.geojson")[["GEOID", "geometry"]]
            u["GEOID"] = u["GEOID"].astype(str).str.zfill(5)
            pt = u.geometry.representative_point()
            acs = pd.read_csv(ROOT / "external/acs/acs5_2018_zcta.csv", dtype={"zcta": str}).set_index("zcta")
        u = u.rename(columns={"GEOID": "unit"}).reset_index(drop=True)
        u_m = u.to_crs(CRS_M)
        pm = pt.to_crs(CRS_M)
        df = pd.DataFrame({"unit": u["unit"], "x": pm.x.to_numpy(), "y": pm.y.to_numpy()})
        df["area_km2"] = u["aland_km2"].to_numpy() if "aland_km2" in u else u_m.area.to_numpy() / 1e6
        a = acs.reindex(df["unit"])
        df["population"] = a["population"].fillna(0).to_numpy()
        df["income"] = a["med_hh_income"].to_numpy()
        df["hh_no_vehicle_share"] = (a["hh_no_vehicle"] / a["households"]).to_numpy()
        df["ses"] = ses_tertile(df.set_index("unit")["income"], df.set_index("unit")["population"]).to_numpy()

        # LODES7 2018 WAC jobs (2010 blocks; LODES8 uses 2020 blocks and does not match 2010 tracts) -> unit (tract by GEOID prefix; ZCTA by block internal point
        # is not available, so ZCTA jobs come from tract jobs split by tract-ZCTA area overlap)
        wac = pd.read_csv(ROOT / "external/lodes/lodes7/ca_wac_S000_JT00_2018.csv.gz", usecols=["w_geocode", "C000"], dtype={"w_geocode": str})
        tract_jobs = wac.groupby(wac["w_geocode"].str[:11])["C000"].sum().rename(index=RENAME)
        if scale == "tract":
            df["jobs"] = tract_jobs.reindex(df["unit"]).fillna(0).to_numpy()
        else:
            tr = gpd.read_file(TIGER, columns=["GEOID10", "geometry"]).to_crs(CRS_M)
            tr["GEOID10"] = tr["GEOID10"].replace(RENAME)
            tr = tr[tr.intersects(u_m.union_all())]
            tr["jobs"] = tract_jobs.reindex(tr["GEOID10"]).fillna(0).to_numpy()
            tr["a"] = tr.area
            ov = gpd.overlay(tr[["GEOID10", "jobs", "a", "geometry"]], u_m[["unit", "geometry"]], how="intersection")
            ov["jobs_part"] = ov["jobs"] * ov.area / ov["a"]
            df["jobs"] = ov.groupby("unit")["jobs_part"].sum().reindex(df["unit"]).fillna(0).to_numpy()

        roads = load_roads(u.to_crs("EPSG:4326").total_bounds)
        df["road_km"] = road_km(u_m.geometry.to_numpy(), roads)

        xy = df[["x", "y"]].to_numpy()
        D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) / 1000.0
        np.save(out / "D.npy", D.astype(np.float32))
        K = np.exp(-D / POT_SCALE_KM)
        np.fill_diagonal(K, 0)
        df["pop_potential"] = K @ df["population"].to_numpy()

        flows_dir = ROOT / f"derived/cuebiq_ca/flows/{region}"
        uc = pd.read_csv(flows_dir / f"units_{scale}.csv", dtype={"unit": str})
        if scale == "zcta":
            uc["unit"] = uc["unit"].str.zfill(5)
        df = df.merge(uc[["unit", "home_devices", "origin_included"]], on="unit", how="left")
        for snap in ("ca190101", "ca_latest"):
            fac = pd.read_csv(ROOT / f"derived/osm/{snap}/{region}_{scale}_facilities.csv", dtype={"unit": str})
            if scale == "zcta":
                fac["unit"] = fac["unit"].str.zfill(5)
            fac = fac.drop(columns=["area_km2"]).set_index("unit")
            suffix = "" if snap == "ca190101" else "_latest"
            df = df.join(fac.add_suffix(suffix), on="unit")
        df.to_parquet(out / "units.parquet", index=False)

        idx = pd.Index(df["unit"])
        fl = pd.read_parquet(flows_dir / f"flows_{scale}.parquet")
        if scale == "zcta":
            fl["o"], fl["d"] = fl["o"].str.zfill(5), fl["d"].str.zfill(5)
        fl = fl[~fl["date"].isin(BAD_DAYS)]
        bud = pd.read_parquet(flows_dir / f"budget_{scale}.parquet")
        rep = {"region": region, "scale": scale, "units": int(len(df)), "splits": {}}
        for sp, (a0, a1) in SPLITS.items():
            f = fl[(fl["date"] >= a0) & (fl["date"] <= a1)]
            oi, di = idx.get_indexer(f["o"]), idx.get_indexer(f["d"])
            mats = {}
            for col in ("n_dev_5", "n_dev_10", "n_dev_20"):
                M = np.zeros((len(df), len(df)), dtype=np.float32)
                np.add.at(M, (oi, di), f[col].to_numpy())
                mats[col] = M
            np.savez_compressed(out / f"T_{sp}.npz", **mats)
            b = bud[(bud["date"] >= a0) & (bud["date"] <= a1) & ~bud["date"].isin(BAD_DAYS)]
            rep["splits"][sp] = {"dates": int(f["date"].nunique()), "visits": int(mats["n_dev_5"].sum()),
                                 "active_device_days": int(b["active_devices"].sum())}
        rep["origins_included"] = int(df["origin_included"].sum())
        rep["ses_counts"] = df["ses"].value_counts().sort_index().to_dict()
        (out / "prepare_report.json").write_text(json.dumps(rep, indent=1, default=int))
        print(json.dumps(rep, default=int))


if __name__ == "__main__":
    main(sys.argv[1])
