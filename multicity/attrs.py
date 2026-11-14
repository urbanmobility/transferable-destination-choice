"""National tract attributes on 2010 geography for the counties of the city table (attribute data).

download(): LODES7 WAC (2010 blocks), TIGER 2010 tracts per state, TIGER 2010 AREAWATER per city county — skips
            files already present; ACS is fetched on the development machine with scripts/fetch_acs_api.py (the key
            lives there) and rsynced.
build():    derived/multicity/tract_attrs.parquet  tract, state, county, population, income, households, hh_no_vehicle,
                                                   jobs, aland_km2, x, y (EPSG:5070 metres of the TIGER internal point)
            units/M/tracts_<SS>.gpkg               GEOID, aland_km2, geometry (EPSG:4326), ALAND10 > 0 only
Usage: python -I attrs.py download | build
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ACS_YEAR, CITIES_CSV, CONUS_STATES, CRS_M, DERIVED, LODES_YEAR, ROOT  # noqa: E402

RENAME = {"06037930401": "06037137000"}        # 2012 Census correction of an LA County tract code (round 1)
LODES_URL = "https://lehd.ces.census.gov/data/lodes/LODES7/{st}/wac/{st}_wac_S000_JT00_{y}.csv.gz"
TRACT_URL = "https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_{ss}_tract10.zip"
# TIGER2010 has no per-county AREAWATER download (404 on 11-03); TIGER2019 is the last vintage on 2010-era county codes
WATER_URL = "https://www2.census.gov/geo/tiger/TIGER2019/AREAWATER/tl_2019_{c}_areawater.zip"


def _counties(cities_csv, states=None):
    c = pd.read_csv(cities_csv, dtype=str)
    out = sorted({k for s in c["counties"] for k in s.split(";")})
    return [k for k in out if states is None or k[:2] in states]


def _fetch(url, dest, tries=4):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "404"
            time.sleep(10 * (a + 1))
        except Exception:
            time.sleep(10 * (a + 1))
    return "fail"


def download(root, cities_csv=CITIES_CSV, states=None, lodes_year=LODES_YEAR):
    root = Path(root)
    states = states or sorted({c[:2] for c in _counties(cities_csv)})
    rep = {"lodes": {}, "tract": {}, "water": {}, "fallback": []}
    for ss in states:
        st = CONUS_STATES[ss].lower()
        r = _fetch(LODES_URL.format(st=st, y=lodes_year), root / f"external/lodes/lodes7/{st}_wac_S000_JT00_{lodes_year}.csv.gz")
        if r == "404":
            rep["fallback"].append(ss)
            r = _fetch(LODES_URL.format(st=st, y=lodes_year - 1),
                       root / f"external/lodes/lodes7/{st}_wac_S000_JT00_{lodes_year - 1}.csv.gz")
        rep["lodes"][ss] = r
        rep["tract"][ss] = _fetch(TRACT_URL.format(ss=ss), root / f"external/tiger/raw/tl_2010_{ss}_tract10.zip")
    for c in _counties(cities_csv, states):
        rep["water"][c] = _fetch(WATER_URL.format(c=c), root / f"external/tiger/areawater/tl_2019_{c}_areawater.zip")
    return rep


def build(root, cities_csv, out_dir, geom_dir, states=None, lodes_year=LODES_YEAR, acs_year=ACS_YEAR):
    root, out_dir, geom_dir = Path(root), Path(out_dir), Path(geom_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geom_dir.mkdir(parents=True, exist_ok=True)
    counties = _counties(cities_csv, states)
    states = sorted({c[:2] for c in counties})
    parts = []
    for ss in states:
        abbr = CONUS_STATES[ss]
        keep = {c for c in counties if c[:2] == ss}
        g = gpd.read_file("zip://" + str(root / f"external/tiger/raw/tl_2010_{ss}_tract10.zip"))
        g = g[g["GEOID10"].str[:5].isin(keep) & (g["ALAND10"].astype(float) > 0)].copy()
        g["GEOID"] = g["GEOID10"].replace(RENAME)
        g["aland_km2"] = g["ALAND10"].astype(float) / 1e6
        g[["GEOID", "aland_km2", "geometry"]].to_crs("EPSG:4326").to_file(geom_dir / f"tracts_{ss}.gpkg", driver="GPKG")
        pt = gpd.GeoSeries(gpd.points_from_xy(g["INTPTLON10"].astype(float), g["INTPTLAT10"].astype(float)),
                           crs="EPSG:4326").to_crs(CRS_M)
        df = pd.DataFrame({"tract": g["GEOID"].to_numpy(), "aland_km2": g["aland_km2"].to_numpy(),
                           "x": pt.x.to_numpy(), "y": pt.y.to_numpy()})
        acs = pd.read_csv(root / f"external/acs/acs5_{acs_year}_tract_{abbr}.csv", dtype={"tract": str}).set_index("tract")
        a = acs.reindex(df["tract"])
        df["population"] = a["population"].fillna(0).to_numpy()
        df["income"] = a["med_hh_income"].to_numpy()
        df["households"] = a["households"].fillna(0).to_numpy()
        df["hh_no_vehicle"] = a["hh_no_vehicle"].fillna(0).to_numpy()
        lodes = root / f"external/lodes/lodes7/{abbr.lower()}_wac_S000_JT00_{lodes_year}.csv.gz"
        if not lodes.exists():
            lodes = root / f"external/lodes/lodes7/{abbr.lower()}_wac_S000_JT00_{lodes_year - 1}.csv.gz"
        wac = pd.read_csv(lodes, usecols=["w_geocode", "C000"], dtype={"w_geocode": str})
        jobs = wac.groupby(wac["w_geocode"].str[:11])["C000"].sum().rename(index=RENAME)
        df["jobs"] = jobs.reindex(df["tract"]).fillna(0).to_numpy()
        parts.append(df)
    t = pd.concat(parts, ignore_index=True)
    t["state"], t["county"] = t["tract"].str[:2], t["tract"].str[:5]
    t = t[["tract", "state", "county", "population", "income", "households", "hh_no_vehicle", "jobs", "aland_km2", "x", "y"]]
    t.to_parquet(out_dir / "tract_attrs.parquet", index=False)
    return {"states": states, "tracts": int(len(t)), "counties": len(counties)}


if __name__ == "__main__":
    if sys.argv[1] == "download":
        print(json.dumps(download(ROOT), indent=1))
    else:
        print(json.dumps(build(ROOT, CITIES_CSV, DERIVED, ROOT / "units/M")))
