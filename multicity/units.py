"""Per-city standard tables (design §5.1), isomorphic to the round-1 model_inputs so sr_data.py reads them unchanged.

derived/model_inputs/M/<city>/tract/
  units.parquet        unit, x, y (EPSG:5070 m), area_km2, population, income, hh_no_vehicle_share, ses, jobs, road_km,
                       pop_potential, home_devices, origin_included, n_<cls> x 8, n_other, N, n_transit_osm, park_share
  D.npy                distances (km) between tract internal points
  T_<split>.npz        n_dev_5 (all trips), n_work_5, n_nonwork_5: origin x destination device-day counts, both in the city
  T_groups_<split>.npz T (20, N, N): n_dev_5 by device group
  prepare_report.json  counts, coverage of destinations outside the city, trips whose origin is not a unit
  form.parquet, S.npy, form_report.json   via scripts/sr/form_vars.py (ODSR_LINE=M)
units/M/<city>_tracts.geojson             GEOID, aland_km2 (EPSG:4326)
Usage: ODSR_ROOT=... ODSR_LINE=M python -I units.py CITY [CITY ...]
"""
import json
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from config import CITIES_CSV, CLASSES, MIN_ORIGIN_DEVICES, N_GROUPS, ROOT, SPLITS  # noqa: E402
from cities import counties_of  # noqa: E402
from prepare_l_inputs import POT_SCALE_KM, ses_tertile  # noqa: E402
import form_vars  # noqa: E402

OSM_COLS = ("road_km", "n_transit_osm", "park_share")


def load_city_tracts(city, root, cities_csv=CITIES_CSV):
    """Tract polygons of the city from the per-state gpkg files (Task 6), sorted by GEOID."""
    counties = set(counties_of(city, cities_csv))
    parts = []
    for ss in sorted({c[:2] for c in counties}):
        g = gpd.read_file(root / f"units/M/tracts_{ss}.gpkg")
        parts.append(g[g["GEOID"].str[:5].isin(counties)])
    return pd.concat(parts).sort_values("GEOID").reset_index(drop=True)


def _trips(root, city, units, counties, states, derived=None):
    """Trips with an origin in the city's counties: (date, g, o, d, work, n) with o in units, plus the unmatched count.
    derived: the line's national tables (default root/derived/multicity, the M line)."""
    derived = Path(derived) if derived is not None else Path(root) / "derived/multicity"
    con = duckdb.connect()
    con.execute("SET threads=16")
    con.execute("CREATE TABLE u AS SELECT * FROM (VALUES " + ", ".join(f"('{t}')" for t in units) + ") t(unit)")
    src = f"read_parquet('{derived}/trips/o_state=*/*.parquet', hive_partitioning=true)"
    st = ", ".join(f"'{s}'" for s in states)
    cs = ", ".join(f"'{c}'" for c in counties)
    t = con.execute(f"""
        SELECT date, g, o, d, work, n FROM {src} JOIN u ON o = u.unit
        WHERE CAST(o_state AS VARCHAR) IN ({st})""").df()
    unmatched = con.execute(f"""
        SELECT coalesce(sum(n), 0) FROM {src}
        WHERE CAST(o_state AS VARCHAR) IN ({st}) AND substr(o, 1, 5) IN ({cs}) AND o NOT IN (SELECT unit FROM u)""").fetchone()[0]
    bud = con.execute(f"""
        SELECT date, o, active_devices FROM read_parquet('{derived}/budget/*.parquet') JOIN u ON o = u.unit""").df()
    con.close()
    t["date"], bud["date"] = t["date"].astype(str), bud["date"].astype(str)
    return t, int(unmatched), bud


def build_city(city, root=ROOT, cities_csv=CITIES_CSV, osm=None, n_groups=N_GROUPS, line=None):
    """line: the data line (default config.LINE). Trips, budget, homes and POI come from the line's national tables;
    tract attributes, tract geometry and the OSM features are shared (read from the M line); every output goes to the
    line's own directories. A line without work trips (M_nw, M19) drops them before the matrices are built."""
    import config
    root = Path(root)
    line = config.line_of(line or config.LINE)
    L = config.line_constants(line, root)
    base = L["MI"] / city / "tract"
    base.mkdir(parents=True, exist_ok=True)
    counties = counties_of(city, cities_csv)
    states = sorted({c[:2] for c in counties})
    g = load_city_tracts(city, root, cities_csv)
    L["UNITS_DIR"].mkdir(parents=True, exist_ok=True)
    g[["GEOID", "aland_km2", "geometry"]].to_crs("EPSG:4326").to_file(L["UNITS_DIR"] / f"{city}_tracts.geojson", driver="GeoJSON")
    units = list(g["GEOID"])
    a = pd.read_parquet(L["SHARED"] / "tract_attrs.parquet").set_index("tract").reindex(units)
    df = pd.DataFrame({"unit": units, "x": a["x"].to_numpy(), "y": a["y"].to_numpy(), "area_km2": a["aland_km2"].to_numpy(),
                       "population": a["population"].fillna(0).to_numpy(), "income": a["income"].to_numpy(),
                       "hh_no_vehicle_share": (a["hh_no_vehicle"] / a["households"].replace(0, np.nan)).to_numpy(),
                       "jobs": a["jobs"].fillna(0).to_numpy()})
    df["ses"] = ses_tertile(df.set_index("unit")["income"], df.set_index("unit")["population"]).to_numpy()
    xy = df[["x", "y"]].to_numpy(float)
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) / 1000.0
    np.save(base / "D.npy", D.astype(np.float32))
    K = np.exp(-D / POT_SCALE_KM)
    np.fill_diagonal(K, 0)
    df["pop_potential"] = K @ df["population"].to_numpy()
    hd = pd.read_parquet(L["DERIVED"] / "home_tracts.parquet").set_index("tract")["home_devices"]
    df["home_devices"] = hd.reindex(units).fillna(0).astype(int).to_numpy()
    df["origin_included"] = df["home_devices"] >= MIN_ORIGIN_DEVICES
    poi = pd.read_parquet(L["DERIVED"] / "poi_tracts.parquet").set_index("tract").reindex(units).fillna(0)
    for c in [f"n_{k}" for k in CLASSES] + ["n_other", "N"]:
        df[c] = poi[c].astype(int).to_numpy() if c in poi else 0
    osm_file = L["MI_M"] / city / "tract" / "osm_features.parquet"      # OSM features are shared: the M line's file
    if osm is None and osm_file.exists():
        osm = pd.read_parquet(osm_file)
    if osm is None and line != "M":      # plan 7 final review: never zero-fill a sealed line's OSM columns silently
        raise RuntimeError(f"no shared OSM features for {city} ({osm_file}): run osm_city features on the M line first")
    if osm is not None:
        o = osm.set_index("unit").reindex(units)
        for c in OSM_COLS:
            df[c] = o[c].fillna(0).to_numpy()
    else:
        for c in OSM_COLS:
            df[c] = 0.0
    df.to_parquet(base / "units.parquet", index=False)

    t, unmatched, bud = _trips(root, city, units, counties, states, derived=L["DERIVED"])
    if not L["INCLUDE_WORK"]:
        t = t[~t["work"].astype(bool)]
    idx = pd.Index(units)
    N = len(units)
    rep = {"city": city, "units": N, "origins_included": int(df["origin_included"].sum()), "states": states,
           "counties": counties, "o_unmatched_trips": unmatched, "splits": {}}
    for sp, (a0, a1) in L["SPLITS"].items():
        f = t[(t["date"] >= a0) & (t["date"] <= a1)]
        inside = f["d"].isin(idx)
        fi = f[inside]
        oi, di = idx.get_indexer(fi["o"]), idx.get_indexer(fi["d"])
        n = fi["n"].to_numpy(float)
        work = fi["work"].to_numpy(bool)
        mats = {}
        for key, m in (("n_dev_5", np.ones(len(fi), bool)), ("n_work_5", work), ("n_nonwork_5", ~work)):
            M = np.zeros((N, N), dtype=np.float32)
            np.add.at(M, (oi[m], di[m]), n[m])
            mats[key] = M
        np.savez_compressed(base / f"T_{sp}.npz", **mats)
        G = np.zeros((n_groups, N, N), dtype=np.float32)
        np.add.at(G, (fi["g"].to_numpy(int), oi, di), n)
        np.savez_compressed(base / f"T_groups_{sp}.npz", T=G)
        b = bud[(bud["date"] >= a0) & (bud["date"] <= a1)]
        tot = float(f["n"].sum())
        rep["splits"][sp] = {"dates": int(f["date"].nunique()), "visits": int(mats["n_dev_5"].sum()),
                             "work_visits": int(mats["n_work_5"].sum()),
                             "active_device_days": int(b["active_devices"].sum()),
                             "coverage_out_of_city": float(f.loc[~inside, "n"].sum() / tot) if tot else float("nan")}
    (base / "prepare_report.json").write_text(json.dumps(rep, indent=1))
    if form_vars.ROOT != root or form_vars.LINE != line:
        raise RuntimeError(f"form_vars reads {form_vars.ROOT} line {form_vars.LINE}; set ODSR_ROOT / ODSR_LINE={line}")
    form_vars.main(city, scales=("tract",))
    return rep


def main(cities):
    failed = []
    for c in cities:
        try:
            r = build_city(c)
            print(c, json.dumps({k: r[k] for k in ("units", "origins_included", "o_unmatched_trips")},
                                ), json.dumps({sp: v["visits"] for sp, v in r["splits"].items()}), flush=True)
        except Exception as e:  # noqa: BLE001 — one city must not stop the batch; the failure is listed
            failed.append(c)
            print(c, "FAILED", repr(e), flush=True)
    import config
    log = config.results_dir() / "logs/units_failed.txt"          # the line's own results (plan 7 final review)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(failed) + ("\n" if failed else ""))
    print("UNITS_DONE failed", len(failed), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
