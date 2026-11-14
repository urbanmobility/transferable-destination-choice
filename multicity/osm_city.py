"""OSM features per city (road_km, transit stops, park share for M2; main roads for the form metrics).

download_states(): Geofabrik state extracts -> external/osm/us2024/<name>-latest.osm.pbf (development machine; the compute
                   server has no route to Geofabrik).
extract_state():   osmium tags-filter + export -> derived/osm/us2024/<ST>_{transit,parks,roads}.geojsonseq
tract_features():  derived/model_inputs/M/<city>/tract/osm_features.parquet (unit, road_km, n_transit_osm, park_share)
                   and roads_builtup.gpkg (main roads inside tracts with >= 30 home devices, EPSG:5070)
Transit stops are classified with the round-1 rules (scripts/osm/l1a_rules.classify == "transit"); parks are
l1a_rules.PARK_TAGS polygons, dissolved; roads are motorway..tertiary incl. links (prepare_l_inputs.road_km).
Usage: python -I osm_city.py download | extract ST [ST ...] | features CITY [CITY ...]
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "osm"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from config import CITIES_CSV, CONUS_STATES, CRS_M, MIN_ORIGIN_DEVICES, ROOT  # noqa: E402
from l1a_rules import classify, is_park  # noqa: E402
from prepare_l_inputs import road_km  # noqa: E402
from units import load_city_tracts  # noqa: E402

_NAMES = {"CA": ["california/norcal", "california/socal"], "DC": ["district-of-columbia"], "NY": ["new-york"],
          "NC": ["north-carolina"], "ND": ["north-dakota"], "NH": ["new-hampshire"], "NJ": ["new-jersey"],
          "NM": ["new-mexico"], "RI": ["rhode-island"], "SC": ["south-carolina"], "SD": ["south-dakota"],
          "WV": ["west-virginia"]}
_FULL = {"AL": "alabama", "AZ": "arizona", "AR": "arkansas", "CO": "colorado", "CT": "connecticut", "DE": "delaware",
         "FL": "florida", "GA": "georgia", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
         "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan",
         "MN": "minnesota", "MS": "mississippi", "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
         "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania", "TN": "tennessee", "TX": "texas",
         "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington", "WI": "wisconsin", "WY": "wyoming"}
GEOFABRIK = {**{k: [v] for k, v in _FULL.items()}, **_NAMES}
URL = "https://download.geofabrik.de/north-america/us/{name}-latest.osm.pbf"
PBF_DIR = "external/osm/us2024"
EXTRACT_DIR = "derived/osm/us2024"
TRANSIT_FILTER = ["n/highway=bus_stop", "n/railway=station,halt,tram_stop,subway_entrance", "nwr/public_transport=station",
                  "nwr/amenity=bus_station,ferry_terminal", "nwr/aeroway=aerodrome"]
PARK_FILTER = ["wr/leisure=park,dog_park", "wr/landuse=recreation_ground"]
ROAD_FILTER = ["w/highway=motorway,trunk,primary,secondary,tertiary,motorway_link,trunk_link,primary_link,secondary_link,tertiary_link"]


def pbf_paths(abbr, root=ROOT):
    return [Path(root) / PBF_DIR / f"{Path(n).name}-latest.osm.pbf" for n in GEOFABRIK[abbr]]


def download_states(abbrs, root=ROOT):
    rep = {}
    for ab in abbrs:
        for name, dest in zip(GEOFABRIK[ab], pbf_paths(ab, root)):
            if dest.exists() and dest.stat().st_size > 0:
                rep[name] = "skip"
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(URL.format(name=name), timeout=300) as r, open(tmp, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 24), b""):
                    f.write(chunk)
            tmp.rename(dest)
            rep[name] = "ok"
            print(name, "ok", dest.stat().st_size, flush=True)
    return rep


def extract_state(ss, root=ROOT):
    """osmium tags-filter + export of the three feature sets for one state (FIPS code). Idempotent: a state whose
    three extracts exist and are non-empty is skipped (returns "skip"); extracts are written to a .part file and
    renamed, so a failed run (e.g. no pbf on the compute server, where the extracts are synced in) never truncates an existing one."""
    ab = CONUS_STATES[ss]
    out = Path(root) / EXTRACT_DIR
    out.mkdir(parents=True, exist_ok=True)
    targets = {name: out / f"{ss}_{name}.geojsonseq" for name in ("transit", "parks", "roads")}
    if all(t.exists() and t.stat().st_size > 0 for t in targets.values()):
        return "skip"
    pbfs = pbf_paths(ab, root)
    missing = [p for p in pbfs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"state {ss}: extracts incomplete and pbf missing: {[str(p) for p in missing]}")
    for name, flt in (("transit", TRANSIT_FILTER), ("parks", PARK_FILTER), ("roads", ROAD_FILTER)):
        part = targets[name].with_suffix(".part")
        with open(part, "w") as f:
            for pbf in pbfs:
                tmp = out / f"{ss}_{name}_{pbf.stem}.pbf"
                subprocess.run(["osmium", "tags-filter", "-o", str(tmp), "--overwrite", str(pbf), *flt], check=True)
                subprocess.run(["osmium", "export", "-f", "geojsonseq", "-a", "type,id", str(tmp)], check=True, stdout=f)
                tmp.unlink()
        part.rename(targets[name])
    return out


def _read(path):
    """(geometries, properties) of a geojsonseq file, one object per (@type, @id): overlapping extracts (CA norcal /
    socal) are deduplicated, and a closed way that osmium export writes twice (LineString, then the assembled area)
    keeps its area geometry."""
    geoms, props, seen = [], [], {}
    path = Path(path)
    if not path.exists():
        return geoms, props
    with open(path) as f:
        for line in f:
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            ft = json.loads(line)
            p = ft["properties"]
            key = (p.get("@type"), p.get("@id"))
            gm = shapely.geometry.shape(ft["geometry"])
            if key in seen:
                i = seen[key]
                if gm.geom_type in ("Polygon", "MultiPolygon") and geoms[i].geom_type not in ("Polygon", "MultiPolygon"):
                    geoms[i], props[i] = gm, p
                continue
            seen[key] = len(geoms)
            geoms.append(gm)
            props.append(p)
    return geoms, props


def tract_features(city, root=ROOT, cities_csv=CITIES_CSV):
    root = Path(root)
    g = load_city_tracts(city, root, cities_csv)
    tracts = gpd.GeoDataFrame(g[["GEOID"]], geometry=g.geometry, crs=g.crs or "EPSG:4326").to_crs("EPSG:4326")
    tracts_m = tracts.to_crs(CRS_M)
    states = sorted({t[:2] for t in tracts["GEOID"]})
    units = list(tracts["GEOID"])
    out = pd.DataFrame({"unit": units, "road_km": 0.0, "n_transit_osm": 0, "park_share": 0.0}).set_index("unit")
    ex = root / EXTRACT_DIR
    # transit stops: points (representative points of areas) classified as transit, counted per tract
    pts = []
    for ss in states:
        geoms, props = _read(ex / f"{ss}_transit.geojsonseq")
        for gm, p in zip(geoms, props):
            if classify(p) == "transit":
                pts.append(gm if gm.geom_type == "Point" else gm.representative_point())
    if pts:
        hit = gpd.sjoin(gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326"), tracts, predicate="within", how="inner")
        cnt = hit.groupby("GEOID").size()
        out["n_transit_osm"] = cnt.reindex(units).fillna(0).astype(int).to_numpy()
    # parks: dissolved polygons, share of each tract's area
    parks = []
    for ss in states:
        geoms, props = _read(ex / f"{ss}_parks.geojsonseq")
        parks += [gm for gm, p in zip(geoms, props) if gm.geom_type in ("Polygon", "MultiPolygon") and is_park(p)]
    if parks:
        pu = gpd.GeoSeries(parks, crs="EPSG:4326").to_crs(CRS_M).union_all()
        inter = shapely.area(shapely.intersection(tracts_m.geometry.to_numpy(), pu))
        out["park_share"] = inter / tracts_m.area.to_numpy()
    # main roads: km per tract; the built-up subset is kept for the orientation metric
    roads = []
    for ss in states:
        geoms, _ = _read(ex / f"{ss}_roads.geojsonseq")
        roads += [gm for gm in geoms if gm.geom_type in ("LineString", "MultiLineString")]
    base = root / f"derived/model_inputs/M/{city}/tract"
    base.mkdir(parents=True, exist_ok=True)
    if roads:
        rm = gpd.GeoSeries(roads, crs="EPSG:4326").to_crs(CRS_M).to_numpy()
        out["road_km"] = road_km(tracts_m.geometry.to_numpy(), rm)
        hd = pd.read_parquet(root / "derived/multicity/home_tracts.parquet").set_index("tract")["home_devices"]
        built = tracts_m[hd.reindex(units).fillna(0).to_numpy() >= MIN_ORIGIN_DEVICES].union_all()
        clipped = shapely.intersection(rm, built)
        keep = ~shapely.is_empty(clipped)
        gpd.GeoDataFrame(geometry=clipped[keep], crs=CRS_M).to_file(base / "roads_builtup.gpkg", driver="GPKG")
    else:
        gpd.GeoDataFrame(geometry=[], crs=CRS_M).to_file(base / "roads_builtup.gpkg", driver="GPKG")
    res = out.reset_index()
    res.to_parquet(base / "osm_features.parquet", index=False)
    return res


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "download":
        print(json.dumps(download_states(args or list(CONUS_STATES.values()))))
    elif cmd == "extract":
        for ss in args:
            print(ss, "skip (extracts present)" if extract_state(ss) == "skip" else "extracted", flush=True)
    else:
        for c in args:
            f = tract_features(c)
            print(c, json.dumps({"road_km": float(f.road_km.sum()), "transit": int(f.n_transit_osm.sum()),
                                 "park_share_mean": float(f.park_share.mean())}), flush=True)
