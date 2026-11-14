"""A synthetic two-state city (TEST1: fake counties 06999 and 32999, three 1-km tracts each) with every national
input of the pipeline laid out under one root, for the units / qc / osm / tiers tests."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

TRACTS = ["06999000100", "06999000200", "06999000300", "32999000100", "32999000200", "32999000300"]
OUT_CITY = "06073000100"      # a destination outside the city
UNMATCHED = "06999999900"     # an origin tract that is not in the unit table


def _square(lon, lat, w=0.009):
    return shapely.box(lon, lat, lon + w, lat + w)


def make_city(root: Path, cities_csv: Path):
    root = Path(root)
    (root / "derived/multicity/trips/o_state=06").mkdir(parents=True)
    (root / "derived/multicity/trips/o_state=32").mkdir(parents=True)
    (root / "derived/multicity/budget").mkdir(parents=True)
    (root / "units/M").mkdir(parents=True)
    pd.DataFrame({"city": ["TEST1"], "name": ["Test City"], "counties": ["06999;32999"], "states": ["06;32"],
                  "n_counties": [2]}).to_csv(cities_csv, index=False)
    # geometry: 06 tracts along a row at lat 34, 32 tracts along a row at lat 36 (both 1 km apart)
    for ss, lon0, lat0 in (("06", -118.0, 34.0), ("32", -115.0, 36.0)):
        geo = [_square(lon0 + 0.01 * k, lat0) for k in range(3)]
        gpd.GeoDataFrame({"GEOID": [t for t in TRACTS if t.startswith(ss)], "aland_km2": [1.0] * 3},
                         geometry=geo, crs="EPSG:4326").to_file(root / f"units/M/tracts_{ss}.gpkg", driver="GPKG")
    pts = gpd.GeoSeries([_square(-118.0 + 0.01 * k, 34.0).centroid for k in range(3)]
                        + [_square(-115.0 + 0.01 * k, 36.0).centroid for k in range(3)], crs="EPSG:4326").to_crs("EPSG:5070")
    pd.DataFrame({"tract": TRACTS, "state": [t[:2] for t in TRACTS], "county": [t[:5] for t in TRACTS],
                  "population": [100, 200, 300, 400, 500, 600], "income": [50000, 60000, np.nan, 80000, 90000, 100000],
                  "households": [40, 80, 120, 160, 200, 240], "hh_no_vehicle": [4, 8, 12, 16, 20, 24],
                  "jobs": [10, 20, 30, 40, 50, 60], "aland_km2": [1.0] * 6,
                  "x": pts.x.to_numpy(), "y": pts.y.to_numpy()}).to_parquet(root / "derived/multicity/tract_attrs.parquet", index=False)
    pd.DataFrame({"tract": TRACTS, "home_devices": [40, 10, 35, 50, 5, 30]}).to_parquet(root / "derived/multicity/home_tracts.parquet", index=False)
    pd.DataFrame({"tract": TRACTS, "n_food": [1, 3, 1, 1, 1, 1], "n_retail": [0, 2, 0, 0, 0, 0], "n_health": 0,
                  "n_education": 0, "n_leisure": 0, "n_service": 0, "n_office": 0, "n_transit": 0,
                  "n_other": [0, 1, 0, 0, 0, 0], "N": [1, 6, 1, 1, 1, 1]}).to_parquet(root / "derived/multicity/poi_tracts.parquet", index=False)
    rows = [  # date, g, o, d, work, n
        ("2024-03-01", 0, TRACTS[0], TRACTS[1], False, 2),
        ("2024-03-01", 1, TRACTS[0], TRACTS[3], True, 1),
        ("2024-04-10", 2, TRACTS[3], TRACTS[4], False, 3),
        ("2024-04-10", 3, TRACTS[2], TRACTS[0], True, 1),
        ("2024-03-15", 4, TRACTS[5], TRACTS[3], False, 1),
        ("2024-03-01", 0, TRACTS[0], OUT_CITY, False, 1),           # destination outside the city
        ("2024-03-01", 7, UNMATCHED, TRACTS[0], False, 1),           # origin not in the unit table
        ("2024-05-05", 5, TRACTS[0], TRACTS[2], False, 2),           # val
        ("2024-05-20", 6, TRACTS[0], TRACTS[1], False, 1),           # test
    ]
    t = pd.DataFrame(rows, columns=["date", "g", "o", "d", "work", "n"])
    t["g"] = t["g"].astype("int32")
    for ss in ("06", "32"):
        part = t[t["o"].str[:2] == ss]
        part.to_parquet(root / f"derived/multicity/trips/o_state={ss}/2024-03_0.parquet", index=False)
    pd.DataFrame({"date": ["2024-03-01", "2024-03-01", "2024-04-10", "2024-05-05", "2024-05-20"],
                  "o": [TRACTS[0], UNMATCHED, TRACTS[3], TRACTS[0], TRACTS[0]],
                  "active_devices": [3, 1, 4, 2, 1]}).to_parquet(root / "derived/multicity/budget/2024-03.parquet", index=False)
    return root
