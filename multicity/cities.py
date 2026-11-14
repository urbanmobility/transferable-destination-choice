"""City table: county -> CBSA (OMB 2020 standards, July 2023 delineation file), the nine-county Bay Area as one
city (BAY9; the nine counties leave their own CBSAs, a CBSA with no county left is dropped), CONUS only.

Output: config/cities.csv   city, name, counties (';'-joined 5-digit FIPS), states (';'-joined 2-digit), n_counties
Usage: python -I cities.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BAY9, CITIES_CSV, CONUS_STATES, ROOT  # noqa: E402

METRO = "Metropolitan Statistical Area"


def build(xlsx, out_csv):
    d = pd.read_excel(xlsx, skiprows=2, dtype=str).dropna(subset=["FIPS State Code"])
    d = d[d["Metropolitan/Micropolitan Statistical Area"] == METRO].copy()
    d["county"] = d["FIPS State Code"].str.zfill(2) + d["FIPS County Code"].str.zfill(3)
    d["city"], d["name"] = d["CBSA Code"], d["CBSA Title"]
    d.loc[d["county"].isin(BAY9), ["city", "name"]] = ["BAY9", "SF Bay Area (9 county)"]
    g = d.groupby("city").agg(name=("name", "first"),
                              counties=("county", lambda s: ";".join(sorted(s))),
                              states=("county", lambda s: ";".join(sorted({c[:2] for c in s}))))
    g["n_counties"] = g["counties"].str.count(";") + 1
    conus = g["states"].apply(lambda s: all(st in CONUS_STATES for st in s.split(";")))
    g = g[conus].reset_index()
    out_csv = Path(out_csv)
    if out_csv.exists():   # keep the roles the tiers stage wrote (the list is frozen once written)
        old = pd.read_csv(out_csv, dtype=str)
        keep = [c for c in ("role", "tier", "substitute_for") if c in old.columns]
        if keep:
            g = g.merge(old[["city"] + keep], on="city", how="left")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out_csv, index=False)
    return g


def table(csv=None):
    return pd.read_csv(csv or CITIES_CSV, dtype=str).set_index("city")


def counties_of(city, csv=None):
    return table(csv).loc[city, "counties"].split(";")


def tracts_of(city, tract_index, csv=None):
    """Members of a tract index (11-digit GEOIDs) whose county is in the city."""
    c = set(counties_of(city, csv))
    return tract_index[[t[:5] in c for t in tract_index]]


if __name__ == "__main__":
    g = build(ROOT / "external/cbsa/list1_2023.xlsx", CITIES_CSV)
    print(len(g), "cities;", g.loc[g["city"] == "BAY9", "n_counties"].item(), "Bay Area counties")
