"""Paths and constants of the multi-city pipeline (design §3, §5.1, §5.2).

Everything date-, threshold- or list-like lives here; the ledger records this file's sha before the search starts.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
from sr_data import CLASSES, ROOT  # noqa: E402,F401  (ROOT honours ODSR_ROOT)
import sr_data as _sr  # noqa: E402

VERASET = Path(os.environ.get("ODSR_VERASET", ROOT / "raw/veraset"))   # the Veraset (Dewey) download: visits/, work-visits/, home-location/
MIN_DWELL_MIN = 5
MIN_ORIGIN_DEVICES = 30
MIN_ORIGIN_TRACTS = 150
N_GROUPS = 20
GROUP_SEED = 20261103
CRS_M = "EPSG:5070"                     # CONUS Albers, metres
ACS_YEAR = 2019                         # ACS 2015-2019 five-year, 2010 tracts
LODES_YEAR = 2019                       # LODES7 WAC, 2010 blocks

BAY9 = {"06001", "06013", "06041", "06055", "06075", "06081", "06085", "06095", "06097"}
SEARCH = ["35620", "16980", "31080", "BAY9", "26420", "12060", "42660", "33460"]      # NYC Chicago LA Bay Houston Atlanta Seattle Minneapolis
CONFIRM = ["38060", "37980", "19740", "41740", "38300", "40900"]                      # Phoenix Philadelphia Denver San Diego Pittsburgh Sacramento

CONUS_STATES = {
    "01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA",
    "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY",
}

# Data lines (plan 7, design §6.2): M = 2024 all purposes (the line of plans 1-6, unchanged), M_nw = 2024
# non-work trips (bridge check), M19 = 2019-04 non-work trips (2019 work visits carry no location; home / work rows
# are inline in visits). Each line writes its own directories; SHARED (tract attributes, tract geometry) and the
# M line's OSM files are read-only inputs of every line. ODSR_LINE selects the line (unknown, e.g. round-1 L -> M).
_MONTHS_M = ("2024-03", "2024-04", "2024-05")
_SPLITS_M = {"train": ("2024-03-01", "2024-04-30"), "val": ("2024-05-01", "2024-05-15"), "test": ("2024-05-16", "2024-05-31")}
LINE_SPEC = {
    "M": {"months": _MONTHS_M, "splits": _SPLITS_M, "min_train_trips": 500_000, "derived": "derived/multicity",
          "results": "results/multicity", "cities_csv": "cities.csv", "include_work": True, "hw_inline": False,
          "poi_id": "placekey"},
    "M_nw": {"months": _MONTHS_M, "splits": _SPLITS_M, "min_train_trips": 500_000, "derived": "derived/multicity",
             "results": "results/multicity_nw", "cities_csv": "cities.csv", "include_work": False, "hw_inline": False,
             "poi_id": "placekey"},
    "M19": {"months": ("2019-04",),
            "splits": {"train": ("2019-04-01", "2019-04-20"), "val": ("2019-04-21", "2019-04-25"),
                       "test": ("2019-04-26", "2019-04-30")},
            "min_train_trips": round(500_000 * 20 / 61), "derived": "derived/multicity19",
            "results": "results/multicity19", "cities_csv": "cities_M19.csv", "include_work": False, "hw_inline": True,
            "poi_id": "safegraph_place_id"},
}
SHARED_REL = "derived/multicity"


def line_of(sr_line) -> str:
    return sr_line if sr_line in LINE_SPEC else "M"


def line_constants(line: str, root=None) -> dict:
    """Months, splits, threshold, switches and paths of a data line under root (default: sr_data.ROOT now)."""
    s, root = LINE_SPEC[line_of(line)], Path(root) if root is not None else Path(_sr.ROOT)
    name = line_of(line)
    return {"MONTHS": s["months"], "SPLITS": s["splits"], "MIN_TRAIN_TRIPS": s["min_train_trips"],
            "INCLUDE_WORK": s["include_work"], "HW_INLINE": s["hw_inline"], "POI_ID": s["poi_id"],
            "DERIVED": root / s["derived"], "SHARED": root / SHARED_REL, "RESULTS": root / s["results"],
            "MI": root / f"derived/model_inputs/{name}", "MI_M": root / "derived/model_inputs/M",
            "UNITS_DIR": root / f"units/{name}", "CITIES_CSV": Path(__file__).resolve().parent / "config" / s["cities_csv"]}


def results_dir(line: str | None = None) -> Path:
    """The line's results directory under sr_data.ROOT as it is now (tests re-point the root)."""
    return line_constants(line or LINE)["RESULTS"]


LINE = line_of(_sr.LINE)
_C = line_constants(LINE, ROOT)
MONTHS, SPLITS, MIN_TRAIN_TRIPS = _C["MONTHS"], _C["SPLITS"], _C["MIN_TRAIN_TRIPS"]
INCLUDE_WORK, HW_INLINE, POI_ID = _C["INCLUDE_WORK"], _C["HW_INLINE"], _C["POI_ID"]
MI, DERIVED, SHARED, RESULTS = _C["MI"], _C["DERIVED"], _C["SHARED"], _C["RESULTS"]
CITIES_CSV = _C["CITIES_CSV"]

SPLIT_SEED = 20261105                   # plan 4 Task 1: tie-break seed of the discovery / lawtest split


def cities_with_role(role: str) -> list[str]:
    """City ids with the given role in config/cities.csv (search, discovery, lawtest, confirm, rest, excluded, ...)."""
    import pandas as pd
    c = pd.read_csv(CITIES_CSV, dtype=str)
    return c.loc[c["role"] == role, "city"].tolist()
