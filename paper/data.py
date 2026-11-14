"""Load the exported per-panel tables (see multicity/paper_figdata.py and multicity/od_maps.py)."""
import json

import pandas as pd

import style

_CSV = ("cities", "bands", "shares", "size_laws", "p7", "e1_pairs", "e2_effects", "axis1", "types", "p9_target",
        "p9_summary", "p9_refs", "p9_diffs", "c3_cities", "consts", "turning")
_JSON = ("e1_summary", "p9_contrasts", "c3_summary", "alldest_summary")


def load(root=None) -> dict:
    root = root or style.DATA
    d = {k: pd.read_csv(root / f"{k}.csv", dtype={"city": str}) for k in _CSV}
    d.update({k: json.loads((root / f"{k}.json").read_text()) for k in _JSON})
    d["od"] = load_od(DENVER, root)                # Fig. 6
    return d


DENVER = "19740"


def od_cities(root=None) -> list:
    """The cities whose OD-flow tables are on this machine (the confirmation cities), Denver first."""
    od = (root or style.DATA) / "od_maps"
    cs = sorted(p.name for p in od.iterdir() if (p / "pairs.csv").exists()) if od.exists() else []
    return [c for c in cs if c == DENVER] + [c for c in cs if c != DENVER]


def load_od(city: str, root=None) -> dict:
    """Classes and aggregates of the observed and predicted OD flows of one city (multicity/od_maps.py)."""
    od = (root or style.DATA) / "od_maps" / city
    return {"city": city, "pairs": pd.read_csv(od / "pairs.csv", dtype={"o": str, "d": str}), "overlap": pd.read_csv(od / "overlap.csv"),
            "dist": pd.read_csv(od / "dist.csv"), "summary": json.loads((od / "summary.json").read_text()), "tracts": od / "tracts.geojson"}


def available(root=None) -> bool:
    root = root or style.DATA
    return all((root / f"{k}.csv").exists() for k in _CSV) and (root / "od_maps" / DENVER / "pairs.csv").exists()
