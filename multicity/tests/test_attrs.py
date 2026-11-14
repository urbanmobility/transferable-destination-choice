"""Task 6: national tract attributes (ACS + LODES + TIGER 2010) for the counties of the city table."""
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(os.environ.get("ODSR_ROOT", Path(__file__).resolve().parents[3]))
pytestmark = pytest.mark.skipif(not (ROOT / "external/tiger/raw").exists(), reason="needs the downloaded ACS, LODES and TIGER inputs")


def test_build_attrs_ca(tmp_path):
    from attrs import build
    from config import CITIES_CSV
    r = build(ROOT, CITIES_CSV, tmp_path, tmp_path / "geom", states=("06",), lodes_year=2018, acs_year=2018)
    t = pd.read_parquet(tmp_path / "tract_attrs.parquet").set_index("tract")
    assert len(t.loc[t.county == "06037"]) == 2343                      # matches the round-1 LA unit count
    assert t.loc["06037137000", "jobs"] > 0 and "06037930401" not in t.index
    assert (t.aland_km2 > 0).all() and t.x.between(-2.5e6, 2.5e6).all() and t.y.between(0, 3.5e6).all()
    assert t.loc["06037137000", "population"] > 0 and {"income", "households", "hh_no_vehicle", "state"} <= set(t.columns)
    assert (tmp_path / "geom/tracts_06.gpkg").exists() and r["states"] == ["06"] and r["tracts"] == len(t)
    assert set(t.county.str[:2]) == {"06"} and "06073" in set(t.county)                    # San Diego county is in a city
