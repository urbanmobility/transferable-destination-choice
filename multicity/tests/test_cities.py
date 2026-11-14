"""Task 2: city table from the 2023 CBSA delineation file (OMB 2020 standards), Bay Area as one city, CONUS only."""
import os
from pathlib import Path

import pytest

XLSX = Path(os.environ.get("ODSR_ROOT", Path(__file__).resolve().parents[3])) / "external/cbsa/list1_2023.xlsx"
pytestmark = pytest.mark.skipif(not XLSX.exists(), reason="needs the CBSA delineation file")


def test_build_cities(tmp_path):
    from cities import build
    df = build(XLSX, tmp_path / "cities.csv").set_index("city")
    assert df.loc["BAY9", "n_counties"] == 9 and "06085" in df.loc["BAY9", "counties"]
    assert "41860" not in df.index and "42220" not in df.index          # fully absorbed into BAY9
    assert df.loc["41940", "counties"] == "06069"                       # San Jose CBSA keeps San Benito only
    assert df.loc["31080", "counties"] == "06037;06059"
    assert df.loc["35620", "states"] == "34;36" and df.loc["28140", "n_counties"] >= 10   # NY-NJ, Kansas City MO-KS
    assert "46520" not in df.index and "11260" not in df.index           # Honolulu, Anchorage (non-CONUS)
    assert (df["n_counties"] >= 1).all()
    assert (tmp_path / "cities.csv").exists()


def test_counties_and_tracts_of(tmp_path):
    import pandas as pd
    from cities import build, counties_of, tracts_of
    build(XLSX, tmp_path / "cities.csv")
    assert counties_of("31080", tmp_path / "cities.csv") == ["06037", "06059"]
    idx = pd.Index(["06037000100", "06059000200", "06073000300", "32003000400"])
    assert list(tracts_of("31080", idx, tmp_path / "cities.csv")) == ["06037000100", "06059000200"]


def test_build_preserves_role_columns(tmp_path):
    """Final review (Important 4): regenerating the city table must not drop the roles written by the tiers stage."""
    import pandas as pd
    from cities import build
    csv = tmp_path / "cities.csv"
    build(XLSX, csv)
    df = pd.read_csv(csv, dtype=str)
    df["role"], df["tier"], df["substitute_for"] = "rest", "3", ""
    df.loc[df["city"] == "31080", "role"] = "search"
    df.to_csv(csv, index=False)
    build(XLSX, csv)
    out = pd.read_csv(csv, dtype=str).set_index("city")
    assert out.loc["31080", "role"] == "search" and out.loc["BAY9", "role"] == "rest" and "tier" in out.columns
