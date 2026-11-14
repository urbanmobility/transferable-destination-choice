"""Task 8: per-city QC — support mismatch between destination blocks, representativeness, eligibility flags."""
import importlib

import numpy as np
import pandas as pd

from test_units import city_env  # noqa: F401  (fixture: env + synthetic city)


def test_qc_support_mismatch_and_eligibility(city_env):  # noqa: F811
    root, csv, units = city_env
    units.build_city("TEST1", root, cities_csv=csv)
    import qc
    import sr_data
    importlib.reload(qc)
    d = sr_data.load("TEST1", "tract")
    f = sr_data.folds(d)
    base = root / "derived/model_inputs/M/TEST1/tract"
    u = pd.read_parquet(base / "units.parquet")
    j = int(np.flatnonzero((f == 1) & d["dest"])[0])          # a destination of the validation block
    u.loc[j, "jobs"] = 0
    u.to_parquet(base / "units.parquet", index=False)
    q = qc.run_city("TEST1")
    assert "E_j" in q["support_mismatch"] and "d" not in q["support_mismatch"]
    assert q["eligible"] is False and "ineligible" in q["flags"] and "support_mismatch" in q["flags"]
    assert 0 <= q["representativeness"]["r2"] <= 1 and set(q["representativeness"]["coverage_ratio_by_ses"]) == {"1", "2", "3"}
    assert q["n_origins"] == 3 and q["trips"]["train"] == 8 and q["coverage_out_of_city"] > 0   # tract 3 has no income -> ses 0
    assert (base / "qc.json").exists()
    assert q["support"]["E_j"]["val"]["zeros"] >= 1 and q["support"]["E_j"]["search"]["zeros"] == 0   # zeros count rows


def test_qc_main_writes_table(city_env):  # noqa: F811
    root, csv, units = city_env
    units.build_city("TEST1", root, cities_csv=csv)
    import qc
    importlib.reload(qc)
    out = qc.main(["TEST1"])
    t = pd.read_csv(root / "results/multicity/qc/cities_qc.csv", dtype={"city": str})
    assert list(t.columns[:5]) == ["city", "eligible", "n_origins", "train_trips", "population"] and t.loc[0, "city"] == "TEST1"
    assert t.loc[0, "population"] == 2100
    assert out.loc["TEST1", "eligible"] == False  # noqa: E712


def test_qc_flags_nonfinite_variable(city_env):  # noqa: F811
    """Final review (Critical 1): a NaN feature must be counted per block and flagged, not hidden by nanmin."""
    root, csv, units = city_env
    units.build_city("TEST1", root, cities_csv=csv)
    import qc
    importlib.reload(qc)
    base = root / "derived/model_inputs/M/TEST1/tract"
    u = pd.read_parquet(base / "units.parquet")
    u["jobs"] = np.nan
    u.to_parquet(base / "units.parquet", index=False)
    q = qc.run_city("TEST1")
    assert q["support"]["E_j"]["search"]["nonfinite"] == q["support"]["E_j"]["search"]["rows"] > 0
    assert "E_j" in q["nonfinite"] and "nonfinite" in q["flags"]


def test_qc_main_records_failed_cities_and_fails_on_a_listed_city(city_env):  # noqa: F811
    """Final review (Important 6): a city whose QC crashes is listed in qc_failed.txt; a search / confirmation city
    failing QC fails the run instead of silently becoming no_qc."""
    import pytest
    root, csv, units = city_env
    units.build_city("TEST1", root, cities_csv=csv)
    import config
    import qc
    importlib.reload(qc)
    out = qc.main(["TEST1", "NOPE"])
    assert list(out.index) == ["TEST1"]
    assert (root / "results/multicity/qc/qc_failed.txt").read_text().split() == ["NOPE"]
    with pytest.raises(RuntimeError, match=config.SEARCH[0]):
        qc.main(["TEST1", config.SEARCH[0]])
