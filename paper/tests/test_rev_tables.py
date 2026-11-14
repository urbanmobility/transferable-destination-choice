"""Revision 2026-11-13, Task R3: the pure parts of rev_tables.py."""
import numpy as np
import pytest

import rev_tables as rt


def test_mde_is_z_sum_times_se():
    assert rt.mde(1.0) == pytest.approx(1.959964 + 0.841621, rel=1e-5)
    assert rt.mde(2.0, alpha=0.05 / 10) == pytest.approx(2.0 * (2.807034 + 0.841621), rel=1e-5)


def test_se_from_ci_inverts_a_normal_interval():
    assert rt.se_from_ci(-1.96, 1.96) == pytest.approx(1.0, rel=1e-3)


def test_ols_boot_recovers_a_slope_and_controls_a_covariate():
    rng = np.random.default_rng(0)
    x, z = rng.normal(size=200), rng.normal(size=200)
    y = 2.0 * x - 3.0 * z + rng.normal(scale=0.1, size=200)
    s, lo, hi = rt.ols_boot(y, x, B=300, seed=1)
    assert s == pytest.approx(2.0, abs=0.5)
    s, lo, hi = rt.ols_boot(y, x, z, B=300, seed=1)
    assert s == pytest.approx(2.0, abs=0.05) and lo < 2.0 < hi


def test_loglog_slope_of_a_power_law():
    x = np.log(np.array([1e6, 2e6, 4e6, 8e6]))
    y = 10.0 * np.exp(x) ** -0.5
    assert rt.loglog_slope(x, y) == pytest.approx(-0.5)


def test_rise_slope_counts_positive_profiles():
    import pandas as pd
    t = pd.DataFrame({"city": ["a"] * 3 + ["b"] * 3, "d_ref": [1, 3, 10] * 2, "eN": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1]})
    s = rt.rise_slopes(t)
    assert s["a"] > 0 > s["b"]


def test_r1_tables_read_the_summary(tmp_path, monkeypatch):
    import json
    s = {"line": "all", "models": {"sk|local": {"target": {"n": 32, "share": 0.95, "nll": 3.4, "pseudo_r2": 0.3},
                                                "all": {"n": 67, "share": 0.96, "nll": 3.5, "pseudo_r2": 0.31}}},
         "contrasts": {"sk|sep_N|local|target": {"label": "coupling", "n": 32, "median": 0.01, "ci_lo": 0.0, "ci_hi": 0.02,
                                                 "wins": 20, "n_failed": 0, "p": 0.01}}}
    (tmp_path / "summary_models_all.json").write_text(json.dumps(s))
    monkeypatch.setattr(rt, "RV", tmp_path)
    med, con = rt.t_r1("all")
    assert med.iloc[0]["model"] == "sk" and med.iloc[0]["share (32)"] == 0.95
    assert con.iloc[0]["a"] == "sk" and con.iloc[0]["dNLL median"] == 0.01 and con.iloc[0]["a better"] == "20/32"
    assert rt.t_r1("nonwork") == (None, None)


def test_paired_nll_counts_a_failure_as_a_loss():
    import pandas as pd
    t = pd.DataFrame({"city": ["x", "x", "y", "y"], "model": ["A", "B"] * 2, "level": "local", "nll": [1.0, 1.1, np.nan, 2.0]})
    r = rt.paired_nll(t, ("A", "local"), ("B", "local"), ["x", "y"], B=50, seed=1)
    assert r["n"] == 2 and r["wins"] == 1 and r["losses"] == 1 and r["n_failed"] == 1


def test_figS1_has_four_panels_with_rising_counts():
    import matplotlib.pyplot as plt
    import figS1
    if not (figS1.RV / "banded_fixed.parquet").exists():
        pytest.skip("needs the exported banded-model tables")
    fig, src = figS1.build()
    assert set(src) == {"fixed", "emp", "empN", "nonwork"}
    texts = [t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)]
    assert sum(s.startswith("rising ") for s in texts) == 4
    plt.close(fig)
