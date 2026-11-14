"""Revision 2026-11-13, Task R2: the pure parts of rev_banded.py."""
import math

import numpy as np
import pandas as pd
import pytest

import rev_banded as rb


def test_fixed_refs_use_band_midpoints_and_one_pooled_point():
    pooled = {"N_j": 30.0, "a_j": 2.0, "E_j": 800.0}
    F = rb.fixed_refs(pooled)
    assert [round(F[b]["d"], 2) for b in range(6)] == [0.5, 1.41, 3.16, 7.07, 14.14, 28.28]
    assert all(F[b]["N_j"] == 30.0 and F[b]["a_j"] == 2.0 for b in range(6))
    assert F[3]["logd"] == pytest.approx(math.log(F[3]["d"]))


def test_with_mass_moves_the_chosen_mass_into_N_j():
    F = rb.fixed_refs({"N_j": 30.0, "E_j": 800.0})
    G = rb.with_mass(F, "E_j")
    assert G[2]["N_j"] == 800.0 and F[2]["N_j"] == 30.0          # the input is not modified


def test_design_cols_swaps_the_banded_mass():
    u = pd.DataFrame({"N": [0.0, 9.0], "jobs": [99.0, 0.0]})
    cols = {"log_pop": np.zeros(2), "log_jobs": np.log1p(u["jobs"].to_numpy()), "log_area": np.ones(2)}
    c, m = rb.design_cols(cols, "fixed", u)
    assert "log_jobs" in c and "log_N" not in c and m == pytest.approx(np.log1p([0.0, 9.0]))
    c, m = rb.design_cols(cols, "emp", u)
    assert "log_jobs" not in c and "log_N" not in c and m == pytest.approx(np.log1p([99.0, 0.0]))
    c, m = rb.design_cols(cols, "empN", u)
    assert "log_jobs" not in c and c["log_N"] == pytest.approx(np.log1p([0.0, 9.0]))
    assert "log_jobs" in cols                                       # the input dict is not modified


def test_profile_rows_give_gamma_times_n_over_n_plus_one():
    from choice_model import bspline_basis
    kn, lo, hi = [0.0, 1.0, 2.0], -3.0, 4.0
    nb = bspline_basis(np.array([0.0]), np.array(kn), lo=lo, hi=hi).shape[1]
    fit = {"gamma": {b: 0.1 * (b + 1) for b in range(6)}, "spline_coef": list(np.linspace(0, -3, nb)),
           "knots": (kn, lo, hi)}
    F = rb.fixed_refs({"N_j": 4.0, "E_j": 9.0})
    rows = rb.profile_rows("x", "Flexible", fit, F, "fixed", {b: 5000.0 for b in range(6)})
    eN = {r["band"]: r["eN"] for r in rows}
    assert eN[0] == pytest.approx(0.1 * 4 / 5) and eN[5] == pytest.approx(0.6 * 4 / 5)
    rows = rb.profile_rows("x", "Flexible", fit, rb.with_mass(F, "E_j"), "fixed", {b: 5000.0 for b in range(6)})
    assert {r["band"]: r["eN"] for r in rows}[1] == pytest.approx(0.2 * 9 / 10)
