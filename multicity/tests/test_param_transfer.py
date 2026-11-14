"""Plan 9, Task 3: constants for a city without trip data — uniform (median), size (Theil-Sen), size + form (ridge)."""
import numpy as np
import pandas as pd
import pytest

import param_transfer as pt

FORM = ["f1", "f2"]


def _world(n=14, seed=0):
    rng = np.random.default_rng(seed)
    cities = [f"c{k}" for k in range(n)]
    x = np.linspace(12, 16, n)
    cov = pd.DataFrame({"log_population": x, "f1": rng.normal(size=n), "f2": rng.normal(size=n)}, index=cities)
    K = {c: [2 + 0.5 * x[k], 1.0, float(rng.normal())] for k, c in enumerate(cities)}
    return K, cov, cities


def _predict(K, cov, city, **kw):
    return pt.predict(K, cov, city, "log_population", FORM, **kw)


def test_levels():
    assert pt.LEVELS == ("local", "uniform", "size", "form")


def test_uniform_is_elementwise_median():
    K, cov, cs = _world()
    out = _predict(K, cov, "c0")
    assert set(out) == {"uniform", "size", "form"}
    assert np.allclose(out["uniform"], np.median([K[c] for c in cs[1:]], axis=0))


def test_size_recovers_exact_line():
    K, cov, _ = _world()
    cov.loc["new"] = [14.0, 0.0, 0.0]
    assert _predict(K, cov, "new")["size"][0] == pytest.approx(9.0, abs=1e-9)


def test_target_city_never_a_donor():
    K, cov, _ = _world()
    a = _predict(K, cov, "c3")
    b = _predict({**K, "c3": [1e6, 1e6, 1e6]}, cov, "c3")
    for lv in a:
        assert np.array_equal(a[lv], b[lv]), lv


def test_constant_column_passes_through():
    K, cov, _ = _world()
    out = _predict(K, cov, "c0")
    assert all(out[lv][1] == 1.0 for lv in out)


def test_too_few_donors_gives_empty():
    K, cov, cs = _world(n=10)
    assert _predict(K, cov, "c0") == {}                                   # 9 donors
    assert _predict(K, cov, "c0", min_donors=9) != {}


def test_donors_need_covariates():
    K, cov, cs = _world(n=11)
    assert _predict(K, cov.drop(index="c5"), "c0") == {}                  # 10 donors, one without covariates -> 9


def test_city_without_covariates_gives_empty():
    K, cov, _ = _world()
    assert _predict(K, cov, "nowhere") == {}


def test_form_clips_outlier_donor():
    K, cov, cs = _world()
    K["c7"] = [1e6, 1.0, 0.0]
    rest = [K[c][0] for c in cs[1:] if c != "c7"]
    v = _predict(K, cov, "c0")["form"][0]
    assert min(rest) - 1.0 <= v <= max(rest) + 1.0


def test_form_handles_missing_and_constant_covariates():
    K, cov, _ = _world()
    cov.loc["c4", "f1"] = np.nan
    cov["f2"] = 3.0
    out = _predict(K, cov, "c0")
    assert all(np.isfinite(out[lv]).all() for lv in out)
