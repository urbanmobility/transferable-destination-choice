"""Paper figures, Task 3: the small statistics the panels show."""
import numpy as np
import pytest

import stats


def test_replication_ratio():
    assert stats.replication_ratio(-0.16, -0.08, -0.12, -0.04) == pytest.approx((0.5, 0.25, 0.75))
    assert stats.replication_ratio(2.0, 1.0, -1.0, 3.0) == pytest.approx((0.5, -0.5, 1.5))
    r = stats.replication_ratio(-2.0, 1.0, -1.0, 3.0)                       # negative denominator: ends swap
    assert r == pytest.approx((-0.5, -1.5, 0.5)) and r[1] <= r[0] <= r[2]
    assert all(np.isnan(v) for v in stats.replication_ratio(0.0, 1.0, 0.5, 1.5))
    assert all(np.isnan(v) for v in stats.replication_ratio(float("nan"), 1.0, 0.5, 1.5))


def test_ols_band_on_a_known_line():
    rng = np.random.default_rng(0)
    x = np.linspace(13, 17, 40)
    y = 1.0 - 0.15 * x + rng.normal(0, 0.05, 40)
    g = np.array([13.0, 15.0, 17.0])
    fit, lo, hi, slope = stats.ols_band(x, y, g)
    assert slope == pytest.approx(-0.15, abs=0.02) and np.allclose(fit, 1.0 - 0.15 * g, atol=0.05)
    assert (lo < fit).all() and (fit < hi).all() and (hi - lo)[1] < (hi - lo)[0]            # narrowest at the centre
    exact = stats.ols_band(x, 2 + 0 * x, g)
    assert np.allclose(exact[1], exact[2])


def test_ols_band_ignores_missing():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats.ols_band(x, y, np.array([5.0]))[0] == pytest.approx([5.0])


def test_running_median():
    x = np.arange(9.0)
    y = np.array([0, 0, 0, 10, 0, 0, 0, 0, 0.0])
    xs, ys = stats.running_median(x[::-1], y[::-1], k=3)
    assert list(xs) == sorted(xs) and len(xs) == 9 and ys[3] == 0.0 and ys.max() == 0.0     # one outlier never wins


def test_size_terciles():
    lab = stats.size_terciles(np.arange(9.0))
    assert list(lab) == ["small"] * 3 + ["medium"] * 3 + ["large"] * 3
    assert list(stats.size_terciles(np.array([3.0, 1.0, 2.0]))) == ["large", "small", "medium"]
