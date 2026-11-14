"""Paper figures, Task 1: the small pure helpers of the figure-data export (the export itself only reads result
files that exist on the server)."""
import numpy as np
import pytest

import paper_figdata as pf


def test_ols_slope():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert pf.ols_slope(x, 2.0 - 0.5 * x) == pytest.approx(-0.5)
    assert np.isnan(pf.ols_slope([1.0, np.nan], [2.0, 3.0]))                 # fewer than 3 finite points
    assert pf.ols_slope([1, 2, 3, np.nan], [1, 2, 3, 9]) == pytest.approx(1.0)


def test_weighted_centroid():
    assert pf.weighted_centroid([0.0, 10.0], [0.0, 4.0], [1.0, 3.0]) == pytest.approx((7.5, 3.0))
    assert pf.weighted_centroid([0.0, 10.0], [0.0, 4.0], [0.0, 0.0]) == pytest.approx((5.0, 2.0))   # no weight: plain mean


def test_quantile_class_is_rank_based():
    v = np.array([5, 1, 1, 9, 3, 7, 2, 8, 4, 6], float)
    c = pf.quantile_class(v, 5)
    assert set(c) == {1, 2, 3, 4, 5} and np.bincount(c)[1:].tolist() == [2, 2, 2, 2, 2]
    assert c[np.argmax(v)] == 5 and c[np.argsort(v)[0]] == 1


def test_flow_classes_carry_no_counts():
    counts = np.array([0, 50, 3, 0, 20, 7, 1, 90, 0, 12], float)
    c = pf.flow_classes(counts, top=6)
    assert c.dtype.kind == "i" and set(c) <= {0, 1, 2, 3}
    assert c[7] == 3 and c[1] == 3 and c[4] == 2 and c[9] == 2 and c[5] == 1 and c[2] == 1      # ranks 1-2, 3-4, 5-6
    assert c[6] == 0 and (c[counts == 0] == 0).all()                       # beyond the top 6, and unvisited
    assert (pf.flow_classes(np.zeros(4), top=3) == 0).all()


def test_set_of_role():
    assert [pf.set_of(r) for r in ("search", "discovery", "lawtest", "confirm")] == ["discovery", "discovery", "heldout", "confirmation"]


def test_central_busy_origin():
    x = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    y = np.zeros(5)
    pop = np.array([1.0, 1.0, 1.0, 1.0, 0.0])                    # centre at x = 1.5
    orig = np.array([0, 1, 3, 4])
    assert pf.central_busy_origin(x, y, pop, np.array([5.0, 1.0, 9.0, 9.0]), orig) == 3      # busy and nearest
    assert pf.central_busy_origin(x, y, pop, np.array([9.0, 9.0, 9.0, 9.0]), orig) == 1      # all busy: nearest
