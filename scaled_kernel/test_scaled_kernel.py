import numpy as np
import pytest

import scaled_kernel as sk


def test_worked_example_of_table_s5():
    p = {k: round(v, 3) for k, v in sk.DEFAULT.items()}
    assert float(sk.utility(5.0, 71, 2.6, p)) == pytest.approx(-5.935, abs=5e-4)


def test_probabilities_sum_to_one_and_exclude_the_origin():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 20, (6, 2))
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    P = sk.choice_probabilities(d, rng.integers(1, 200, 6), rng.uniform(0.5, 5, 6))
    np.testing.assert_allclose(P.sum(1), 1.0)
    assert np.all(np.diag(P) == 0)


def test_size_elasticity_rises_with_distance_and_matches_a_numerical_derivative():
    d = np.array([0.5, 2.0, 8.0, 30.0])
    e_d, e_n = sk.elasticities(d, 50)
    assert np.all(np.diff(e_n) > 0) and np.all(np.diff(e_d) < 0)
    h = 1e-6
    num = (sk.utility(d, 50 * np.exp(h), 1.0) - sk.utility(d, 50 * np.exp(-h), 1.0)) / (2 * h)
    np.testing.assert_allclose(e_n, num, rtol=1e-5)


def test_distance_term_is_d_over_scale_to_the_theta1():
    p = sk.DEFAULT
    ell = sk.distance_scale([10, 100, 1000])
    assert np.all(np.diff(ell) > 0)
    d = 7.0
    term = 1.0 - sk.utility(d, 100, 1.0)                        # area term is 1 for a_j = 1
    assert float(term) == pytest.approx((d / ell[1]) ** p["theta1"])
