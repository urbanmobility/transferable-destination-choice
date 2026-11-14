"""Revision 2026-11-13, Task R4: the pure parts of rev_flows.py."""
import numpy as np
import pytest

import rev_flows as rf


def test_uniform_destinations_split_each_origin_total_evenly():
    y = np.array([3.0, 1.0, 0.0, 2.0, 2.0])          # origin 0: 3 destinations, origin 1: 2
    off = np.array([0, 3, 5])
    mu = rf.uniform_flows(y, off)
    np.testing.assert_allclose(mu, [4 / 3] * 3 + [2.0, 2.0])


def test_population_totals_keep_the_city_total_and_the_choice_shares():
    y = np.array([3.0, 1.0, 0.0, 2.0, 2.0])
    off = np.array([0, 3, 5])
    p = np.array([0.5, 0.25, 0.25, 0.5, 0.5])
    pop = np.array([100.0, 300.0])
    mu = rf.population_totals(p, off, pop, y.sum())
    assert mu.sum() == pytest.approx(8.0)
    np.testing.assert_allclose(mu[:3], np.array([0.5, 0.25, 0.25]) * 2.0)    # origin 0 gets 8 * 100/400


def test_cpc_of_the_uniform_baseline_by_hand():
    y = np.array([4.0, 0.0])
    off = np.array([0, 2])
    mu = rf.uniform_flows(y, off)
    assert rf.cpc(y, mu) == pytest.approx(2 * 2.0 / 8.0)
