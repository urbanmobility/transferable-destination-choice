"""od_maps.py: the pure parts — flows, rank tiers, classes, overlap. No city data is read."""
import numpy as np
import pytest

import od_maps as om


def test_flows_keep_every_origin_total_and_follow_eta():
    y = np.array([3.0, 1.0, 0.0, 2.0, 2.0])
    off = np.array([0, 3, 5])
    mu = om.flows(np.array([0.0, 0.0, 0.0, 1.0, 1.0]), y, off)
    assert mu[:3] == pytest.approx([4 / 3] * 3) and mu[3:] == pytest.approx([2.0, 2.0])
    mu = om.flows(np.log([2.0, 1.0, 1.0, 3.0, 1.0]), y, off)
    assert mu == pytest.approx([2.0, 1.0, 1.0, 3.0, 1.0])
    assert om.cpc(y, y) == 1.0 and 0 < om.cpc(y, mu) < 1


def test_nll_matches_the_definition():
    y = np.array([3.0, 1.0, 0.0, 2.0, 2.0])
    off = np.array([0, 3, 5])
    mu = om.flows(np.zeros(5), y, off)
    assert om.nll(y, mu, off) == pytest.approx(-(4 * np.log(1 / 3) + 4 * np.log(1 / 2)) / 8)


def test_tiers_are_nested_and_skip_zeros():
    v = np.array([5.0, 0.0, 9.0, 1.0, 7.0, 3.0])
    assert om.tiers(v, cuts=(1, 3, 4)).tolist() == [2, 0, 1, 0, 2, 3]
    assert om.tiers(v, cuts=(10, 20)).tolist() == [1, 0, 1, 1, 1, 1]            # fewer positive values than the first cut
    assert om.top(v, 2).tolist() == [2, 4]


def test_ratio_class_is_symmetric_and_handles_zeros():
    obs = np.array([10.0, 10.0, 10.0, 10.0, 0.0, 0.0])
    pred = np.array([10.0, 21.0, 4.0, 12.0, 3.0, 0.0])
    assert om.ratio_class(pred, obs).tolist() == [0, 3, -3, 1, 3, 0]


def test_log_class_puts_zeros_below_the_first_edge():
    edges = np.array([-1.0, 0.0, 1.0, 2.0])
    assert om.log_class(np.array([0.0, 0.05, 0.5, 5.0, 50.0, 5000.0]), edges).tolist() == [-1, -1, 0, 1, 2, 2]


def test_histogram_and_calibration_count_pairs():
    rng = np.random.default_rng(0)
    mu = rng.uniform(1, 9, 200)
    y = rng.poisson(mu).astype(float)
    h = om.histogram(y, mu)
    assert int(h["n_pairs"].sum()) == 200
    c = om.calibration(y, mu, min_pairs=20)
    assert (c["n_pairs"] >= 20).all() and c["obs_mean"].between(1, 9).all()
    assert om.calibration(y, mu, min_pairs=1000).empty                      # small classes are not written


def test_distance_shares_sum_to_one_and_overlap_is_one_for_a_perfect_model():
    d = np.array([0.2, 1.0, 5.0, 500.0])
    s = om.distance_shares(d, np.ones(4), edges=np.array([0.5, 2.0, 10.0, 100.0]))
    assert s == pytest.approx([0.5, 0.25, 0.25])
    y = np.arange(50, dtype=float)
    r = om.overlap(y, y, ks=(10,))[0]
    assert r["overlap"] == 1.0 and r["trips_on_predicted_top"] == pytest.approx(r["trips_on_observed_top"])
    r = om.overlap(y, y[::-1].copy(), ks=(10,))[0]
    assert r["overlap"] == 0.0 and r["trips_on_predicted_top"] < r["trips_on_observed_top"]


def test_rank_thirds():
    v = np.array([0.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    assert om.rank_thirds(v, 6).tolist() == [0, 3, 3, 2, 2, 1, 1]
