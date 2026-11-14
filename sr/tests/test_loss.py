import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pysr_setup import design_matrix, grouped_nll_julia, make_model, numpy_nll  # noqa: E402
from sr_data import VARS, build  # noqa: E402


@pytest.fixture(scope="module")
def small():
    return build(["SFBAY"], "zcta", "train", "search", 50, 0)


def test_numpy_nll_matches_definition(small):
    eta = np.random.default_rng(0).normal(size=len(small.y))
    tot, vis = 0.0, 0.0
    for a, b in zip(small.offsets[:-1], small.offsets[1:]):
        e = eta[a:b]
        lp = e - np.log(np.exp(e - e.max()).sum()) - e.max()
        tot -= (small.y[a:b] * lp).sum()
        vis += small.y[a:b].sum()
    assert np.isclose(numpy_nll(eta, small), tot / vis, rtol=1e-12)


def test_loss_matches_numpy(small):
    eta = np.random.default_rng(1).normal(size=len(small.y)) * 2
    X = design_matrix(small, "city2")
    assert np.isclose(grouped_nll_julia(eta, small.y, X), numpy_nll(eta, small), rtol=1e-10)


def test_loss_nonfinite_penalised(small):
    eta = np.zeros(len(small.y))
    eta[3] = np.nan
    assert grouped_nll_julia(eta, small.y, design_matrix(small, "city2")) == 1e9


def test_city_column_coding():
    s = build(["LA", "SFBAY"], "zcta", "train", "search", 10, 0)
    X2, X6 = design_matrix(s, "city2"), design_matrix(s, "cityses2")
    k = len(VARS)
    assert np.array_equal(X2[:, k + 1], s.city + 1)                  # cat column, 1-based
    assert np.array_equal(X6[:, k + 1], s.city * 3 + s.ses + 1)
    gid = X2[:, k + 2]
    for a, b in zip(s.offsets[:-1], s.offsets[1:]):                   # one gid per origin, distinct
        assert len(set(gid[a:b])) == 1
    assert len(set(gid[s.offsets[:-1]])) == len(s.offsets) - 1


def test_hall_of_fame_losses_match_numpy(small):
    m = make_model(small, "A", "single", seed=0, niterations=2, procs=4)
    m.fit(design_matrix(small, "single"), small.y)
    X = design_matrix(small, "single")
    for k in range(len(m.equations_)):
        eta = m.predict(X, index=k)
        if np.isfinite(eta).all():
            assert np.isclose(m.equations_.iloc[k]["loss"], numpy_nll(eta, small), rtol=1e-5)


def test_template_recovers_synthetic():
    s = build(["LA", "SFBAY"], "zcta", "train", "search", 60, 0)
    rng = np.random.default_rng(7)
    lE = np.log(s.X[:, VARS.index("E_j")] + 1.0)
    p = np.array([0.6, 0.3])
    eta = -1.5 * s.X[:, VARS.index("logd")] + p[s.city] * lE
    y = np.zeros(len(s.y))
    for a, b in zip(s.offsets[:-1], s.offsets[1:]):
        pr = np.exp(eta[a:b] - eta[a:b].max())
        y[a:b] = rng.multinomial(2000, pr / pr.sum())
    s.y = y
    true = numpy_nll(eta, s)
    m = make_model(s, "A", "city2", seed=0, niterations=40, procs=16)
    m.fit(design_matrix(s, "city2"), s.y)
    assert m.equations_["loss"].min() <= true * 1.002


def test_huge_constant_eta_is_uniform(small):
    """A constant eta must give the uniform-choice loss, however large (no cancellation exploit)."""
    uni = numpy_nll(np.zeros(len(small.y)), small)
    big = np.full(len(small.y), 1.16e24)
    X = design_matrix(small, "single")
    assert np.isclose(numpy_nll(big, small), uni, rtol=1e-9)
    assert np.isclose(grouped_nll_julia(big, small.y, X), uni, rtol=1e-9)
