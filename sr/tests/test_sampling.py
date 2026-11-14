import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pysr_setup import design_matrix, grouped_nll_julia, numpy_nll  # noqa: E402
from sr_data import VARS, build  # noqa: E402


def _ref_eta(s):
    return -1.5 * s.X[:, VARS.index("logd")] + np.log(s.X[:, VARS.index("E_j")] + 1.0)


def test_sampled_rows_keep_all_positives():
    full = build(["SFBAY"], "tract", "train", "search", 15, 0)
    smp = build(["SFBAY"], "tract", "train", "search", 15, 0, neg_rate=0.1, neg_seed=3)
    key = lambda s: set(zip(s.origin_unit[s.y > 0], s.dest_unit[s.y > 0]))  # noqa: E731
    assert key(full) == key(smp)
    assert np.allclose(smp.logw[smp.y > 0], 0.0)
    assert np.allclose(smp.logw[smp.y == 0], np.log(10.0))
    assert len(smp.y) < 0.6 * len(full.y)
    assert np.allclose(full.logw, 0.0)


def test_sampled_loss_close_to_full():
    full = build(["SFBAY"], "tract", "train", "search", 30, 0)
    smp = build(["SFBAY"], "tract", "train", "search", 30, 0, neg_rate=0.1, neg_seed=3)
    a, b = numpy_nll(_ref_eta(full), full), numpy_nll(_ref_eta(smp), smp)
    assert abs(a - b) / a < 5e-3


def test_julia_loss_uses_weights():
    smp = build(["SFBAY"], "tract", "train", "search", 10, 0, neg_rate=0.1, neg_seed=3)
    e = _ref_eta(smp)
    X = design_matrix(smp, "single")
    assert np.isclose(grouped_nll_julia(e, smp.y, X, np.exp(smp.logw)), numpy_nll(e, smp), rtol=1e-10)


def test_pysr_weighted_turbo_losses_match_numpy():
    from pysr_setup import make_model
    smp = build(["LA", "SFBAY"], "tract", "train", "search", 6, 0, neg_rate=0.1, neg_seed=3)
    X = design_matrix(smp, "city2")
    m = make_model(smp, "A", "city2", seed=0, niterations=2, procs=4, turbo=True)
    m.fit(X, smp.y, weights=np.exp(smp.logw))
    n = 0
    for k in range(len(m.equations_)):
        e = m.predict(X, index=k)
        if np.isfinite(e).all():
            assert np.isclose(m.equations_.iloc[k]["loss"], numpy_nll(e, smp), rtol=1e-5)
            n += 1
    assert n >= 2


def test_julia_loss_keeps_the_weight_correction_for_large_float32_eta():
    """PySR runs with precision=32: eta and the sample weights reach the loss as Float32. Adding log(w) in Float32
    drops the negative-sampling correction once |eta| is large (spacing 128 at 1.1e9), which gave degenerate
    huge-constant equations a spuriously low loss (6.14 vs 7.13 in test_pysr_weighted_turbo_losses_match_numpy)."""
    from pysr import jl
    from pysr_setup import K
    smp = build(["SFBAY"], "tract", "train", "search", 10, 0, neg_rate=0.1, neg_seed=3)
    X = design_matrix(smp, "single")
    grouped_nll_julia(np.zeros(len(smp.y)), smp.y, X)                     # loads the Julia loss core
    e = np.full(len(smp.y), 1.1e9)
    got = float(jl._odsr_gnll_weighted(e.astype(np.float32), smp.y.astype(np.float32),
                                       X[:, K + 2].astype(np.float32), np.exp(smp.logw).astype(np.float32)))
    assert np.isclose(got, numpy_nll(e, smp), rtol=1e-6)


def test_block_search23_excludes_fold_4():
    """Plan 4 Task 2: new searches use destination folds 2, 3; fold 4 is held out for ranking (design §5.1)."""
    from sr_data import dest_mask, folds, load
    d = load("SFBAY", "tract")
    f = folds(d)
    m23, m4, ms = dest_mask(d, f, "search23"), dest_mask(d, f, "fold:4"), dest_mask(d, f, "search")
    assert m23.any() and m4.any() and not (m23 & m4).any() and ((m23 | m4) == ms).all()
