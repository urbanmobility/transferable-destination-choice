import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pysr_setup import design_matrix, make_model, numpy_nll  # noqa: E402
from refit_select import Candidate, eta, loss, parse_equation, refit  # noqa: E402
from sr_data import build  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    s = build(["LA", "SFBAY"], "zcta", "train", "search", 15, 0)
    X = design_matrix(s, "city2")
    m = make_model(s, "A", "city2", seed=0, niterations=3, procs=8)
    m.fit(X, s.y)
    return s, X, m


def test_export_matches_pysr(fitted):
    s, X, m = fitted
    checked = 0
    for k in range(len(m.equations_)):
        r = m.equations_.iloc[k]
        c = parse_equation(r["equation"], "city2", "A", int(r["complexity"]))
        ref = m.predict(X, index=k)
        mine = eta(c, s, "city2")
        if not np.isfinite(ref).all():
            continue
        assert np.allclose(mine, ref, rtol=1e-4, atol=1e-4)
        assert np.isclose(loss(c, s, "city2"), r["loss"], rtol=1e-4)
        checked += 1
    assert checked >= 3


def test_refit_nonfinite():
    s = build(["SFBAY"], "zcta", "train", "search", 10, 0)
    c = parse_equation("f = log(#1 - 5.0)", "single", "A", 4)
    r = refit(c, s, "single")
    assert np.isfinite(loss(r, s, "single"))


def test_refit_improves(fitted):
    s, X, m = fitted
    best = m.equations_.sort_values("loss").iloc[0]
    c = parse_equation(best["equation"], "city2", "A", int(best["complexity"]))
    base = loss(c, s, "city2")
    pert = dataclasses.replace(c, consts=c.consts * 1.3)
    lp = loss(pert, s, "city2")
    r = refit(pert, s, "city2")
    assert loss(r, s, "city2") <= lp + 1e-9
    assert loss(r, s, "city2") <= base + 1e-3


def test_numpy_loss_consistent(fitted):
    s, X, m = fitted
    best = m.equations_.sort_values("loss").iloc[0]
    c = parse_equation(best["equation"], "city2", "A", int(best["complexity"]))
    assert np.isclose(loss(c, s, "city2"), numpy_nll(eta(c, s, "city2"), s))


def test_refit_free_mask_keeps_fixed_constants(fitted):
    s, X, m = fitted
    best = m.equations_.sort_values("loss").iloc[0]
    c = parse_equation(best["equation"], "city2", "A", int(best["complexity"]))
    pert = dataclasses.replace(c, consts=c.consts * 1.2)
    free = np.zeros(len(c.consts), bool)
    free[c.n_lit:] = True                                   # only the city parameters move
    r = refit(pert, s, "city2", free=free)
    assert np.array_equal(r.consts[:c.n_lit], pert.consts[:c.n_lit])
    assert loss(r, s, "city2") <= loss(pert, s, "city2") + 1e-12


def test_refit_power_of_column_with_zeros():
    from refit_select import PENALTY, _obj
    from sr_data import VARS
    s = build(["SFBAY"], "zcta", "train", "search", 10, 0)
    s.X[::7, VARS.index("pd_j")] = 0.0                     # d(x^c)/dc = x^c log x is 0 * (-inf) at x = 0
    c = parse_equation("f = (#19 ^ 0.3) * -0.5", "single", "A", 5)
    f0, g0 = _obj(c.consts, c, s)
    assert f0 < PENALTY and np.isfinite(g0).all()
    for i in range(len(c.consts)):
        h = 1e-5 * max(1.0, abs(c.consts[i]))
        up, dn = c.consts.copy(), c.consts.copy()
        up[i] += h
        dn[i] -= h
        fd = (loss(c, s, consts=up) - loss(c, s, consts=dn)) / (2 * h)
        assert np.isclose(g0[i], fd, rtol=1e-3, atol=1e-7)
    r = refit(c, s, "single")
    assert loss(r, s, "single") < loss(c, s, "single") - 1e-6


def test_select_cache_reuses_refits(monkeypatch):
    import refit_select as rs
    from sr_data import ROOT
    run = ["results/sr/L/runs/zcta_A_single-SFBAY_s0"]
    kw = dict(regions=("SFBAY",), B=20, procs=1, max_total=None, per_cx=None)   # procs=1: the session runs Julia
    try:
        a = rs.select(run, "zcta", "A", out_name="tmp_cache_a", **kw)
        assert a["regions"] == ["SFBAY"]

        def boom(*args, **kwargs):
            raise AssertionError("refit called despite cache")
        monkeypatch.setattr(rs, "refit", boom)
        b = rs.select(run, "zcta", "A", out_name="tmp_cache_b", cache=["tmp_cache_a"], **kw)
        assert b["chosen"] == a["chosen"]
        assert [x["val_nll"] for x in b["candidates"]] == pytest.approx([x["val_nll"] for x in a["candidates"]], rel=1e-9)
        with pytest.raises(ValueError):                       # a cache from other data is refused
            rs.select(run, "zcta", "A", out_name="tmp_cache_c", cache=["tmp_cache_a"], **{**kw, "regions": ("LA",)})
    finally:
        for n in "abc":
            (ROOT / f"results/sr/L/tmp_cache_{n}.json").unlink(missing_ok=True)


def _logit_data(scale=1, seed=0, n_orig=60, n_dest=12):
    """Synthetic conditional-logit rows: x1 ~ U(0, 3), true beta = -0.8, counts drawn per origin; 20 device groups
    with each origin's visits in one group. scale multiplies every count (and the group counts)."""
    from refit_select import XS  # noqa: F401
    from sr_data import SRData, VARS
    rng = np.random.default_rng(seed)
    X = np.zeros((n_orig * n_dest, len(VARS)), np.float32)
    x1 = rng.uniform(0, 3, n_orig * n_dest)
    X[:, 0] = x1
    y = np.zeros(n_orig * n_dest)
    G = np.zeros((20, n_orig * n_dest))
    for i in range(n_orig):
        s = slice(i * n_dest, (i + 1) * n_dest)
        p = np.exp(-0.8 * x1[s])
        p /= p.sum()
        y[s] = rng.multinomial(30, p)
        G[i % 20, s] = y[s]
    off = np.arange(0, n_orig * n_dest + 1, n_dest)
    d = SRData(X=X, names=list(VARS), y=y * scale, w=np.ones(len(y)), offsets=off, city=np.zeros(len(y), int),
               ses=np.zeros(len(y), int), offset_B=np.zeros(len(y)), origin_unit=np.repeat(np.arange(n_orig), n_dest),
               dest_unit=np.tile(np.arange(n_dest), n_orig))
    return d, G * scale


def test_refit_se_matches_fisher_information_and_sandwich_is_cluster_scale_free():
    """Plan 4 Task 11: refit(return_se=True) gives the inverse-Hessian SE of the constants (one-parameter logit:
    1 / sqrt(sum_i O_i Var_p(x))); with device-group counts Yg it gives the cluster-robust sandwich SE, which does not
    shrink when every count is multiplied (the groups, not the visits, are the independent units)."""
    from refit_select import eta as eta_
    d, G = _logit_data()
    c = parse_equation("f = -1.0 * #1; p1 = []; p2 = []", None, "A", 3)
    r, se, cov = refit(c, d, return_se=True)
    e = eta_(r, d)
    info = 0.0
    for a, b in zip(d.offsets[:-1], d.offsets[1:]):
        p = np.exp(e[a:b] - e[a:b].max())
        p /= p.sum()
        x = d.X[a:b, 0].astype(float)
        info += d.y[a:b].sum() * (p @ x ** 2 - (p @ x) ** 2)
    assert abs(r.consts[0] - 0.8) < 0.1 and se[0] == pytest.approx(1 / np.sqrt(info), rel=1e-3)
    assert cov.shape == (1, 1) and cov[0, 0] == pytest.approx(se[0] ** 2)
    d4, G4 = _logit_data(scale=4)
    _, se4, _ = refit(c, d4, return_se=True)
    _, sw, _ = refit(c, d, return_se=True, Yg=G)
    _, sw4, _ = refit(c, d4, return_se=True, Yg=G4)
    assert se4[0] == pytest.approx(se[0] / 2, rel=1e-3)
    assert sw[0] > 0 and sw4[0] == pytest.approx(sw[0], rel=1e-3)
