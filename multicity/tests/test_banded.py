"""Plan 5 Tasks 4-5 (pre-freeze wrap-up): elasticity profiles of the skeletons by distance band, and the
distance-band flexible model that adjudicates between them."""
import math

import numpy as np
import pytest

from sr_data import VARS, SRData
from test_search_param import two_cities  # noqa: F401

import banded

C0079C1 = {"cid": "c0079c1", "template": None, "variant": "A", "complexity": 13,
           "equation": "f = 0.219*log(#7) - 2.19*log(#1 + 1.12) + 0.781*log(#6 + 1); p1 = []; p2 = []"}
C0413C1 = {"cid": "c0413c1", "template": None, "variant": "A", "complexity": 27,
           "equation": "f = #12 - 0.111*#2*log(#18*#6 + #26 + 1) + 0.114*log(#7) + log(#6 + 1); p1 = []; p2 = []"}


def _ref(d, slope, **kw):
    r = {v: 1.0 for v in VARS}
    r.update(kw)
    r.update(d=d, logd=math.log(d), dlogS_dlogd=slope, n_rows=10)
    return r


def _srdata(d):
    d = np.asarray(d, float)
    X = np.ones((len(d), len(VARS)), np.float32)
    X[:, VARS.index("d")] = d
    X[:, VARS.index("logd")] = np.log(d)
    X[:, VARS.index("S_ij")] = 100.0 * d
    n = len(d)
    z = np.zeros(n, int)
    return SRData(X=X, names=list(VARS), y=np.zeros(n), w=np.ones(n), offsets=np.array([0, n]), city=z, ses=z,
                  offset_B=np.zeros(n), origin_unit=z, dest_unit=np.arange(n))


# ---------------------------------------------------------------------------------------------------------- Task 4
def test_bands_are_the_plan_bands():
    assert banded.BANDS == ((0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, math.inf))


def test_profile_c0079c1_matches_closed_form():
    c = [0.18, 2.23, 2.23, 0.88, 6.93]
    ed, en = banded.skeleton_profile(C0079C1, c, {2: _ref(3.0, 1.6, N_j=40.0, S_ij=5e4)})[2]
    assert ed == pytest.approx(-2.23 * 3.0 / (3.0 + 2.23)) and en == pytest.approx(0.88 * 40 / (40 + 6.93))


def test_profile_c0413c1_matches_closed_form():     # the formula of the 11-09 diagnostic sk_elast.py
    c = [0.107, 1.25, 0.162, 7.9]
    d, N, rho, S, sl = 5.0, 30.0, 2000.0, 2e5, 1.7
    M = c[1] + rho * N + S
    ed, en = banded.skeleton_profile(C0413C1, c, {3: _ref(d, sl, N_j=N, pd_i=rho, S_ij=S)})[3]
    assert ed == pytest.approx(-c[0] * (math.log(M) + math.log(d) * sl * S / M), rel=1e-9)
    assert en == pytest.approx(N / (c[3] + N) - c[0] * math.log(d) * rho * N / M, rel=1e-9)


def test_profile_none_band_stays_none():
    p = banded.skeleton_profile(C0079C1, [0.18, 2.23, 2.23, 0.88, 6.93], {0: None, 1: _ref(1.5, 1.0, N_j=10.0)})
    assert p[0] is None and p[1] is not None


def test_band_refs_empty_band_is_none():
    r = banded.band_refs(_srdata([0.5, 0.7, 3.0, 4.0]))            # no row beyond 5 km
    assert set(r) == set(range(len(banded.BANDS)))
    assert r[1] is None and r[3] is None and r[5] is None
    assert r[0]["n_rows"] == 2 and r[0]["d"] == pytest.approx(0.6) and r[0]["logd"] == pytest.approx(math.log(0.6))
    assert r[2]["n_rows"] == 2 and r[2]["S_ij"] == pytest.approx(350.0)
    assert r[0]["dlogS_dlogd"] == pytest.approx(np.polyfit(np.log([0.5, 0.7, 3.0, 4.0]),
                                                           np.log(100 * np.array([0.5, 0.7, 3.0, 4.0]) + 1), 1)[0])


# ---------------------------------------------------------------------------------------------------------- Task 5
def _grid_city(n, step, seed):
    rng = np.random.default_rng(seed)
    g = np.arange(n) * step
    xy = np.array([(a, b) for a in g for b in g])
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    N = rng.integers(0, 300, len(xy)).astype(float)
    X = rng.normal(size=(len(xy), 1))
    m = np.ones(len(xy), bool)
    return D, N, X, m, m.copy()


def _simulate(D, N, X, gamma, beta_d, trips_per_origin, seed):
    """Truth inside the model class: centred log1p(N) per band (banded.fit_banded's convention)."""
    rng = np.random.default_rng(seed)
    ln = np.log1p(N) - np.log1p(N).mean()
    eta = beta_d * np.log(np.maximum(D, 0.05)) + 0.3 * X[:, 0][None, :]
    for b, g in gamma.items():
        lo, hi = banded.BANDS[b]
        eta = eta + g * ((D >= lo) & (D < hi)) * ln[None, :]
    np.fill_diagonal(eta, -np.inf)
    P = np.exp(eta - eta.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    return np.vstack([rng.multinomial(trips_per_origin, p) for p in P]).astype(float)


def _pairs_srdata(D, N):
    i, j = np.where(~np.eye(len(D), dtype=bool))
    X = np.ones((len(i), len(VARS)), np.float32)
    X[:, VARS.index("d")] = D[i, j]
    X[:, VARS.index("logd")] = np.log(D[i, j])
    X[:, VARS.index("N_j")] = N[j]
    X[:, VARS.index("S_ij")] = 50.0 * D[i, j] ** 2
    z = np.zeros(len(i), int)
    return SRData(X=X, names=list(VARS), y=np.zeros(len(i)), w=np.ones(len(i)), offsets=np.array([0, len(i)]),
                  city=z, ses=z, offset_B=np.zeros(len(i)), origin_unit=i, dest_unit=j)


def test_fit_banded_recovers_rising_mass_coefficient():
    D, N, X, orig, dest = _grid_city(n=12, step=0.6, seed=1)
    gamma_true = {0: 0.5, 1: 0.6, 2: 0.75, 3: 0.9}                 # bands 4, 5 have no pair (max d 9.3 km)
    T = _simulate(D, N, X, gamma_true, beta_d=-1.5, trips_per_origin=20000, seed=2)
    Bd, knots = banded.distance_spline(D, dest)
    fit = banded.fit_banded(Bd, X, D, np.log1p(N), T, orig, dest, knots)
    for b, g in gamma_true.items():
        assert fit["gamma"][b] == pytest.approx(g, abs=0.05)
    assert fit["gamma"][4] is None and fit["gamma"][5] is None    # empty bands dropped, no singular fit
    assert fit["converged"] and np.isfinite(fit["train_nll"])
    refs = banded.band_refs(_pairs_srdata(D, N))
    prof = banded.flex_profile(fit, refs)
    assert prof[2][0] == pytest.approx(-1.5, abs=0.15)             # distance elasticity of the -1.5 log d core
    Nr = refs[2]["N_j"]
    assert prof[2][1] == pytest.approx(fit["gamma"][2] * Nr / (Nr + 1))
    assert prof[4] is None and prof[5] is None


def test_distance_spline_matches_val_refs_knots():
    D, N, X, orig, dest = _grid_city(n=5, step=1.0, seed=3)
    Bd, (kn, lo, hi) = banded.distance_spline(D, dest)
    logd = np.log(np.maximum(D, 0.05))
    off = ~np.eye(len(D), dtype=bool)
    from run_gate_l import DIST_DF
    assert np.allclose(kn, np.quantile(logd[off], np.linspace(0, 1, DIST_DF - 1)[1:-1]))
    assert lo == pytest.approx(logd[off].min()) and hi == pytest.approx(logd[off].max())
    from choice_model import bspline_basis
    assert np.array_equal(Bd, bspline_basis(logd.ravel(), kn, lo=lo, hi=hi).reshape(len(D), len(D), -1))


def test_city_banded_refuses_cities_outside_search_and_discovery(tmp_path, monkeypatch):
    import config
    import sr_data
    csv = tmp_path / "cities.csv"
    csv.write_text("city,role\nS1,search\nD1,discovery\nL1,lawtest\nC1,confirm\n")
    monkeypatch.setattr(config, "CITIES_CSV", csv)
    monkeypatch.setattr(sr_data, "load", lambda *a, **k: pytest.fail("data read for a sealed city"))
    for c in ("L1", "C1"):
        with pytest.raises(ValueError):
            banded.city_banded(c)


def test_city_banded_runs_on_the_synthetic_city(two_cities, monkeypatch):
    import importlib
    import json
    import config
    importlib.reload(config)
    csv = two_cities / "roles.csv"
    csv.write_text("city,role\nTEST1,discovery\nTEST2,lawtest\n")
    monkeypatch.setattr(config, "CITIES_CSV", csv)
    (config.RESULTS / "refs").mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "refs/TEST1.json").write_text(json.dumps({"models": {"M2_glm": {"nll_per_visit": 1.0}}}))
    out = banded.city_banded("TEST1", out_dir=two_cities / "banded")
    assert set(out["gamma"]) == set(range(len(banded.BANDS))) and out["m2_val_nll"] == 1.0
    assert np.isfinite(out["train_nll"]) and np.isfinite(out["val_nll"])
    assert set(out["visits"]) == set(out["refs"]) == set(out["profile"])
    back = json.loads((two_cities / "banded/TEST1.json").read_text())
    assert back["city"] == "TEST1" and set(back["gamma"]) == {str(b) for b in range(len(banded.BANDS))}


def test_city_banded_allowed_list_never_admits_a_confirmation_city(tmp_path, monkeypatch):
    import config
    import sr_data
    csv = tmp_path / "cities.csv"
    csv.write_text("city,role\nS1,search\nL1,lawtest\nC1,confirm\n")
    monkeypatch.setattr(config, "CITIES_CSV", csv)
    monkeypatch.setattr(sr_data, "load", lambda *a, **k: pytest.fail("data read"))
    with pytest.raises(ValueError):
        banded.city_banded("C1", allowed=["C1", "L1"])
    with pytest.raises(ValueError):
        banded.city_banded(config.CONFIRM[0], allowed=[config.CONFIRM[0]])
    with pytest.raises(ValueError):
        banded.city_banded("L1")                                          # default: search + discovery only
