"""Plan 9, Tasks 1-2: the earlier models — equation path (refit_select rows) and matrix path (radiation family, Kwon)."""
import numpy as np
import pytest

import prior_models as pm
from p9_fixture import toy_city, toy_inputs, toy_rows
from refit_select import parse_equation
from sr_data import SRData, VARS


# ---------------------------------------------------------------------------------------------- Task 1: equation path
def test_registry_has_the_thirteen_equation_models():
    assert len(pm.EQ_MODELS) == 13 and "S0_N" in pm.EQ_MODELS and "S0_P" not in pm.EQ_MODELS
    assert {k for k, s in pm.EQ_MODELS.items() if s.original} == {
        "grav_power_P", "grav_exp_P", "strexp1_P", "cab_pow_P", "cab_exp_P"}


@pytest.mark.parametrize("name", sorted(pm.EQ_MODELS))
def test_free_mask_matches_numerals(name):
    s = pm.EQ_MODELS[name]
    assert len(s.free) == len(parse_equation(s.equation, None, "A", 9).consts)


@pytest.mark.parametrize("name", sorted(pm.EQ_MODELS))
def test_poi_versions_do_not_read_destination_population(name):
    s = pm.EQ_MODELS[name]
    assert s.mass == name.rsplit("_", 1)[1]
    assert ("#4" not in s.equation) if s.mass == "N" else ("#6" not in s.equation)


def _rows(y, n_dest):
    n = len(y)
    z = np.zeros(n, int)
    return SRData(X=np.ones((n, len(VARS)), np.float32), names=list(VARS), y=np.asarray(y, float), w=np.ones(n),
                  offsets=np.arange(0, n + 1, n_dest), city=z, ses=z, offset_B=np.zeros(n), origin_unit=z, dest_unit=z)


def test_score_rows_uniform():
    s = pm.score_rows(np.zeros(8), _rows([1, 1, 1, 1, 2, 0, 0, 2], 4))
    assert s["nll"] == pytest.approx(np.log(4)) and s["cpc"] == pytest.approx(0.75)


def test_score_rows_nonfinite_is_nan():
    e = np.zeros(8)
    e[3] = np.nan
    s = pm.score_rows(e, _rows([1, 1, 1, 1, 2, 0, 0, 2], 4))
    assert np.isnan(s["nll"]) and np.isnan(s["cpc"])


def test_score_rows_ignores_empty_origin():
    e = np.array([0.3, -0.2, 0.1, 0.0])
    a = pm.score_rows(np.r_[e, e, e], _rows([1, 2, 0, 1, 0, 0, 0, 0, 3, 0, 1, 0], 4))
    b = pm.score_rows(np.r_[e, e], _rows([1, 2, 0, 1, 3, 0, 1, 0], 4))
    assert a["nll"] == pytest.approx(b["nll"]) and a["cpc"] == pytest.approx(b["cpc"])


@pytest.fixture(scope="module")
def toy():
    return toy_inputs(0)


def test_fit_eq_recovers_gravity(toy):
    s = pm.EQ_MODELS["grav_power_N"]
    start = pm.score_eq(s.equation, parse_equation(s.equation, None, "A", 9).consts, toy["tr"])["nll"]
    nll, consts = pm.fit_eq(s.equation, s.free, toy["tr"])
    assert consts[2] == pytest.approx(1.5, abs=0.15) and consts[1] == 1.0 and nll <= start


def test_fit_eq_none_when_never_finite(toy):
    assert pm.fit_eq("f = log(#1 - 1000.0)", (True,), toy["tr"]) is None


@pytest.mark.parametrize("name", sorted(pm.EQ_MODELS))
def test_every_model_finite_at_start(name, toy):
    s = pm.EQ_MODELS[name]
    r = pm.score_eq(s.equation, parse_equation(s.equation, None, "A", 9).consts, toy["tr"])
    assert np.isfinite(r["nll"]) and 0 < r["cpc"] <= 1


def test_equation_path_handles_zero_mass_units():
    d = toy_city(3, zero_mass=True)
    tr = toy_rows(d, np.arange(10), "train", "search")
    for name, s in pm.EQ_MODELS.items():
        r = pm.score_eq(s.equation, parse_equation(s.equation, None, "A", 9).consts, tr)
        assert np.isfinite(r["nll"]), name


# ---------------------------------------------------------------------------------------------- Task 2: matrix path
def _brute_S(D, m):
    """Mass strictly closer to i than j, both ends excluded."""
    N = len(m)
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            S[i, j] = sum(m[k] for k in range(N) if k not in (i, j) and D[i, k] < D[i, j])
    return S


def test_registry_has_the_eleven_matrix_models():
    assert len(pm.MAT_MODELS) == 11
    assert {k for k, s in pm.MAT_MODELS.items() if s.needs_trips} == {"kwon"}
    assert {k for k, s in pm.MAT_MODELS.items() if s.original} == {"rad_P", "ext_rad_P", "io_P", "pwo_P", "ops_P", "kwon"}
    assert pm.MAT_MODELS["io_N"].bounds == [(np.log(1e-9), np.log(10.0))]
    assert pm.MAT_MODELS["ext_rad_P"].bounds == [(0.01, 5.0)]


def test_dest_centred_bruteforce():
    d = toy_city(1, n_units=8)
    m = d["u"]["N"].to_numpy(float) + 1.0
    SJ = pm.dest_centred(d["D"], m)
    for i in range(8):
        for j in range(8):
            assert SJ[i, j] == pytest.approx(sum(m[k] for k in range(8) if d["D"][j, k] <= d["D"][i, j]))


@pytest.mark.parametrize("mass, col", [("P", "population"), ("N", "N")])
def test_radiation_ops_pwo_bruteforce(mass, col):
    t = toy_inputs(1, n_units=8, n_orig=8)
    A, D = t["A"], t["d"]["D"]
    raw = t["d"]["u"][col].to_numpy(float)
    m, S = raw + 1.0, _brute_S(D, raw)
    io = np.arange(8)
    rad, ops, pwo = (pm.mat_eta(f"{k}_{mass}", [], A, io) for k in ("rad", "ops", "pwo"))
    M = m.sum()
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            sj = sum(m[k] for k in range(8) if D[j, k] <= D[i, j])
            assert rad[i, j] == pytest.approx(np.log(m[j] / ((m[i] + S[i, j]) * (m[i] + m[j] + S[i, j]))), abs=1e-9)
            assert ops[i, j] == pytest.approx(np.log(m[j] / (m[i] + S[i, j] + m[j])), abs=1e-9)
            if sj < M:
                assert pwo[i, j] == pytest.approx(np.log(m[j] * (1 / sj - 1 / M)), abs=1e-9)
            else:                                   # the whole city lies within d_ij of j: probability 0, floored
                assert pwo[i, j] < -600


def test_poi_version_ignores_population():
    from form_vars import intervening
    d = toy_city(2)
    d2 = dict(d, u=d["u"].assign(population=d["u"]["population"] * 7))
    A, A2 = (pm.city_arrays(x, x["fold"], "val", "val", intervening(x["D"], x["u"]["population"].to_numpy(float)),
                            n_orig=10) for x in (d, d2))
    io = A["io"]["ev"]
    for fam, th in (("rad", []), ("ext_rad", [0.5]), ("io", [np.log(1e-3)]), ("pwo", []), ("ops", [])):
        assert np.array_equal(pm.mat_eta(f"{fam}_N", th, A, io), pm.mat_eta(f"{fam}_N", th, A2, io)), fam
        assert not np.allclose(pm.mat_eta(f"{fam}_P", th, A, io), pm.mat_eta(f"{fam}_P", th, A2, io)), fam


def test_zero_mass_units_are_finite():
    t = toy_inputs(3, zero_mass=True)
    A = t["A"]
    for name, s in pm.MAT_MODELS.items():
        e = pm.mat_eta(name, s.start, A, A["io"]["ev"])
        assert np.isfinite(e[A["av"]["ev"]]).all(), name
        assert np.isfinite(pm.score_mat(e, A["T"]["ev"], A["av"]["ev"])["nll"]), name


def test_io_fit_recovers_L():
    from form_vars import intervening

    def io_eta(d):
        poi = d["u"]["N"].to_numpy(float)
        return -0.02 * intervening(d["D"], poi) + np.log(-np.expm1(-0.02 * (poi + 1.0)))[None, :]
    t = toy_inputs(4, n_units=60, n_orig=40, trips=2000, eta_fn=io_eta)
    th = pm.fit_mat("io_N", t["A"])
    assert 0.01 < np.exp(th[0]) < 0.04


def test_kwon_groups_equal_count():
    rng = np.random.default_rng(0)
    vol = rng.uniform(0, 100, 47)
    dest = np.ones(47, bool)
    dest[[3, 9]] = False
    g = pm.kwon_groups(vol, dest)
    cnt = np.bincount(g[dest], minlength=10)
    assert cnt.max() - cnt.min() <= 1 and set(g[dest]) == set(range(10)) and (g[~dest] == 0).all()
    o = np.argsort(vol[dest], kind="stable")
    assert (np.diff(g[dest][o]) >= 0).all()


def test_kwon_pair_index():
    g = np.repeat(np.arange(10), 2)
    P = pm.kwon_pairs(g)
    assert np.array_equal(P, P.T) and set(np.unique(P)) == set(range(55))


def test_kwon_gradient_matches_finite_differences():
    A = toy_inputs(5, n_units=40, n_orig=20)["A"]
    rng = np.random.default_rng(0)
    th = np.r_[rng.normal(1.0, 0.2, 55), rng.normal(0.0, 0.2, 55)]
    f0, g = pm.kwon_objective(th, A)
    for k in rng.choice(110, 5, replace=False):
        h = np.zeros(110)
        h[k] = 1e-5
        num = (pm.kwon_objective(th + h, A)[0] - pm.kwon_objective(th - h, A)[0]) / 2e-5
        assert g[k] == pytest.approx(num, rel=1e-4, abs=1e-8)


def test_kwon_fit_lowers_nll_and_stays_finite():
    A = toy_inputs(5, n_units=20, n_orig=6)["A"]
    s = pm.MAT_MODELS["kwon"]
    start = pm.score_mat(pm.mat_eta("kwon", s.start, A, A["io"]["tr"]), A["T"]["tr"], A["av"]["tr"])["nll"]
    th = pm.fit_mat("kwon", A)
    fit = pm.score_mat(pm.mat_eta("kwon", th, A, A["io"]["tr"]), A["T"]["tr"], A["av"]["tr"])["nll"]
    assert len(th) == 110 and np.isfinite(th).all() and fit < start
    seen = np.unique(A["kw_g"][A["io"]["tr"]][A["av"]["tr"]])          # 6 origins: some of the 55 group pairs are empty
    empty = sorted(set(range(55)) - set(seen))
    assert empty and all(th[k] == 1.0 and th[55 + k] == 0.0 for k in empty)
    assert np.isfinite(pm.score_mat(pm.mat_eta("kwon", th, A, A["io"]["ev"]), A["T"]["ev"], A["av"]["ev"])["nll"])


def test_score_mat_matches_score_rows(toy):
    A = toy["A"]
    s = pm.EQ_MODELS["grav_power_N"]
    c = parse_equation(s.equation, None, "A", 9).consts
    e = c[0] * np.log(A["mass"]["N"])[None, :] - c[2] * A["logd"][A["io"]["ev"]]
    a, b = pm.score_mat(e, A["T"]["ev"], A["av"]["ev"]), pm.score_eq(s.equation, c, toy["ev"])
    assert a["nll"] == pytest.approx(b["nll"], abs=1e-6) and a["cpc"] == pytest.approx(b["cpc"], abs=1e-6)


def test_fit_mat_parameter_counts(toy):
    assert pm.fit_mat("rad_P", toy["A"]) == [] and pm.fit_mat("ops_N", toy["A"]) == []
    th = pm.fit_mat("ext_rad_N", toy["A"])
    assert len(th) == 1 and 0.01 <= th[0] <= 5.0


def test_ext_rad_nonpositive_alpha_is_a_failure(toy):
    A = toy["A"]
    for a in (0.0, -0.3):
        e = pm.mat_eta("ext_rad_N", [a], A, A["io"]["ev"])
        assert np.isnan(pm.score_mat(e, A["T"]["ev"], A["av"]["ev"])["nll"])
    assert np.isfinite(pm.score_mat(pm.mat_eta("ext_rad_N", [0.01], A, A["io"]["ev"]), A["T"]["ev"], A["av"]["ev"])["nll"])


def test_io_and_ext_rad_bruteforce():
    t = toy_inputs(1, n_units=8, n_orig=8)
    A, D = t["A"], t["d"]["D"]
    raw = t["d"]["u"]["N"].to_numpy(float)
    m, S = raw + 1.0, _brute_S(D, raw)
    io_, er = pm.mat_eta("io_N", [np.log(0.01)], A, np.arange(8)), pm.mat_eta("ext_rad_N", [0.7], A, np.arange(8))
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            a, b = m[i] + S[i, j], m[i] + S[i, j] + m[j]
            assert io_[i, j] == pytest.approx(np.log(np.exp(-0.01 * S[i, j]) - np.exp(-0.01 * (S[i, j] + m[j]))), abs=1e-9)
            assert er[i, j] == pytest.approx(np.log((b ** 0.7 - a ** 0.7) * (m[i] ** 0.7 + 1) / ((a ** 0.7 + 1) * (b ** 0.7 + 1))), abs=1e-9)


def test_cab_exp_poi_version_starts_at_poi_scale():
    """Source-city audit (2026-11-11): from the population-scaled start the POI version stopped 0.02-0.04 nats above
    plain gravity in 4 of 35 cities; from this start it reaches it."""
    assert pm.EQ_MODELS["cab_exp_N"].equation == "f = 4.0*(1.0 + exp(-5.5)*((#6 + exp(2.0))*(#3 + exp(9.0)))^0.54/#1)^0.26"
    assert pm.EQ_MODELS["cab_exp_P"].equation == "f = 4.0*(1.0 + exp(-8.0)*((#4 + exp(5.0))*(#3 + exp(9.0)))^0.54/#1)^0.26"
    assert pm.EQ_MODELS["cab_exp_N"].free == pm.EQ_MODELS["cab_exp_P"].free
