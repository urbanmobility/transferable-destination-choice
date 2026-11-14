"""Exploration Tasks 2-3 (form effects beyond size): methods fixed before computing, the 67-city
frame, E-1 (signal left beyond size) and E-2 (effect sizes, FDR, shrinkage, form axes)."""
import json

import numpy as np
import pandas as pd
import pytest

import explore_form as ef
import explore_p7 as ex

CFG = json.loads(ef.CONFIG.read_text())


def zs(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / a.std(ddof=1)


def test_config_lists():
    assert len(CFG["outcomes"]) == 30 and len(CFG["covariates"]) == 17 and CFG["seed"] == 20261110
    assert sum(len(v) for v in CFG["e3"]["families"].values()) == 26


def test_fix_config_and_require_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(ef, "CONFIG", tmp_path / "c.json")
    monkeypatch.setattr(ef, "FIXED", tmp_path / "out/CONFIG_FIXED")
    ef.CONFIG.write_text('{"seed": 1}')
    with pytest.raises(RuntimeError):
        ef.require_fixed()
    sha = ef.fix_config(log=lambda s: None)
    assert ef.require_fixed() == {"seed": 1} and ef.fix_config(log=lambda s: None) == sha
    assert len(json.loads(ef.FIXED.read_text())) == 1                                  # same sha: no new entry
    ef.CONFIG.write_text('{"seed": 2}')
    with pytest.raises(RuntimeError):
        ef.require_fixed()
    with pytest.raises(ValueError):
        ef.fix_config(log=lambda s: None)
    ef.fix_config(reason="typo in seed", log=lambda s: None)
    assert ef.require_fixed() == {"seed": 2} and len(json.loads(ef.FIXED.read_text())) == 2


def _frame_files(root):
    pd.DataFrame({"city": ["a", "a", "b", "z"], "block": ["test_dest", "val", "test_dest", "test_dest"],
                  "y": [1.0, 9.0, 2.0, 5.0], "log_population": [5.0, 5.0, 6.0, 7.0]}).to_parquet(root / "city_outcomes.parquet")
    pd.DataFrame({"city": ["a", "b"], "x": [0.1, 0.2]}).to_csv(root / "covariates.csv", index=False)
    pd.DataFrame({"city": ["c"], "block": ["test_dest"], "y": [3.0], "log_population": [7.0]}).to_parquet(
        root / "city_outcomes_lawtest.parquet")
    pd.DataFrame({"city": ["c"], "x": [0.3]}).to_csv(root / "covariates_lawtest.csv", index=False)


def test_load_frame_is_complete_and_unique(tmp_path, monkeypatch):
    _frame_files(tmp_path)
    roles = {"search": ["a"], "discovery": ["b"], "lawtest": ["c"], "confirm": []}
    monkeypatch.setattr(ef.config, "cities_with_role", lambda r: roles[r])
    monkeypatch.setattr(ef.laws, "covariates", lambda cs, allow_confirm=False: pd.DataFrame(
        {"city": cs, "log_population": np.arange(len(cs), dtype=float)}))
    out, cov, g = ef.load_frame(("discovery", "lawtest"), root=tmp_path)
    assert sorted(out.city) == ["a", "b", "c"] and g == {"a": "discovery", "b": "discovery", "c": "lawtest"}
    assert "block" not in out and "log_population" not in out and cov.city.is_unique
    assert out.set_index("city").loc["a", "y"] == 1.0                                 # test_dest row only; z ignored
    roles["lawtest"] = ["c", "b"]
    with pytest.raises(ValueError):
        ef.load_frame(("discovery", "lawtest"), root=tmp_path)                       # b in two groups
    roles["lawtest"] = ["c"]
    pd.DataFrame({"city": [], "x": []}).to_csv(tmp_path / "covariates_lawtest.csv", index=False)
    with pytest.raises(ValueError):
        ef.load_frame(("discovery", "lawtest"), root=tmp_path)                       # c lacks a covariate row


def test_std_partial_slope_matches_explore_and_needs_the_control():
    rng = np.random.default_rng(0)
    n = 60
    lp = rng.normal(size=n)
    x = 0.7 * lp + rng.normal(size=n)
    y = 2 * x - 3 * lp + rng.normal(scale=0.1, size=n)
    cities = [f"c{i}" for i in range(n)]
    out = pd.DataFrame({"city": cities, "y": y})
    cov = pd.DataFrame({"city": cities, "x": x, "log_population": lp})
    s = ef.std_partial_slopes(out, cov, [("y", "x")])[0]
    assert s == pytest.approx(ex._slope(zs(y), zs(x), lp), abs=1e-12)
    assert abs(s - ex._slope(zs(y), zs(x))) > 0.3                                      # no control -> different
    two = ef.std_partial_slopes(out, cov.assign(t=rng.normal(size=n)), [("y", "x")],
                                controls=("log_population", "t"))[0]
    assert two == pytest.approx(s, abs=0.05)                                           # an irrelevant 2nd control barely moves it


def _rel():
    cols = ["outcome", "covariate", "ci_lo", "ci_hi", "pci_lo", "pci_hi", "partial_slope", "r_size"]
    rows = [("o1", "log_population", 0.1, 0.2, np.nan, np.nan, np.nan, 1.0),
            ("o1", "tract_area", 0.0, 1.0, 0.1, 0.3, 0.2, 0.3),
            ("o1", "x", 0.0, 1.0, 0.1, 0.3, 0.2, 0.9),
            ("o1", "y", 0.0, 1.0, 0.1, 0.3, 0.2, 0.2),
            ("o2", "y", 0.0, 1.0, -0.1, 0.3, 0.1, 0.2),
            ("o3", "w", 0.0, 1.0, -0.3, -0.1, -0.2, 0.1),
            ("o4", "log_jobs", 0.0, 1.0, 0.1, 0.2, 0.15, 0.95)]
    return pd.DataFrame(rows, columns=cols)


def test_e1_pairs_exclusions():
    reg = [{"outcome": "o3", "covariate": "w"}]
    pairs = lambda v: list(zip(*[ef.e1_pairs(_rel(), reg, CFG, v)[k] for k in ("outcome", "covariate")]))  # noqa: E731
    assert pairs("main") == [("o1", "y")]
    assert pairs("with_registered") == [("o1", "y"), ("o3", "w")]
    assert pairs("with_confounders") == [("o1", "tract_area"), ("o1", "x"), ("o1", "y")]


def test_e1_summary_values():
    s = ef.e1_summary(np.array([1.0, -1.0, 2.0, -0.5]), np.array([0.5, -0.25, 1.0, 0.25]), ["x", "x", "y", "y"])
    assert s["n_pairs"] == 4 and s["agree"] == 0.75 and s["median_ratio"] == pytest.approx(0.375)
    assert s["by_covariate"]["x"]["agree"] == 1.0 and s["by_covariate"]["y"]["agree"] == 0.5


def test_freedman_lane_permutes_only_the_residuals():
    rng = np.random.default_rng(1)
    lp = rng.normal(size=30)
    x = 1 + 2 * lp + rng.normal(size=30)
    cov = pd.DataFrame({"city": [f"c{i}" for i in range(30)], "x": x, "log_population": lp, "other": rng.normal(size=30)})
    b, a = np.polyfit(lp, x, 1)
    fitted, resid = a + b * lp, x - (a + b * lp)
    perm = rng.permutation(30)
    p = ef.freedman_lane(cov, ["x"], "log_population", perm)
    assert p.x.to_numpy() == pytest.approx(fitted + resid[perm])
    pd.testing.assert_series_equal(p.other, cov.other)
    pd.testing.assert_frame_equal(ef.freedman_lane(cov, ["x"], "log_population", np.arange(30)), cov)


def _perm_world(signal, seed):
    rng = np.random.default_rng(seed)
    n, k = 26, 20
    cities = [f"c{i}" for i in range(n)]
    X = rng.normal(size=(n, k))
    Y = (0.8 * X if signal else 0.0) + rng.normal(scale=0.6, size=(n, k))
    cov = pd.DataFrame(X, columns=[f"x{j}" for j in range(k)]).assign(city=cities, log_population=rng.normal(size=n))
    out = pd.DataFrame(Y, columns=[f"y{j}" for j in range(k)]).assign(city=cities)
    return out, cov, [(f"y{j}", f"x{j}") for j in range(k)]


def test_e1_permutation_detects_signal_and_null():
    out, cov, pairs = _perm_world(True, 0)
    obs, null, p = ef.e1_permutation(out, cov, pairs, np.ones(len(pairs)), B=200, seed=1)
    assert obs == 1.0 and p <= 0.01 and len(null) == 200
    out, cov, pairs = _perm_world(False, 0)
    _, _, p = ef.e1_permutation(out, cov, pairs, np.ones(len(pairs)), B=200, seed=1)
    assert p >= 0.05


def test_stage_e1_refuses_unfixed_config(tmp_path, monkeypatch):
    monkeypatch.setattr(ef, "FIXED", tmp_path / "CONFIG_FIXED")
    with pytest.raises(RuntimeError):
        ef.stage_e1()


# ------------------------------------------------------------------ Task 3: E-2
def test_bh_known_values():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert ef.bh(np.array(p), 0.05).tolist() == [True, True] + [False] * 8
    assert ef.bh(np.array([np.nan, 0.001]), 0.05).tolist() == [False, True]


def test_eb_shrink():
    one = np.ones(4)
    assert ef.eb_shrink(np.array([1.0, -1, 1, -1]), one, ["a"] * 4) == pytest.approx(0.0)
    assert ef.eb_shrink(np.array([3.0, -3, 3, -3]), one, ["a"] * 4) == pytest.approx([8 / 3, -8 / 3, 8 / 3, -8 / 3])
    assert ef.eb_shrink(np.array([3.0, -3]), np.zeros(2), ["a", "a"]) == pytest.approx([3.0, -3.0])
    mixed = ef.eb_shrink(np.array([1.0, -1, 3, -3]), one, ["a", "a", "b", "b"])                 # per-group prior
    assert mixed == pytest.approx([0.0, 0.0, 8 / 3, -8 / 3])


def test_form_axes_two_factors():
    rng = np.random.default_rng(3)
    n = 60
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    cols = [f"v{j}" for j in range(6)]
    M = np.column_stack([f1, f1, f1, f2, f2, f2]) * np.array([1.0, 0.9, 0.8, 1.0, 0.9, 0.8]) + rng.normal(scale=0.05, size=(n, 6))
    cov = pd.DataFrame(M, columns=cols).assign(city=[f"c{i}" for i in range(n)], log_population=rng.normal(size=n))
    scores, load, expl = ef.form_axes(cov, cols, "log_population", 0.7, 5)
    assert list(scores.columns) == ["axis1", "axis2"] and expl[:2].sum() >= 0.7 and expl[0] < 0.7
    assert all(load[a].abs().idxmax() == load[a].idxmax() for a in ("axis1", "axis2"))   # largest |loading| positive
    assert list(scores.index) == list(cov.city)


def test_effects_size_row_uses_plain_slope():
    rng = np.random.default_rng(4)
    n = 40
    lp = rng.normal(size=n)
    x = 0.5 * lp + rng.normal(size=n)
    y = x + lp + rng.normal(scale=0.5, size=n)
    cities = [f"c{i}" for i in range(n)]
    out = pd.DataFrame({"city": cities, "y": y})
    cov = pd.DataFrame({"city": cities, "x": x, "log_population": lp})
    e = ef.effects(out, cov, ["y"], ["x", "log_population"], B=200).set_index("covariate")
    assert e.loc["log_population", "std_slope"] == pytest.approx(np.corrcoef(y, lp)[0, 1], abs=1e-9)
    assert e.loc["x", "std_slope"] == pytest.approx(ex._slope(zs(y), zs(x), lp), abs=1e-9)
    assert (e.ci_lo <= e.std_slope).all() and (e.std_slope <= e.ci_hi).all() and (e.se > 0).all()
    assert (e.n == 40).all()


def test_render_effects_counts_and_no_nan():
    e1 = {v: {"n_pairs": 10, "agree": 0.8, "median_ratio": 0.5, "pearson": 0.6, "spearman": 0.5, "by_covariate": {}}
          for v in ("main", "with_registered", "with_confounders")}
    e1["main"].update(perm_obs=0.8, perm_p=0.004, perm_B=2000, perm_null_mean=0.5, perm_null_q95=0.65)
    e2 = pd.DataFrame({"outcome": ["y", "y"], "covariate": ["x", "w"], "n": [67, 67], "std_slope": [0.4, np.nan],
                       "se": [0.1, np.nan], "ci_lo": [0.2, np.nan], "ci_hi": [0.6, np.nan], "p": [0.001, np.nan],
                       "p_norm": [0.0001, np.nan], "role": ["form", "form"],
                       "bh": [True, False], "eb": [0.35, np.nan], "registered": [False, False]})
    axes = {"explained": [0.4, 0.3], "loadings": {"axis1": {"x": 0.7, "w": 0.7}, "axis2": {"x": -0.7, "w": 0.7}},
            "effects": [{"outcome": "y", "term": "axis1", "coef": 0.3, "ci_lo": 0.1, "ci_hi": 0.5}]}
    md = ef.render_effects(e1, e2, axes)
    assert "exploratory" in md and "nan" not in md.lower() and "q = 0.10" in md and "permutation" in md and "0.004" in md


# ------------------------------------------------------------------ 2026-11-11: axis-level tract-area sensitivity
def test_axis_effects_extra_control_absorbs_a_confounded_axis():
    """y follows tract_area only; axis1 is correlated with tract_area: without the control axis1 looks like an
    effect, with it the axis coefficient goes to ~0 and tract_area gets its own term."""
    rng = np.random.default_rng(11)
    n = 80
    cities = [f"c{i}" for i in range(n)]
    ta = rng.normal(size=n)
    scores = pd.DataFrame({"axis1": 0.8 * ta + 0.6 * rng.normal(size=n), "axis2": rng.normal(size=n)}, index=cities)
    cov = pd.DataFrame({"city": cities, "log_population": rng.normal(size=n), "tract_area": ta})
    out = pd.DataFrame({"city": cities, "y": ta + rng.normal(scale=0.3, size=n)})
    plain = ef.axis_effects(out, scores, cov, ["y"], B=200, seed=1).set_index("term")
    ctl = ef.axis_effects(out, scores, cov, ["y"], B=200, seed=1, extra=("tract_area",)).set_index("term")
    assert list(plain.index) == ["log_population", "axis1", "axis2"]
    assert list(ctl.index) == ["log_population", "axis1", "axis2", "tract_area"]
    assert plain.loc["axis1", "ci_lo"] > 0.3 and abs(ctl.loc["axis1", "coef"]) < 0.15
    assert ctl.loc["axis1", "ci_lo"] < 0 < ctl.loc["axis1", "ci_hi"] and ctl.loc["tract_area", "ci_lo"] > 0.5
