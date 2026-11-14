"""Exploration Task 4 (form effects and city types): city types by skeleton-family support
and by behaviour curves, the stability and prediction-strength gates, and the description of the types."""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

import city_types as ct

REFS = {c: {"grav_power": {"nll_per_visit": 4.2}, "grav_exp": {"nll_per_visit": 4.25}, "ext_rad": {"nll_per_visit": 4.3},
            "M2_glm": {"nll_per_visit": 4.0}, "S0": {"nll_per_visit": 4.1}} for c in ("A", "B")}


def blobs(k, n_each, seed, sd=0.4, dim=4):
    rng = np.random.default_rng(seed)
    centres = np.eye(dim)[:k] * 10.0
    X = np.vstack([c + rng.normal(scale=sd, size=(n_each, dim)) for c in centres])
    return pd.DataFrame(X, index=[f"s{seed}_{i}" for i in range(len(X))])


def test_module_imports_without_matplotlib(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    importlib.reload(ct)


def test_family_profile_lag_and_floor():
    cv = pd.DataFrame([("A", "k1", 4.2 - 0.99 * 0.2, True), ("A", "o1", 4.2 - 0.95 * 0.2, True),
                       ("B", "k1", 4.2 - 0.90 * 0.2, True), ("B", "o1", np.nan, False)],
                      columns=["city", "cid", "val_nll", "finite"])
    prof = ct.family_profile(cv, REFS, {"K": ["k1"], "O": ["o1"]})
    assert prof.loc["A", "K"] == 0 and prof.loc["A", "O"] == pytest.approx(-0.04) and prof.loc["B", "K"] == 0
    assert prof.loc["B", "O"] == pytest.approx(-0.04)                                 # floor = global minimum lag


def test_prediction_strength_perfect_and_singletons():
    Xtr, Xte = blobs(3, 12, seed=0), blobs(3, 9, seed=1)
    assert ct.prediction_strength(Xtr, Xte, 3) == pytest.approx(1.0)
    far = pd.DataFrame([[-50.0, -50.0, -50.0, -50.0]], index=["far"])
    one = pd.concat([blobs(2, 9, seed=2), far])                                       # two blobs + a lone far point
    assert ct.prediction_strength(Xtr, one, 3) == pytest.approx(1.0)                  # singleton cluster not counted
    assert np.isnan(ct.prediction_strength(Xtr, Xte.iloc[[0, 20]], 2))                # only singleton clusters


def test_choose_k_on_separated_blobs():
    # seeds 10/20: with seeds 0/1 both halves happen to split one blob the same way, so k = 4 also reaches
    # prediction strength 1.0 - a known small-sample limit of a single train/test split (ledgered, reported)
    Xtr, Xte = blobs(3, 12, seed=10), blobs(3, 9, seed=20)
    X = pd.concat([Xtr, Xte])
    res = ct.choose_k(X, Xtr, Xte, [2, 3, 4, 5, 6], 0.6, 0.8, B=50, frac=0.8, seed=0)
    assert res["k"] == 3 and [r["k"] for r in res["table"]] == [2, 3, 4, 5, 6]


def test_choose_k_reports_none_for_noise():
    rng = np.random.default_rng(5)
    U = pd.DataFrame(rng.normal(size=(61, 5)), index=[f"u{i}" for i in range(61)])
    res = ct.choose_k(U, U.iloc[:35], U.iloc[35:], [2, 3, 4, 5, 6], 0.6, 0.8, B=50, frac=0.8, seed=0)
    assert res["k"] is None


def test_behaviour_profile_drops_missing_bands():
    bands = ["0_1km", "1_2km", "2_5km", "5_10km", "10_20km"]
    cols = {f"{q}_{b}": [-1.0, -1.1, -1.2] for b in bands for q in ("ed", "eN")}
    cols["ed_10_20km"] = [-2.0, np.nan, -2.2]
    out = pd.DataFrame({"city": ["a", "b", "c"], "ed_20km": [np.nan] * 3, **cols})
    prof = ct.behaviour_profile(out, bands)
    assert list(prof.index) == ["a", "c"] and prof.shape[1] == 10


def test_render_types_without_stable_types():
    res = {"config_sha": "x" * 64, "features": {
        "skeleton": {"n": 67, "dropped": [], "k": None,
                     "table": [{"k": 2, "silhouette": 0.3, "stability": 0.5, "ps": float("nan")}]},
        "behaviour": {"n": 60, "dropped": ["a"], "k": None,
                      "table": [{"k": 2, "silhouette": 0.2, "stability": 0.7, "ps": 0.6}]}}}
    md = ct.render_types(res)
    assert "Exploratory" in md and "No stable types" in md and "nan" not in md.lower()
