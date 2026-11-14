"""Plan 4 Task 1: the 53 rest cities are split into discovery / lawtest."""
import numpy as np
import pandas as pd

from split_rest import split_rest


def _fixture(n_rest=53, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"{10000 + 7 * k}" for k in range(n_rest)]
    cities = pd.DataFrame({"city": ids + ["S1", "S2", "C1"], "name": "x",
                           "role": ["rest"] * n_rest + ["search", "search", "confirm"],
                           "tier": [3.0] * n_rest + [1.0, 1.0, 2.0]})
    cities = pd.concat([cities, pd.DataFrame({"city": ["E1"], "name": "x", "role": ["excluded"], "tier": [3.0]})],
                       ignore_index=True)
    form = pd.DataFrame({"city": ids, "population": rng.lognormal(14, 1, n_rest),
                         "pop_density_gini": rng.uniform(0.2, 0.6, n_rest).round(3)})
    if n_rest > 4:
        form.loc[3, "pop_density_gini"] = form.loc[4, "pop_density_gini"]      # a tie
    return cities, form


def test_split_rest_alternates_within_population_tiers():
    cities, form = _fixture(n_rest=53)
    out = split_rest(cities, form)
    rest = out[out.role.isin(["discovery", "lawtest"])].merge(form, on="city")
    assert len(rest) == 53 and abs((rest.role == "discovery").sum() - 26.5) <= 0.5
    assert rest.split_tier.nunique() == 3
    tiers = rest.groupby("split_tier").population.agg(["min", "max"]).sort_index()
    assert (tiers["max"].to_numpy()[:-1] < tiers["min"].to_numpy()[1:]).all()   # terciles of population
    for _, g in rest.groupby("split_tier"):                     # split_tier = population tercile computed inside
        order = g.sort_values("split_rank")
        assert order.pop_density_gini.is_monotonic_increasing    # rank = gini ascending (ties broken by the seed)
        assert order.role.tolist() == [("discovery", "lawtest")[k % 2] for k in range(len(order))]
    keep = cities.role.isin(["search", "confirm", "excluded"])
    assert (out.loc[keep, "role"] == cities.loc[keep, "role"]).all()
    assert (out["tier"] == cities["tier"]).all()                # the frozen population-tier column is untouched


def test_split_rest_tie_break_uses_seed_and_alternates():
    cities, form = _fixture(n_rest=53)
    a, b = split_rest(cities, form, seed=1), split_rest(cities, form, seed=2)
    for out in (a, b):
        rest = out[out.role.isin(["discovery", "lawtest"])].merge(form, on="city")
        for _, g in rest.groupby("split_tier"):
            assert g.sort_values("split_rank").role.tolist() == [("discovery", "lawtest")[k % 2] for k in range(len(g))]


def test_split_rest_is_deterministic():
    cities, form = _fixture(n_rest=53)
    assert split_rest(cities, form).role.tolist() == split_rest(cities, form).role.tolist()


def test_split_rest_handles_missing_form_metric():
    cities, form = _fixture(n_rest=53)
    gone = cities[cities.role == "rest"].city.iloc[0]
    form_gini_missing = form.copy()
    form_gini_missing.loc[form_gini_missing.city == cities[cities.role == "rest"].city.iloc[1], "pop_density_gini"] = np.nan
    for f in (form[form.city != gone], form_gini_missing):
        out = split_rest(cities, f)
        assert set(out[out.role.isin(["discovery", "lawtest"])].city) == set(cities[cities.role == "rest"].city)
        assert out.role.tolist() == split_rest(cities, f).role.tolist()
    out = split_rest(cities, form_gini_missing)
    miss = cities[cities.role == "rest"].city.iloc[1]
    g = out[out.split_tier == out.loc[out.city == miss, "split_tier"].iloc[0]].sort_values("split_rank")
    assert g.city.iloc[-1] == miss                               # missing gini goes last in its tier


def test_main_refuses_to_resplit(tmp_path, monkeypatch):
    import pytest
    import split_rest as sr
    cities, form = _fixture(n_rest=10)
    cities.loc[0, "role"] = "discovery"
    monkeypatch.setattr(sr, "CITIES_CSV", tmp_path / "cities.csv")
    cities.to_csv(tmp_path / "cities.csv", index=False)
    with pytest.raises(SystemExit):
        sr.main()


def test_cities_with_role(tmp_path, monkeypatch):
    import config
    cities, _ = _fixture(n_rest=4)
    cities.to_csv(tmp_path / "cities.csv", index=False)
    monkeypatch.setattr(config, "CITIES_CSV", tmp_path / "cities.csv")
    assert config.cities_with_role("search") == ["S1", "S2"]
