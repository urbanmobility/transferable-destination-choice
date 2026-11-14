"""Supplementary check (added 2026-11-12): the comparison with every destination in the choice set — the pure parts."""
import numpy as np
import pandas as pd
import pytest

import alldest_check as ac


def _scores():
    rows = []
    for c, (g, m2, a, b) in {"x": (4.0, 3.0, 3.1, 3.3), "y": (5.0, 4.0, 4.05, 4.02), "z": (3.0, 2.5, 2.6, 2.9)}.items():
        rows += [{"city": c, "model": "grav_ref", "level": "local", "nll": g, "cpc": 0.5},
                 {"city": c, "model": "M2_ref", "level": "local", "nll": m2, "cpc": 0.7},
                 {"city": c, "model": "A", "level": "local", "nll": a, "cpc": 0.66},
                 {"city": c, "model": "B", "level": "local", "nll": b, "cpc": 0.60}]
    return pd.DataFrame(rows)


def test_add_share_uses_the_same_block_references():
    t = ac.add_share(_scores()).set_index(["city", "model"])
    assert t.loc[("x", "A"), "share"] == pytest.approx(0.9) and t.loc[("x", "B"), "share"] == pytest.approx(0.7)
    assert t.loc[("x", "grav_ref"), "share"] == pytest.approx(0.0) and t.loc[("x", "M2_ref"), "share"] == pytest.approx(1.0)


def test_add_share_refuses_a_city_without_a_positive_gap():
    s = _scores()
    s.loc[(s["city"] == "y") & (s["model"] == "M2_ref"), "nll"] = 5.2
    with pytest.raises(RuntimeError, match="y"):
        ac.add_share(s)


def test_paired_counts_wins_over_the_given_cities():
    t = ac.add_share(_scores())
    r = ac.paired(t, ("A", "local"), ("B", "local"), ["x", "y", "z"], B=200, seed=1)
    assert r["n"] == 3 and r["wins"] == 2 and r["median"] == pytest.approx(0.2)        # x +0.2, y -0.03, z +0.6
    assert ac.paired(t, ("A", "local"), ("B", "local"), ["y"], B=50, seed=1)["wins"] == 0
    with pytest.raises(ValueError):
        ac.paired(t, ("A", "local"), ("B", "local"), ["x", "nowhere"], B=50, seed=1)


def test_agreement_between_blocks():
    a = pd.Series({"x": 0.9, "y": 0.95, "z": 0.8})
    b = pd.Series({"x": 0.85, "y": 0.97, "z": 0.7, "w": 0.1})
    r = ac.agreement(a, b)
    assert r["n"] == 3 and r["spearman"] == pytest.approx(1.0) and r["median_diff"] == pytest.approx(-0.05)
