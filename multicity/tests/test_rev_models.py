"""Revision 2026-11-13, Task R1: the pure parts of rev_models.py."""
import numpy as np
import pandas as pd
import pytest

import rev_models as rm
from refit_select import parse_equation


def _write_city(root, line="M", city="c1", n=4):
    base = root / f"derived/model_inputs/{line}/{city}/tract"
    base.mkdir(parents=True)
    pd.DataFrame({"unit": range(n), "x": np.arange(n, dtype=float), "y": 0.0, "population": 100.0, "N": 5.0,
                  "origin_included": True, "ses": 1}).to_parquet(base / "units.parquet")
    np.save(base / "D.npy", np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(np.float32))
    for k, sp in enumerate(("train", "val", "test")):
        np.savez(base / f"T_{sp}.npz", n_dev_5=np.full((n, n), 10.0 + k), n_work_5=np.full((n, n), 3.0),
                 n_nonwork_5=np.full((n, n), 7.0 + k))
    return city


def test_use_counts_switches_every_loader_and_restores(tmp_path, monkeypatch):
    import sr_data
    import val_refs
    monkeypatch.setattr(sr_data, "ROOT", tmp_path)
    monkeypatch.setattr(sr_data, "LINE", "M")
    monkeypatch.setattr(val_refs, "load", sr_data.load)    # another test may have reloaded sr_data
    c = _write_city(tmp_path)
    orig = sr_data.load
    assert sr_data.load(c, "tract")["T"]["train"][0, 0] == 10.0
    with rm.use_counts("n_nonwork_5"):
        assert sr_data.load(c, "tract")["T"]["train"][0, 0] == 7.0
        assert sr_data.load(c, "tract")["T"]["test"][0, 0] == 9.0
        assert val_refs.load(c, "tract")["T"]["val"][0, 0] == 8.0     # imported by name elsewhere: patched too
    assert sr_data.load is orig and val_refs.load is orig
    with rm.use_counts("n_dev_5"):                                      # the default column: no patch at all
        assert sr_data.load is orig


def test_uniform_consts_leaves_the_city_out():
    K = {"a": [1.0, 10.0], "b": [3.0, 30.0], "c": [5.0, 50.0], "d": [100.0, 0.0]}
    np.testing.assert_allclose(rm.uniform_consts(K, "d"), [3.0, 30.0])
    np.testing.assert_allclose(rm.uniform_consts(K, "a"), [5.0, 30.0])
    np.testing.assert_allclose(rm.uniform_consts(K, "elsewhere"), [4.0, 20.0])


def test_paired_dnll_is_b_minus_a_in_nats():
    rows = []
    for c, (a, b) in {"x": (3.0, 3.1), "y": (4.0, 3.98), "z": (2.0, 2.5), "w": (np.nan, 2.0)}.items():
        rows += [{"city": c, "model": "A", "level": "local", "nll": a}, {"city": c, "model": "B", "level": "local", "nll": b}]
    t = pd.DataFrame(rows)
    r = rm.paired_dnll(t, ("A", "local"), ("B", "local"), ["x", "y", "z", "w"], B=200, seed=1)
    assert r["n"] == 4 and r["n_failed"] == 1                          # w: A failed -> a loss for A, as in p9_stats
    assert r["wins"] == 2 and r["losses"] == 2                          # x +0.1, y -0.02, z +0.5, w -inf
    assert r["median"] == pytest.approx(0.04)                           # median of (-inf, -0.02, 0.1, 0.5)
    both = t.copy()
    both.loc[(both["city"] == "w") & (both["model"] == "B"), "nll"] = np.nan
    r2 = rm.paired_dnll(both, ("A", "local"), ("B", "local"), ["x", "y", "z", "w"], B=200, seed=1)
    assert r2["ties"] == 1 and r2["n_failed"] == 1                     # both failed: a tie


def test_every_free_mask_matches_its_equation():
    for name, (eq, free) in rm.REV_EQ.items():
        c = parse_equation(eq, None, "A", 9)
        assert len(free) == len(c.consts), name


def test_separable_twin_has_the_reach_models_inputs_and_size():
    sk, sep = rm.REV_EQ["sk"], rm.REV_EQ["sep_N"]
    assert sum(sk[1]) == sum(sep[1]) == 5
    for v in ("#1", "#6", "#7"):
        assert v in sk[0] and v in sep[0]
    assert set(rm.LINE_MODELS["nonwork"]) <= set(rm.REV_EQ)


def test_use_line_poitrain_swaps_the_poi_counts_only(tmp_path, monkeypatch):
    import sr_data
    import val_refs
    monkeypatch.setattr(sr_data, "ROOT", tmp_path)
    monkeypatch.setattr(sr_data, "LINE", "M")
    monkeypatch.setattr(val_refs, "load", sr_data.load)    # another test may have reloaded sr_data
    c = _write_city(tmp_path)
    tab = pd.DataFrame({"tract": [0, 2], "N": [11, 13], "n_food": [1, 2], "n_other": [0, 1]})
    tab["tract"] = tab["tract"].astype(str)
    tab.to_parquet(tmp_path / "poi_train.parquet")
    monkeypatch.setattr(rm, "POI_TRAIN", tmp_path / "poi_train.parquet")
    orig = sr_data.load
    with rm.use_line("poitrain"):
        u = sr_data.load(c, "tract")["u"]
        assert u["N"].tolist() == [11.0, 0.0, 13.0, 0.0]                  # tracts absent from the table: 0
        assert val_refs.load(c, "tract")["T"]["train"][0, 0] == 10.0      # counts unchanged on this line
    assert sr_data.load is orig and val_refs.load is orig
    with rm.use_line("nonwork"):
        assert sr_data.load(c, "tract")["u"]["N"].tolist() == [5.0] * 4
        assert sr_data.load(c, "tract")["T"]["train"][0, 0] == 7.0


def test_fixed_shift_variants_keep_the_utility_defined():
    for name in ("sep_E1", "sk_E1", "sk_plus1"):
        eq, free = rm.REV_EQ[name]
        assert "+ 1.0)" in eq and len(free) == len(__import__("refit_select").parse_equation(eq, None, "A", 9).consts)
    assert sum(rm.REV_EQ["sep_E1"][1]) == sum(rm.REV_EQ["sk_E1"][1]) == 4
    assert rm.LINES["all2"] == ("n_dev_5", False)
