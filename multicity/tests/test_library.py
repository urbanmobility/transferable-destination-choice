"""Plan 2 Task 3: candidate pool (union of selections, structural dedup) and per-search-city multi-start refits with
1-SE flags and device-group bootstrap matrices."""
import dataclasses
import importlib
import json

import numpy as np
import pandas as pd

from test_search_param import two_cities  # noqa: F401  (fixture: TEST1 / TEST2 under a tmp ODSR_ROOT)

A = "f = log(#5 + 1.0); g = nexp(#5); h = log(#1 + 2.0); p1 = [0.1, 0.2]; p2 = [-2.0, -2.5]"
B1 = "f = log(#5 + 3.0); g = nexp(#5); h = log(#1 + 4.0); p1 = [0.3]; p2 = [-1.0]"       # same structure as A
B2 = "f = #6 * 2.0; g = nexp(#5); h = log(#1 + 1.0); p1 = [0.0]; p2 = [-1.0]"
C0 = "f = log(#5); g = nexp(#5); h = log(#1 + 2.0); p1 = [0.0]; p2 = [-2.0]"             # log(E_j): E_j = 0 rows exist


def _write(root):
    a = {"variant": "A", "candidates": [{"equation": A, "template": "city2", "complexity": 10,
                                         "consts_refit": [1.0, 2.0, 0.1, 0.2, -2.0, -2.5], "within_1se": True}]}
    b = {"variant": "A", "candidates": [{"equation": B1, "template": None, "complexity": 10,
                                         "consts_refit": [3.0, 4.0, 0.3, -1.0], "within_1se": False},
                                        {"equation": B2, "template": None, "complexity": 8,
                                         "consts_refit": [2.0, 1.0, 0.0, -1.0], "within_1se": True}]}
    pa, pb = root / "a.json", root / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    return pa, pb


def test_pool_dedups_by_structure(two_cities):
    import library
    importlib.reload(library)
    pa, pb = _write(two_cities)
    p = library.pool([pa, pb])
    assert len(p) == 2 and p[0]["cid"] == "c0001" and sorted(p[0]["source"]) == ["a.json", "b.json"]
    assert p[1]["cid"] == "c0002" and p[1]["complexity"] == 8 and p[1]["source"] == ["b.json"]
    assert p[0]["equation"] == A and p[0]["template"] == "city2"
    saved = json.loads((two_cities / "results/multicity/library/pool.json").read_text())
    assert [x["cid"] for x in saved] == ["c0001", "c0002"]


def test_to_single_averages_city_parameters():
    import library
    import refit_select
    c = refit_select.parse_equation(A, "city2", "A", 10)
    s = library.to_single(c)
    assert s.n_cat == 1 and s.template is None and s.n_lit == 2
    assert np.allclose(s.consts, [1.0, 2.0, 0.15, -2.25])


def test_refit_city_multistart_and_inf(two_cities):
    import library
    import refit_select
    importlib.reload(library)
    c = refit_select.parse_equation(A, "city2", "A", 10)
    r = library.refit_city(library.to_single(c), "TEST1")
    assert r["finite"] and np.isfinite(r["val_nll"]) and len(r["consts"]) == 4 and np.isfinite(r["train_nll"])
    # log(E_j): the zero-jobs tract is a validation-block destination -> refit on the training block succeeds, the
    # validation score is not finite (Review Focus 2: recorded as not fittable, never a crash)
    c0 = refit_select.parse_equation(C0, None, "A", 6)
    r0 = library.refit_city(c0, "TEST1")
    assert r0["finite"] is False and r0["val_nll"] is None and len(r0["consts"]) == 3
    # log(P_j - 1000): not finite on any training row at any start -> no constants at all
    c1 = refit_select.parse_equation("f = log(#4 - 1000.0); p1 = []; p2 = []", None, "A", 4)
    r1 = library.refit_city(c1, "TEST1")
    assert r1["finite"] is False and r1["val_nll"] is None and r1["consts"] is None and r1["train_nll"] is None


def test_search_city_fits_table(two_cities):
    import library
    importlib.reload(library)
    pa, pb = _write(two_cities)
    pc = two_cities / "c.json"
    pc.write_text(json.dumps({"variant": "A", "candidates": [{"equation": C0, "template": None, "complexity": 6,
                                                               "consts_refit": [2.0, 0.0, -2.0]}]}))
    pool = library.pool([pa, pb, pc])
    df, boot = library.search_city_fits(pool, ["TEST1", "TEST2"], procs=1)
    assert set(df.columns) >= {"cid", "city", "train_nll", "val_nll", "finite", "consts", "se_vs_best", "within_1se"}
    assert len(df) == 6 and df[df.cid == "c0003"]["finite"].eq(False).all() and df[df.cid == "c0003"]["val_nll"].isna().all()
    assert df.groupby("city")["within_1se"].any().all()                 # the best candidate of each city is in its 1-SE set
    assert (df[df.within_1se]["finite"]).all()
    assert boot["TEST1"].shape == (200, 3) and np.isfinite(boot["TEST1"][:, :2]).any()
    assert (boot["TEST1"][:, 2] == library.PENALTY).all()              # not fittable: penalty in every replicate
    lib = two_cities / "results/multicity/library"
    assert (lib / "search_city_fits.parquet").exists() and (lib / "boot_TEST1.npy").exists() and (lib / "boot_TEST2.npy").exists()


def test_boot_nll_matches_the_replicate_loop(two_cities):
    """Speed (ETA review 11-03): the bootstrap only reweights the counts, the log-probabilities stay; the vectorised
    boot_nll must equal the loop over replicates exactly (NaN where a replicate carries no visit)."""
    import sr_data
    import refit_select as rs
    import library
    importlib.reload(library)
    va = sr_data.build(["TEST1"], "tract", "val", "val", None, 0)
    Yg = rs._device_group_y(va, ["TEST1"], "tract", "val")
    W = library._boot_weights(50)
    c = library.to_single(rs.parse_equation(A, "city2", "A", 10))
    e = rs.eta(c, va)
    y0, loop = va.y.copy(), []
    for b in range(len(W)):
        va.y = W[b] @ Yg
        with np.errstate(all="ignore"):
            loop.append(rs._grouped(e, va)[0])
    va.y = y0
    fast = rs.boot_nll(e, va, Yg, W)
    assert fast.shape == (50,) and np.allclose(fast, loop, equal_nan=True, rtol=1e-12)
    assert (rs.boot_nll(np.full(len(e), np.inf), va, Yg, W) == rs.PENALTY).all()


def test_training_refits_use_sampled_origins(two_cities, monkeypatch):
    """ETA review 11-03: refits run on n_orig sampled origins per city (default 300, like the selection); the
    validation scoring uses every origin."""
    import sr_data
    import library
    importlib.reload(library)
    calls, real = [], sr_data.build

    def spy(regions, scale, split, block, n_orig, seed, *a, **k):
        calls.append((split, block, n_orig))
        return real(regions, scale, split, block, n_orig, seed, *a, **k)
    monkeypatch.setattr(sr_data, "build", spy)
    library.city_data("TEST1")
    assert ("train", "search", library.LIB_N_ORIG) in calls and ("val", "val", None) in calls and library.LIB_N_ORIG == 300
    calls.clear()
    library.city_data("TEST1", n_orig=2)
    assert ("train", "search", 2) in calls


def test_parallel_over_city_and_candidate_is_deterministic(two_cities):
    import library
    importlib.reload(library)
    pa, pb = _write(two_cities)
    pool = library.pool([pa, pb])
    d1, b1 = library.search_city_fits(pool, ["TEST1", "TEST2"], procs=1, B=40)
    d2, b2 = library.search_city_fits(pool, ["TEST1", "TEST2"], procs=3, B=40)
    cols = ["cid", "city", "train_nll", "val_nll", "finite", "within_1se"]
    pd.testing.assert_frame_equal(d1[cols].reset_index(drop=True), d2[cols].reset_index(drop=True))
    for c in ("TEST1", "TEST2"):
        assert np.allclose(b1[c], b2[c], equal_nan=True)


def test_city_data_inner_uses_folds_23_and_4(two_cities, monkeypatch):
    """Plan 4 Task 2: inner mode = training period only; refits on folds 2, 3 (sampled origins), scoring on fold 4
    (every origin), device-group counts of the training period."""
    import sr_data
    import library
    importlib.reload(library)
    calls, real = [], sr_data.build

    def spy(regions, scale, split, block, n_orig, seed, *a, **k):
        calls.append((split, block, n_orig))
        return real(regions, scale, split, block, n_orig, seed, *a, **k)
    monkeypatch.setattr(sr_data, "build", spy)
    D = library.city_data("TEST1", n_orig=10, mode="inner")
    assert calls == [("train", "search23", 10), ("train", "fold:4", None)]
    d = sr_data.load("TEST1", "tract")
    f4 = set(np.flatnonzero(sr_data.dest_mask(d, sr_data.folds(d), "fold:4")))
    assert set(D["va"].dest_unit) <= f4 and set(D["tr"].dest_unit).isdisjoint(f4)
    T = d["T"]["train"]
    assert np.allclose(D["va"].y, T[D["va"].origin_unit, D["va"].dest_unit])       # training-period counts
    assert D["Yg"].shape == (20, len(D["va"].y))
    assert np.allclose(D["Yg"].sum(0), D["va"].y)
    calls.clear()
    library.city_data("TEST1", n_orig=10)                                            # default mode unchanged
    assert calls == [("train", "search", 10), ("val", "val", None)]


def test_select_eval_blocks(two_cities, monkeypatch, tmp_path):
    """Plan 4 Task 2: refit_select.select(train_block=search23, eval_split=train, eval_block=fold:4) refits on folds
    2, 3 and scores on fold 4 of the training period (the inner-loop ranking); defaults unchanged."""
    import refit_select
    import sr_data
    importlib.reload(refit_select)
    rd = tmp_path / "runs/tract_A_single-TEST1_s0"
    rd.mkdir(parents=True)
    (two_cities / "results/sr/M").mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps({"template": "single", "active_vars": None}))
    pd.DataFrame({"complexity": [7], "loss": [5.0], "equation": ["f = log(#6 + 1.0) - 1.5 * log(#1 + 1.0)"]}
                 ).to_csv(rd / "hall_of_fame.csv", index=False)
    calls, real = [], refit_select.build

    def spy(regions, scale, split, block, n_orig, seed, *a, **k):
        calls.append((split, block))
        return real(regions, scale, split, block, n_orig, seed, *a, **k)
    monkeypatch.setattr(refit_select, "build", spy)
    ys = []
    monkeypatch.setattr(refit_select, "_device_group_y", lambda data, regions, scale, split: ys.append(split)
                        or np.ones((20, len(data.y))))
    out = refit_select.select([rd], "tract", "A", regions=("TEST1",), procs=1, out_name="sel_inner",
                              train_block="search23", eval_split="train", eval_block="fold:4")
    assert calls == [("train", "search23"), ("train", "fold:4")] and ys == ["train"]
    assert out["eval"] == {"train_block": "search23", "split": "train", "block": "fold:4"}
    calls.clear(), ys.clear()
    refit_select.select([rd], "tract", "A", regions=("TEST1",), procs=1, out_name="sel_val")
    assert calls == [("train", "search"), ("val", "val")] and ys == ["val"]


def test_multistart_records_dispersion(two_cities):
    """Plan 4 Task 7 (design §5.4.4): every refit records how many starts reached the best loss, the SD of the start
    losses and the per-constant SD over the starts at the best loss (identifiability evidence); default 5 starts."""
    import refit_select
    import library
    importlib.reload(library)
    c = library.to_single(refit_select.parse_equation(A, "city2", "A", 10))
    r = library.refit_city(c, "TEST1")
    assert 1 <= r["n_converged"] <= 5 and r["n_starts"] == 5 and r["loss_sd_starts"] >= 0
    assert len(r["consts_sd"]) == len(r["consts"]) and all(v >= 0 for v in r["consts_sd"])
    # two constants on the same term: only their sum is identified -> the starts land on different splits
    cc = refit_select.parse_equation("f = 1.0 * log(#6 + 1.0) + 0.5 * log(#6 + 1.0) - 1.5 * log(#1 + 1.0)", None, "A", 9)
    cc.comps["f"] = refit_select.sympy.sympify("c0*log(x6 + 1) + c2*log(x6 + 1) - c4*log(x1 + c5)",
                                               locals={"x6": refit_select.XS[5], "x1": refit_select.XS[0]})
    cc = dataclasses.replace(cc, consts=np.array([1.0, 1.0, 0.5, 1.0, 1.5, 1.0]))
    rc = library.refit_city(cc, "TEST1")
    assert rc["n_converged"] >= 2 and max(rc["consts_sd"][0], rc["consts_sd"][2]) > 1e-3 and rc["loss_sd_starts"] < 1e-4
    assert library.FIT_COLUMNS[-3:] == ["n_converged", "loss_sd_starts", "consts_sd"]


def test_fit_cities_inner_mode(two_cities, monkeypatch):
    """Plan 4 Task 14: the loop's inner fits use library.fit_cities(mode="inner") (folds 2, 3 -> fold 4)."""
    import library
    importlib.reload(library)
    seen = []
    real = library.city_data
    monkeypatch.setattr(library, "city_data", lambda c, n_orig=None, mode="val": seen.append(mode) or real(c, n_orig, mode))
    pa, pb = _write(two_cities)
    out = library.fit_cities(library.pool([pa, pb])[:1], ["TEST1"], procs=1, B=10, mode="inner")
    assert seen == ["inner"] and set(out) == {"TEST1"}
