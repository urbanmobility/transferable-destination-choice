"""Plan 2 Task 7: per-city library fits (validation set) — primary skeleton, 1-SE set, gain share, trust flag,
coverage; main() over the non-confirmation cities."""
import dataclasses
import importlib
import json

import numpy as np
import pandas as pd

from test_library import A, B2, C0
from test_search_param import two_cities  # noqa: F401

REFS = {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 4.0}}


def _pool():
    return [{"cid": "c0001", "equation": A, "template": "city2", "complexity": 10, "consts_refit": [1.0, 2.0, 0.1, 0.2, -2.0, -2.5]},
            {"cid": "c0002", "equation": B2, "template": None, "complexity": 8, "consts_refit": [2.0, 1.0, 0.0, -1.0]},
            {"cid": "c0003", "equation": C0, "template": None, "complexity": 6, "consts_refit": [2.0, 0.0, -2.0]}]


def test_share():
    import fit_library
    import pytest
    assert fit_library.share(4.2, REFS) == pytest.approx(0.8) and fit_library.share(None, REFS) is None
    assert fit_library.share(3.0, {"grav_power": {"nll_per_visit": 4.0}, "M2_glm": {"nll_per_visit": 4.0}}) is None


def test_fit_city_primary_share_and_trust(two_cities):
    import fit_library
    importlib.reload(fit_library)
    rows, summ = fit_library.fit_city("TEST1", _pool(), refs=REFS, qc={"support_mismatch": ["E_j"]})
    assert len(rows) == 3 and sum(r["primary"] for r in rows) == 1
    prim = next(r for r in rows if r["primary"])
    assert prim["finite"] and prim["near_best"]                     # plan 4: the simplest near-best candidate
    assert prim["complexity"] == min(r["complexity"] for r in rows if r["near_best"])
    assert all(r["share_grav"] is None and r["share_s0"] is None for r in rows if not r["finite"])
    by = {r["cid"]: r for r in rows}
    assert by["c0003"]["finite"] is False and by["c0003"]["trust"] is False      # log(E_j): not finite when E_j = 0
    assert by["c0001"]["trust"] is True and by["c0002"]["trust"] is True            # log(E_j + 1), exp(-E_j), N_j: safe at 0
    assert summ["city"] == "TEST1" and summ["primary"] == prim["cid"] and summ["n_1se"] >= 1 and summ["n_near"] >= 1
    assert summ["share_grav"] == prim["share_grav"] and summ["covered"] == (prim["val_nll"] < 5.0)   # floor: grav
    assert summ["support_flags"] == ["E_j"] and summ["grav"] == 5.0 and summ["m2"] == 4.0 and summ["trust"] is True


def test_refit_from_a_wrong_start_recovers(two_cities):
    """A refit started from constants 3x off still reaches the training loss of the good start."""
    import library
    import refit_select
    importlib.reload(library)
    c = library.to_single(refit_select.parse_equation(A, "city2", "A", 10))
    good = library.refit_city(c, "TEST1")
    bad = library.refit_city(dataclasses.replace(c, consts=c.consts * 3.0), "TEST1")
    assert good["finite"] and bad["finite"] and abs(good["train_nll"] - bad["train_nll"]) < 1e-3


def test_main_skips_confirmation_cities(two_cities):
    import fit_library
    importlib.reload(fit_library)
    root = two_cities
    lib = root / "results/multicity/library"
    lib.mkdir(parents=True)
    (lib / "pool.json").write_text(json.dumps(_pool()))
    (lib / "candidates_gate1.json").write_text(json.dumps({"candidates": [
        {"cid": "c0001", "gate1": True}, {"cid": "c0002", "gate1": True}, {"cid": "c0003", "gate1": False}]}))
    refs_dir = root / "results/multicity/refs"
    refs_dir.mkdir(parents=True)
    for c in ("TEST1", "TEST2"):
        (refs_dir / f"{c}.json").write_text(json.dumps({"city": c, "models": REFS}))
    out = fit_library.main(["TEST1", "TEST2"], confirm={"TEST2"}, procs=1)
    assert list(out["city"]) == ["TEST1"] and out.iloc[0]["primary"] in ("c0001", "c0002")
    rows = pd.read_parquet(root / "results/multicity/cities_val.parquet")
    assert sorted(rows["cid"].unique()) == ["c0001", "c0002"] and set(rows["city"]) == {"TEST1"}
    assert (root / "results/multicity/cities_val_summary.csv").exists()
    unc = (root / "results/multicity/uncovered.txt").read_text().split()
    assert unc == ([] if out.iloc[0]["covered"] else ["TEST1"])
    assert np.isfinite(out.iloc[0]["share_grav"])


def test_main_parallel_over_pairs_matches_sequential(two_cities):
    """ETA review 11-03: per-city fits run in parallel over (city, candidate) pairs (New York alone needs ~30 min per
    candidate); the result must not depend on the number of processes."""
    import fit_library
    importlib.reload(fit_library)
    root = two_cities
    lib = root / "results/multicity/library"
    lib.mkdir(parents=True)
    (lib / "pool.json").write_text(json.dumps(_pool()))
    (lib / "candidates_gate1.json").write_text(json.dumps({"candidates": [{"cid": c, "gate1": True} for c in ("c0001", "c0002", "c0003")]}))
    (root / "results/multicity/refs").mkdir(parents=True)
    for c in ("TEST1", "TEST2"):
        (root / f"results/multicity/refs/{c}.json").write_text(json.dumps({"city": c, "models": REFS}))
    s1 = fit_library.main(["TEST1", "TEST2"], confirm=set(), procs=1)
    r1 = pd.read_parquet(root / "results/multicity/cities_val.parquet")
    s2 = fit_library.main(["TEST1", "TEST2"], confirm=set(), procs=3)
    r2 = pd.read_parquet(root / "results/multicity/cities_val.parquet")
    cols = ["city", "cid", "val_nll", "finite", "within_1se", "primary", "trust"]
    pd.testing.assert_frame_equal(r1[cols].sort_values(["city", "cid"]).reset_index(drop=True),
                                  r2[cols].sort_values(["city", "cid"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(s1[["city", "primary", "covered"]].sort_values("city").reset_index(drop=True),
                                  s2[["city", "primary", "covered"]].sort_values("city").reset_index(drop=True))


def test_primary_is_simplest_near_best():
    """Plan 4 Task 7: the near-best set = within max(1 SE, delta_city) of the city's best; the
    primary skeleton is its simplest member; covered = the primary passes the floor; shares on S0 and classic zero."""
    import fit_library
    importlib.reload(fit_library)
    refs = {"grav_power": {"nll_per_visit": 5.0}, "M2_glm": {"nll_per_visit": 4.0}, "S0": {"nll_per_visit": 4.1},
            "ext_rad": {"nll_per_visit": 4.15}}
    rows = [{"cid": c, "city": "T", "train_nll": v, "val_nll": v, "finite": True, "consts": "[1.0]", "trust": True}
            for c, v in (("c1", 4.10), ("c2", 4.12), ("c3", 4.20))]
    boot = np.tile(np.array([4.10, 4.12, 4.20]), (20, 1))
    out, summ = fit_library._finish("T", rows, boot, refs, [], {"c1": 10, "c2": 6, "c3": 3})
    by = {r["cid"]: r for r in out}
    assert by["c1"]["near_best"] and by["c2"]["near_best"] and not by["c3"]["near_best"]
    assert summ["primary"] == "c2" and by["c2"]["primary"] and summ["n_near"] == 2
    assert summ["covered"] is True and abs(summ["share_s0"] - (-0.2)) < 1e-9 and abs(summ["share_grav"] - 0.88) < 1e-9
    refs2 = {**refs, "ext_rad": {"nll_per_visit": 4.11}}                     # primary 4.12 is above the floor 4.11
    assert fit_library._finish("T", rows, boot, refs2, [], {"c1": 10, "c2": 6, "c3": 3})[1]["covered"] is False


def test_fit_library_zero_candidates(two_cities):
    """No GATE 1 candidate -> main returns without a process pool, an empty summary is written and
    every city is listed as uncovered."""
    import fit_library
    importlib.reload(fit_library)
    root = two_cities
    lib = root / "results/multicity/library"
    lib.mkdir(parents=True)
    (lib / "pool.json").write_text(json.dumps(_pool()))
    (lib / "candidates_gate1.json").write_text(json.dumps({"candidates": [{"cid": "c0001", "gate1": False}]}))
    import multiprocessing
    def boom(*a, **k):
        raise AssertionError("no pool for zero candidates")
    orig = multiprocessing.get_context
    multiprocessing.get_context = boom
    try:
        out = fit_library.main(["TEST1", "TEST2"], confirm=set(), procs=4)
    finally:
        multiprocessing.get_context = orig
    assert len(out) == 0
    assert len(pd.read_csv(root / "results/multicity/cities_val_summary.csv")) == 0
    assert (root / "results/multicity/uncovered.txt").read_text().split() == ["TEST1", "TEST2"]


def test_main_takes_candidates_and_output_dir(two_cities):
    """The agent loop passes its top candidates and an output directory (discovery and lawtest fits apart)."""
    import fit_library
    importlib.reload(fit_library)
    root = two_cities
    (root / "results/multicity/refs").mkdir(parents=True)
    for c in ("TEST1", "TEST2"):
        (root / f"results/multicity/refs/{c}.json").write_text(json.dumps({"city": c, "models": REFS}))
    out_dir = root / "results/multicity/loop/c1_discovery"
    s = fit_library.main(["TEST1"], confirm=set(), procs=1, cands=_pool()[:2], out_dir=out_dir, starts=2)
    rows = pd.read_parquet(out_dir / "cities_val.parquet")
    assert sorted(rows["cid"].unique()) == ["c0001", "c0002"] and (out_dir / "cities_val_summary.csv").exists()
    assert {"n_converged", "loss_sd_starts", "consts_sd", "near_best", "share_s0", "share_grav", "complexity",
            "consts_se", "consts_cov"} <= set(rows)
    ok = rows[rows.finite.astype(bool)]
    se = json.loads(ok.consts_se.iloc[0])
    assert len(se) == len(json.loads(ok.consts.iloc[0])) and len(json.loads(ok.consts_cov.iloc[0])) == len(se)
    assert list(s["city"]) == ["TEST1"]


def test_empty_outputs_keep_the_fit_columns(two_cities):
    """An empty candidate list writes cities_val.parquet with the fit columns, so param_table and the
    ablations degrade to empty instead of KeyError."""
    import fit_library
    import laws
    importlib.reload(fit_library)
    out_dir = two_cities / "results/multicity/loop/c1_discovery"
    fit_library.main(["TEST1"], confirm=set(), procs=2, cands=[], out_dir=out_dir)
    cv = pd.read_parquet(out_dir / "cities_val.parquet")
    assert {"cid", "city", "finite", "consts", "val_nll"} <= set(cv.columns) and len(cv) == 0
    assert len(laws.param_table(cv, [])) == 0
