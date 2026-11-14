"""Plan 7 Task 4: the city-level outcome table (what the city-form hypotheses explain): skeleton class membership,
performance, identifiable parameters, the size-law residual and the direct-transfer loss."""
import json

import numpy as np
import pandas as pd
import pytest

import city_outcomes as co

REFS = {c: {"grav_power": {"nll_per_visit": 4.2}, "grav_exp": {"nll_per_visit": 4.25}, "ext_rad": {"nll_per_visit": 4.3},
            "M2_glm": {"nll_per_visit": 4.0}, "S0": {"nll_per_visit": 4.1}} for c in "ABCD"}


def _cv(rows):
    return pd.DataFrame([{"cid": cid, "city": city, "val_nll": v, "finite": True, "se_vs_best": 0.0, "within_1se": False,
                          "complexity": 10, "consts": "[1.0, 2.0, 3.0]"} for city, cid, v in rows])


def test_families():
    assert co.KERNEL == ("c0353", "c0353c1", "c0353c1r1", "c0354", "c0355", "c0355r1")
    assert co.OPP == ("c0413", "c0413c1", "c0413c1r1", "c0413c1r2", "c0413c1r3", "c0413c1r4")


def test_membership_classes():
    # delta at 0.03 = 0.006: near-best = within 0.006 of the city's best finite row; floor = below 4.2
    cv = _cv([("A", "c0355r1", 4.000), ("A", "c0413", 4.050),
              ("B", "c0355r1", 4.050), ("B", "c0413c1r2", 4.000),
              ("C", "c0353", 4.000), ("C", "c0413c1", 4.003),
              ("D", "c0355r1", 4.300), ("D", "c0413", 4.301)])           # D: all above the floor models
    m = co.membership(cv, REFS, 0.03).set_index("city")
    assert m["cls"].to_dict() == {"A": "kernel", "B": "opp", "C": "both", "D": "neither"}
    assert m.loc["C", "kernel_near"] and m.loc["C", "opp_near"] and not m.loc["D", "kernel_near"]


def test_performance_arithmetic():
    cv = _cv([("A", "c0355r1", 4.05), ("A", "c0262", 4.02), ("A", "c0413c1r1", 4.00), ("A", "c0514c1", 4.10)])
    p = co.performance(cv, REFS, clean=["c0355r1", "c0262", "c0514c1"]).set_index("city")
    assert p.loc["A", "gain_grav_m2"] == pytest.approx(0.2)
    assert p.loc["A", "share_grav_best_clean"] == pytest.approx((4.2 - 4.02) / 0.2)      # c0413c1r1 is not clean
    assert p.loc["A", "share_s0_main"] == pytest.approx((4.1 - 4.05) / 0.1)


def test_size_residual_uses_fit_cities_only():
    df = pd.DataFrame({"city": ["a", "b", "c", "x"], "log_population": [14.0, 15.0, 16.0, 15.0]})
    df["q"] = 1 - 0.1 * df["log_population"]
    df.loc[df["city"] == "x", "q"] += 0.05
    r = co.size_residual(df, "q", ["a", "b", "c"])
    assert np.allclose(r[df["city"] != "x"], 0, atol=1e-12) and r[df["city"] == "x"].iloc[0] == pytest.approx(0.05, abs=1e-12)


def test_loo_transfer_consts_excludes_the_city():
    cv = pd.DataFrame({"cid": "c0355r1", "city": ["a", "b", "c"], "finite": True, "consts": ["[1.0]", "[2.0]", "[10.0]"]})
    assert co.loo_transfer_consts(cv, "c0355r1", "c") == [1.5]
    assert co.loo_transfer_consts(cv, "c0355r1", "a") == [6.0]


def test_turning_distance_is_consts_1():
    cv = pd.DataFrame({"cid": ["c0514c1", "c0514c1"], "city": ["a", "b"], "finite": [True, False],
                       "consts": ["[0.19, 5.147, 19.6]", "[0.2, 3.0, 10.0]"]})
    t = co.turning_distance(cv)
    assert t == {"a": pytest.approx(5.147)}


def test_outcome_table_refuses_lawtest_before_signing(tmp_path, monkeypatch):
    import p7seal
    monkeypatch.setattr(p7seal, "REG", tmp_path / "r.json")
    monkeypatch.setattr(p7seal, "SIGNED", tmp_path / "S")
    monkeypatch.setattr(p7seal.config, "cities_with_role", lambda r: ["10900"] if r == "lawtest" else [])
    with pytest.raises(RuntimeError):
        co.outcome_table(["12060", "10900"], "test_dest")
    with pytest.raises(ValueError):
        co.outcome_table(["12060"], "test_time")                 # blocks: test_dest (primary) and val only


def test_outcome_defs_cover_every_column_name():
    for k in ("cls", "kernel_near", "opp_near", "gain_grav_m2", "share_grav_best_clean", "share_s0_main",
              "ed_5_10km", "slope_eN_logd", "ed_5_10km_c0355r1", "ed_5_10km_c0262", "ed_5_10km_c0514c1",
              "theta_turn_c0514c1", "ed_5_10km_resid", "transfer_loss_c0355r1", "transfer_frac_c0355r1"):
        assert k in co.OUTCOME_DEFS and co.OUTCOME_DEFS[k]
    assert json.dumps(co.OUTCOME_DEFS, ensure_ascii=False)


def test_transfer_loss_refuses_lawtest_before_signing(tmp_path, monkeypatch):
    import p7seal
    monkeypatch.setattr(p7seal, "REG", tmp_path / "r.json")
    monkeypatch.setattr(p7seal, "SIGNED", tmp_path / "S")
    monkeypatch.setattr(co, "_cv", lambda b: (_ for _ in ()).throw(AssertionError("data read before the guard")))
    with pytest.raises(RuntimeError):
        co.transfer_loss("10900")


# ------------------------------------------------------------------ plan 8 Task 1: law-test cities, tagged outputs
import config  # noqa: E402


def _capture_ledger(monkeypatch):
    import ledger
    logged = []
    monkeypatch.setattr(ledger, "add", logged.append)
    return logged


def test_cities_and_paths_by_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(co, "OUT", tmp_path)
    assert co.cities_for("lawtest") == config.cities_with_role("lawtest") and len(co.cities_for("lawtest")) == 26
    assert co.cities_for("discovery") == co.discovery()
    assert co.paths_for("discovery") == {"outcomes": tmp_path / "city_outcomes.parquet",
                                         "transfer": tmp_path / "transfer_loss.parquet"}
    assert [p.name for p in co.paths_for("lawtest").values()] == ["city_outcomes_lawtest.parquet",
                                                                  "transfer_loss_lawtest.parquet"]
    for bad in ("line", "rest"):
        with pytest.raises(ValueError):
            co.cities_for(bad)
        with pytest.raises(ValueError):
            co.paths_for(bad)


def test_size_residual_with_a_separate_fit_table():
    fit = pd.DataFrame({"city": ["a", "b", "c"], "log_population": [13.0, 14.0, 15.0], "ed_5_10km": [-1.0, -2.0, -3.0]})
    df = pd.DataFrame({"city": ["x"], "log_population": [14.0], "ed_5_10km": [-1.0]})
    assert co.size_residual(df, "ed_5_10km", ["a", "b", "c"], fit_df=fit).iloc[0] == pytest.approx(1.0)
    assert co.size_residual(df, "ed_5_10km", ["a", "b"], fit_df=fit).isna().all()          # < 3 fit points


def test_outcomes_stage_needs_the_transfer_file(tmp_path, monkeypatch):
    monkeypatch.setattr(co, "OUT", tmp_path)
    monkeypatch.setattr(co, "cities_for", lambda tag: ["12060"])
    monkeypatch.setattr(co, "outcome_table", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    with pytest.raises(FileNotFoundError):
        co.stage_outcomes("lawtest")


def test_lawtest_tag_never_overwrites_discovery_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(co, "OUT", tmp_path)
    (tmp_path / "city_outcomes.parquet").write_bytes(b"sentinel")
    pd.DataFrame({"city": ["L1"], "transfer_loss_c0355r1": [0.01]}).to_parquet(tmp_path / "transfer_loss_lawtest.parquet")
    _capture_ledger(monkeypatch)
    monkeypatch.setattr(co, "cities_for", lambda tag: ["L1"])
    seen = []

    def fake_table(cities, block, transfer_path=None):
        seen.append((block, transfer_path))
        return pd.DataFrame({"city": cities, "ed_5_10km": [-1.0], "log_population": [14.0], "cls": ["kernel"]})

    monkeypatch.setattr(co, "outcome_table", fake_table)
    monkeypatch.setattr(co, "_discovery_fit_table", lambda: pd.DataFrame(
        {"city": ["a", "b", "c"], "log_population": [13.0, 14.0, 15.0], "ed_5_10km": [-1.0, -2.0, -3.0]}))
    monkeypatch.setattr(co, "discovery", lambda: ["a", "b", "c"])
    co.stage_outcomes("lawtest")
    assert (tmp_path / "city_outcomes.parquet").read_bytes() == b"sentinel"
    out = pd.read_parquet(tmp_path / "city_outcomes_lawtest.parquet")
    assert seen == [("test_dest", tmp_path / "transfer_loss_lawtest.parquet")] and out["block"].tolist() == ["test_dest"]
    assert out["ed_5_10km_resid"].iloc[0] == pytest.approx(1.0)


def test_transfer_stage_writes_the_tag_file(tmp_path, monkeypatch):
    import frozen
    monkeypatch.setattr(co, "OUT", tmp_path)
    monkeypatch.setenv("LOOP_OUTER_PROCS", "1")
    monkeypatch.setattr(frozen, "require", lambda: [])
    monkeypatch.setattr(co, "cities_for", lambda tag: ["L1", "L2"])
    monkeypatch.setattr(co, "transfer_loss", lambda c: {"city": c, "transfer_loss_c0355r1": 0.01})
    _capture_ledger(monkeypatch)
    co.stage_transfer("lawtest")
    assert pd.read_parquet(tmp_path / "transfer_loss_lawtest.parquet")["city"].tolist() == ["L1", "L2"]
    assert not (tmp_path / "transfer_loss.parquet").exists()


# ------------------------------------------------------------------ plan 8 final review fixes
def test_stages_refuse_existing_outputs(tmp_path, monkeypatch):
    import frozen
    monkeypatch.setattr(co, "OUT", tmp_path)
    monkeypatch.setattr(frozen, "require", lambda: [])
    monkeypatch.setattr(co, "cities_for", lambda tag: ["L1"])
    monkeypatch.setattr(co, "transfer_loss", lambda c: (_ for _ in ()).throw(AssertionError("computed")))
    monkeypatch.setattr(co, "outcome_table", lambda *a, **k: (_ for _ in ()).throw(AssertionError("computed")))
    for tag in ("discovery", "lawtest"):
        for p in co.paths_for(tag).values():
            p.write_bytes(b"x")
        with pytest.raises(FileExistsError):
            co.stage_transfer(tag)
        with pytest.raises(FileExistsError):
            co.stage_outcomes(tag)


def test_cli_needs_a_tag():
    for argv in (["outcomes"], ["transfer"], ["outcomes", "line"], ["meta", "lawtest"]):
        with pytest.raises(SystemExit):
            co.main(argv)


def test_outcomes_ledger_reports_absent_keys_as_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(co, "OUT", tmp_path)
    pd.DataFrame({"city": ["L1"], "transfer_loss_c0355r1": [0.01]}).to_parquet(tmp_path / "transfer_loss_lawtest.parquet")
    logged = _capture_ledger(monkeypatch)
    monkeypatch.setattr(co, "cities_for", lambda tag: ["L1"])
    monkeypatch.setattr(co, "outcome_table", lambda cities, block, transfer_path=None: pd.DataFrame(
        {"city": cities, "ed_5_10km": [-1.0], "log_population": [14.0]}))
    monkeypatch.setattr(co, "_discovery_fit_table", lambda: pd.DataFrame(
        {"city": ["a", "b", "c"], "log_population": [13.0, 14.0, 15.0], "ed_5_10km": [-1.0, -2.0, -3.0]}))
    monkeypatch.setattr(co, "discovery", lambda: ["a", "b", "c"])
    co.stage_outcomes("lawtest")
    assert "'ed_5_10km_c0262': 0" in logged[-1] and "'ed_5_10km': 1" in logged[-1]


# ------------------------------------------------------------------ exploration Task 1: confirmation-city tag
def test_confirm_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(co, "OUT", tmp_path)
    assert co.cities_for("confirm") == config.cities_with_role("confirm") and len(co.cities_for("confirm")) == 6
    assert [p.name for p in co.paths_for("confirm").values()] == ["city_outcomes_confirm.parquet",
                                                                  "transfer_loss_confirm.parquet"]
    (tmp_path / "city_outcomes.parquet").write_bytes(b"sentinel")
    pd.DataFrame({"city": ["C1"], "transfer_loss_c0355r1": [0.01]}).to_parquet(tmp_path / "transfer_loss_confirm.parquet")
    _capture_ledger(monkeypatch)
    monkeypatch.setattr(co, "cities_for", lambda tag: ["C1"])
    seen = []

    def fake_table(cities, block, transfer_path=None):
        seen.append((block, transfer_path))
        return pd.DataFrame({"city": cities, "ed_5_10km": [-1.0], "log_population": [14.0]})

    monkeypatch.setattr(co, "outcome_table", fake_table)
    monkeypatch.setattr(co, "_discovery_fit_table", lambda: pd.DataFrame(
        {"city": ["a", "b", "c"], "log_population": [13.0, 14.0, 15.0], "ed_5_10km": [-1.0, -2.0, -3.0]}))
    monkeypatch.setattr(co, "discovery", lambda: ["a", "b", "c"])
    co.stage_outcomes("confirm")
    assert (tmp_path / "city_outcomes.parquet").read_bytes() == b"sentinel"
    out = pd.read_parquet(tmp_path / "city_outcomes_confirm.parquet")
    assert seen == [("test_dest", tmp_path / "transfer_loss_confirm.parquet")] and out["block"].tolist() == ["test_dest"]
    assert out["ed_5_10km_resid"].iloc[0] == pytest.approx(1.0)
