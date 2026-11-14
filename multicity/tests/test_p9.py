"""Plan 9, Tasks 4 and 6: the config is fixed first, source stages never see a target city, target stages wait for
the opponents, failures are recorded as failures, and the stages run end to end on a synthetic world."""
import json

import numpy as np
import pandas as pd
import pytest

import ledger
import p9
import param_transfer as pt
import prior_models as pm
from p9_fixture import toy_inputs
from refit_select import parse_equation

SOURCE = [f"s{k:02d}" for k in range(12)]
LAW, CONF = ["l0", "l1"], ["f0", "f1"]
CITIES = SOURCE + LAW + CONF
EQ = ("grav_power_N", "S0_N", "strexp3_N", "cab_pow_P")
MAT = ("rad_P", "io_N", "kwon")
OURS = {"oursA": "f = 1.0*log(#6 + 1.0) - 1.5*#2", "oursB": "f = 1.0*log(#6 + 2.0) - 1.0*#1^0.5"}
LOGPOP = {c: 12.0 + 0.3 * k for k, c in enumerate(CITIES)}


@pytest.fixture(scope="module")
def data():
    """Per city and block: toy inputs (distance decay grows with city size) and the 'frozen' skeleton fits."""
    inp, fits = {}, {"val": {}, "test_dest": {}}
    for k, c in enumerate(CITIES):
        beta = 1.0 + 0.25 * (LOGPOP[c] - 12.0)
        inp[(c, "val")] = toy_inputs(100 + k, beta=beta)
        inp[(c, "test_dest")] = toy_inputs(100 + k, beta=beta, eval_split="test", eval_block="fold:0")
        for cid, eq in OURS.items():
            _, consts = pm.fit_eq(eq, [True] * len(parse_equation(eq, None, "A", 9).consts), inp[(c, "val")]["tr"])
            for b in fits:
                nll = pm.score_eq(eq, consts, inp[(c, b)]["ev"])["nll"]
                if np.isfinite(nll):                 # like the real files' `finite` filter: no row, a failure
                    fits[b].setdefault(cid, {})[c] = {"consts": consts, "nll": nll}
    return inp, fits


@pytest.fixture
def world(data, tmp_path, monkeypatch):
    inp, fits = data
    cfg = json.loads(p9.CONFIG.read_text())
    cfg.update(prior=list(EQ) + list(MAT), bootstrap=200, starts=2, ours={"primary": "oursA", "reported": list(OURS)})
    cfg["regression"] = dict(cfg["regression"], median_share=[])
    cfg_path = tmp_path / "p9.json"
    cfg_path.write_text(json.dumps(cfg))
    rng = np.random.default_rng(0)
    cov = pd.DataFrame({"log_population": LOGPOP}).assign(
        **{f: rng.normal(size=len(CITIES)) for f in cfg["form_covariates"]})
    monkeypatch.setattr(p9, "CONFIG", cfg_path)
    monkeypatch.setattr(p9, "OUT", tmp_path / "p9")
    monkeypatch.setattr(pm, "EQ_MODELS", {k: pm.EQ_MODELS[k] for k in EQ})
    monkeypatch.setattr(pm, "MAT_MODELS", {k: pm.MAT_MODELS[k] for k in MAT})
    monkeypatch.setattr(p9, "group_cities", lambda g: {"discovery": SOURCE, "lawtest": LAW, "confirm": CONF}[g])
    monkeypatch.setattr(p9, "city_inputs", lambda city, block: inp[(city, block)])
    monkeypatch.setattr(p9, "ours_fits", lambda block: fits[block])
    monkeypatch.setattr(p9, "ours_candidates", lambda: {k: parse_equation(e, None, "A", 9) for k, e in OURS.items()})
    monkeypatch.setattr(p9, "refs", lambda city, block: {
        "grav": fits[block]["oursA"][city]["nll"] + 0.3, "m2": fits[block]["oursA"][city]["nll"] - 0.01, "m2_cpc": 0.6})
    monkeypatch.setattr(p9, "covariates", lambda: cov)
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "LEDGER.md")
    monkeypatch.setenv("LOOP_OUTER_PROCS", "1")
    monkeypatch.delenv("P9_CITIES", raising=False)
    return {"cfg_path": cfg_path, "out": tmp_path / "p9", "fits": fits, "inp": inp, "cov": cov, "tmp": tmp_path}


def _run(*stages):
    for s in stages:
        p9.main([s])


# ------------------------------------------------------------------------------------------------ Task 4
def test_fix_then_require(world):
    with pytest.raises(RuntimeError, match="not fixed"):
        p9.require_fixed()
    sha = p9.fix_config()
    assert p9.fix_config() == sha and p9.require_fixed()["bootstrap"] == 200
    assert len(json.loads((world["out"] / "CONFIG_FIXED").read_text())) == 1


def test_changed_config_needs_reason(world):
    p9.fix_config()
    cfg = json.loads(world["cfg_path"].read_text())
    world["cfg_path"].write_text(json.dumps(dict(cfg, bootstrap=300)))
    with pytest.raises(RuntimeError, match="not fixed"):
        p9.require_fixed()
    with pytest.raises(ValueError, match="reason"):
        p9.fix_config()
    p9.fix_config("more bootstrap draws")
    ent = json.loads((world["out"] / "CONFIG_FIXED").read_text())
    assert len(ent) == 2 and ent[1]["reason"] == "more bootstrap draws" and p9.require_fixed()["bootstrap"] == 300


def test_prior_list_must_match_registry(world):
    cfg = json.loads(world["cfg_path"].read_text())
    world["cfg_path"].write_text(json.dumps(dict(cfg, prior=cfg["prior"][:-1])))
    with pytest.raises(ValueError, match="registry"):
        p9.fix_config()


def test_guard_val_refuses_target_city(world):
    p9.fix_config()
    p9.guard("val", SOURCE[:3])
    with pytest.raises(RuntimeError, match="source"):
        p9.guard("val", SOURCE[:2] + LAW[:1])


def test_guard_test_dest_refuses_before_opponents(world):
    p9.fix_config()
    with pytest.raises(RuntimeError, match="opponents"):
        p9.guard("test_dest", LAW)
    (world["out"] / "opponents.json").write_text("{}")              # a file that is not bound to this run
    with pytest.raises(RuntimeError, match="opponents"):
        p9.guard("test_dest", LAW)


def test_guard_test_dest_refuses_source_city(world):
    _run("fix", "source", "prerun", "select")
    p9.guard("test_dest", LAW + CONF)
    with pytest.raises(RuntimeError, match="target"):
        p9.guard("test_dest", LAW + SOURCE[:1])


def test_workers_are_guarded(world):
    p9.fix_config()
    p9.prepare("val")
    with pytest.raises(RuntimeError, match="source"):
        p9.fit_city(("l0", "val"))
    with pytest.raises(RuntimeError, match="opponents"):
        p9.fit_city(("l0", "test_dest"))
    with pytest.raises(RuntimeError, match="opponents"):
        p9.transfer_city(("l0", "test_dest"))


def test_config_levels_and_blocks_must_match_code(world):
    cfg = json.loads(world["cfg_path"].read_text())
    world["cfg_path"].write_text(json.dumps(dict(cfg, levels=["local", "uniform"])))
    with pytest.raises(ValueError, match="levels"):
        p9.fix_config()
    world["cfg_path"].write_text(json.dumps(dict(cfg, blocks={"source": "val", "target": "val"})))
    with pytest.raises(ValueError, match="blocks"):
        p9.fix_config()


def test_matrix_arrays_must_agree_with_rows(world):
    a, b = world["inp"][("s00", "val")], world["inp"][("s01", "val")]
    p9.check_paths("s00", a["ev"], a["A"])
    with pytest.raises(RuntimeError, match="disagree"):
        p9.check_paths("s00", a["ev"], b["A"])


def test_scored_requires_positive_reference_gap(world, monkeypatch):
    rows = pd.DataFrame([{"city": "s00", "model": "m", "kind": "eq", "level": "local", "consts": "[]", "nll": 3.0, "cpc": 0.5}])
    monkeypatch.setattr(p9, "refs", lambda city, block: {"grav": 3.0, "m2": 3.1, "m2_cpc": 0.6})
    with pytest.raises(RuntimeError, match="gravity"):
        p9.scored(rows, "val")
    monkeypatch.setattr(p9, "refs", lambda city, block: {"grav": float("nan"), "m2": 3.1, "m2_cpc": 0.6})
    with pytest.raises(RuntimeError, match="gravity"):
        p9.scored(rows, "val")


def test_guard_needs_fixed_config(world):
    with pytest.raises(RuntimeError, match="not fixed"):
        p9.guard("val", SOURCE)


def test_fit_city_rows(world):
    p9.fix_config()
    p9.prepare("val")
    rows = pd.DataFrame(p9.fit_city(("s03", "val"))).set_index("model")
    assert list(rows.index) == list(EQ) + list(MAT) + list(OURS)
    assert rows.loc["grav_power_N", "nll"] < rows.loc["rad_P", "nll"]
    assert rows.loc["kwon", "n_free"] == 110 and rows.loc["rad_P", "n_free"] == 0 and rows.loc["S0_N", "n_free"] == 3
    assert set(rows["kind"]) == {"eq", "mat", "ours"} and np.isfinite(rows["cpc"]).all()
    assert json.loads(rows.loc["oursA", "consts"]) == world["fits"]["val"]["oursA"]["s03"]["consts"]


def test_ours_nll_mismatch_raises(world):
    p9.fix_config()
    world["fits"]["val"]["oursA"]["s04"] = dict(world["fits"]["val"]["oursA"]["s04"],
                                               nll=world["fits"]["val"]["oursA"]["s04"]["nll"] + 0.01)
    try:
        p9.prepare("val")
        with pytest.raises(RuntimeError, match="oursA"):
            p9.fit_city(("s04", "val"))
    finally:
        world["fits"]["val"]["oursA"]["s04"]["nll"] -= 0.01


def _K(world, model="grav_power_N", cities=SOURCE):
    out = {}
    for c in cities:
        s = pm.EQ_MODELS[model]
        out[c] = pm.fit_eq(s.equation, s.free, world["inp"][(c, "val")]["tr"], starts=1)[1]
    return out


def test_transfer_city_levels_and_loo(world, monkeypatch):
    p9.fix_config()
    K = {("grav_power_N", "eq"): _K(world)}
    seen = []
    real = pt.predict
    monkeypatch.setattr(pt, "predict", lambda K_, cov, city, *a, **k: (seen.append((city, sorted(K_))), real(K_, cov, city, *a, **k))[1])
    p9.prepare("val", K=K)
    rows = pd.DataFrame(p9.transfer_city(("s05", "val")))
    assert list(rows["level"]) == ["uniform", "size", "form"] and np.isfinite(rows["nll"]).all()
    d = dict(zip(rows["level"], rows["consts"].map(json.loads)))
    others = np.array([K[("grav_power_N", "eq")][c] for c in SOURCE if c != "s05"])
    assert np.allclose(d["uniform"], np.median(others, axis=0))            # s05's own fit is not used
    assert seen[0][0] == "s05"


def test_out_of_domain_constants_are_failures(world):
    p9.fix_config()
    K = {("S0_N", "eq"): {c: [-100.0 - k, 1.5, 1.0] for k, c in enumerate(SOURCE)}}      # log(N_j - 100): undefined
    p9.prepare("val", K=K)
    t = p9.scored(pd.DataFrame(p9.transfer_city(("s05", "val"))), "val")
    assert len(t) == 3 and t["failed"].all() and (t["share"] == -np.inf).all() and t["nll"].isna().all()


def test_too_few_donors_marks_all_levels_failed(world):
    p9.fix_config()
    K = {("grav_power_N", "eq"): _K(world, cities=SOURCE[:9])}
    p9.prepare("val", K=K)
    t = p9.scored(pd.DataFrame(p9.transfer_city(("s11", "val"))), "val")
    assert list(t["level"]) == ["uniform", "size", "form"] and t["failed"].all()


# ------------------------------------------------------------------------------------------------ Task 6
def test_stages_end_to_end(world):
    _run("fix", "source", "prerun", "select", "target", "contrasts", "report")
    out = world["out"]
    models = list(EQ) + list(MAT) + list(OURS)
    pre, tg = pd.read_parquet(out / "prerun.parquet"), pd.read_parquet(out / "target.parquet")
    assert set(pre["city"]) == set(SOURCE) and set(tg["city"]) == set(LAW + CONF)
    for t, n in ((pre, 12), (tg, 4)):
        assert len(t) == n * len(models) * 4 and not t.duplicated(["city", "model", "level"]).any()
        assert set(t["level"]) == set(pt.LEVELS)
    opp = json.loads((out / "opponents.json").read_text())
    assert set(opp["opponents"]) == {"fit", "zero_shot", "fit_original", "zero_shot_original"}
    assert opp["opponents"]["zero_shot"]["model"] != "kwon" and opp["opponents"]["fit_original"]["model"] in ("cab_pow_P", "rad_P", "kwon")
    c = json.loads((out / "contrasts.json").read_text())
    assert set(c["contrasts"]) == {"C1", "C2", "C3", "C4", "D1", "D2", "D4"}
    assert all(c["contrasts"][k]["verdict"] in ("better", "worse", "no distinguishable difference") for k in ("C1", "C2", "C3", "C4"))
    assert c["contrasts"]["D1"]["verdict"] is None and c["contrasts"]["C1"]["n"] == 4
    assert c["contrasts"]["C3"]["a"] == ["oursA", "size"] and c["contrasts"]["C3"]["b"] == ["oursA", "uniform"]
    md = (out / "report_p9.md").read_text()
    for s in ("C1", "C2", "C3", "C4", "Failed cities", "The skeletons were selected on these cities", "CPC", opp["opponents"]["fit"]["model"]):
        assert s in md, s
    led = (world["tmp"] / "LEDGER.md").read_text()
    assert "opponents" in led and "contrasts" in led


def test_target_refuses_before_select(world):
    _run("fix", "source", "prerun")
    with pytest.raises(RuntimeError, match="opponents"):
        p9.main(["target"])
    assert not (world["out"] / "fits_target.parquet").exists()


def test_select_and_contrasts_write_once(world):
    _run("fix", "source", "prerun", "select")
    with pytest.raises(FileExistsError):
        p9.main(["select"])
    _run("target", "contrasts")
    before = (world["out"] / "contrasts.json").read_text()
    with pytest.raises(FileExistsError):
        p9.main(["contrasts"])
    assert (world["out"] / "contrasts.json").read_text() == before


def test_stage_refuses_changed_config(world):
    _run("fix")
    cfg = json.loads(world["cfg_path"].read_text())
    world["cfg_path"].write_text(json.dumps(dict(cfg, bootstrap=999)))
    for s in ("source", "prerun", "select", "target", "contrasts", "report"):
        with pytest.raises(RuntimeError, match="not fixed"):
            p9.main([s])


def test_partial_city_run_cannot_select(world, monkeypatch):
    monkeypatch.setenv("P9_CITIES", ",".join(SOURCE[:11]))
    _run("fix", "source", "prerun")
    assert set(pd.read_parquet(world["out"] / "prerun.parquet")["city"]) == set(SOURCE[:11])
    with pytest.raises(RuntimeError, match="P9_CITIES"):
        p9.main(["select"])
    monkeypatch.delenv("P9_CITIES")
    with pytest.raises(RuntimeError, match="source cities"):          # the files on disk are still the partial run
        p9.main(["select"])


def test_zero_parameter_models_identical_across_levels(world):
    _run("fix", "source", "prerun")
    t = pd.read_parquet(world["out"] / "prerun.parquet")
    r = t[t["model"] == "rad_P"].pivot(index="city", columns="level", values="nll")
    assert len(r) == 12 and all(np.array_equal(r["local"], r[lv]) for lv in ("uniform", "size", "form"))


def test_target_donors_are_source_cities_only(world, monkeypatch):
    _run("fix", "source", "prerun", "select")
    seen = set()
    real = pt.predict
    monkeypatch.setattr(pt, "predict", lambda K, *a, **k: (seen.update(K), real(K, *a, **k))[1])
    _run("target")
    assert seen == set(SOURCE)


def test_unknown_stage_exits(world):
    with pytest.raises(SystemExit):
        p9.main(["nonsense"])


# ------------------------------------------------------------------------------------------------ review fixes (I1-I3)
def _touch_parquet(p):
    t = pd.read_parquet(p)
    t.loc[t.index[0], "cpc"] = float(t["cpc"].iloc[0]) + 1e-3
    t.to_parquet(p, index=False)


def test_select_requires_the_prerun_of_these_fits(world):
    _run("fix", "source", "prerun")
    _touch_parquet(world["out"] / "fits_source.parquet")                  # the source fits were redone after the pre-run
    with pytest.raises(RuntimeError, match="fits_source"):
        p9.main(["select"])
    assert not (world["out"] / "opponents.json").exists()


def test_target_refuses_stale_opponents(world):
    _run("fix", "source", "prerun", "select")
    _touch_parquet(world["out"] / "fits_source.parquet")
    with pytest.raises(RuntimeError, match="fits_source"):
        p9.main(["target"])
    assert not (world["out"] / "fits_target.parquet").exists()


def test_prerun_redone_after_select_blocks_target(world):
    _run("fix", "source", "prerun", "select")
    _touch_parquet(world["out"] / "prerun.parquet")
    with pytest.raises(RuntimeError, match="prerun"):
        p9.main(["target"])


def test_config_amendment_after_select_blocks_target_and_contrasts(world):
    _run("fix", "source", "prerun", "select")
    cfg = json.loads(world["cfg_path"].read_text())
    world["cfg_path"].write_text(json.dumps(dict(cfg, ours={"primary": "oursB", "reported": list(OURS)})))
    p9.fix_config("switch the primary skeleton")
    for s in ("target", "contrasts"):
        with pytest.raises(RuntimeError, match="config"):
            p9.main([s])


def test_code_change_after_select_blocks_target(world, monkeypatch):
    _run("fix", "source", "prerun", "select")
    monkeypatch.setattr(p9, "code_sha", lambda: {"p9.py": "changed"})
    with pytest.raises(RuntimeError, match="code"):
        p9.main(["target"])


def test_target_runs_once_whole_and_is_ledgered(world, monkeypatch):
    _run("fix", "source", "prerun", "select")
    monkeypatch.setenv("P9_CITIES", "l0,f0")
    with pytest.raises(RuntimeError, match="P9_CITIES"):
        p9.main(["target"])
    monkeypatch.delenv("P9_CITIES")
    _run("target")
    assert "target" in (world["tmp"] / "LEDGER.md").read_text().splitlines()[-1]
    with pytest.raises(FileExistsError):
        p9.main(["target"])
    with pytest.raises(FileExistsError):                                  # select cannot be redone either
        (world["out"] / "opponents.json").unlink()
        p9.main(["select"])


def test_report_refuses_a_target_file_that_is_not_the_contrasts_one(world):
    _run("fix", "source", "prerun", "select", "target", "contrasts")
    _touch_parquet(world["out"] / "target.parquet")
    with pytest.raises(RuntimeError, match="target"):
        p9.main(["report"])


def test_target_preflight_needs_every_source_donor_of_the_primary(world):
    _run("fix", "source", "prerun", "select")
    f = world["fits"]["test_dest"]["oursA"]
    keep = f.pop("s02")
    try:
        with pytest.raises(RuntimeError, match="oursA"):
            p9.main(["target"])
        assert not (world["out"] / "fits_target.parquet").exists()
    finally:
        f["s02"] = keep


def test_target_preflight_needs_matching_donor_constants(world):
    _run("fix", "source", "prerun", "select")
    f = world["fits"]["test_dest"]["oursA"]
    keep = f["s02"]
    f["s02"] = dict(keep, consts=[keep["consts"][0] + 1e-3] + keep["consts"][1:])
    try:
        with pytest.raises(RuntimeError, match="constants"):
            p9.main(["target"])
    finally:
        f["s02"] = keep


def test_contrasts_refuse_a_primary_without_its_local_fit(world):
    _run("fix", "source", "prerun", "select", "target")
    p = world["out"] / "target.parquet"
    t = pd.read_parquet(p)
    k = (t["model"] == "oursA") & (t["level"] == "local") & (t["city"] == "l0")
    t.loc[k, ["nll", "failed", "share"]] = [np.nan, True, -np.inf]
    t.to_parquet(p, index=False)
    with pytest.raises(RuntimeError, match="local"):
        p9.main(["contrasts"])
    assert not (world["out"] / "contrasts.json").exists()


def test_select_regression_gate(world):
    cfg = json.loads(world["cfg_path"].read_text())
    cfg["regression"] = dict(cfg["regression"], median_share=[["oursA", "local", 5.0]])
    world["cfg_path"].write_text(json.dumps(cfg))
    _run("fix", "source", "prerun")
    with pytest.raises(RuntimeError, match="regression"):
        p9.main(["select"])
    assert not (world["out"] / "opponents.json").exists()


def test_report_labels(world):
    cfg = json.loads(world["cfg_path"].read_text())
    assert p9._kind("oursA", cfg) == "ours (primary)" and p9._kind("oursB", cfg) == "ours"
    assert p9._kind("kwon", cfg) == "earlier, as published (needs the city's own trips)"
    assert p9._kind("rad_P", cfg) == "earlier, as published"
    assert p9._kind("io_N", cfg) == "earlier, POI count" and p9._kind("grav_power_N", cfg) == "earlier, POI count"
    assert p9._kind("S0_N", cfg) == "earlier, relaxed or reference"
    assert p9._kind("strexp3_N", cfg) == "earlier, relaxed or reference"
