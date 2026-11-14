"""Plan 6 Task 1: the freeze record and its guard. Nothing reads the confirmation cities or the test period before
results/multicity/FROZEN exists and matches frozen/library.json."""
import json

import pytest

import frozen

POOL = {c: {"cid": c, "equation": f"f = log(#6 + 1.0); p1 = []; p2 = []  # {c}", "template": None, "variant": "A",
            "complexity": k} for c, k in (("c0355r1", 14), ("c0262", 23), ("c0514c1", 15), ("c0413c1r1", 24),
                                          ("c0079c1", 13))}
LIB = {"entries": [{"cid": "c0355r1", "role": "main"}, {"cid": "c0262", "role": "second"},
                   {"cid": "c0514c1", "role": "optional"}, {"cid": "c0413c1r1", "role": "comparison"},
                   {"cid": "c0079c1", "role": "reference"}]}


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, "FROZEN", tmp_path / "FROZEN")
    monkeypatch.setattr(frozen, "FROZEN_DIR", tmp_path / "frozen")
    cfg, proto = tmp_path / "lib.json", tmp_path / "plan6.md"
    cfg.write_text(json.dumps(LIB))
    proto.write_text("# plan 6\n")
    return tmp_path, cfg, proto


def test_require_and_unseal_refuse_before_freeze(sealed):
    assert not frozen.is_frozen()
    with pytest.raises(RuntimeError):
        frozen.require()
    with pytest.raises(RuntimeError):
        frozen.unseal_env()


def test_freeze_writes_record_and_require_checks_sha(sealed):
    root, cfg, proto = sealed
    lines = []
    sha = frozen.freeze(cfg, proto, POOL, log=lines.append)
    assert frozen.is_frozen() and (root / "FROZEN").read_text().strip() == sha and len(lines) == 1 and sha[:12] in lines[0]
    ent = frozen.require()
    assert [(e["cid"], e["role"], e["complexity"]) for e in ent][:2] == [("c0355r1", "main", 14), ("c0262", "second", 23)]
    assert all(e["equation"] == POOL[e["cid"]]["equation"] for e in ent)
    rec = json.loads((root / "frozen/library.json").read_text())
    assert rec["protocol"]["path"] == str(proto) and len(rec["protocol"]["sha"]) == 64
    assert {"cities_csv_sha", "scoring_sha", "frozen_at"} <= set(rec)
    assert frozen.clean_library() == ["c0355r1", "c0262", "c0514c1"]
    assert frozen.unseal_env() == {"ODSR_UNSEAL": "1"}
    p = root / "frozen/library.json"
    p.write_text(p.read_text().replace("c0262", "c0263"))
    with pytest.raises(RuntimeError):
        frozen.require()


def test_freeze_refuses_twice_and_bad_entries(sealed, tmp_path):
    root, cfg, proto = sealed
    bad = tmp_path / "bad.json"
    for entries in ([{"cid": "c9999", "role": "main"}],                          # not in the pool
                    [{"cid": "c0355r1", "role": "main"}, {"cid": "c0262", "role": "favourite"}],
                    [{"cid": "c0262", "role": "second"}]):                       # no main
        bad.write_text(json.dumps({"entries": entries}))
        with pytest.raises(ValueError):
            frozen.freeze(bad, proto, POOL, log=lambda s: None)
    assert not frozen.is_frozen()
    frozen.freeze(cfg, proto, POOL, log=lambda s: None)
    with pytest.raises(FileExistsError):
        frozen.freeze(cfg, proto, POOL, log=lambda s: None)


# ------------------------------------------------------------------------------------------ Task 2: post-freeze paths
@pytest.fixture
def unfrozen(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, "FROZEN", tmp_path / "FROZEN")
    monkeypatch.setattr(frozen, "FROZEN_DIR", tmp_path / "frozen")
    return tmp_path


@pytest.fixture
def frozen_now(sealed):
    root, cfg, proto = sealed
    frozen.freeze(cfg, proto, POOL, log=lambda s: None)
    return root


def test_city_data_test_modes_refuse_before_freeze(unfrozen):
    import library
    for m in ("test_dest", "test_time"):
        with pytest.raises(RuntimeError):
            library.city_data("12060", mode=m)


def test_city_data_test_modes_after_freeze(frozen_now, monkeypatch):
    import library
    calls = []
    monkeypatch.setattr(library.sr_data, "build", lambda cities, scale, split, block, n, seed: (split, block, n))
    monkeypatch.setattr(library, "_device_group_y", lambda va, regions, scale, split: ("Yg", split))
    d = library.city_data("12060", 50, mode="test_dest")
    assert d["tr"] == ("train", "search", 50) and d["va"] == ("test", "fold:0", None) and d["Yg"] == ("Yg", "test")
    assert library.city_data("12060", 50, mode="test_time")["va"] == ("test", "val", None)
    assert calls == []


def test_fit_library_guards(unfrozen, tmp_path, monkeypatch):
    import config
    import fit_library
    with pytest.raises(RuntimeError):
        fit_library.main(cities=["12060"], cands=[POOL["c0262"]], mode="test_dest", out_dir=tmp_path / "o")
    with pytest.raises(RuntimeError):
        fit_library.main(cities=[config.CONFIRM[0]], cands=[POOL["c0262"]], allow_confirm=True, out_dir=tmp_path / "o")
    assert fit_library._eligible(["12060", config.CONFIRM[0]], {config.CONFIRM[0]}, False) == ["12060"]
    assert fit_library._eligible(["12060", config.CONFIRM[0]], {config.CONFIRM[0]}, True) == ["12060", config.CONFIRM[0]]
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs_test").mkdir()
    (tmp_path / "refs/12060.json").write_text(json.dumps({"models": {"a": 1}}))
    (tmp_path / "refs_test/12060.json").write_text(json.dumps({"test_dest": {"b": 2}}))
    assert fit_library._refs("12060") == {"a": 1} and fit_library._refs("12060", "test_dest") == {"b": 2}


def test_refs_test_refs_writes_both_blocks_after_freeze(unfrozen, tmp_path, monkeypatch):
    import refs
    with pytest.raises(RuntimeError):
        refs.test_refs("12060")
    root = tmp_path / "data"
    monkeypatch.setattr(refs.sr_data, "ROOT", root)
    d = root / "results/multicity/refs"
    d.mkdir(parents=True)
    (d / "12060.json").write_text(json.dumps({"city": "12060", "models": {"keep": 1}}))
    cfg, proto = tmp_path / "lib.json", tmp_path / "p.md"
    cfg.write_text(json.dumps(LIB))
    proto.write_text("p")
    frozen.freeze(cfg, proto, POOL, log=lambda s: None)
    seen = []
    monkeypatch.setattr(refs, "glm_refs", lambda c, s, transit_col=None, train_block=None, eval_split=None, eval_block=None,
                        models=None: seen.append((train_block, eval_split, eval_block)) or {"grav_power": {"nll_per_visit": 4.0}})
    monkeypatch.setattr(refs, "ext_rad_val", lambda c, tb, es, eb: {"nll_per_visit": 4.1, "block": (tb, es, eb)})
    monkeypatch.setattr(refs.sr_data, "build", lambda cities, scale, split, block, n, seed: (split, block))
    monkeypatch.setattr(refs, "s0_val", lambda c, tr=None, va=None: {"nll_per_visit": 3.9, "tr": tr, "va": va})
    before = (d / "12060.json").read_bytes()
    out = refs.test_refs("12060")
    assert (d / "12060.json").read_bytes() == before                         # pre-freeze file untouched
    r = json.loads((root / "results/multicity/refs_test/12060.json").read_text())
    assert set(out) == {"test_dest", "test_time"} and set(r) >= {"test_dest", "test_time"}
    assert seen == [("search", "test", "fold:0"), ("search", "test", "val")]
    assert r["test_dest"]["ext_rad"]["block"] == ["search", "test", "fold:0"]
    assert r["test_time"]["S0"]["va"] == ["test", "val"] and r["test_dest"]["S0"]["tr"] == ["train", "search"]


def test_covariates_banded_and_scan_open_only_after_freeze(unfrozen, tmp_path, monkeypatch):
    import config
    import laws
    import report
    import banded
    c = config.CONFIRM[0]
    with pytest.raises(ValueError):
        laws.covariates([c])
    with pytest.raises(RuntimeError):
        laws.covariates([c], allow_confirm=True)
    with pytest.raises(RuntimeError):
        report._scan(f"city {c}", lawtest=set(), allow_confirm=True)
    with pytest.raises(RuntimeError):
        banded.city_banded(c, allowed=[c], allow_confirm=True)
    cfg, proto = tmp_path / "lib.json", tmp_path / "p.md"
    cfg.write_text(json.dumps(LIB))
    proto.write_text("p")
    frozen.freeze(cfg, proto, POOL, log=lambda s: None)
    fm = tmp_path / "form_metrics.csv"
    fm.write_text("city," + ",".join(k for k in laws.COVARIATES if not k.startswith("log_")) + ",population,jobs\n"
                  + c + "," + ",".join("1" for k in laws.COVARIATES if not k.startswith("log_")) + ",1000,500\n")
    monkeypatch.setattr(laws, "RESULTS", tmp_path)
    assert list(laws.covariates([c], allow_confirm=True)["city"]) == [c]
    report._scan(f"city {c}", lawtest=set(), allow_confirm=True)
    with pytest.raises(ValueError):
        report._scan("city 10740", lawtest={"10740"}, allow_confirm=True)       # lawtest ids still refused
    with pytest.raises(ValueError):
        banded.city_banded(c)                                                  # default: still sealed

    class Reached(Exception):
        pass

    def stop(*a, **k):
        raise Reached
    monkeypatch.setattr(banded, "_load_city", stop, raising=False)
    import sr_data
    monkeypatch.setattr(sr_data, "load", stop)
    with pytest.raises(Reached):
        banded.city_banded(c, allowed=[c], allow_confirm=True)


def test_require_rechecks_every_pinned_input(sealed, tmp_path):
    root, cfg, proto = sealed
    pin = tmp_path / "eval.parquet"
    pin.write_bytes(b"abc")
    frozen.freeze(cfg, proto, POOL, log=lambda s: None, pinned={"eval_set": pin})
    rec = json.loads((root / "frozen/library.json").read_text())
    assert {"protocol", "eval_set", "scoring"} <= set(rec["pinned"])
    frozen.require()
    pin.write_bytes(b"abd")
    with pytest.raises(RuntimeError):
        frozen.require()
    pin.write_bytes(b"abc")
    frozen.require()
    proto.write_text("# plan 6 changed\n")
    with pytest.raises(RuntimeError):
        frozen.require()


def test_freeze_refuses_a_missing_pinned_file(sealed, tmp_path):
    root, cfg, proto = sealed
    with pytest.raises(ValueError):
        frozen.freeze(cfg, proto, POOL, log=lambda s: None, pinned={"eval_set": tmp_path / "nope.parquet"})
    assert not frozen.is_frozen()


def test_run_city_of_a_confirmation_city_needs_the_freeze_too(unfrozen, monkeypatch):
    import config
    import refs
    monkeypatch.setenv("ODSR_UNSEAL", "1")
    with pytest.raises(RuntimeError):
        refs.run_city(config.CONFIRM[0])
    with pytest.raises(RuntimeError):
        refs.inner_refs(config.CONFIRM[0])
