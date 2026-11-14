"""Tolerance test (11-09): the loop's rules re-run with the tolerance multiplier 0.05 (instead of the registered 0.03) as a
sensitivity test — gate on the search cities, discovery fits, coverage at both multipliers."""
import pandas as pd
import pytest

import scoring
import sensitivity as sv

REFS = {c: {"grav_power": {"nll_per_visit": 3.2}, "grav_exp": {"nll_per_visit": 3.3}, "ext_rad": {"nll_per_visit": 3.25},
            "M2_glm": {"nll_per_visit": 3.0}, "S0": {"nll_per_visit": 3.1}} for c in ("A", "B")}     # gap 0.2


def _cv():
    rows = [("c1", "A", 3.000, 20), ("c2", "A", 3.008, 10), ("c3", "A", 3.020, 5),
            ("c1", "B", 3.004, 20), ("c2", "B", 3.000, 10), ("c3", "B", 3.0, 5)]
    df = pd.DataFrame(rows, columns=["cid", "city", "val_nll", "complexity"])
    df["finite"] = [True, True, True, True, True, False]
    df["se_vs_best"] = 0.0005
    df["within_1se"] = False
    return df


def test_recompute_near_uses_the_given_multiplier():
    n3 = sv.recompute_near(_cv(), REFS, 0.03)                    # delta 0.006
    n5 = sv.recompute_near(_cv(), REFS, 0.05)                    # delta 0.010
    near = lambda df: set(zip(df.cid[df.near_best], df.city[df.near_best]))  # noqa: E731
    assert near(n3) == {("c1", "A"), ("c1", "B"), ("c2", "B")}
    assert near(n5) == {("c1", "A"), ("c2", "A"), ("c1", "B"), ("c2", "B")}
    prim = lambda df: set(zip(df.cid[df.primary], df.city[df.primary]))  # noqa: E731
    assert prim(n3) == {("c1", "A"), ("c2", "B")} and prim(n5) == {("c2", "A"), ("c2", "B")}
    assert not n5[(n5.cid == "c3") & (n5.city == "B")].near_best.any()        # not finite: never near-best


def test_with_margin_is_temporary():
    r = REFS["A"]
    assert scoring.delta_city(r) == pytest.approx(0.006)
    with sv.with_margin(0.05):
        assert scoring.delta_city(r) == pytest.approx(0.010)
    assert scoring.delta_city(r) == pytest.approx(0.006) and scoring.MARGIN == 0.03


def test_gate_cids_keeps_every_representative_with_enough_cities():
    ranked = [{"cid": "a", "n_near": 6}, {"cid": "b", "n_near": 4}, {"cid": "c", "n_near": 3}, {"cid": "d", "n_near": 5}]
    assert sv.gate_cids(ranked) == ["a", "b", "d"] and sv.gate_cids(ranked, 3) == ["a", "b", "c", "d"]


def test_recompute_near_without_refs_uses_the_se_only():
    n = sv.recompute_near(_cv(), {}, 0.05)
    assert set(zip(n.cid[n.near_best], n.city[n.near_best])) == {("c1", "A"), ("c2", "B")}


def _outputs(root, city):
    import json
    (root / "discovery").mkdir(parents=True)
    (root / "loop").mkdir()
    (root / "gate.json").write_text(json.dumps({"margin": 0.05, "gate": [{"cid": "c1", "n_near": 5, "n_flags": 0,
                                                                         "complexity": 13, "merged_from": []}],
                                                "gate_0.03": ["c1"]}))
    cv = _cv().assign(city=city, share_s0=0.7, consts="[1.0]")
    cv.to_parquet(root / "discovery/cities_val.parquet")
    (root / "discovery/coverage.json").write_text(json.dumps({"0.03": {"all": [], "clean": []},
                                                              "0.05": {"all": [], "clean": []}, "reproduction": {}}))
    (root / "loop/pool.json").write_text(json.dumps([{"cid": c, "complexity": 10, "equation": "f = log(#6 + 1.0); p1 = []; p2 = []",
                                                      "template": None, "variant": "A"} for c in ("c1", "c2", "c3")]))
    (root / "loop/flags.json").write_text("{}")


def test_build_report_writes_sections_and_refuses_sealed_ids(tmp_path, monkeypatch):
    import report
    monkeypatch.setattr(report, "_lawtest_ids", lambda: {"99999"})
    monkeypatch.setattr(report, "_confirm_ids", lambda: set())
    _outputs(tmp_path / "ok", "12060")
    s = sv.build_report(tmp_path / "ok/r.md", root=tmp_path / "ok")
    text = (tmp_path / "ok/r.md").read_text()
    assert s == {"gate": 1, "candidates": 3}
    assert [ln.split()[1] for ln in text.splitlines() if ln.startswith("## ")] == ["1.", "2.", "3.", "4."]
    _outputs(tmp_path / "bad", "99999")
    with pytest.raises(RuntimeError):
        sv.build_report(tmp_path / "bad/r.md", root=tmp_path / "bad")
