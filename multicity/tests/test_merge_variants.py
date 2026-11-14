"""Design decision of 2026-11-09 (variants examined side by side): three merge-rule variants of the tolerance
test — near-best at 0.05 with the equivalence merge at 0.03 (chain, = plan 5's merge), at 0.05 without chaining,
at 0.03 without chaining."""
import numpy as np
import pandas as pd
import pytest

import merge
import merge_variants as mv
import scoring
import sensitivity as sv

CITIES = ["T1", "T2", "T3"]
CIDS = ["c0001", "c0002", "c0003", "c0004", "c0005"]
BASE = {"c0001": 4.000, "c0002": 4.004, "c0003": 4.008, "c0004": 4.000, "c0005": 4.009}
CX = {"c0001": 5, "c0002": 7, "c0003": 9, "c0004": 6, "c0005": 8}
REFS = {"T1": {"grav_power": {"nll_per_visit": 4.2}, "M2_glm": {"nll_per_visit": 4.0}},     # delta 0.006 / 0.010
        "T2": {"grav_power": {"nll_per_visit": 4.2}, "M2_glm": {"nll_per_visit": 4.0}},
        "T3": {"grav_power": {"nll_per_visit": 4.2}}}                                       # no M2: the SD alone


@pytest.fixture
def world():
    """c0001-c0002-c0003 a chain at 0.03 (neighbours 0.004 apart, the ends 0.008); c0004 not fittable in T2;
    c0005 absent from T3; T3 has wide bootstrap noise (its tolerance is the SD); etas of c0001-c0003 are monotone
    transforms of one base (rank-close), c0004 / c0005 independent."""
    rng = np.random.default_rng(3)
    rows, boot, etas = [], {}, {}
    offsets = np.array([0, 6, 13, 21])
    for city in CITIES:
        cids = [c for c in CIDS if not (city == "T3" and c == "c0005")]
        common = rng.normal(size=(40, 1)) * 0.01
        wide = 0.015 if city == "T3" else 0.0005
        mat = np.column_stack([BASE[c] + common[:, 0] + rng.normal(size=40) * wide for c in cids])
        if city == "T1":
            mat[:4, cids.index("c0002")] = np.nan
        boot[city] = mat
        for c in cids:
            rows.append({"cid": c, "city": city, "val_nll": BASE[c], "finite": not (city == "T2" and c == "c0004"),
                         "consts": "[1.0]", "se_vs_best": 0.0005, "within_1se": False})
        base = rng.normal(size=offsets[-1])
        etas[city] = {"offsets": offsets, "eta": {"c0001": base, "c0002": 2 * base + 1, "c0003": np.exp(base),
                                                  "c0004": rng.normal(size=offsets[-1]),
                                                  "c0005": rng.normal(size=offsets[-1])}}
    pool = [{"cid": c, "equation": "f = log(#6 + 1.0); p1 = []; p2 = []", "template": None, "complexity": CX[c]}
            for c in CIDS]
    return pool, pd.DataFrame(rows), boot, etas


def _sets(reps):
    return {r["cid"]: tuple(r["merged_from"]) for r in reps}


def test_loss_table_matches_merge_loss_close(world):
    _, fits, boot, _ = world
    lt, lt0 = mv.LossTable(fits, boot, CITIES, REFS), mv.LossTable(fits, boot, CITIES, None)
    seen = set()
    for m in (0.03, 0.05):
        with sv.with_margin(m):
            for i in CIDS:
                for j in CIDS:
                    if i != j:
                        got = lt.close(i, j, m)
                        assert got == merge.loss_close(i, j, fits, boot, CITIES, REFS), (m, i, j)
                        assert lt0.close(i, j, m) == merge.loss_close(i, j, fits, boot, CITIES, None), (m, i, j)
                        seen.add((m, i, j, got))
    assert (0.03, "c0001", "c0003", False) in seen and (0.05, "c0001", "c0003", True) in seen
    assert (0.03, "c0001", "c0002", True) in seen and not any(g for m, i, j, g in seen if "c0004" in (i, j))


def test_loss_table_pairs_follow_the_merge_order(world):
    _, fits, boot, _ = world
    lt = mv.LossTable(fits, boot, CITIES, REFS)
    with sv.with_margin(0.03):
        want = [(a, b) for k, a in enumerate(CIDS) for b in CIDS[k + 1:]
                if merge.loss_close(a, b, fits, boot, CITIES, REFS)]
    assert lt.pairs(CIDS, 0.03) == want == [("c0001", "c0002"), ("c0002", "c0003")]


def test_cluster_chain_reproduces_merge_merge(world):
    pool, fits, boot, etas = world
    lt = mv.LossTable(fits, boot, CITIES, REFS)
    for m in (0.03, 0.05):
        pairs = lt.pairs(CIDS, m)
        rclose = mv.lookup(mv.rank_table(pairs, etas, CITIES))
        with sv.with_margin(m):
            want = merge.merge(pool, fits, boot, etas, CITIES, REFS)
        assert _sets(mv.cluster_chain(pool, pairs, rclose)) == _sets(want)
    assert _sets(want)["c0001"] == ("c0002", "c0003")


def test_cluster_leader_breaks_the_chain(world):
    pool, fits, boot, etas = world
    lt = mv.LossTable(fits, boot, CITIES, REFS)
    pairs = lt.pairs(CIDS, 0.05)
    rclose = mv.lookup(mv.rank_table(pairs, etas, CITIES))
    s03 = set(lt.pairs(CIDS, 0.03))
    lead = mv.cluster_leader(pool, mv.member_of(s03), rclose)
    chain = mv.cluster_chain(pool, lt.pairs(CIDS, 0.03), rclose)
    assert _sets(chain) == {"c0001": ("c0002", "c0003"), "c0004": (), "c0005": ()}
    assert _sets(lead) == {"c0001": ("c0002",), "c0003": (), "c0004": (), "c0005": ()}       # c0003 ~ c0002 only
    assert _sets(mv.cluster_leader(pool, mv.member_of(set(pairs)), rclose))["c0001"] == ("c0002", "c0003")


def test_cluster_leader_joins_the_first_representative_in_complexity_order():
    pool = [{"cid": c, "complexity": k} for c, k in (("a", 3), ("b", 1), ("c", 2), ("d", 5))]
    yes = lambda i, j: True  # noqa: E731
    near = {frozenset(p) for p in (("d", "b"), ("d", "c"))}
    lc = lambda i, j: frozenset((i, j)) in near  # noqa: E731
    assert _sets(mv.cluster_leader(pool, lc, yes)) == {"a": (), "b": ("d",), "c": ()}
    assert _sets(mv.cluster_leader(pool, yes, yes)) == {"b": ("a", "c", "d")}


def test_rank_table_parallel_equals_serial_and_lookup_refuses_unknown_pairs(world):
    _, fits, boot, etas = world
    pairs = [(a, b) for k, a in enumerate(CIDS) for b in CIDS[k + 1:]]
    t1, t2 = mv.rank_table(pairs, etas, CITIES, procs=1), mv.rank_table(pairs, etas, CITIES, procs=2)
    assert t1 == t2 and t1[("c0001", "c0003")] and not t1[("c0001", "c0004")]
    f = mv.lookup({("a", "b"): True})
    assert f("a", "b") and f("b", "a")
    with pytest.raises(KeyError):
        f("a", "c")


def _gate_world(gap):
    cities = ["S1", "S2", "S3", "S4"]
    refs = {c: {"grav_power": {"nll_per_visit": 4.2}, "grav_exp": {"nll_per_visit": 4.25},
                "ext_rad": {"nll_per_visit": 4.3}, "M2_glm": {"nll_per_visit": 4.0}} for c in cities}
    rows = [{"cid": cid, "city": c, "val_nll": 4.0 + d, "finite": True, "se_vs_best": 0.0, "within_1se": cid == "b1"}
            for c in cities for cid, d in (("b1", 0.0), ("r1", gap))]
    return pd.DataFrame(rows), refs


def test_gate_for_applies_the_near_best_multiplier_and_restores_it():
    fits, refs = _gate_world(0.008)                                                # delta 0.006 at 0.03, 0.010 at 0.05
    reps = [{"cid": "b1", "complexity": 9, "merged_from": []}, {"cid": "r1", "complexity": 5, "merged_from": []}]
    assert mv.gate_for(reps, fits, refs, {}, 0.05)[1] == ["r1", "b1"]
    assert mv.gate_for(reps, fits, refs, {}, 0.03)[1] == ["b1"] and scoring.MARGIN == 0.03
    fits2, _ = _gate_world(0.05)
    merged = [{"cid": "r1", "complexity": 5, "merged_from": ["b1"]}]                # a member's cities count
    assert mv.gate_for(merged, fits2, refs, {}, 0.03)[1] == ["r1"]


def test_cluster_mismatches_and_rep_of():
    reps = [{"cid": "a", "merged_from": ["b"]}, {"cid": "c", "merged_from": []}]
    assert mv.cluster_mismatches(reps, [{"cid": "a", "merged_from": ["b"]}, {"cid": "c", "merged_from": []}]) == []
    assert len(mv.cluster_mismatches(reps, [{"cid": "a", "merged_from": ["b", "c"]}])) == 1
    assert len(mv.cluster_mismatches(reps, [{"cid": "z", "merged_from": []}])) == 1
    assert mv.rep_of(reps) == {"a": "a", "b": "a", "c": "c"}


def _outputs(root, city):
    import json
    (root / "discovery").mkdir(parents=True)
    (root / "loop").mkdir()
    var = {v: {**mv.VARIANTS[v], "near_margin": 0.05, "n_reps": 3,
               "gate": [{"cid": "c1", "n_near": 5, "n_flags": 0, "complexity": 13, "merged_from": []}],
               "rep_of": {"c1": "c1", "c2": "c1" if v == "V2" else "c2"}} for v in mv.VARIANTS}
    (root / "gates.json").write_text(json.dumps({"variants": var, "reference": {"chain_0.05": {"c1": "c1", "c2": "c1"}},
                                                 "checks": {"loss_close_pairs": {"0.03": 1, "0.05": 2}}}))
    df = pd.DataFrame({"cid": ["c1", "c2"], "city": city, "val_nll": [3.0, 3.01], "finite": True, "complexity": [13, 15],
                       "se_vs_best": 0.0, "within_1se": False, "share_s0": 0.6})
    df.to_parquet(root / "discovery/cities_val.parquet")
    cov = {v: {s: {m: {"all": [], "clean": []} for m in ("0.03", "0.05")} for s in ("gate", "plan5+gate")}
           for v in mv.VARIANTS}
    (root / "discovery/coverage.json").write_text(json.dumps({"variants": cov, "plan5": ["c2"], "reproduction": {}}))
    (root / "loop/pool.json").write_text(json.dumps([{"cid": c, "complexity": k, "equation": "f = log(#6 + 1.0); p1 = []; p2 = []",
                                                      "template": None} for c, k in (("c1", 13), ("c2", 15))]))
    (root / "loop/flags.json").write_text("{}")


def test_build_report_writes_sections_and_refuses_sealed_ids(tmp_path, monkeypatch):
    import report
    monkeypatch.setattr(report, "_lawtest_ids", lambda: {"99999"})
    monkeypatch.setattr(report, "_confirm_ids", lambda: set())
    _outputs(tmp_path / "ok", "12060")
    s = mv.build_report(tmp_path / "ok/r.md", root=tmp_path / "ok")
    text = (tmp_path / "ok/r.md").read_text()
    assert s == {"V1": 1, "V2": 1, "V3": 1}
    assert [ln.split()[1] for ln in text.splitlines() if ln.startswith("## ")] == ["1.", "2.", "3.", "4."]
    assert "12060" in text
    _outputs(tmp_path / "bad", "99999")
    with pytest.raises(RuntimeError):
        mv.build_report(tmp_path / "bad/r.md", root=tmp_path / "bad")
