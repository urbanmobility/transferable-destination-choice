"""Plan 9 post-hoc exploration (added 2026-11-11): the source cities re-scored on the test block (leave-one-city-out, no
refit), kept apart from the pinned plan-9 outputs, and the 67-city same-block analysis of where size-predicted
constants help."""
import json

import numpy as np
import pandas as pd
import pytest

import ledger
import p9
import p9_explore as px
import param_transfer as pt
from test_p9 import CONF, EQ, LAW, MAT, OURS, SOURCE, _run, data, world  # noqa: F401  (fixtures)

ALL = ("fix", "source", "prerun", "select", "target", "contrasts")


@pytest.fixture
def done(world, monkeypatch):
    monkeypatch.setattr(px, "search_cities", lambda: SOURCE[-3:])
    _run(*ALL)
    return world


def test_score_refuses_before_the_contrasts(world, monkeypatch):
    monkeypatch.setattr(px, "search_cities", lambda: SOURCE[-3:])
    _run("fix", "source", "prerun", "select", "target")
    with pytest.raises(RuntimeError, match="contrasts"):
        px.main(["score"])


def test_score_rows_loo_and_no_refit(done):
    px.main(["score"])
    t = pd.read_parquet(done["out"] / "explore" / "source_test.parquet")
    models = list(EQ) + list(MAT) + list(OURS)
    assert set(t["city"]) == set(SOURCE) and len(t) == 12 * len(models) * 4
    assert not t.duplicated(["city", "model", "level"]).any() and set(t["level"]) == set(pt.LEVELS)
    src = pd.read_parquet(done["out"] / "fits_source.parquet").set_index(["city", "model"])
    loc = t[t["level"] == "local"].set_index(["city", "model"])
    for m in ("grav_power_N", "kwon"):                                   # the training-block fits, not refitted
        assert loc.loc[("s03", m), "consts"] == src.loc[("s03", m), "consts"]
    f = done["fits"]["test_dest"]["oursA"]["s03"]                         # skeleton: the stored test-block score
    assert loc.loc[("s03", "oursA"), "nll"] == pytest.approx(f["nll"], abs=1e-9)
    u = t[(t["model"] == "grav_power_N") & (t["level"] == "uniform")].set_index("city")["consts"].map(json.loads)
    others = np.array([json.loads(src.loc[(c, "grav_power_N"), "consts"]) for c in SOURCE if c != "s05"])
    assert np.allclose(u["s05"], np.median(others, axis=0))              # leave-one-city-out
    assert loc.loc[("s03", "grav_power_N"), "nll"] != pytest.approx(src.loc[("s03", "grav_power_N"), "nll"])   # another block


def test_score_leaves_the_pinned_outputs_alone(done):
    keep = {f: ledger.sha(done["out"] / f) for f in ("target.parquet", "contrasts.json", "opponents.json",
                                                    "fits_source.parquet", "prerun.parquet", "CONFIG_FIXED")}
    code = p9.code_sha()
    px.main(["score"])
    px.main(["analyze"])
    assert keep == {f: ledger.sha(done["out"] / f) for f in keep} and code == p9.code_sha()
    assert "p9_explore.py" not in p9.CODE
    assert set(p.name for p in (done["out"] / "explore").iterdir()) == {"source_test.parquet", "c3_explore.json", "c3_explore.md"}
    md = (done["out"] / "explore" / "c3_explore.md").read_text()
    assert "exploration" in md and "not a test" in md


def _frame(cities, gain, block_share=0.9):
    rows = []
    for c, g in zip(cities, gain):
        for lv, s in (("local", block_share + 0.02), ("uniform", block_share), ("size", block_share + g), ("form", block_share)):
            rows.append({"city": c, "model": "m", "level": lv, "share": s, "nll": 4 - s, "cpc": 0.6, "failed": False})
    return pd.DataFrame(rows)


def test_gain_table_and_dose_response():
    src, tgt = [f"s{k}" for k in range(6)], [f"t{k}" for k in range(6)]
    logpop = dict(zip(src + tgt, [13.0, 13.5, 14.0, 14.5, 15.0, 16.0, 13.2, 13.8, 14.1, 14.4, 15.2, 15.9]))
    cov = pd.DataFrame({"log_population": logpop})
    med = np.median([logpop[c] for c in src])
    gs = [abs(logpop[c] - med) / 10 for c in src]
    gt = [abs(logpop[c] - med) / 10 for c in tgt]
    g = px.gain_table({"source": _frame(src, gs), "target": _frame(tgt, gt)}, cov, "m", src, {"s5": "search"})
    assert list(g.columns[:6]) == ["set", "role", "logpop", "dist", "gain", "gain_nll"] and len(g) == 12
    assert g.loc["s5", "role"] == "search" and g.loc["s0", "role"] == "source" and g.loc["t0", "role"] == "target"
    assert g.loc["t5", "dist"] == pytest.approx(abs(15.9 - med)) and g.loc["s3", "gain"] == pytest.approx(gs[3])
    assert g.loc["s3", "loss_uniform"] == pytest.approx(0.02) and g.loc["s3", "gain_nll"] == pytest.approx(gs[3])
    r = px.dose_response(g, B=200, seed=1)
    assert r["n"] == 12 and r["spearman"] == pytest.approx(1.0) and r["ci_lo"] > 0.5 and r["slope"] == pytest.approx(0.1, abs=1e-6)
    assert [b["bin"] for b in r["terciles"]] == ["near", "mid", "far"] and sum(b["n"] for b in r["terciles"]) == 12
    assert r["terciles"][2]["gain_med"] > r["terciles"][0]["gain_med"]


def test_gain_table_drops_failures_and_says_so():
    src = [f"s{k}" for k in range(4)]
    f = _frame(src, [0.01] * 4)
    f.loc[(f["city"] == "s1") & (f["level"] == "size"), ["failed", "share", "nll"]] = [True, -np.inf, np.nan]
    cov = pd.DataFrame({"log_population": dict(zip(src, [13.0, 14.0, 15.0, 16.0]))})
    g = px.gain_table({"source": f}, cov, "m", src, {})
    assert "s1" not in g.index and g.attrs["dropped"] == {"source": ["s1"]}


def test_block_comparison_is_paired():
    src = [f"s{k}" for k in range(5)]
    a, b = _frame(src, [0.02, 0.01, 0.03, -0.01, 0.00]), _frame(src, [0.01, 0.01, -0.02, -0.01, 0.02], block_share=0.8)
    r = px.block_comparison(a, b, "m")
    assert r["n"] == 5 and r["wins_val"] == 3 and r["wins_test"] == 3 and r["both"] == 2
    assert r["gain_val_med"] == pytest.approx(0.01) and r["gain_test_med"] == pytest.approx(0.01)
