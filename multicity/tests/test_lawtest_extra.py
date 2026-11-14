"""Decision of 2026-11-09: two executor-registered hypotheses on the flexible distance-band model (H3-01, H3-02), tested
on the lawtest cities with Holm over the cumulative count (the 10 loop hypotheses + these)."""
import numpy as np
import pandas as pd
import pytest

import laws
import lawtest_extra as lx


def test_extra_hypotheses_are_well_formed():
    assert [h["parameter"] for h in lx.HYPS] == ["ed_5_10km", "slope_eN_logd"]
    assert [(h["test"], h["direction"]) for h in lx.HYPS] == [("slope_ci", "-"), ("median_sign_ci", "+")]
    assert all(h["skeleton"] == "flex_banded" for h in lx.HYPS)
    assert lx.HYPS[0]["covariate"] == "log_population" and lx.HYPS[1]["covariate"] == "none"
    for h in lx.HYPS:
        assert all(str(h[f]).strip() for f in laws.FIELDS)


def test_register_extra_assigns_ids_sha_and_refuses_repeats():
    old = [{"statement": "x", "skeleton": "c0413c1", "parameter": "elasticity_5km", "covariate": "log_population",
            "direction": "-", "test": "slope_ci", "criterion": "x", "id": "H1-01"}]
    lines = []
    new = lx.register_extra(lx.HYPS, old, [], log=lines.append)
    assert [h["id"] for h in new] == ["H3-01", "H3-02"] and all(h["cycle"] == 3 for h in new)
    assert [h["sha"] for h in new] == [laws._hsha(h) for h in lx.HYPS]
    assert all(h["source"] == lx.SOURCE for h in new) and len(lines) == 1 and "H3-01" in lines[0]
    with pytest.raises(ValueError):
        lx.register_extra(lx.HYPS, old, new, log=lines.append)                 # nothing is registered twice


def test_median_sign_ci():
    rng = np.random.default_rng(0)
    pos = lx.median_sign_ci(0.12 + 0.03 * rng.normal(size=26), "+", "k", B=2000)
    assert pos["pass_raw"] and pos["ci_lo"] > 0 and pos["p"] <= 2 / 2000 and pos["n"] == 26
    mix = lx.median_sign_ci(0.1 * rng.normal(size=26), "+", "k", B=2000)
    assert not mix["pass_raw"] and mix["ci_lo"] < 0 < mix["ci_hi"]
    neg = lx.median_sign_ci(-0.12 + 0.03 * rng.normal(size=26), "+", "k", B=2000)
    assert not neg["pass_raw"]
    assert lx.median_sign_ci(np.array([0.1, np.nan, 0.2]), "+", "k", B=200)["n"] == 2


def test_run_extra_slope_ci_uses_the_loop_test():
    rng = np.random.default_rng(1)
    z = rng.normal(14.5, 0.8, 26)
    q = pd.DataFrame({"city": [f"C{k}" for k in range(26)], "value": -0.3 * z + 0.05 * rng.normal(size=26)})
    cov = pd.DataFrame({"city": q["city"], "log_population": z})
    r = lx.run_extra(lx.HYPS[0], q, cov, B=2000)
    assert r["pass_raw"] and r["stat"] == pytest.approx(np.polyfit(z, q["value"], 1)[0]) and r["ci_hi"] < 0


def test_holm_is_cumulative_over_all_registered_hypotheses():
    ps = {"H1-01": 0.002, "H1-02": 0.11, "H1-03": 0.66, "H1-04": 0.69, "H1-05": 0.059, "H2-01": 0.90,
          "H2-02": 0.021, "H2-03": 0.0008, "H2-04": 0.045, "H2-05": 0.42}
    prev = [{"id": k, "p": p, "pass_raw": k in ("H1-01", "H2-03", "H2-02"), "pass": k in ("H1-01", "H2-03")}
            for k, p in ps.items()]
    ext = [{"id": "H3-01", "p": 0.0004, "pass_raw": True}, {"id": "H3-02", "p": 0.0004, "pass_raw": False}]
    out = lx.holm_cumulative(prev, ext)
    assert [r["m"] for r in out] == [12, 12]
    assert out[0]["holm"] and out[0]["pass"] and out[1]["holm"] and not out[1]["pass"]
    assert out[0]["prev_still_pass"] == {"H1-01": True, "H2-03": True}
