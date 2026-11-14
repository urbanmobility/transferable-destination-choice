"""Executor-registered hypotheses on the flexible distance-band model (design decision of 2026-11-09: register them as
new hypotheses, extending the testing budget if necessary). Two exploratory regularities of the 35 discovery cities (11-09 diagnostic) are registered BEFORE any
lawtest data is read, then tested once on the lawtest cities:

  H3-01  ed_5_10km ~ log_population (-), slope_ci — the e_d of the 5-10 km band gets more negative with city size
  H3-02  slope_eN_logd > 0, median_sign_ci     — the destination-size elasticity rises with distance

Quantities: prefreeze.flex_quantities on banded.city_banded outputs (the plan-5 flexible model). Tests: slope_ci is
the loop's own laws.run_test; median_sign_ci (new, for a sign claim that has no covariate) = city-bootstrap 95% CI of
the median excludes 0 in the stated direction, two-sided bootstrap p as in laws.run_test. Multiple testing: Holm
over the cumulative count, the 10 loop hypotheses (laws/results.json, raw p) plus these. Files: laws/
registered_extra.json, laws/results_extra.json (the loop's registered.json / results.json are not touched); fits of
the lawtest cities: results/multicity/prefreeze/lawtest_banded/. No confirmation city, no test period, no LLM.

Usage (compute server, loop env): python -I lawtest_extra.py register | banded | test
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import config  # noqa: E402
import laws  # noqa: E402

CYCLE = 3
SOURCE = "executor (design decision 2026-11-09)"      # provenance label stored in registered_extra.json
LAWS = config.RESULTS / "laws"
# The statements and criteria are registered verbatim (content sha laws._hsha, registered_extra.json), so they stay in
# the language of the registration. In English:
#   H3-01 statement: in the flexible banded model (the M2 design plus a size term per distance band), the distance
#     elasticity e_d of the 5-10 km band becomes more negative as log metro population grows, i.e. mid-to-long-range
#     decay is steeper in large cities.
#   H3-01 criterion: OLS with ed_5_10km (prefreeze.flex_quantities: e_d of band 3, 5-10 km, with >= 1000 training-period
#     trips) as response and log_population as the only predictor; on the test cities the city-bootstrap (B = 5000)
#     95% CI excludes 0 and the slope is < 0; Holm over all hypotheses registered so far (H1-01 to H2-05 and this
#     round), alpha = 0.05.
#   H3-02 statement: in the flexible banded model the destination-size elasticity e_N rises with distance: the median
#     over cities of the per-city OLS slope of e_N on ln d is greater than 0.
#   H3-02 criterion: the median over the test cities of slope_eN_logd (prefreeze.flex_quantities: OLS slope of e_N on
#     ln d_ref with >= 1000 training-period trips and at least 3 bands); lower bound of the city-bootstrap (B = 5000)
#     95% CI > 0; Holm as above, alpha = 0.05.
HYPS = [
    {"statement": "灵活分段模型（M2 设计加各距离段的规模项）在 5–10 km 段的距离弹性 e_d 随都市区人口对数增大而更负，"
                  "即大城市的中远距离衰减更陡。",
     "skeleton": "flex_banded", "parameter": "ed_5_10km", "covariate": "log_population", "direction": "-",
     "test": "slope_ci",
     "criterion": "以 ed_5_10km（prefreeze.flex_quantities：第 3 段 5–10 km、训练期出行 ≥ 1000 时的 e_d）为响应、"
                  "log_population 为唯一预测变量做 OLS；检验城的城市 bootstrap（B = 5000）95% CI 不含 0 且斜率 < 0；"
                  "Holm 按累计登记的全部假设（H1-01 至 H2-05 与本轮）校正，α = 0.05。"},
    {"statement": "灵活分段模型中，目的地规模弹性 e_N 随距离上升：各城 e_N 对 ln d 的 OLS 斜率的城市中位数大于 0。",
     "skeleton": "flex_banded", "parameter": "slope_eN_logd", "covariate": "none", "direction": "+",
     "test": "median_sign_ci",
     "criterion": "slope_eN_logd（prefreeze.flex_quantities：训练期出行 ≥ 1000、至少 3 段时 e_N 对 ln d_ref 的 OLS "
                  "斜率）在检验城上的城市中位数，城市 bootstrap（B = 5000）95% CI 下限 > 0；Holm 同上，α = 0.05。"},
]


def register_extra(hyps: list, registered: list, extra: list, log) -> list:
    """extra + the new hypotheses (id H3-<k>, cycle 3, content sha as laws._hsha, source, time); any (skeleton,
    parameter, covariate) already in registered or extra -> ValueError (nothing is registered twice)."""
    have = {(h["skeleton"], h["parameter"], h["covariate"]) for h in list(registered) + list(extra)}
    new = []
    for h in hyps:
        key = (h["skeleton"], h["parameter"], h["covariate"])
        if key in have:
            raise ValueError(f"already registered: {key}")
        have.add(key)
        new.append({**{k: h[k] for k in laws.FIELDS}, "id": f"H{CYCLE}-{len(extra) + len(new) + 1:02d}", "cycle": CYCLE,
                    "sha": laws._hsha(h), "source": SOURCE, "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log("Hypotheses of 2026-11-09 registered by the executor before any lawtest data was read: "
        + "; ".join(f"{h['id']} sha {h['sha']} [{h['skeleton']} {h['parameter']} ~ {h['covariate']} ({h['direction']}) "
                    f"by {h['test']}]" for h in new))
    return list(extra) + new


def median_sign_ci(values, direction: str, key: str, B: int = laws.TEST_B, seed: int = laws.SEED) -> dict:
    y = np.asarray(values, float)
    y = y[np.isfinite(y)]
    sgn = laws._sign(direction)
    out = {"stat": None, "p": 1.0, "pass_raw": False, "n": int(len(y)), "ci_lo": None, "ci_hi": None, "B": B}
    if len(y) < laws.MIN_CITIES:
        return {**out, "note": "too few cities"}
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    bs = np.median(y[rng.integers(0, len(y), (B, len(y)))], axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return {**out, "stat": float(np.median(y)), "p": p, "ci_lo": float(lo), "ci_hi": float(hi),
            "pass_raw": bool(lo > 0 if sgn > 0 else hi < 0)}


def run_extra(h: dict, q: pd.DataFrame, cov: pd.DataFrame, B: int = laws.TEST_B) -> dict:
    """q: city, value (the hypothesis' quantity per lawtest city); cov: city + covariates."""
    q = q[np.isfinite(q["value"].astype(float))]
    if h["test"] == "slope_ci":
        pt = pd.DataFrame({"cid": h["skeleton"], "param": h["parameter"], "city": q["city"], "value": q["value"],
                           "se": None, "sd_starts": 0.0, "identifiable": True})
        return laws.run_test(h, pt, cov, None, B=B)
    if h["test"] == "median_sign_ci":
        return median_sign_ci(q["value"].to_numpy(float), h["direction"], f"{h['skeleton']}|{h['parameter']}", B=B)
    raise ValueError(f"unknown test {h['test']}")


def holm_cumulative(previous: list, extra_results: list) -> list:
    """Holm over the previous (loop) raw p values and these; each extra result gets m, holm, pass; prev_still_pass =
    for the loop hypotheses that passed before, whether they still pass with the larger m."""
    allr = list(previous) + list(extra_results)
    ok = laws.holm([float(r["p"]) for r in allr])
    still = {r["id"]: bool(k and r.get("pass_raw", r.get("pass"))) for r, k in zip(previous, ok) if r.get("pass")}
    out = []
    for r, k in zip(extra_results, ok[len(previous):]):
        out.append({**r, "m": len(allr), "holm": bool(k), "pass": bool(k and r["pass_raw"]), "prev_still_pass": still})
    return out


# ================================================================================================ CLI stages
def _read(p, default):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def stage_register():
    import ledger
    import prefreeze
    registered = _read(LAWS / "registered.json", [])
    extra = _read(LAWS / "registered_extra.json", [])
    ev = {}
    q = {}
    for c in prefreeze.cities():                            # discovery evidence (already seen), for the record only
        b = _read(prefreeze.ROOT / f"banded/{c}.json", None)
        if b:
            q[c] = prefreeze.flex_quantities(b)
    cov = laws.covariates(prefreeze.cities())
    for h in HYPS:
        d = pd.DataFrame({"city": list(q), "value": [q[c][h["parameter"]] for c in q]}).dropna()
        ev[h["parameter"]] = run_extra(h, d, cov, B=2000)
    new = register_extra(HYPS, registered, extra, log=ledger.add)
    p = LAWS / "registered_extra.json"
    p.write_text(json.dumps(new, ensure_ascii=False, indent=1))
    ledger.add(f"Decision of 2026-11-09: budget extended for H3-01 / H3-02 (registered by the executor outside the loop's "
               f"2 outer cycles; Holm over the cumulative count, now 12; no LLM call). Discovery evidence (already seen, "
               f"not a test): " + "; ".join(f"{k}: stat {v['stat']:.4f} CI [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}] n {v['n']}"
                                            for k, v in ev.items()) + f"; registered_extra.json sha {ledger.sha(p)}")
    return 0


def _banded_one(args):
    import banded
    import prefreeze
    city, allowed = args
    try:
        banded.city_banded(city, out_dir=prefreeze.ROOT / "lawtest_banded", allowed=allowed)
        return city, None
    except Exception as e:  # noqa: BLE001
        return city, repr(e)[:300]


def stage_banded():
    import multiprocessing as mp
    import os
    import fit_library
    import ledger
    import prefreeze
    extra = _read(LAWS / "registered_extra.json", [])
    if {h["parameter"] for h in extra} != {h["parameter"] for h in HYPS}:
        sys.exit("register H3-01 / H3-02 first: no lawtest data is read before the hypotheses are registered")
    law = config.cities_with_role("lawtest")
    out = prefreeze.ROOT / "lawtest_banded"
    out.mkdir(parents=True, exist_ok=True)
    todo = sorted((c for c in law if not (out / f"{c}.json").exists()), key=lambda c: -fit_library._size(c))
    if todo:
        with mp.get_context("fork").Pool(min(int(os.environ.get("BANDED_PROCS", "4")), len(todo)),
                                         maxtasksperchild=1) as P:
            res = list(P.imap_unordered(_banded_one, [(c, law) for c in todo], chunksize=1))
        errs = {c: e for c, e in res if e}
    else:
        errs = {}
    ledger.add(f"Hypotheses of 2026-11-09: flexible distance-band model fitted on {len(todo) - len(errs)} of "
               f"{len(todo)} lawtest cities ({len(list(out.glob('*.json')))} present){'' if not errs else f'; errors {len(errs)}'}")
    return 4 if errs else 0


def stage_test():
    import ledger
    import prefreeze
    extra = _read(LAWS / "registered_extra.json", [])
    prev = _read(LAWS / "results.json", [])
    law = config.cities_with_role("lawtest")
    q = {c: prefreeze.flex_quantities(json.loads((prefreeze.ROOT / f"lawtest_banded/{c}.json").read_text()))
         for c in law if (prefreeze.ROOT / f"lawtest_banded/{c}.json").exists()}
    cov = laws.covariates(law)
    res = []
    for h in extra:
        d = pd.DataFrame({"city": list(q), "value": [q[c][h["parameter"]] for c in q]}).dropna()
        res.append({"id": h["id"], "parameter": h["parameter"], "covariate": h["covariate"],
                    "direction": h["direction"], "test": h["test"], **run_extra(h, d, cov)})
    res = holm_cumulative(prev, res)
    p = LAWS / "results_extra.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    ledger.add("Hypotheses of 2026-11-09 on the lawtest cities (Holm over m = " + str(res[0]["m"] if res else 0)
               + "): " + "; ".join(f"{r['id']} {'PASS' if r['pass'] else 'fail'} (stat {r['stat']}, CI [{r['ci_lo']}, "
                                   f"{r['ci_hi']}], p {r['p']:.4f}, n {r['n']})" for r in res)
               + f"; loop laws still passing at the larger m: {res[0]['prev_still_pass'] if res else {}}"
               + f"; results_extra.json sha {ledger.sha(p)}")
    return 0


STAGES = {"register": stage_register, "banded": stage_banded, "test": stage_test}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: lawtest_extra.py {{{'|'.join(STAGES)}}}")
    sys.exit(STAGES[sys.argv[1]]())
