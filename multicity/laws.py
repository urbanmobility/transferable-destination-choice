"""Outer loop — cross-city analysis, hypotheses and parameterised skeletons (plan 4 Tasks 11-13; design
§4.10, §5.4).

Task 11 (code only, city-level):
covariates(): the urban-form covariates of the given cities from form_metrics.csv (+ log population, log jobs);
  confirmation cities refused.
param_table(): one row per (candidate, constant, city) of the library fits (cities_val.parquet): value, SE (device-
  group cluster-robust sandwich, refit_select.const_cov), SD over the converged starts; identifiable = |value| > 2 SE
  and SD_starts < SE. Parameter names "<comp>:c<i>" (comp f / g / h, or p1 / p2 for template parameters).
composites(): candidates with both a distance and an S_ij term — the effective distance elasticity d eta / d log d at
  d = REF_D_KM along the city's S-distance relation (S_ij moves with d: dlogS/dlogd from the city's rows), SE by the
  delta method; the identifiable quantity when log(d + c) and log(S_ij + c) are collinear.
cross_city_summary(): univariate OLS of every identifiable parameter / composite on every covariate over the cities
  where it is identifiable (>= MIN_CITIES), city-bootstrap percentile CI (B, seeded) and two-sided bootstrap p.
term_existence(): every ablated top-level term (stable key = the term with symbolic constants) in every city:
  exists = removing it costs >= delta_city (design §4.10, effect size, not significance).
analysis_payload(): the city-level summary the agent reads (no tract data, no lawtest / confirmation city).
"""
import dataclasses
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import config  # noqa: E402
import llm  # noqa: E402
import scoring  # noqa: E402
from config import RESULTS  # noqa: E402
from library import candidate  # noqa: E402
from refit_select import XS, parse_equation  # noqa: E402
from sr_data import VARS  # noqa: E402

COVARIATES = ("pop_density_gini", "cbd_job_share", "subcentre_job_share", "n_subcentres", "orientation_entropy",
              "water_share", "log_population", "log_jobs")
# COV_MEANING and METHODS are part of the agent's input (analysis_payload), and COV_MEANING enters the statements of
# registered hypotheses (whose sha covers the statement), so both are kept verbatim in the original Chinese.
# English glosses:
#   pop_density_gini     population-weighted Gini coefficient of population density (low = evenly spread)
#   cbd_job_share        job share of the CBD subcentre
#   subcentre_job_share  job share of all employment subcentres (degree of polycentricity)
#   n_subcentres         number of employment subcentres
#   orientation_entropy  orientation entropy of the main roads (low = grid-like)
#   water_share          water share inside the convex hull of the built-up area
#   log_population       log of the metro population
#   log_jobs             log of the metro jobs
COV_MEANING = {"pop_density_gini": "人口密度的人口加权基尼系数（低 = 分布均匀）", "cbd_job_share": "CBD 次中心的就业份额",
               "subcentre_job_share": "所有就业次中心的就业份额（多中心程度）", "n_subcentres": "就业次中心个数",
               "orientation_entropy": "主干道路走向熵（低 = 网格化）", "water_share": "建成区凸包内水面比例",
               "log_population": "都市区人口的对数", "log_jobs": "都市区就业岗位的对数"}
REF_D_KM, MIN_CITIES, SEED = 5.0, 5, 20261105
_D, _LOGD, _S = XS[VARS.index("d")], XS[VARS.index("logd")], XS[VARS.index("S_ij")]


def _lawtest_ids():
    return set(config.cities_with_role("lawtest"))


def _confirm_ids():
    return set(config.CONFIRM) | set(config.cities_with_role("confirm"))


def covariates(cities, allow_confirm: bool = False) -> pd.DataFrame:
    """allow_confirm (plan 6): confirmation cities only after the freeze."""
    cities = [str(c) for c in cities]
    sealed = sorted(set(cities) & _confirm_ids())
    if allow_confirm:
        import frozen
        frozen.require()
    elif sealed:
        raise ValueError(f"confirmation cities {sealed} are sealed")
    f = pd.read_csv(RESULTS / "form_metrics.csv", dtype={"city": str}).set_index("city")
    f["log_population"] = np.log(f["population"].astype(float))
    f["log_jobs"] = np.log(f["jobs"].astype(float))
    out = f.reindex(cities)[list(COVARIATES)].reset_index()
    return out.rename(columns={"index": "city"})


def _param_names(c):
    names = []
    for i in range(c.n_lit):
        s = sympy.Symbol(f"c{i}")
        comp = next((k for k, e in c.comps.items() if s in e.free_symbols), "f")
        names.append(f"{comp}:c{i}")
    if "g" in c.comps:
        names += [f"p1:c{c.n_lit}", f"p2:c{c.n_lit + 1}"]
    return names


def _load(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return json.loads(v) if isinstance(v, str) else v


def param_table(cities_val: pd.DataFrame, pool: list) -> pd.DataFrame:
    by = {x["cid"]: x for x in pool}
    rows = []
    for r in cities_val[cities_val["finite"].astype(bool)].itertuples():
        cs, se, sd = _load(r.consts), _load(getattr(r, "consts_se", None)), _load(getattr(r, "consts_sd", None))
        if cs is None or r.cid not in by:
            continue
        names = _param_names(candidate({**by[r.cid], "consts_refit": None}))
        for i, v in enumerate(cs):
            s_ = se[i] if se and i < len(se) else None
            d_ = sd[i] if sd and i < len(sd) else 0.0
            ident = bool(s_ is not None and np.isfinite(s_) and abs(v) > 2 * s_ and (d_ or 0.0) < s_)
            rows.append({"cid": r.cid, "param": names[i] if i < len(names) else f"c{i}", "city": r.city,
                         "value": float(v), "se": s_, "sd_starts": d_, "identifiable": ident})
    return pd.DataFrame(rows, columns=["cid", "param", "city", "value", "se", "sd_starts", "identifiable"])


def reference_points(datas: dict) -> dict:
    """{city: {"d": REF_D_KM, "S_ij": median S_ij of the rows with d in [4, 6] km, "dlogS_dlogd": OLS slope of
    log(S_ij + 1) on log d over the rows, other variables: their medians on those rows}}."""
    out = {}
    for city, D in datas.items():
        X = D["tr"].X.astype(float)
        d = X[:, VARS.index("d")]
        near = (d >= 4) & (d <= 6)
        sel = X[near] if near.any() else X
        ref = {v: float(np.median(sel[:, k])) for k, v in enumerate(VARS)}
        ld, ls = np.log(np.maximum(d, 0.05)), np.log(X[:, VARS.index("S_ij")] + 1.0)
        ref.update(d=REF_D_KM, logd=float(np.log(REF_D_KM)), dlogS_dlogd=float(np.polyfit(ld, ls, 1)[0]))
        out[city] = ref
    return out


def _elasticity_expr(c):
    """d eta / d log d along S_ij(d): x1 d/dx1 + d/dx2 + slope * x26 d/dx26 (template parameters included)."""
    e = c.comps["f"]
    if "g" in c.comps:
        e = e + sympy.Symbol(f"c{c.n_lit}") * c.comps["g"] + sympy.Symbol(f"c{c.n_lit + 1}") * c.comps["h"]
    slope = sympy.Symbol("slope")
    return sympy.diff(e, _D) * _D + sympy.diff(e, _LOGD) + slope * _S * sympy.diff(e, _S)


def composites(cid: str, cities_val: pd.DataFrame, pool: list, ref_points: dict) -> pd.DataFrame:
    cols = ["cid", "param", "city", "value", "se", "sd_starts", "identifiable"]
    x = next(p for p in pool if p["cid"] == cid)
    c = parse_equation(x["equation"], x.get("template"), x.get("variant", "A"), int(x["complexity"]))
    from library import to_single
    c = to_single(c)
    syms = set().union(*[e.free_symbols for e in c.comps.values()])
    if not (_S in syms and syms & {_D, _LOGD}):
        return pd.DataFrame(columns=cols)
    ex = _elasticity_expr(c)
    n = len(c.consts)
    cs_ = [sympy.Symbol(f"c{i}") for i in range(n)]
    fn = sympy.lambdify(list(XS) + cs_ + [sympy.Symbol("slope")], ex, "numpy")
    rows = []
    for r in cities_val[(cities_val["cid"] == cid) & cities_val["finite"].astype(bool)].itertuples():
        cs, cov = _load(r.consts), _load(getattr(r, "consts_cov", None))
        ref = ref_points.get(r.city)
        if cs is None or ref is None or len(cs) != n:
            continue
        xs = [float(ref.get(v, 0.0)) for v in VARS]
        f = lambda th: float(fn(*xs, *th, float(ref["dlogS_dlogd"])))  # noqa: E731
        th = np.asarray(cs, float)
        v = f(th)
        se = None
        if cov is not None:
            C = np.array([[np.nan if a is None else a for a in row] for row in cov], float)
            g = np.array([(f(th + h) - f(th - h)) / (2 * h[i]) for i, h in
                          enumerate(np.diag(1e-6 * np.maximum(1.0, np.abs(th))))])
            q = float(g @ C @ g)
            se = float(np.sqrt(q)) if np.isfinite(q) and q >= 0 else None
        rows.append({"cid": cid, "param": f"elasticity_{REF_D_KM:g}km", "city": r.city, "value": v, "se": se,
                     "sd_starts": 0.0, "identifiable": bool(se is not None and abs(v) > 2 * se)})
    return pd.DataFrame(rows, columns=cols)


def _ols_slope(x, y):
    x = x - x.mean()
    den = float((x * x).sum())
    return float((x * (y - y.mean())).sum() / den) if den > 0 else np.nan


def cross_city_summary(pt: pd.DataFrame, cov: pd.DataFrame, B: int = 200, seed: int = SEED,
                       min_cities: int = MIN_CITIES) -> list:
    covi = cov.set_index("city")
    out = []
    for (cid, param), g in pt[pt["identifiable"]].groupby(["cid", "param"], sort=True):
        g = g[g["city"].isin(covi.index)]
        if len(g) < min_cities:
            continue
        for k in COVARIATES:
            z = covi.loc[g["city"], k].to_numpy(float)
            y = g["value"].to_numpy(float)
            ok = np.isfinite(z) & np.isfinite(y)
            if ok.sum() < min_cities:
                continue
            z, y = z[ok], y[ok]
            slope = _ols_slope(z, y)
            rng = np.random.default_rng([seed, zlib.crc32(f"{cid}|{param}|{k}".encode())])     # replayable
            bs = []
            for _ in range(B):
                i = rng.integers(0, len(z), len(z))
                bs.append(_ols_slope(z[i], y[i]))
            bs = np.array([b for b in bs if np.isfinite(b)])
            lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan))
            p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))) if len(bs) else 1.0
            out.append({"cid": cid, "param": param, "covariate": k, "slope": slope, "ci_lo": float(lo),
                        "ci_hi": float(hi), "p": p, "n": int(len(z))})
    return out


def term_existence(ablation_by_city: dict, refs: dict) -> pd.DataFrame:
    rows = []
    for (cid, city), terms in ablation_by_city.items():
        for t in terms:
            d = t.get("delta_nll")
            rows.append({"cid": cid, "comp": t.get("comp", "f"), "term": t.get("key", t["term"]), "city": city,
                         "delta_nll": d, "exists": bool(scoring.term_exists(d, refs[city]))})
    return pd.DataFrame(rows, columns=["cid", "comp", "term", "city", "delta_nll", "exists"])


# METHODS (glosses):
#   slope_ci         univariate regression slope of the parameter (or composite) on the city covariate; city-bootstrap
#                    95% CI excludes 0 and the sign agrees with direction
#   group_diff       cities split at the covariate median; bootstrap 95% CI of the difference in parameter means
#                    excludes 0 and the sign agrees with direction
#   term_exist_prop  share of upper-half cities in which a term "exists" minus the share in the lower half >= 0.3
#                    (direction +; <= -0.3 for -)
METHODS = {"slope_ci": "参数（或合成量）对城市特征的单变量回归斜率，城市 bootstrap 95% CI 不含 0 且符号与 direction 一致",
           "group_diff": "按城市特征中位数分上下两半，参数均值之差的 bootstrap 95% CI 不含 0 且符号与 direction 一致",
           "term_exist_prop": "某项在城市特征上半城市中'存在'的比例减下半城市的比例 ≥ 0.3（direction 为 + 时；- 时 ≤ −0.3）"}


def analysis_payload(summary: list, existence: pd.DataFrame, pt: pd.DataFrame, pool: list, registered=None,
                     results=None) -> dict:
    """registered / results: hypotheses already registered and their lawtest verdicts — only pass / fail reaches the
    agent (final review #5; design §5.4 step 7)."""
    sealed = _lawtest_ids() | _confirm_ids()
    cities = set(pt["city"]) | set(existence["city"]) if len(existence) else set(pt["city"])
    bad = sorted(cities & sealed)
    if bad:
        raise ValueError(f"lawtest / confirmation cities {bad} in the analysis input")
    by = {x["cid"]: x for x in pool}
    ident = {}
    for (cid, param), g in pt[pt["identifiable"]].groupby(["cid", "param"]):
        ident.setdefault(cid, []).append({"param": param, "n_cities": int(len(g)), "median": round(float(g.value.median()), 5),
                                          "min": round(float(g.value.min()), 5), "max": round(float(g.value.max()), 5)})
    ex = []
    if len(existence):
        for (cid, comp, term), g in existence.groupby(["cid", "comp", "term"]):
            ex.append({"cid": cid, "comp": comp, "term": term, "n_exists": int(g["exists"].sum()), "n_cities": int(len(g)),
                       "exists_in": sorted(g.loc[g["exists"], "city"])})
    skel = {}
    for cid in sorted(set(pt["cid"]) | {e["cid"] for e in ex}):
        if cid in by:
            c = candidate({**by[cid], "consts_refit": None})
            skel[cid] = str(c.comps["f"]) + (f" + c{c.n_lit}*({c.comps['g']}) + c{c.n_lit + 1}*({c.comps['h']})"
                                             if "g" in c.comps else "")
    p = {"n_cities": int(len(cities)), "skeletons": skel, "covariates": dict(COV_MEANING), "methods": dict(METHODS),
         "identifiable": ident,
         "regressions": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()} for s in summary],
         "term_existence": ex,
         "registered": [{"id": h["id"], **{k: h[k] for k in ("skeleton", "parameter", "covariate", "direction", "test")},
                         "pass": (bool((results or {})[h["id"]]["pass"]) if h["id"] in (results or {}) else None)}
                        for h in (registered or [])]}
    llm.check_input(p)
    return p


# ---- Task 12: hypothesis registration and tests on the lawtest cities (design §5.4 steps 6-7) ----
import hashlib  # noqa: E402
import time  # noqa: E402

TESTS = ("slope_ci", "group_diff", "term_exist_prop")
FIELDS = ("statement", "skeleton", "parameter", "covariate", "direction", "test", "criterion")
HYPS_PER_CYCLE, EXIST_PROP_MIN, ALPHA = 5, 0.3, 0.05
TEST_B = 5000          # final review #6: p resolution 1 / B well below the smallest Holm threshold (0.05 / 10)


def _sign(d):
    # also accepts the Chinese words an agent may write for positive / increase / rise and negative / decrease / fall
    d = str(d).strip().lower()
    if d in ("+", "positive", "pos", "increase", "正", "增大", "上升"):
        return 1
    if d in ("-", "−", "negative", "neg", "decrease", "负", "减小", "下降"):
        return -1
    return None


def _hsha(h):
    body = {k: h[k] for k in FIELDS}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


def _laws_dir(out_dir):
    d = Path(out_dir) if out_dir else RESULTS / "laws"
    d.mkdir(parents=True, exist_ok=True)
    return d


def register(hyps: list, cycle: int, pt: pd.DataFrame, registered: list, terms: set | None = None, out_dir=None,
             log=None, cap: int = HYPS_PER_CYCLE) -> tuple:
    """(accepted, rejected with reason). Accepted ones get id H<cycle>-<k>, a content sha, and are appended to
    laws/registered.json and the ledger; nothing registered is ever changed."""
    if log is None:
        import ledger
        log = ledger.add
    d = _laws_dir(out_dir)
    pairs = {(h["skeleton"], h["parameter"], h["covariate"]) for h in registered}
    n_cycle = sum(int(h.get("cycle", 0)) == cycle for h in registered)
    ident = pt[pt["identifiable"]].groupby(["cid", "param"]).size() if len(pt) else pd.Series(dtype=int)
    acc, rej = [], []
    for h in hyps:
        miss = [f for f in FIELDS if not str(h.get(f, "") or "").strip()]
        key = (h.get("skeleton"), h.get("parameter"), h.get("covariate"))
        reason = None
        if miss:
            reason = f"missing fields {miss}"
        elif h["test"] not in TESTS:
            reason = "unknown test"
        elif h["covariate"] not in COVARIATES:
            reason = "unknown covariate"
        elif _sign(h["direction"]) is None:
            reason = "bad direction"
        elif h["test"] != "term_exist_prop" and int(ident.get((h["skeleton"], h["parameter"]), 0)) < MIN_CITIES:
            reason = "parameter not identifiable"
        elif h["test"] == "term_exist_prop" and terms is not None and (h["skeleton"], h["parameter"]) not in terms:
            reason = "unknown term"
        elif key in pairs or key in {(a["skeleton"], a["parameter"], a["covariate"]) for a in acc}:
            reason = "pair already registered"
        elif n_cycle + len(acc) >= cap:
            reason = "cycle cap reached"
        if reason:
            rej.append({**h, "reason": reason})
            continue
        x = {k: h[k] for k in FIELDS}
        x.update(id=f"H{cycle}-{n_cycle + len(acc) + 1:02d}", cycle=cycle, sha=_hsha(h),
                 registered_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        acc.append(x)
    p = d / "registered.json"
    p.write_text(json.dumps(list(registered) + acc, ensure_ascii=False, indent=1))
    if acc or rej:
        import ledger
        log(f"Plan 4 cycle {cycle} hypotheses: registered "
            + ("; ".join(f"{a['id']} sha {a['sha']} [{a['skeleton']} {a['parameter']} ~ {a['covariate']} "
                         f"({a['direction']}) by {a['test']}]" for a in acc) or "none")
            + f"; refused {[(r.get('skeleton'), r.get('parameter'), r.get('covariate'), r['reason']) for r in rej]}"
            + f"; registered.json sha {ledger.sha(p)}")
    return acc, rej


def holm(pvals: list, alpha: float = ALPHA) -> list:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [False] * m
    for k, i in enumerate(order):
        if pvals[i] <= alpha / (m - k):
            out[i] = True
        else:
            break
    return out


def _boot(fn, n, B, rng):
    vals = []
    for _ in range(B):
        v = fn(rng.integers(0, n, n))
        if np.isfinite(v):
            vals.append(v)
    return np.array(vals)


def run_test(h: dict, pt: pd.DataFrame, cov: pd.DataFrame, existence, B: int = TEST_B, seed: int = SEED) -> dict:
    """One registered hypothesis on the given (lawtest) cities: stat, two-sided bootstrap p (permutation p for
    term_exist_prop, one-sided in the stated direction), pass_raw = the registered criterion of the method."""
    sgn = _sign(h["direction"])
    covi = cov.set_index("city")
    rng = np.random.default_rng([seed, zlib.crc32(f"{h['skeleton']}|{h['parameter']}|{h['covariate']}".encode())])
    out = {"stat": None, "p": 1.0, "pass_raw": False, "n": 0, "ci_lo": None, "ci_hi": None, "B": B}
    if h["test"] == "term_exist_prop":
        e = existence[(existence["cid"] == h["skeleton"]) & (existence["term"] == h["parameter"])] if existence is not None else []
        if len(e) == 0:
            return {**out, "note": "term not ablated in these cities"}
        e = e[e["city"].isin(covi.index)]
        z = covi.loc[e["city"], h["covariate"]].to_numpy(float)
        x = e["exists"].to_numpy(bool).astype(float)
        ok = np.isfinite(z)
        z, x = z[ok], x[ok]
        if len(z) < MIN_CITIES:
            return {**out, "n": int(len(z)), "note": "too few cities"}
        up = z > np.median(z)
        stat = float(x[up].mean() - x[~up].mean()) if up.any() and (~up).any() else np.nan
        perm = np.array([(lambda u: x[u].mean() - x[~u].mean())(rng.permutation(up)) for _ in range(B)])
        p = float((1 + np.sum(sgn * perm >= sgn * stat)) / (B + 1))
        return {**out, "stat": stat, "p": p, "n": int(len(z)), "pass_raw": bool(sgn * stat >= EXIST_PROP_MIN)}
    g = pt[(pt["cid"] == h["skeleton"]) & (pt["param"] == h["parameter"]) & pt["identifiable"]]
    g = g[g["city"].isin(covi.index)]
    z = covi.loc[g["city"], h["covariate"]].to_numpy(float)
    y = g["value"].to_numpy(float)
    ok = np.isfinite(z) & np.isfinite(y)
    z, y = z[ok], y[ok]
    if len(z) < MIN_CITIES:
        return {**out, "n": int(len(z)), "note": "too few identifiable cities"}
    if h["test"] == "slope_ci":
        stat = _ols_slope(z, y)
        bs = _boot(lambda i: _ols_slope(z[i], y[i]), len(z), B, rng)
    else:                                                                  # group_diff
        up = z > np.median(z)
        if not up.any() or up.all():
            return {**out, "n": int(len(z)), "note": "covariate does not split the cities"}
        a, b = y[up], y[~up]
        stat = float(a.mean() - b.mean())
        bs = np.array([a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
                       for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    ok_dir = lo > 0 if sgn > 0 else hi < 0
    return {**out, "stat": float(stat), "p": p, "n": int(len(z)), "ci_lo": float(lo), "ci_hi": float(hi),
            "pass_raw": bool(ok_dir)}


def test_registered(registered: list, pt: pd.DataFrame, cov: pd.DataFrame, existence, previous: dict | None = None,
                    out_dir=None, log=None, B: int = TEST_B) -> list:
    """Every registered hypothesis tested once (stored results reused), Holm over the cumulative count m; writes
    laws/results.json and a ledger line. The agent gets agent_view() only (pass / fail)."""
    if log is None:
        import ledger
        log = ledger.add
    d = _laws_dir(out_dir)
    if previous is None:
        f = d / "results.json"
        previous = {r["id"]: r for r in json.loads(f.read_text())} if f.exists() else {}
    res = []
    for h in registered:
        r = dict(previous[h["id"]]) if h["id"] in previous else run_test(h, pt, cov, existence, B=B)
        r.update(id=h["id"], test=h["test"], skeleton=h["skeleton"], parameter=h["parameter"],
                 covariate=h["covariate"], direction=h["direction"])
        res.append(r)
    ok = holm([float(r["p"]) for r in res])
    for r, k in zip(res, ok):
        r.update(m=len(res), holm=bool(k), **{"pass": bool(k and r["pass_raw"])})
        r.setdefault("pass_at_cycle", r["pass"])
    p = d / "results.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    import ledger
    log(f"Plan 4 lawtest results (Holm over m = {len(res)}): "
        + "; ".join(f"{r['id']} {'PASS' if r['pass'] else 'fail'} (p {r['p']:.4f}, raw {r['pass_raw']})" for r in res)
        + f"; results.json sha {ledger.sha(p)}")
    return res


def agent_view(results: list) -> dict:
    return {r["id"]: bool(r["pass"]) for r in results}


# ---- Task 13: parameterised skeletons (design §5.4 step 8) ----
from scipy.optimize import minimize  # noqa: E402


def parameterize(x: dict, param: str, covariate: str) -> dict:
    return {**x, "law": {"param": param, "covariate": covariate}}


def _param_index(param):
    return int(param.split(":c")[-1])


def _zscores(cov, cities, covariate):
    z = cov.set_index("city").loc[list(cities), covariate].astype(float)
    m, s = float(z.mean()), float(z.std(ddof=0))
    s = s if s > 0 else 1.0
    return {c: float((z[c] - m) / s) for c in cities}, m, s


def joint_refit(lx: dict, datas: dict, cov: pd.DataFrame, refs: dict | None = None, starts: int = 3,
                seed: int = SEED) -> dict:
    """All cities share the constants; the law's parameter is b0 + b1 z_city (z = the covariate standardised over these
    cities); objective = sum over cities of the per-visit training NLL (refit_select._obj gradients, L-BFGS-B, multi-
    start). Scored per city on datas[city]["va"]."""
    from refit_select import _obj, loss
    from library import to_single, PERTURB_SD
    c = to_single(candidate(lx))
    i = _param_index(lx["law"]["param"])
    cities = list(datas)
    z, zm, zs = _zscores(cov, cities, lx["law"]["covariate"])
    base = np.asarray(c.consts, float)
    n = len(base)

    def unpack(th):
        shared, b0, b1 = th[:n], th[n], th[n + 1]
        out = {}
        for city in cities:
            v = shared.copy()
            v[i] = b0 + b1 * z[city]
            out[city] = v
        return out

    def obj(th):
        tot, g = 0.0, np.zeros(n + 2)
        for city, v in unpack(th).items():
            f, gc = _obj(v, c, datas[city]["tr"])
            tot += f
            g[:n] += gc
            g[n] += gc[i]
            g[n + 1] += gc[i] * z[city]
        g[i] = 0.0                                   # the shared slot of the law parameter is unused
        return tot, g
    rng = np.random.default_rng(seed)
    best = None
    for k in range(starts):
        th0 = np.r_[base, base[i], 0.0]
        if k:
            th0[:n + 1] = th0[:n + 1] * np.exp(rng.normal(0.0, PERTURB_SD, n + 1))
        r = minimize(obj, th0, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-9, "ftol": 1e-13})
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    th = best.x
    per = {}
    for city, v in unpack(th).items():
        cc = dataclasses.replace(c, consts=np.asarray(v, float))
        nll = loss(cc, datas[city]["va"])
        nll = float(nll) if np.isfinite(nll) and nll < 1e8 else None
        per[city] = {"nll": nll, "share_s0": scoring.share(nll, refs[city], "S0") if refs and city in refs else None,
                     "param": float(v[i])}
    return {"cid": lx["cid"], "param": lx["law"]["param"], "covariate": lx["law"]["covariate"],
            "consts_shared": [float(v) for v in th[:n]], "b0": float(th[n]), "b1": float(th[n + 1]),
            "z": z, "z_mean": zm, "z_sd": zs, "per_city": per, "n_params": n + 1, "objective": float(best.fun)}


def _named(e):
    """sympy expression in x1..x26 -> seed string with variable names, nexp for exp(-z), ^ for powers."""
    e = e.replace(lambda a: isinstance(a, sympy.exp), lambda a: sympy.Function("nexp")(-a.args[0]))
    e = e.xreplace({XS[k]: sympy.Symbol(v) for k, v in enumerate(VARS)})
    return str(e).replace("**", "^")


def substituted_seeds(lx: dict, jr: dict, cov: pd.DataFrame, cities: list) -> dict:
    """{city: [{"f": eta with the law's parameter at b0 + b1 z_city (z on the joint fit's scale), "g": "1.0",
    "h": "1.0"}]} for the cities' search runs (design §5.4 step 8: the substituted form seeds the city's run)."""
    from library import to_single
    c = to_single(candidate(lx))
    i = _param_index(jr["param"])
    zc = cov.set_index("city").loc[list(cities), jr["covariate"]].astype(float)
    out = {}
    for city in cities:
        v = np.asarray(jr["consts_shared"], float).copy()
        v[i] = jr["b0"] + jr["b1"] * float((zc[city] - jr["z_mean"]) / jr["z_sd"])
        lits = {sympy.Symbol(f"c{k}"): sympy.Float(float(x)) for k, x in enumerate(v)}
        e = c.comps["f"]
        if "g" in c.comps:
            e = e + sympy.Symbol(f"c{c.n_lit}") * c.comps["g"] + sympy.Symbol(f"c{c.n_lit + 1}") * c.comps["h"]
        out[city] = [{"f": _named(e.xreplace(lits)), "g": "1.0", "h": "1.0", "law": f"{jr['cid']} {jr['param']} ~ "
                      f"{jr['covariate']}"}]
    return out
