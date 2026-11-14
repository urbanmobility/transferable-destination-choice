"""Exploration of form effects beyond size (design and plan of 2026-11-10):
how much city-form signal beyond size is left, and how large the effects are. Exploratory only - nothing here is a
test that "passes"; E-1's permutation test is the one pre-specified exploratory global test.

- Methods are fixed in config/explore_form.json before any quantity is computed: `fix` records its sha in
  results/multicity/explore/CONFIG_FIXED; every later stage refuses a config whose sha differs from the last fixed one;
  a change needs `fix --reason TEXT` (ledgered).
- Sample: the 67 cities (discovery 35, law-test 26, confirmation 6), test_dest block, the 30 plan-7 outcomes and the
  17 covariates; E-1 uses discovery -> law-test only.
- E-1: discovery-significant pairs beyond size, their standardized partial slopes (control log_population) on the
  law-test cities, sign agreement, shrinkage ratio; a permutation of the size-residualized covariates (rows permuted
  jointly, size fit kept) gives the null of the agreement rate.

- E-2: on all 67 cities, every outcome x covariate standardized partial slope with a city-bootstrap CI;
  Benjamini-Hochberg (q from the config) and normal-normal empirical-Bayes shrinkage (prior variance per outcome) over the
  form cells = the cells beyond size without the confounders (tract_area, the measurement unit: kept in the full table
  with role "confounder", never in BH, shrinkage or the ranking). BH takes the normal-approximation p
  2*Phi(-|estimate / bootstrap SE|): the bootstrap p cannot go below 1/B, which is above the rank-1 BH threshold q/m
  for a few hundred cells, so an isolated strong cell could never pass; the intervals stay bootstrap percentile ones.
  A tract-area-controlled sensitivity; principal "form axes" of the size-residualized form covariates and each
  outcome on size + axes.
- axes_ta (added 2026-11-11, after the config was first fixed, so it needed fix --reason): each outcome
  on size + axes + tract_area, written to its own file e2_axis_effects_ta.csv; the E-2 files are not rewritten.

Usage (run_stage.sh, M line): explore_form.py fix [--reason TEXT] | e1 | e2 | axes_ta | report
"""
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import explore_p7 as ex  # noqa: E402
import laws  # noqa: E402

CONFIG = Path(__file__).resolve().parent / "config" / "explore_form.json"
OUT = config.results_dir("M") / "explore"
FIXED = OUT / "CONFIG_FIXED"
GROUP_FILES = {"discovery": ("city_outcomes.parquet", "covariates.csv"),
               "lawtest": ("city_outcomes_lawtest.parquet", "covariates_lawtest.csv"),
               "confirm": ("city_outcomes_confirm.parquet", "covariates_confirm.csv")}


# ------------------------------------------------------------------------------------------- methods fixed first
def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _entries() -> list:
    return json.loads(FIXED.read_text()) if FIXED.exists() else []


def fix_config(reason: str = None, log=print) -> str:
    """Record the config's sha (first time, or a change with a reason); the same sha again is a no-op."""
    sha, ent = _sha(CONFIG), _entries()
    if ent and ent[-1]["sha"] == sha:
        return sha
    if ent and not reason:
        raise ValueError("the exploration config changed after it was fixed: give a reason (fix --reason TEXT)")
    ent.append({"sha": sha, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reason": reason})
    FIXED.parent.mkdir(parents=True, exist_ok=True)
    FIXED.write_text(json.dumps(ent, ensure_ascii=False, indent=1))
    log(f"Exploration config {'fixed' if len(ent) == 1 else 'amended'}: {CONFIG.name} sha {sha}"
        + (f"; reason: {reason}" if reason else ""))
    return sha


def require_fixed() -> dict:
    ent = _entries()
    if not ent or ent[-1]["sha"] != _sha(CONFIG):
        raise RuntimeError(f"{CONFIG.name} is not fixed (or changed after fixing): run explore_form.py fix first")
    return json.loads(CONFIG.read_text())


# --------------------------------------------------------------------------------------------- the 67-city frame
def group_cities(group: str) -> list:
    if group == "discovery":
        return [str(c) for c in config.cities_with_role("search") + config.cities_with_role("discovery")]
    if group in ("lawtest", "confirm"):
        return [str(c) for c in config.cities_with_role(group)]
    raise ValueError(f"city group {group!r}: one of {tuple(GROUP_FILES)}")


def load_frame(groups=("discovery", "lawtest", "confirm"), root=None) -> tuple:
    """(outcomes, covariates, {city: group}) of the groups' role cities: test_dest rows of each group's outcome table
    (without block / log_population) and laws.covariates left-joined with the groups' plan-7 covariate files.
    ValueError when a city is in two groups or lacks its outcome or covariate row."""
    root = Path(root) if root is not None else config.results_dir("M") / "p7"
    outs, covs, group_of = [], [], {}
    for g in groups:
        cs = group_cities(g)
        for c in cs:
            if c in group_of:
                raise ValueError(f"city {c} is in groups {group_of[c]} and {g}")
            group_of[c] = g
        fo, fc = GROUP_FILES[g]
        o = pd.read_parquet(root / fo)
        o = o.assign(city=o["city"].astype(str))
        o = o[(o["block"] == "test_dest") & o["city"].isin(cs)]
        c_ = pd.read_csv(root / fc, dtype={"city": str})
        c_ = c_[c_["city"].isin(cs)]
        for what, have in (("outcome", set(o["city"])), ("covariate", set(c_["city"]))):
            miss = sorted(set(cs) - have)
            if miss:
                raise ValueError(f"group {g}: no {what} row for {miss[:5]}")
        outs.append(o)
        covs.append(c_)
    out = pd.concat(outs, ignore_index=True).drop(columns=["block", "log_population"], errors="ignore")
    cities = list(group_of)
    cov = laws.covariates(cities, allow_confirm=True).merge(pd.concat(covs, ignore_index=True), on="city", how="left")
    if not (out["city"].is_unique and cov["city"].is_unique):
        raise ValueError("one row per city is required")
    return out.reset_index(drop=True), cov.reset_index(drop=True), group_of


# --------------------------------------------------------------------------------------------- slopes
def _z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    sd = a.std(ddof=1)
    return (a - a.mean()) / sd if sd > 0 else a * np.nan


def _fw_slope(y, x, Z) -> float:
    """Partial slope of y on x given the columns of Z (Frisch-Waugh, with an intercept)."""
    A = np.column_stack([np.ones(len(y)), Z])
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    sxx = float(rx @ rx)
    return float(rx @ ry) / sxx if sxx > 1e-12 else np.nan


def _pair_arrays(d: pd.DataFrame, o: str, c: str, controls) -> tuple:
    ctrl = [k for k in controls if k != c]
    sub = ex._num(d[[o, c] + ctrl]).replace([np.inf, -np.inf], np.nan).dropna()
    return _z(sub[o].to_numpy()), _z(sub[c].to_numpy()), sub[ctrl].to_numpy(float), len(sub)


def _slope_of(y, x, Z) -> float:
    if Z.shape[1] == 0:
        return ex._slope(y, x)
    if Z.shape[1] == 1:
        return ex._slope(y, x, Z[:, 0])
    return _fw_slope(y, x, Z)


def std_partial_slopes(out: pd.DataFrame, cov: pd.DataFrame, pairs, controls=("log_population",)) -> np.ndarray:
    """Standardized partial slope (y and x z-scored on the pair's complete rows; controls as they are) of each
    (outcome, covariate) pair; a covariate that is itself a control gets the plain standardized slope; NaN with
    fewer than 5 complete cities."""
    d = out.merge(cov, on="city", how="inner", suffixes=("", "_cov"))
    res = []
    for o, c in pairs:
        y, x, Z, n = _pair_arrays(d, o, c, controls)
        res.append(_slope_of(y, x, Z) if n >= 5 else np.nan)
    return np.array(res, float)


# --------------------------------------------------------------------------------------------- E-1
def _sig(r, size: str) -> bool:
    lo, hi = (r["ci_lo"], r["ci_hi"]) if r["covariate"] == size else (r["pci_lo"], r["pci_hi"])
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def e1_pairs(rel: pd.DataFrame, registered: list, cfg: dict, variant: str = "main") -> pd.DataFrame:
    """Discovery-significant pairs of relations.csv (rel order) for an E-1 variant: main drops the size covariates,
    the confounders (tract_area), size proxies (|r_size| > size_proxy_r) and the registered pairs; with_registered
    keeps the registered pairs; with_confounders keeps the confounders and size proxies."""
    e1, size = cfg["e1"], cfg["size"]
    if variant not in e1["variants"]:
        raise ValueError(f"E-1 variant {variant!r}: one of {e1['variants']}")
    conf = set(e1["confounders"])
    always = set(e1["exclude_covariates"]) - conf
    reg = {(h["outcome"], h["covariate"]) for h in registered}
    rows = []
    for _, r in rel.iterrows():
        if not _sig(r, size) or r["covariate"] in always:
            continue
        confounded = r["covariate"] in conf or abs(float(r["r_size"])) > e1["size_proxy_r"]
        if confounded and variant != "with_confounders":
            continue
        if (r["outcome"], r["covariate"]) in reg and variant != "with_registered":
            continue
        rows.append({"outcome": r["outcome"], "covariate": r["covariate"], "d_slope": float(r["partial_slope"])})
    return pd.DataFrame(rows, columns=["outcome", "covariate", "d_slope"])


def _agree(d, l) -> float:
    ok = np.isfinite(d) & np.isfinite(l)
    return float(np.mean(np.sign(d[ok]) == np.sign(l[ok]))) if ok.any() else np.nan


def e1_summary(d_std, l_std, covariates) -> dict:
    d, l = np.asarray(d_std, float), np.asarray(l_std, float)
    ok = np.isfinite(d) & np.isfinite(l) & (d != 0)

    def block(m):
        dd, ll = d[m & ok], l[m & ok]
        return {"n": int(len(dd)), "agree": _agree(dd, ll),
                "median_ratio": float(np.median(ll / dd)) if len(dd) else np.nan}

    allm = np.ones(len(d), bool)
    out = {"n_pairs": int(ok.sum()), "agree": _agree(d[ok], l[ok]), "median_ratio": block(allm)["median_ratio"],
           "pearson": float(np.corrcoef(d[ok], l[ok])[0, 1]) if ok.sum() > 2 else np.nan,
           "spearman": float(pd.Series(d[ok]).corr(pd.Series(l[ok]), method="spearman")) if ok.sum() > 2 else np.nan}
    cv = np.asarray(covariates)
    out["by_covariate"] = {c: block(cv == c) for c in dict.fromkeys(covariates)}
    return out


def freedman_lane(cov: pd.DataFrame, cols, control: str, perm) -> pd.DataFrame:
    """Each col = its OLS fit on control (with intercept) + its residuals permuted by `perm` (city rows moved
    jointly for all cols); the size fit is kept, the association beyond size is broken. Other columns unchanged."""
    out = cov.copy()
    z = cov[control].to_numpy(float)
    perm = np.asarray(perm)
    for c in cols:
        x = cov[c].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(z)
        b, a = np.polyfit(z[ok], x[ok], 1)
        fit = a + b * z
        res = x - fit
        out[c] = fit + res[perm]
    return out


def e1_permutation(out, cov, pairs, d_signs, B: int, seed: int, control: str = "log_population") -> tuple:
    """(observed agreement, null agreements, one-sided p) of the law-test signs with the discovery signs."""
    d_signs = np.sign(np.asarray(d_signs, float))
    obs_s = std_partial_slopes(out, cov, pairs, (control,))
    ok = np.isfinite(obs_s) & (d_signs != 0)
    obs = float(np.mean(np.sign(obs_s[ok]) == d_signs[ok]))
    cols = sorted({c for _, c in pairs} - {control})
    rng = np.random.default_rng(seed)
    null = np.empty(B)
    for b in range(B):
        s = std_partial_slopes(out, freedman_lane(cov, cols, control, rng.permutation(len(cov))), pairs, (control,))
        null[b] = np.mean(np.sign(s[ok]) == d_signs[ok])
    return obs, null, float((1 + (null >= obs).sum()) / (B + 1))


def stage_e1() -> dict:
    import ledger
    import p7seal
    cfg = require_fixed()
    rel = pd.read_csv(config.results_dir("M") / "p7" / "relations.csv")
    reg = p7seal.require_signed()
    od, cd, _ = load_frame(("discovery",))
    ol, cl, _ = load_frame(("lawtest",))
    rows, summ = [], {}
    for v in cfg["e1"]["variants"]:
        pr = e1_pairs(rel, reg, cfg, v)
        pairs = list(zip(pr["outcome"], pr["covariate"]))
        d = std_partial_slopes(od, cd, pairs, (cfg["size"],))
        l = std_partial_slopes(ol, cl, pairs, (cfg["size"],))
        summ[v] = e1_summary(d, l, pr["covariate"].tolist())
        if v == "main":
            obs, null, p = e1_permutation(ol, cl, pairs, d, cfg["e1"]["permutations"], cfg["seed"], cfg["size"])
            summ[v].update(perm_obs=obs, perm_p=p, perm_B=int(cfg["e1"]["permutations"]),
                           perm_null_mean=float(null.mean()), perm_null_q95=float(np.quantile(null, 0.95)))
        rows += [{"variant": v, "outcome": o, "covariate": c, "d_std": a, "l_std": b,
                  "agree": bool(np.sign(a) == np.sign(b)) if np.isfinite(a) and np.isfinite(b) else None}
                 for (o, c), a, b in zip(pairs, d, l)]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "e1_pairs.csv", index=False)
    p = OUT / "e1_summary.json"
    p.write_text(json.dumps({"config_sha": _sha(CONFIG), **summ}, ensure_ascii=False, indent=1, default=float))
    m = summ["main"]
    ledger.add(f"Exploration E-1: main {m['n_pairs']} pairs, sign agreement {m['agree']:.2f} (permutation null mean "
               f"{m['perm_null_mean']:.2f}, one-sided p {m['perm_p']:.4f}, B {m['perm_B']}), median ratio "
               f"{m['median_ratio']:.2f}; with_registered {summ['with_registered']['agree']:.2f} "
               f"({summ['with_registered']['n_pairs']}); with_confounders {summ['with_confounders']['agree']:.2f} "
               f"({summ['with_confounders']['n_pairs']}); {p.name} sha {ledger.sha(p)}")
    return summ


# --------------------------------------------------------------------------------------------- E-2
def _boot(y, x, Z, B: int, rng) -> np.ndarray:
    idx = rng.integers(0, len(y), (B, len(y)))
    if Z.shape[1] <= 1:
        return ex._boot_slopes(y, x, Z[:, 0] if Z.shape[1] else None, idx)
    return np.array([_fw_slope(y[i], x[i], Z[i]) for i in idx])


def effects(out, cov, outcomes, covariates, controls=("log_population",), B: int = 2000, seed: int = 20261110) -> pd.DataFrame:
    """Every outcome x covariate: standardized partial slope (plain when the covariate is a control), bootstrap SE,
    percentile CI, two-sided bootstrap p (explore_p7._ci_p, never below 1/B) and the normal-approximation p
    p_norm = 2*Phi(-|slope / SE|); draws per pair over its complete cities, seeded by the pair."""
    from scipy.stats import norm
    d = out.merge(cov, on="city", how="inner", suffixes=("", "_cov"))
    rows = []
    for o in outcomes:
        for c in covariates:
            r = {"outcome": o, "covariate": c, "n": 0, "std_slope": np.nan, "se": np.nan, "ci_lo": np.nan,
                 "ci_hi": np.nan, "p": np.nan, "p_norm": np.nan}
            if o in d and c in d:
                y, x, Z, n = _pair_arrays(d, o, c, controls)
                r["n"] = int(n)
                if n >= 5 and np.isfinite(y).all() and np.isfinite(x).all():
                    rng = np.random.default_rng([seed, zlib.crc32(f"{o}|{c}|{','.join(controls)}".encode())])
                    b = _boot(y, x, Z, B, rng)
                    lo, hi, p = ex._ci_p(b)
                    est, se = _slope_of(y, x, Z), float(np.nanstd(b, ddof=1))
                    r.update(std_slope=est, se=se, ci_lo=lo, ci_hi=hi, p=p,
                             p_norm=float(2 * norm.sf(abs(est / se))) if np.isfinite(est) and se > 0 else np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def bh(p, q: float) -> np.ndarray:
    """Benjamini-Hochberg step-up at level q; NaN p values never pass."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    if not ok.any():
        return out
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    passed = pv[order] <= q * np.arange(1, m + 1) / m
    k = np.nonzero(passed)[0].max() + 1 if passed.any() else 0
    sel = np.zeros(m, bool)
    sel[order[:k]] = True
    out[np.nonzero(ok)[0]] = sel
    return out


def eb_shrink(est, se, groups) -> np.ndarray:
    """Normal-normal empirical Bayes with prior mean 0: per group tau2 = max(0, mean(est^2) - mean(se^2)), shrunk =
    est * tau2 / (tau2 + se^2); se = 0 keeps the estimate; NaN stays NaN."""
    est, se, groups = np.asarray(est, float), np.asarray(se, float), np.asarray(groups)
    out = np.full(len(est), np.nan)
    for g in dict.fromkeys(groups.tolist()):
        m = (groups == g) & np.isfinite(est) & np.isfinite(se)
        if not m.any():
            continue
        tau2 = max(0.0, float(np.mean(est[m] ** 2) - np.mean(se[m] ** 2)))
        den = tau2 + se[m] ** 2
        out[m] = np.where(se[m] == 0, est[m], est[m] * tau2 / np.where(den > 0, den, 1.0))
    return out


def form_axes(cov, cols, control: str, var_target: float, max_axes: int) -> tuple:
    """Principal axes of the size-residualized, standardized cols (cities with any missing value left out):
    (scores indexed by city with columns axis1..axisk, loadings (cols x axes), explained-variance ratios of all
    components); k = first count reaching var_target, at most max_axes; largest |loading| of each axis positive."""
    d = cov[["city", control] + list(cols)].dropna()
    z = d[control].to_numpy(float)
    R = []
    for c in cols:
        x = d[c].to_numpy(float)
        b, a = np.polyfit(z, x, 1)
        R.append(_z(x - (a + b * z)))
    Z = np.column_stack(R)
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    expl = S ** 2 / (S ** 2).sum()
    k = min(int(np.searchsorted(np.cumsum(expl), var_target - 1e-12) + 1), max_axes)
    names = [f"axis{i + 1}" for i in range(k)]
    L = Vt[:k].T.copy()
    for j in range(k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    load = pd.DataFrame(L, index=list(cols), columns=names)
    scores = pd.DataFrame((Z - Z.mean(0)) @ L, index=d["city"].tolist(), columns=names)
    return scores, load, expl


def axis_effects(out, scores, cov, outcomes, B: int, seed: int, size: str = "log_population", extra=()) -> pd.DataFrame:
    """Each standardized outcome on [standardized size + the axes + the standardized extra controls] (OLS with
    intercept), city-bootstrap percentile CIs."""
    d = out.merge(cov[["city", size] + list(extra)], on="city").merge(scores, left_on="city", right_index=True)
    terms = [size] + list(scores.columns) + list(extra)
    rows = []
    for o in outcomes:
        if o not in d:
            continue
        sub = ex._num(d[[o] + terms]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < len(terms) + 3:
            continue
        y = _z(sub[o].to_numpy())
        X = np.column_stack([np.ones(len(sub))] + [_z(sub[t].to_numpy()) for t in terms])
        coef = np.linalg.lstsq(X, y, rcond=None)[0][1:]
        rng = np.random.default_rng([seed, zlib.crc32(o.encode())])
        boot = np.array([np.linalg.lstsq(X[i], y[i], rcond=None)[0][1:]
                         for i in rng.integers(0, len(y), (B, len(y)))])
        for j, t in enumerate(terms):
            lo, hi = np.nanpercentile(boot[:, j], [2.5, 97.5])
            rows.append({"outcome": o, "term": t, "coef": float(coef[j]), "ci_lo": float(lo), "ci_hi": float(hi), "n": len(sub)})
    return pd.DataFrame(rows)


def stage_e2() -> pd.DataFrame:
    import ledger
    import p7seal
    cfg = require_fixed()
    e2, size, seed = cfg["e2"], cfg["size"], cfg["seed"]
    out, cov, _ = load_frame(tuple(cfg["groups"]))
    e = effects(out, cov, cfg["outcomes"], cfg["covariates"], (size,), e2["bootstrap"], seed)
    if e2["fdr_p"] != "normal":
        raise ValueError(f"e2.fdr_p {e2['fdr_p']!r}: only 'normal' (2*Phi(-|estimate / bootstrap SE|))")
    e["role"] = np.where(e["covariate"].isin(e2["family_exclude"]), "size",
                         np.where(e["covariate"].isin(e2["confounders"]), "confounder", "form"))
    fam = e["role"] == "form"
    e["bh"] = False
    e.loc[fam, "bh"] = bh(e.loc[fam, "p_norm"], e2["fdr_q"])
    e["eb"] = e["std_slope"]
    e.loc[fam, "eb"] = eb_shrink(e.loc[fam, "std_slope"], e.loc[fam, "se"], e.loc[fam, "outcome"])
    if e2.get("tract_area_control"):
        covs = [c for c in cfg["covariates"] if c not in (size, "tract_area")]
        ta = effects(out, cov, cfg["outcomes"], covs, (size, "tract_area"), e2["bootstrap"], seed)
        ta = ta.rename(columns={"std_slope": "std_slope_ta", "ci_lo": "ci_lo_ta", "ci_hi": "ci_hi_ta"})
        e = e.merge(ta[["outcome", "covariate", "std_slope_ta", "ci_lo_ta", "ci_hi_ta"]], on=["outcome", "covariate"], how="left")
    reg = {(h["outcome"], h["covariate"]) for h in p7seal.require_signed()}
    e["registered"] = [(o, c) in reg for o, c in zip(e["outcome"], e["covariate"])]
    ax = e2["axes"]
    cols = [c for c in cfg["covariates"] if c not in ax["exclude"]]
    scores, load, expl = form_axes(cov, cols, size, ax["var_target"], ax["max_axes"])
    ae = axis_effects(out, scores, cov, cfg["outcomes"], e2["bootstrap"], seed, size)
    OUT.mkdir(parents=True, exist_ok=True)
    e.to_csv(OUT / "e2_effects.csv", index=False)
    scores.rename_axis("city").to_csv(OUT / "e2_axis_scores.csv")
    ae.to_csv(OUT / "e2_axis_effects.csv", index=False)
    pa = OUT / "e2_axes.json"
    pa.write_text(json.dumps({"config_sha": _sha(CONFIG), "explained": expl.tolist(), "loadings": load.to_dict(),
                              "n_cities": int(len(scores))}, ensure_ascii=False, indent=1))
    beyond = ae[ae["term"] != size]
    nz = int(((beyond["ci_lo"] > 0) | (beyond["ci_hi"] < 0)).sum())
    ledger.add(f"Exploration E-2: {len(out)} cities, {int(fam.sum())} cells beyond size (confounders "
               f"{'/'.join(e2['confounders'])} left out: {int((e['role'] == 'confounder').sum())} cells), BH q {e2['fdr_q']} "
               f"on the normal-approximation p passes "
               f"{int(e.loc[fam, 'bh'].sum())}; form axes {load.shape[1]} (explained {expl[:load.shape[1]].sum():.2f}), "
               f"axis terms with CI excluding 0: {nz} of {len(beyond)}; e2_effects.csv sha {ledger.sha(OUT / 'e2_effects.csv')}")
    return e


def _axis_ta_summary(base, ta) -> dict:
    """Axis terms of the main run against the tract-area-controlled run: how many CIs exclude 0 in each, how many of
    the main run's stay (same sign, CI still excluding 0), and the median coefficient ratio over the main run's."""
    key = ["outcome", "term"]
    b, t = (pd.DataFrame(x) for x in (base, ta))
    b, t = (x[x["term"].astype(str).str.startswith("axis")] for x in (b, t))
    m = b.merge(t, on=key, suffixes=("", "_ta"))
    sig = (m["ci_lo"] > 0) | (m["ci_hi"] < 0)
    sig_ta = (m["ci_lo_ta"] > 0) | (m["ci_hi_ta"] < 0)
    kept = sig & sig_ta & (np.sign(m["coef"]) == np.sign(m["coef_ta"]))
    return {"n_terms": int(len(m)), "n_sig": int(sig.sum()), "n_sig_ta": int(sig_ta.sum()), "n_kept": int(kept.sum()),
            "median_ratio": float((m.loc[sig, "coef_ta"] / m.loc[sig, "coef"]).median()) if sig.any() else np.nan}


def stage_axes_ta() -> pd.DataFrame:
    """Axis-level tract-area sensitivity (added 2026-11-11): the axis regression of E-2 with the confounders
    as extra controls, on the axis scores E-2 wrote; only e2_axis_effects_ta.csv is written."""
    import ledger
    cfg = require_fixed()
    e2, size = cfg["e2"], cfg["size"]
    if not e2["axes"].get("tract_area_control"):
        raise ValueError("e2.axes.tract_area_control is not set in the fixed config: amend it with fix --reason")
    out, cov, _ = load_frame(tuple(cfg["groups"]))
    scores = pd.read_csv(OUT / "e2_axis_scores.csv", dtype={"city": str}).set_index("city")
    ta = axis_effects(out, scores, cov, cfg["outcomes"], e2["bootstrap"], cfg["seed"], size, extra=tuple(e2["confounders"]))
    p = OUT / "e2_axis_effects_ta.csv"
    ta.to_csv(p, index=False)
    s = _axis_ta_summary(pd.read_csv(OUT / "e2_axis_effects.csv"), ta)
    ledger.add(f"Exploration E-2 axis-level tract-area sensitivity (outcome ~ size + axes + {'/'.join(e2['confounders'])}): "
               f"axis terms with CI excluding 0: {s['n_sig_ta']} of {s['n_terms']} (main run {s['n_sig']}); of the main "
               f"run's {s['n_sig']}, {s['n_kept']} keep sign and CI, median coefficient ratio {s['median_ratio']:.2f}; "
               f"{p.name} sha {ledger.sha(p)}")
    return ta


# --------------------------------------------------------------------------------------------- report
def _f(v, nd=3) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{v:.{nd}g}" if np.isfinite(v) else "—"


def render_effects(e1: dict, e2: pd.DataFrame, axes: dict) -> str:
    L = ["# Exploration: form effects beyond size (E-1, E-2)", "",
         "All results are exploratory. E-1's permutation test is the pre-specified exploratory global test (motivated "
         "by plan 8: all 10 registered hypotheses agreed in direction).", "",
         "## E-1 How much signal is left (discovery cities -> law-test cities)", "",
         "| Variant | Pairs | Sign agreement | Slope ratio (median) | Correlation (Pearson) |", "|---|---|---|---|---|"]
    names = {"main": "Main analysis", "with_registered": "With the 10 registered pairs added back",
             "with_confounders": "With tract area and size proxies added back"}
    for v, s in e1.items():
        if v in names:
            L.append(f"| {names[v]} | {s['n_pairs']} | {_f(s['agree'], 2)} | {_f(s['median_ratio'], 2)} | {_f(s['pearson'], 2)} |")
    m = e1.get("main", {})
    L += ["", f"Size-preserving permutation (B = {m.get('perm_B', '—')}): observed agreement rate "
              f"{_f(m.get('perm_obs'), 2)}, null mean "
              f"{_f(m.get('perm_null_mean'), 2)}, 95% quantile {_f(m.get('perm_null_q95'), 2)}, one-sided p = "
              f"{_f(m.get('perm_p'), 2)}.", ""]
    fam = e2[e2["role"] == "form"]
    L += ["## E-2 Effect sizes (all cities, size-controlled standardized partial slopes)", "",
          f"{len(fam)} cells beyond size; {int(fam['bh'].sum())} distinguishable under FDR (Benjamini–Hochberg, "
          "q = 0.10). "
          "The shrunken values (empirical Bayes, grouped by outcome) are the most credible effect sizes. Full table: "
          "e2_effects.csv.", "",
          "The FDR input p uses the normal approximation 2Φ(−|estimate/SE|) (SE = bootstrap standard deviation): the "
          "bootstrap p cannot go below 1/B, "
          "so in a correction over a few hundred cells a single cell, however strong, could never pass the first "
          "threshold; the p column is this normal-approximation p, the intervals are still bootstrap percentile "
          "intervals.", "",
          f"Tract area is a confounder of the measurement unit, not form: its "
          f"{int((e2['role'] == 'confounder').sum())} cells are excluded from FDR, shrinkage and the ranking below, "
          "and kept only in the full table (role = confounder).", "",
          "| Outcome | Covariate | n | Slope [95% CI] | p | FDR | Shrunken | Registered |",
          "|---|---|---|---|---|---|---|---|"]
    top = fam.assign(_a=fam["eb"].abs()).sort_values("_a", ascending=False).head(20)
    for _, r in top.iterrows():
        L.append(f"| {r['outcome']} | {r['covariate']} | {int(r['n'])} | {_f(r['std_slope'])} [{_f(r['ci_lo'])}, {_f(r['ci_hi'])}] | "
                 f"{_f(r['p_norm'], 2)} | {'yes' if r['bh'] else 'no'} | {_f(r['eb'])} | {'yes' if r.get('registered') else 'no'} |")
    ld = axes.get("loadings", {})
    if ld:
        ax = list(ld)
        L += ["", "## Form axes (principal components after size residualization)", "",
              "Explained variance: " + ", ".join(f"{a} {_f(v, 2)}" for a, v in zip(ax, axes.get("explained", []))), "",
              "| Covariate | " + " | ".join(ax) + " |", "|---|" + "---|" * len(ax)]
        for c in ld[ax[0]]:
            L.append(f"| {c} | " + " | ".join(_f(ld[a].get(c), 2) for a in ax) + " |")
        eff = [r for r in axes.get("effects", []) if r["term"].startswith("axis")]
        sig = [r for r in eff if (r["ci_lo"] > 0 or r["ci_hi"] < 0)]
        L += ["", f"Outcome ~ size + axes: {len(eff)} axis terms, {len(sig)} with CI excluding 0 (no multiplicity "
              "correction).", ""]
        if axes.get("effects_ta"):
            t = _axis_ta_summary(axes["effects"], axes["effects_ta"])
            L += [f"With tract area as an extra control (outcome ~ size + axes + tract area): {t['n_sig_ta']} axis "
                  f"terms with CI excluding 0; of the original {t['n_sig']}, "
                  f"{t['n_kept']} keep their sign with CI excluding 0, and the median coefficient is "
                  f"{_f(t['median_ratio'], 2)} times the original. Full table: e2_axis_effects_ta.csv.", ""]
        if sig:
            L += ["| Outcome | Axis | Coefficient [95% CI] |", "|---|---|---|"]
            L += [f"| {r['outcome']} | {r['term']} | {_f(r['coef'])} [{_f(r['ci_lo'])}, {_f(r['ci_hi'])}] |" for r in sig]
    return "\n".join(L) + "\n"


def stage_report() -> Path:
    require_fixed()
    e1 = json.loads((OUT / "e1_summary.json").read_text())
    e2 = pd.read_csv(OUT / "e2_effects.csv")
    axes = json.loads((OUT / "e2_axes.json").read_text())
    axes["effects"] = pd.read_csv(OUT / "e2_axis_effects.csv").to_dict("records")
    if (OUT / "e2_axis_effects_ta.csv").exists():
        axes["effects_ta"] = pd.read_csv(OUT / "e2_axis_effects_ta.csv").to_dict("records")
    p = OUT / "report_effects.md"
    p.write_text(render_effects(e1, e2, axes))
    return p


STAGES = {"e1": stage_e1, "e2": stage_e2, "axes_ta": stage_axes_ta, "report": stage_report}


def main(argv: list):
    if argv[:1] == ["fix"] and (len(argv) == 1 or (len(argv) == 3 and argv[1] == "--reason")):
        import ledger
        return fix_config(argv[2] if len(argv) == 3 else None, log=ledger.add)
    if len(argv) == 1 and argv[0] in STAGES:
        return STAGES[argv[0]]()
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
