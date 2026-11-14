"""Plan 7 Task 6: exploration of outcome ~ covariate on the 35 discovery cities, the
selection rule (fixed in code before the evidence), the registration draft for sign-off, and the test function that
plan 8 applies to the holdout cities after the registration is signed.

Statistics: OLS slope of the outcome on the covariate; for any covariate other than log_population also the partial
slope with log_population as a control (and the partial correlation); city bootstrap (percentile 95% CI, two-sided
p). Pass rule (fixed 2026-11-10): the holdout CI excludes 0 in the registered direction; Holm within the plan-7 family.

Usage (run_stage.sh): explore_p7.py explore MEMO_PATH
"""
import fnmatch
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import laws  # noqa: E402

SIZE = "log_population"
SEED = 20261110
# CRITERION, and the statement built by draft(), are written verbatim into the registration (registered_p7.json, signed
# and hashed), so they stay in the language of the signed file. CRITERION in English: "On the holdout cities, the 95%
# interval of the slope from a city bootstrap (B = 5000) excludes 0 and has the registered direction (controls: see
# control); Holm correction within the plan-7 family of hypotheses (counted separately from the earlier 15); the slope
# size is reported, not judged; primary test (2024 law-test cities) passes = supported, primary test and independent
# replication (cities newly eligible in 2019, non-commute) both pass = confirmed".
CRITERION = ("留出城上按城市 bootstrap（B = 5000）得到的斜率 95% 区间不含 0，且方向与登记一致（控制量见 control）；"
             "在计划 7 一族假设内做 Holm 校正（与旧的 15 条分开计数）；斜率大小只报告不判定；"
             "主检验（2024 年规律检验城）通过 = 支持，主检验与独立复制（2019 年新达标城，非通勤）都通过 = 确认")


def discovery() -> list:
    return config.cities_with_role("search") + config.cities_with_role("discovery")


def _num(df, drop=("city", "block", "cls")):
    out = df.drop(columns=[c for c in drop if c in df], errors="ignore").copy()
    for c in out.columns:
        if out[c].dtype == bool or out[c].dtype == object:
            out[c] = pd.to_numeric(out[c].map(lambda v: float(v) if isinstance(v, (bool, np.bool_)) else v),
                                   errors="coerce")
    return out.astype(float)


def _boot_slopes(y, x, z, idx):
    """Bootstrap slopes of y on x (z is None) or partial slopes of y on x given z, over index matrix idx (B x n)."""
    Y, X = y[idx], x[idx]
    Yc, Xc = Y - Y.mean(1, keepdims=True), X - X.mean(1, keepdims=True)
    if z is None:
        sxx = (Xc * Xc).sum(1)
        return np.where(sxx > 0, (Xc * Yc).sum(1) / np.where(sxx > 0, sxx, 1), np.nan)
    Z = z[idx]
    Zc = Z - Z.mean(1, keepdims=True)
    sxx, szz, sxz = (Xc * Xc).sum(1), (Zc * Zc).sum(1), (Xc * Zc).sum(1)
    sxy, szy = (Xc * Yc).sum(1), (Zc * Yc).sum(1)
    den = sxx * szz - sxz ** 2
    return np.where(den > 1e-12, (sxy * szz - szy * sxz) / np.where(den > 1e-12, den, 1), np.nan)


def _slope(y, x, z=None):
    return float(_boot_slopes(y, x, z, np.arange(len(y))[None, :])[0])


def _ci_p(b):
    b = b[np.isfinite(b)]
    if len(b) < 10:
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(b, [2.5, 97.5])
    p = max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1.0 / len(b))
    return float(lo), float(hi), float(min(p, 1.0))


def _resid(a, z):
    A = np.column_stack([np.ones_like(z), z])
    return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]


def _pair_stats(y, x, z, B, rng, partial: bool) -> dict:
    n = len(y)
    idx = rng.integers(0, n, (B, n))
    out = {"n": n, "spearman_rho": float(pd.Series(y).corr(pd.Series(x), method="spearman")),
           "pearson_r": float(np.corrcoef(x, y)[0, 1]), "slope": _slope(y, x)}
    out["ci_lo"], out["ci_hi"], p_simple = _ci_p(_boot_slopes(y, x, None, idx))
    if partial:
        out["partial_slope"] = _slope(y, x, z)
        out["pci_lo"], out["pci_hi"], out["p"] = _ci_p(_boot_slopes(y, x, z, idx))
        out["partial_rho"] = float(np.corrcoef(_resid(y, z), _resid(x, z))[0, 1])
    else:
        out.update({"partial_slope": np.nan, "pci_lo": np.nan, "pci_hi": np.nan, "partial_rho": np.nan, "p": p_simple})
    return out


def relations(out: pd.DataFrame, cov: pd.DataFrame, cities, B: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Every outcome x covariate on the given cities, which must all be discovery cities (exploration only there)."""
    cities = [str(c) for c in cities]
    bad = sorted(set(cities) - set(discovery()))
    if bad:
        raise ValueError(f"exploration runs on discovery cities only; not discovery: {bad[:5]}")
    _unique(out[out["city"].isin(cities)], cov)
    d = out[out["city"].isin(cities)].merge(cov, on="city", how="inner", suffixes=("", "_cov"))
    Y, Xc = _num(out[out["city"].isin(cities)]), [c for c in cov.columns if c != "city"]
    rows = []
    for o in Y.columns:
        if o in Xc:
            continue
        for c in Xc:
            sub = d[["city", o, c] + ([SIZE] if SIZE in d and c != SIZE else [])].copy()
            sub = _num(sub).assign(city=d["city"].to_numpy()).replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 5:
                continue
            rng = np.random.default_rng([seed, zlib.crc32(f"{o}|{c}".encode())])
            partial = c != SIZE and SIZE in sub
            z = sub[SIZE].to_numpy(float) if partial else None
            st = _pair_stats(sub[o].to_numpy(float), sub[c].to_numpy(float), z, B, rng, partial)
            r_size = 1.0 if c == SIZE else (float(np.corrcoef(sub[c], sub[SIZE])[0, 1]) if SIZE in sub else np.nan)
            rows.append({"outcome": o, "covariate": c, **st, "r_size": r_size})
    return pd.DataFrame(rows)


def _unique(*frames):
    for f in frames:
        if not f["city"].is_unique:
            raise ValueError("one row per city is required (city is not unique): filter the block first")


def _known_hit(o, c, known, exclude=True):
    return [k for k in known if bool(k.get("exclude")) == exclude and k["covariate"] == c and fnmatch.fnmatch(o, k["outcome"])]


def select(rel: pd.DataFrame, known, max_total: int = 10, max_per_outcome: int = 2, min_n: int = 25,
           r_size_max: float = 0.8, max_per_covariate: int = 2) -> list:
    """The selection rule, fixed before the evidence: CI (partial for covariates other than log_population) excludes
    0; not a known relation (exclude=True); not a size proxy (|r(covariate, log_population)| > r_size_max); n >= min_n;
    ranked by |partial_rho| (|pearson_r| for log_population); caps per outcome, per covariate and in total."""
    cand = []
    for r in rel.itertuples(index=False):
        r = r._asdict()
        size = r["covariate"] == SIZE
        lo, hi = (r["ci_lo"], r["ci_hi"]) if size else (r["pci_lo"], r["pci_hi"])
        if not (r["n"] >= min_n and np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)):
            continue
        if _known_hit(r["outcome"], r["covariate"], known, True):
            continue
        if not size and not (abs(r.get("r_size", 0.0)) <= r_size_max):
            continue
        key = abs(r.get("pearson_r", np.nan)) if size else abs(r["partial_rho"])
        if not np.isfinite(key):
            key = abs(r.get("spearman_rho", 0.0))
        stat = r["slope"] if size else r["partial_slope"]
        cand.append((key, r, stat))
    cand.sort(key=lambda t: (-t[0], t[1]["outcome"], t[1]["covariate"]))
    sel, per_o, per_c = [], {}, {}
    for key, r, stat in cand:
        o, c = r["outcome"], r["covariate"]
        if len(sel) >= max_total or per_o.get(o, 0) >= max_per_outcome or per_c.get(c, 0) >= max_per_covariate:
            continue
        per_o[o], per_c[c] = per_o.get(o, 0) + 1, per_c.get(c, 0) + 1
        size = c == SIZE
        flag = [k.get("source", "") for k in _known_hit(o, c, known, False)]
        sel.append({"outcome": o, "covariate": c, "control": None if size else SIZE,
                    "test": "slope_ci" if size else "partial_slope_ci", "direction": "+" if stat > 0 else "-",
                    "evidence": {"n": int(r["n"]), "stat": float(stat),
                                 "ci": [float(r["ci_lo"] if size else r["pci_lo"]), float(r["ci_hi"] if size else r["pci_hi"])],
                                 "rank_r": float(key), "p": float(r["p"]), "plain_slope": float(r["slope"])},
                    "known_flag": flag})
    return sel


def _meaning(name):
    import city_outcomes
    import form_p7
    import laws
    return {**laws.COV_MEANING, **form_p7.DEFS, **city_outcomes.OUTCOME_DEFS}.get(name, name)


def draft(selected: list) -> list:
    out = []
    for k, s in enumerate(selected, 1):
        # statement: "the higher <covariate> of a city, the larger / smaller its <outcome> (controlling for log metro
        # population)"; kept in the language of the signed registration (see CRITERION)
        verb = "越大" if s["direction"] == "+" else "越小"
        ctl = "（控制都市区人口对数）" if s["control"] else ""
        out.append({"id": f"P7-{k:02d}",
                    "statement": f"「{_meaning(s['covariate'])}」越高的城市，「{_meaning(s['outcome'])}」{verb}{ctl}",
                    "outcome": s["outcome"], "covariate": s["covariate"], "control": s["control"],
                    "direction": s["direction"], "test": s["test"], "block": "test_dest", "primary": "lawtest_2024",
                    "replication": "M19_new", "criterion": CRITERION, "evidence": s["evidence"],
                    "known_flag": s.get("known_flag", [])})
    return out


def write_draft(hyps: list, memo_path) -> None:
    """registered_p7.json (unsigned) + a review memo; refused once the registration is signed."""
    import p7seal
    if p7seal.SIGNED.exists():
        raise FileExistsError(f"{p7seal.SIGNED} exists: the registration is signed and cannot be rewritten")
    p7seal.REG.parent.mkdir(parents=True, exist_ok=True)
    p7seal.REG.write_text(json.dumps(hyps, ensure_ascii=False, indent=1))
    lines = ["# Plan 7 registration draft (awaiting sign-off)", "",
             f"Decision rule: {CRITERION}", "",
             "| ID | Statement | Outcome ~ covariate | Control | Direction | Discovery-city evidence (n, slope, 95% CI, "
             "partial correlation) | Literature |",
             "|---|---|---|---|---|---|---|"]
    for h in hyps:
        e = h["evidence"]
        lines.append(f"| {h['id']} | {h['statement']} | {h['outcome']} ~ {h['covariate']} | {h['control'] or '—'} | "
                     f"{h['direction']} | {e['n']}, {e['stat']:.3g}, [{e['ci'][0]:.3g}, {e['ci'][1]:.3g}], {e['rank_r']:.2f} | "
                     f"{'Prior observation (literature or descriptive results of this project): ' + '; '.join(h['known_flag']) if h['known_flag'] else 'none in the literature'} |")
    Path(memo_path).parent.mkdir(parents=True, exist_ok=True)
    Path(memo_path).write_text("\n".join(lines) + "\n")


def run_test(h: dict, out: pd.DataFrame, cov: pd.DataFrame, B: int = 5000, seed: int = SEED) -> dict:
    """One registered hypothesis on the given (holdout) cities: stat (slope or partial slope), percentile CI, two-sided
    bootstrap p, pass_raw = CI excludes 0 in the registered direction."""
    _unique(out, cov)
    d = out.merge(cov, on="city", how="inner", suffixes=("", "_cov"))
    cols = [h["outcome"], h["covariate"]] + ([h["control"]] if h.get("control") else [])
    sub = _num(d[cols]).replace([np.inf, -np.inf], np.nan).dropna()
    res = {"id": h.get("id"), "stat": None, "ci_lo": None, "ci_hi": None, "p": 1.0, "n": int(len(sub)), "B": B,
           "pass_raw": False}
    if len(sub) < 5:
        return {**res, "note": "too few cities"}
    rng = np.random.default_rng([seed, zlib.crc32(f"{h['outcome']}|{h['covariate']}|{h.get('control')}".encode())])
    y, x = sub[h["outcome"]].to_numpy(float), sub[h["covariate"]].to_numpy(float)
    z = sub[h["control"]].to_numpy(float) if h.get("control") else None
    stat = _slope(y, x, z)
    lo, hi, p = _ci_p(_boot_slopes(y, x, z, rng.integers(0, len(y), (B, len(y)))))
    ok = np.isfinite(lo) and ((h["direction"] == "+" and lo > 0) or (h["direction"] == "-" and hi < 0))
    return {**res, "stat": stat, "ci_lo": lo, "ci_hi": hi, "p": p, "pass_raw": bool(ok)}


def frames(oc: pd.DataFrame, cities, cov_csv) -> tuple:
    """(outcomes, covariates) of the given cities as the exploration merged them: the test_dest rows of the outcome
    table without block / log_population, and laws.covariates left-joined with the plan-7 covariate file (plan 8 tests
    the holdout cities with the same merge)."""
    cities = [str(c) for c in cities]
    out = oc[(oc["block"] == "test_dest") & oc["city"].isin(cities)].drop(columns=["block", "log_population"],
                                                                         errors="ignore")
    cov = laws.covariates(cities).merge(pd.read_csv(cov_csv, dtype={"city": str}), on="city", how="left")
    return out.reset_index(drop=True), cov


def stage_explore(memo_path):
    import ledger
    import p7seal
    disc = discovery()
    oc, cov = frames(pd.read_parquet(config.RESULTS / "p7/city_outcomes.parquet"), disc,
                     config.RESULTS / "p7/covariates.csv")
    rel = relations(oc, cov, disc)
    rel.to_csv(config.RESULTS / "p7/relations.csv", index=False)
    known = json.loads((Path(__file__).resolve().parent / "config/p7_known.json").read_text())["known"]
    hyps = draft(select(rel, known))
    write_draft(hyps, memo_path)
    ledger.add(f"Plan 7 P3: {len(rel)} outcome x covariate relations on {len(disc)} discovery cities (relations.csv sha "
               f"{ledger.sha(config.RESULTS / 'p7/relations.csv')}); selection rule -> {len(hyps)} hypotheses; "
               f"registered_p7.json (UNSIGNED) sha {ledger.sha(p7seal.REG)}; memo {memo_path}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["explore"] and len(sys.argv) == 3:
        stage_explore(sys.argv[2])
    else:
        sys.exit(__doc__)
