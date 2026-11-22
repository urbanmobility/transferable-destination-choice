"""Revision 2026-11-13 (response to review, Task R3): supplementary tables from exported results.

Reads data/paper/figdata (+ figdata/rev, pulled from the compute server's
results/multicity/{p7,confirm,p9,refs_test,rev}) and data/paper/source_data; writes data/paper/tables/rev/*.csv and
rev_tables.md. No model is fitted here.
Usage: python rev_tables.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[2] / "paper"
FD, SD = PAPER / "figdata", PAPER / "source_data"
RV, OUT = FD / "rev", PAPER / "tables" / "rev"
B, SEED = 5000, 20261113
SIZE_OUTCOMES = {"ed_5_10km": "Flexible model: e_d at 5-10 km", "ed_5_10km_c0355r1": "Reach model: e_d at 5-10 km",
                 "ed_5_10km_c0262": "Tempered power law: e_d at 5-10 km",
                 "ed_5_10km_c0514c1": "Two-regime power law: e_d at 5-10 km",
                 "theta_turn_c0514c1": "Two-regime power law: turning distance r (km)",
                 "log_theta_turn": "Two-regime power law: ln r (log-log slope)"}
SETS = {"discovery": "Discovery (35)", "heldout": "Held-out (26)", "confirmation": "Confirmation (6)", "all": "All 67"}


# --------------------------------------------------------------------------------------------- pure parts
def mde(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true slope a two-sided level-alpha test detects with the given power (normal approximation)."""
    from scipy.stats import norm
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def se_from_ci(lo: float, hi: float, level: float = 0.95) -> float:
    from scipy.stats import norm
    return float((hi - lo) / (2 * norm.ppf(0.5 + level / 2)))


def ols_boot(y, x, *controls, B: int = B, seed: int = SEED) -> tuple:
    """OLS slope of y on x (with optional controls), city-bootstrap percentile 95% interval -> (slope, lo, hi)."""
    cols = [np.asarray(x, float)] + [np.asarray(c, float) for c in controls]
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.all([np.isfinite(c) for c in cols], axis=0)
    X = np.column_stack([np.ones(ok.sum())] + [c[ok] for c in cols])
    y = y[ok]
    b = np.linalg.lstsq(X, y, rcond=None)[0][1]
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(b), float(lo), float(hi)


def loglog_slope(logpop, y) -> float:
    """Slope of ln y on ln population (logpop is already ln population)."""
    x, y = np.asarray(logpop, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    return float(np.polyfit(x[ok], np.log(y[ok]), 1)[0])


def rise_slopes(t: pd.DataFrame) -> pd.Series:
    """Per city, OLS slope of e_N on ln d_ref (>= 3 bands)."""
    out = {}
    for c, g in t.groupby("city"):
        if len(g) >= 3:
            out[c] = float(np.polyfit(np.log(g["d_ref"].astype(float)), g["eN"].astype(float), 1)[0])
    return pd.Series(out)


def _f(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _md(df: pd.DataFrame) -> list:
    cols = list(df.columns)
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        L.append("| " + " | ".join(_f(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return L


# --------------------------------------------------------------------------------------------- inputs
def cities() -> pd.DataFrame:
    c = pd.read_csv(FD / "cities.csv", dtype={"city": str}).set_index("city")
    ta = pd.concat([pd.read_csv(RV / f, dtype={"city": str}) for f in
                    ("covariates.csv", "covariates_lawtest.csv", "covariates_confirm.csv")]).drop_duplicates("city")
    c["log_tract_area"] = np.log(ta.set_index("city")["tract_area"].reindex(c.index))
    return c


def outcomes(c: pd.DataFrame) -> pd.DataFrame:
    o = pd.concat([pd.read_parquet(RV / "city_outcomes.parquet").query("block == 'test_dest'"),
                   pd.read_parquet(RV / "city_outcomes_lawtest.parquet"),
                   pd.read_parquet(RV / "city_outcomes_confirm.parquet")])
    o["city"] = o["city"].astype(str)
    o = o.drop_duplicates("city").set_index("city")
    o["log_theta_turn"] = np.log(o["theta_turn_c0514c1"].where(o["theta_turn_c0514c1"] > 0))
    o["set"] = c["set"].reindex(o.index)
    o["log_population"] = c["log_population"].reindex(o.index)
    o["log_tract_area"] = c["log_tract_area"].reindex(o.index)
    return o


# --------------------------------------------------------------------------------------------- tables
def t_size_robust(o: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, lab in SIZE_OUTCOMES.items():
        for s in ("discovery", "heldout", "all"):
            g = o if s == "all" else o[o["set"] == s]
            for ctl in ("none", "log mean tract area"):
                args = (g[k], g["log_population"]) + ((g["log_tract_area"],) if ctl != "none" else ())
                b, lo, hi = ols_boot(*args)
                rows.append({"outcome": lab, "cities": SETS[s], "control": ctl, "n": int(g[k].notna().sum()),
                             "slope": b, "ci_lo": lo, "ci_hi": hi})
    p = RV / "banded_nonwork.parquet"                 # the flexible model refitted on non-work visits (Task R2)
    if p.exists():
        t = pd.read_parquet(p)
        t = t[(t["band"] == 3) & (t["ref"] == "band") & (t["visits"] >= 1000)].assign(city=lambda x: x["city"].astype(str))
        g0 = t.set_index("city")[["ed"]].join(o[["set", "log_population", "log_tract_area"]])
        for s in ("discovery", "heldout", "all"):
            g = g0 if s == "all" else g0[g0["set"] == s]
            for ctl in ("none", "log mean tract area"):
                args = (g["ed"], g["log_population"]) + ((g["log_tract_area"],) if ctl != "none" else ())
                b, lo, hi = ols_boot(*args)
                rows.append({"outcome": "Flexible model, non-work visits: e_d at 5-10 km", "cities": SETS[s], "control": ctl,
                             "n": int(g["ed"].notna().sum()), "slope": b, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def t_form_power() -> pd.DataFrame:
    a = pd.read_csv(SD / "fig4_a.csv")
    m = len(a)
    rows = []
    for r in a.itertuples():
        se = se_from_ci(r.ci_lo, r.ci_hi)
        bound = r.ci_hi if r.disc_slope > 0 else r.ci_lo           # the end in the registered direction
        rows.append({"id": r.id, "group": r.group, "outcome ~ covariate": f"{r.outcome} ~ {r.covariate}",
                     "discovery slope": r.disc_slope, "held-out slope": r.test_slope, "se": se,
                     "MDE (80%, alpha .05)": mde(se), "MDE (80%, Holm first step)": mde(se, 0.05 / m),
                     "MDE / |discovery|": mde(se) / abs(r.disc_slope),
                     "largest effect not excluded / discovery": bound / r.disc_slope})
    return pd.DataFrame(rows)


def t_sk_params(c: pd.DataFrame) -> tuple:
    cv = pd.read_parquet(RV / "cities_val.parquet")
    cv = cv[(cv["cid"] == "c0355r1") & cv["finite"].astype(bool)].copy()
    cv["city"] = cv["city"].astype(str)
    K = np.array([json.loads(x) for x in cv["consts"]], float)
    S = np.array([json.loads(x) if x else [np.nan] * 5 for x in cv["consts_se"]], float)
    t = pd.DataFrame({"city": cv["city"].to_numpy(), "set": c["set"].reindex(cv["city"]).to_numpy()})
    for k in range(5):
        t[f"theta{k}"], t[f"se{k}"] = K[:, k], S[:, k]
    t["theta3_over_theta1"] = t["theta3"] / t["theta1"]
    disc = t[t["set"] == "discovery"]
    summ = []
    for k in [f"theta{i}" for i in range(5)] + ["theta3_over_theta1"]:
        summ.append({"parameter": k, "default (discovery median)": float(disc[k].median()),
                     "median (67)": float(t[k].median()), "q25 (67)": float(t[k].quantile(0.25)),
                     "q75 (67)": float(t[k].quantile(0.75)),
                     "cv (67)": float(t[k].std() / abs(t[k].mean())),
                     "median se": float(t[k.replace("theta", "se")].median()) if "_over_" not in k else np.nan})
    return t, pd.DataFrame(summ)


def t_coverage(c: pd.DataFrame, params: pd.DataFrame) -> tuple:
    from scipy.stats import spearmanr
    v = pd.read_csv(RV / "rev_coverage.csv", dtype={"city": str}).set_index("city")
    sh = pd.read_csv(FD / "shares.csv", dtype={"city": str})
    sh = sh[sh["model"] == "Reach model"].set_index("city")["share"]
    t = pd.DataFrame({"set": c["set"].reindex(v.index), "log_population": c["log_population"].reindex(v.index),
                      "poi_per_1k_residents": 1000 * v["poi"] / v["pop"], "poi_per_km2": v["poi"] / v["area"],
                      "jobs_per_poi": v["jobs"] / v["poi"], "zero_poi_tracts": v["zero_poi"],
                      "devices_per_1k_residents": 1000 * v["home_devices"] / v["pop"],
                      "share_reach_model": sh.reindex(v.index)})
    p = params.set_index("city")
    t["theta2"], t["theta3"] = p["theta2"].reindex(t.index), p["theta3"].reindex(t.index)
    rows = []
    for x in ("poi_per_1k_residents", "poi_per_km2", "devices_per_1k_residents"):
        for y in ("share_reach_model", "theta2", "theta3", "log_population"):
            ok = t[x].notna() & t[y].notna()
            r, pv = spearmanr(t.loc[ok, x], t.loc[ok, y])
            rows.append({"coverage": x, "with": y, "spearman": float(r), "p": float(pv), "n": int(ok.sum())})
    return t.reset_index(names="city"), pd.DataFrame(rows)


def t_abs(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in sorted((RV / "refs_test").glob("*.json")):
        m = json.loads(p.read_text())["test_dest"]
        rows.append({"city": p.stem, "nll_gravity_pop": m["grav_power"]["nll_per_visit"],
                     "nll_flexible": m["M2_glm"]["nll_per_visit"]})
    t = pd.DataFrame(rows).set_index("city")
    t = t[t.index.isin(c.index)]
    sh = pd.read_csv(FD / "shares.csv", dtype={"city": str}).pivot(index="city", columns="model", values="nll")
    for m, k in (("Reach model", "nll_reach"), ("Tempered power law", "nll_tempered"), ("Two-regime power law", "nll_two_regime")):
        t[k] = sh[m].reindex(t.index)
    t["gap_nats"] = t["nll_gravity_pop"] - t["nll_flexible"]
    rv = RV / "models_all_scored.parquet"
    if rv.exists():
        s = pd.read_parquet(rv)
        s = s[s["level"] == "local"].pivot(index="city", columns="model", values="nll")
        for m, k in (("null_unif", "nll_uniform_choice"), ("grav_N", "nll_gravity_poi"), ("str3_N", "nll_strexp3_poi")):
            if m in s:
                t[k] = s[m].reindex(t.index)
        if "nll_uniform_choice" in t:
            for k in ("nll_gravity_pop", "nll_gravity_poi", "nll_reach", "nll_flexible"):
                if k in t:
                    t[k.replace("nll_", "pseudo_r2_")] = 1 - t[k] / t["nll_uniform_choice"]
            if "nll_gravity_poi" in t:
                t["share_from_poi_gravity_reach"] = (t["nll_gravity_poi"] - t["nll_reach"]) / (t["nll_gravity_poi"] - t["nll_flexible"])
    t["set"] = c["set"].reindex(t.index)
    t["name"] = c["name"].reindex(t.index)
    return t.reset_index(names="city")


def t_banded(variant: str) -> tuple:
    p = RV / f"banded_{variant}.parquet"
    if not p.exists():
        return None, None
    t = pd.read_parquet(p)
    t["city"] = t["city"].astype(str)
    t = t[t["visits"] >= 1000]
    med = t.groupby(["model", "ref", "band"])[["ed", "eN"]].median().reset_index()
    rise = []
    for (m, ref), g in t.groupby(["model", "ref"]):
        s = rise_slopes(g)
        rise.append({"model": m, "ref": ref, "cities": int(len(s)), "rising": int((s > 0).sum()),
                     "median slope of e_N on ln d": float(s.median())})
    return med, pd.DataFrame(rise)


def t_r1(line: str) -> tuple:
    """Medians and paired NLL differences of rev_models (Task R1) for one line; (None, None) when not pulled yet."""
    p = RV / f"summary_models_{line}.json"
    if not p.exists():
        return None, None
    s = json.loads(p.read_text())
    med = []
    for k, r in s["models"].items():
        m, lv = k.split("|")
        a, b = r.get("target", {}), r.get("all", {})
        med.append({"model": m, "level": lv, "share (32)": a.get("share"), "NLL (32)": a.get("nll"),
                    "pseudo-R2 (32)": a.get("pseudo_r2"), "share (67)": b.get("share"), "NLL (67)": b.get("nll")})
    con = []
    for k, r in s["contrasts"].items():
        a, b, lv, name = k.split("|")
        con.append({"a": a, "b": b, "level": lv, "cities": name, "dNLL median": r["median"], "ci_lo": r["ci_lo"],
                    "ci_hi": r["ci_hi"], "a better": f"{r['wins']}/{r['n']}", "failed": r["n_failed"], "what": r["label"]})
    return pd.DataFrame(med), pd.DataFrame(con)


ALL2_PAIRS = [("sk_plus1", "sk", "+ a separate log N term, shift held at +1"),
              ("sk_E1", "sep_E1", "employment mass, shift held at +1: coupling")]


def paired_nll(t: pd.DataFrame, a: tuple, b: tuple, cities: list, B: int = B, seed: int = SEED) -> dict:
    """NLL_b - NLL_a by city (> 0: a better); a failure (no finite NLL) is a loss for that side, a tie when both fail."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multicity"))
    import p9_stats as st
    side = lambda k: t[(t["model"] == k[0]) & (t["level"] == k[1])].set_index("city")["nll"].astype(float)  # noqa: E731
    A, Bs = side(a), side(b)
    cs = [c for c in cities if c in A.index and c in Bs.index]
    va, vb = A.loc[cs].to_numpy(float), Bs.loc[cs].to_numpy(float)
    fa, fb = ~np.isfinite(va), ~np.isfinite(vb)
    with np.errstate(invalid="ignore"):
        d = np.where(fa & fb, 0.0, np.where(fa, -np.inf, np.where(fb, np.inf, vb - va)))
    r = st.contrast(pd.Series(d, index=cs), B, seed)
    r["n_failed"] = int((fa | fb).sum())
    return r


def all2_contrasts() -> pd.DataFrame:
    """The fixed-shift refits (line all2) against models of line all, from the scored rows of both."""
    p1, p2 = RV / "models_all_scored.parquet", RV / "models_all2_scored.parquet"
    if not (p1.exists() and p2.exists()):
        return None
    t = pd.concat([pd.read_parquet(p1), pd.read_parquet(p2)], ignore_index=True)
    t["city"] = t["city"].astype(str)
    sets = t.drop_duplicates("city").set_index("city")["set"]
    rows = []
    for a, b, lab in ALL2_PAIRS:
        for lv in ("local", "uniform"):
            for name, cs in (("target", list(sets.index[sets == "target"])), ("all", list(sets.index))):
                r = paired_nll(t, (a, lv), (b, lv), cs)
                rows.append({"a": a, "b": b, "level": lv, "cities": name, "dNLL median": r["median"], "ci_lo": r["ci_lo"],
                             "ci_hi": r["ci_hi"], "a better": f"{r['wins']}/{r['n']}", "failed": r["n_failed"], "what": lab})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = cities()
    o = outcomes(c)
    L = ["# Revision tables (post hoc, supplementary)", ""]
    sr = t_size_robust(o)
    sr.to_csv(OUT / "size_robust.csv", index=False)
    L += ["## City-size slopes with and without the mean tract area", ""] + _md(sr) + [""]
    fp = t_form_power()
    fp.to_csv(OUT / "form_power.csv", index=False)
    L += ["## Power of the ten held-out tests (set 2)", ""] + _md(fp) + [""]
    params, summ = t_sk_params(c)
    params.to_csv(OUT / "sk_params.csv", index=False)
    summ.to_csv(OUT / "sk_params_summary.csv", index=False)
    L += ["## Reach-model constants", ""] + _md(summ) + [""]
    cov, rho = t_coverage(c, params)
    cov.to_csv(OUT / "coverage.csv", index=False)
    rho.to_csv(OUT / "coverage_spearman.csv", index=False)
    L += ["## Catalogue coverage", "", "Medians: " + ", ".join(f"{k} {cov[k].median():.3g}" for k in
          ("poi_per_1k_residents", "poi_per_km2", "jobs_per_poi", "zero_poi_tracts", "devices_per_1k_residents")), ""] + _md(rho) + [""]
    ab = t_abs(c)
    ab.to_csv(OUT / "abs_nll.csv", index=False)
    num = ab.select_dtypes("number")
    L += ["## Absolute NLL (nats per visit), test period x reserved block: medians over 67 cities", ""] + \
         [f"- {k}: {num[k].median():.4f} (min {num[k].min():.4f}, max {num[k].max():.4f})" for k in num.columns] + [""]
    L += ["Smallest gravity-to-flexible gaps:", ""] + _md(ab.nsmallest(5, "gap_nats")[["city", "name", "set", "gap_nats"]]) + [""]
    for v in ("fixed", "emp", "empN", "nonwork", "weekend", "poiarea"):
        med, rise = t_banded(v)
        if med is None:
            continue
        med.to_csv(OUT / f"banded_{v}_medians.csv", index=False)
        rise.to_csv(OUT / f"banded_{v}_rise.csv", index=False)
        L += [f"## Banded flexible model, variant {v}", ""] + _md(rise) + [""] + _md(med) + [""]
    p = RV / "banded_strata.parquet"                    # thirds of origins by distance to the CBD (rev_strata.py)
    if p.exists():
        t3 = pd.read_parquet(p)
        for k in ("inner", "middle", "outer"):
            t3[t3["stratum"] == k].drop(columns="stratum").to_parquet(RV / f"banded_strata_{k}.parquet", index=False)
            med, rise = t_banded(f"strata_{k}")
            med.to_csv(OUT / f"banded_strata_{k}_medians.csv", index=False)
            rise.to_csv(OUT / f"banded_strata_{k}_rise.csv", index=False)
    for line in ("all", "nonwork", "poitrain"):
        med, con = t_r1(line)
        if med is None:
            continue
        if line == "all":
            extra = all2_contrasts()
            if extra is not None:
                con = pd.concat([con, extra], ignore_index=True)
        med.to_csv(OUT / f"models_{line}_medians.csv", index=False)
        con.to_csv(OUT / f"models_{line}_contrasts.csv", index=False)
        L += [f"## Model comparisons (R1), line {line}", ""] + _md(med) + [""] + _md(con) + [""]
    (OUT / "rev_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
