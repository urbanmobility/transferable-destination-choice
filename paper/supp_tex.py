"""Revision 2026-11-13 (Task S1): LaTeX tables of Additional file 1 generated from the exported results.

Reads data/paper/tables/rev/*.csv (rev_tables.py), data/paper/figdata (+ rev/), and writes
manuscript/supp/S{1,5,6,7,8,9,10,11,12,13,18}_*.tex. The hand-written tables (S2-S4, S14-S17, S19-S21) live beside them.
Usage: python supp_tex.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[2] / "paper"
FD, TB = PAPER / "figdata", PAPER / "tables"
RT = TB / "rev"
OUT = Path(__file__).resolve().parents[3] / "manuscript" / "supp"
SET = {"discovery": "D", "heldout": "H", "confirmation": "C"}
MODEL = {"null_unif": "Uniform choice", "grav_P": "Power-law gravity, population", "grav_N": "Power-law gravity, POI",
         "str3_N": "Relaxed stretched exp., POI", "str3a_N": "+ $\\ln a_j$",
         "sep_N": "Separable twin of the Reach model", "sk": "Reach model", "sk_lna": "Reach model, $\\theta_4\\ln a_j$",
         "sk_plus": "Reach model $+\\ \\gamma\\ln(N_j+c)$", "sk_plus1": "Reach model $+\\ \\gamma\\ln(N_j+1)$, shift $+1$",
         "sep_E1": "Separable twin, employment, shift $+1$", "sk_E1": "Reach model, employment, shift $+1$", "grav_E": "Power-law gravity, employment",
         "str3_E": "Relaxed stretched exp., employment", "sep_E": "Separable twin, employment", "sk_E": "Reach model, employment",
         "sk_frozen": "Reach model (main-analysis fit)", "tp_frozen": "Tempered power law (main-analysis fit)",
         "tr_frozen": "Two-regime power law (main-analysis fit)"}


def num(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    s = f"{v:,.{nd}f}" if abs(v) >= 1000 else f"{v:.{nd}f}"
    return s.replace("-", "$-$")


def tex_escape(s: str) -> str:
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def table(label: str, caption: str, head: list, rows: list, align: str, long: bool = False, size="\\footnotesize",
          fit: bool = False) -> str:
    """A booktabs table (float) or longtable with a fixed number S<label>."""
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    h = " & ".join(head) + " \\\\"
    if long:
        return (f"{{\\renewcommand{{\\thetable}}{{{label}}}{size}\\setlength{{\\tabcolsep}}{{3pt}}\n\\begin{{longtable}}{{{align}}}\n"
                f"\\caption{{{caption}}}\\label{{tab:{label}}}\\\\\n\\toprule\n{h}\n\\midrule\n\\endfirsthead\n"
                f"\\toprule\n{h}\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n{body}\n\\end{{longtable}}}}\n")
    return (f"\\begin{{table}}[H]\n\\renewcommand{{\\thetable}}{{{label}}}\n\\centering\n{size}\n"
            f"\\caption{{{caption}}}\n\\label{{tab:{label}}}\n" + ("\\resizebox{\\textwidth}{!}{" if fit else "{") +
            f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{h}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n")


def cities() -> pd.DataFrame:
    c = pd.read_csv(FD / "cities.csv", dtype={"city": str})
    q = pd.read_csv(Path(__file__).resolve().parents[3] / "manuscript/sources/cities_qc.csv", dtype={"city": str})
    v = pd.read_csv(FD / "rev" / "rev_coverage.csv", dtype={"city": str})
    return c.merge(q[["city", "train_trips", "val_trips", "test_trips"]], on="city", how="left").merge(v, on="city", how="left")


def s1() -> str:
    c = cities().sort_values(["set", "population"], ascending=[True, False])
    rows = [[tex_escape(r.name), SET[r.set], num(r.population / 1e6, 2), str(int(r.n_units)), num(r.home_devices / 1e3, 0),
             num((r.train_trips + r.val_trips + r.test_trips) / 1e6, 2), num(1000 * r.poi / r.population, 1)]
            for r in c.itertuples()]
    cap = ("The 67 metropolitan areas. Group: D discovery (the eight search cities are New York, Los Angeles, Chicago, "
           "Houston, Atlanta, the San Francisco Bay Area, Seattle and Boston), H held-out, C confirmation. Population: ACS "
           "2015--2019 (millions). Tracts: 2010 census tracts of the metropolitan area. Devices: devices "
           "with an inferred home in the city (thousands). Visits: home-based visits in the three periods (millions). POIs "
           "per 1,000 residents: places with at least one visit in March--May 2024.")
    return table("S1", cap, ["Metropolitan area", "Group", "Pop.", "Tracts", "Devices (k)", "Visits (M)", "POI/1k"],
                 rows, "p{6.2cm}crrrrr", long=True, size="\\scriptsize")


def s5() -> str:
    t = pd.read_csv(RT / "abs_nll.csv", dtype={"city": str}).sort_values(["set", "name"])
    cols = [("nll_uniform_choice", "Uniform"), ("nll_gravity_pop", "Grav. P"), ("nll_gravity_poi", "Grav. N"),
            ("nll_reach", "Reach"), ("nll_tempered", "Tempered"), ("nll_two_regime", "Two-reg."), ("nll_flexible", "Flexible"),
            ("gap_nats", "Gap"), ("pseudo_r2_reach", "$R^2_{\\mathrm{McF}}$")]
    cols = [c for c in cols if c[0] in t]
    rows = [[tex_escape(r["name"].split(",")[0]), SET[r["set"]]] + [num(r[k], 3) for k, _ in cols] for _, r in t.iterrows()]
    cap = ("Absolute fit per city: NLL in nats per visit on the test period and held-out block. Uniform: uniform choice over "
           "the block's destinations. Grav. P and Grav. N: power-law gravity with residential population (the share-0 "
           "reference) and with POI count. Flexible: the 22-parameter reference (share 1). Gap: Grav. P minus Flexible. "
           "$R^2_{\\mathrm{McF}}$: 1 minus the Reach model's NLL over the uniform-choice NLL. Group as in Table~S1.")
    return table("S5", cap, ["City", "G"] + [h for _, h in cols], rows, "lc" + "r" * len(cols), long=True, size="\\scriptsize")


def s6_percity() -> str:
    t = pd.read_csv(RT / "sk_params.csv", dtype={"city": str})
    c = pd.read_csv(FD / "cities.csv", dtype={"city": str}).set_index("city")
    t["name"] = t["city"].map(c["name"]).fillna(t["city"])
    t = t.sort_values(["set", "name"])
    sm = pd.read_csv(RT / "sk_params_summary.csv").set_index("parameter")
    d = sm["default (discovery median)"]
    rows = [[tex_escape(r["name"].split(",")[0]), SET[r["set"]]] +
            [f"{num(r[f'theta{k}'], 3)} ({num(r[f'se{k}'], 3)})" for k in range(5)] for _, r in t.iterrows()]
    rows.append(["\\textbf{Default (discovery median)}", ""] + [f"\\textbf{{{num(d[f'theta{k}'], 3)}}}" for k in range(5)])
    rows.append(["Interquartile range, 67 cities", ""] + [f"{num(sm.loc[f'theta{k}', 'q25 (67)'], 3)}--{num(sm.loc[f'theta{k}', 'q75 (67)'], 3)}" for k in range(5)])
    th = d.round(3)                                   # the example uses the printed (rounded) constants
    ex_d, ex_N, ex_a = 5.0, 71.0, 2.6
    eta = ex_a ** th["theta4"] - th["theta0"] * ex_d ** th["theta1"] / (ex_N + th["theta2"]) ** th["theta3"]
    cap = ("Reach-model constants per city (standard errors in parentheses) in "
           "$\\eta_{ij}=a_j^{\\theta_4}-\\theta_0 d^{\\theta_1}/(N_j+\\theta_2)^{\\theta_3}$, with $d$ in km, $a_j$ in km$^2$ and "
           "$N_j$ the count of places with at least one visit. The default row is the uniform parameter vector used for "
           f"transfer. Worked example with the default: a destination 5~km away with $N_j=71$ and $a_j=2.6$~km$^2$ has "
           f"$\\eta={eta:.3f}$; the choice probability is $\\exp(\\eta_{{ij}})$ normalised over all destinations of the "
           f"origin. Median $\\theta_3/\\theta_1$ (the exponent with which the distance scale grows with $N_j+\\theta_2$): "
           f"{num(sm.loc['theta3_over_theta1', 'median (67)'], 2)} (interquartile range "
           f"{num(sm.loc['theta3_over_theta1', 'q25 (67)'], 2)}--{num(sm.loc['theta3_over_theta1', 'q75 (67)'], 2)}).")
    return table("S6", cap, ["City", "G", "$\\theta_0$", "$\\theta_1$", "$\\theta_2$", "$\\theta_3$", "$\\theta_4$"], rows,
                 "lcrrrrr", long=True, size="\\scriptsize")


BANDS = ["0--1", "1--2", "2--5", "5--10", "10--20", "$>$20"]


def _theta2_bound_change() -> str:
    """Largest change of the held-out-block share when theta_2 >= 1 is imposed in the cities with a negative shift."""
    t = json.loads((FD / "rev" / "theta2_full.json").read_text())
    assert len(t) == 5 and all(np.isfinite(r["bounded"]["all_nll"]) for r in t)
    neg = [r["main"]["consts"][2] for r in t if r["main"]["consts"][2] < 0]
    assert len(neg) == 4 and round(min(neg), 2) == -1.53 and round(max(neg), 2) == -0.05, neg
    d = max(abs(r["bounded"]["block_share"] - r["main"]["block_share"]) for r in t)
    assert d < 0.005, d
    return "0.005"


def s6() -> str:
    parts = []
    bands = pd.read_csv(FD / "bands.csv", dtype={"city": str})
    for ref, lab, med in (("band", "band medians (main figures)", bands.groupby(["model", "band"])[["ed", "eN"]].median().reset_index()),):
        for m in ("Flexible", "Reach model", "Tempered power law", "Two-regime power law"):
            g = med[med["model"] == m].set_index("band")
            parts.append([f"{m}, {lab}", "$e_N$"] + [num(g["eN"].get(b), 2) for b in range(6)])
    for v, lab in (("fixed", "common point"), ("poiarea", "POI count and area only, common point"),
                   ("emp", "employment, common point"), ("empN", "employment, net of $\\ln N$"),
                   ("nonwork", "non-work visits, common point"), ("weekend", "weekend visits, common point"),
                   ("strata_inner", "inner third of origins, common point"), ("strata_middle", "middle third of origins, common point"),
                   ("strata_outer", "outer third of origins, common point")):
        p = RT / f"banded_{v}_medians.csv"
        if not p.exists():
            continue
        g0 = pd.read_csv(p)
        g0 = g0[g0["ref"] == "fixed"]
        for m in (("Flexible", "Reach model", "Tempered power law", "Two-regime power law") if v == "fixed" else ("Flexible",)):
            g = g0[g0["model"] == m].set_index("band")
            parts.append([f"{m}, {lab}", "$e_N$" if v not in ("emp", "empN") else "$e_E$"] + [num(g["eN"].get(b), 2) for b in range(6)])
    rise = []
    for v, lab in (("fixed", "POI count, common point"), ("poiarea", "POI count and area only"), ("emp", "employment"),
                   ("empN", "employment, net of $\\ln N$"),
                   ("nonwork", "POI count, non-work visits"), ("weekend", "POI count, weekend visits"),
                   ("strata_inner", "inner third of origins"), ("strata_middle", "middle third of origins"),
                   ("strata_outer", "outer third of origins")):
        p = RT / f"banded_{v}_rise.csv"
        if p.exists():
            r = pd.read_csv(p)
            r = r[(r["model"] == "Flexible") & (r["ref"] == "fixed")]
            if len(r):
                rise.append(f"{lab}: {int(r['rising'].iloc[0])} of {int(r['cities'].iloc[0])}")
    cap = ("Size elasticity by distance band (km): cross-city medians (per-city profiles in Fig.~S1 and Additional file~2). Band medians: each city's band-specific reference "
           "point. Common point: $d$ at the band's geometric midpoint (0.5, 1.41, 3.16, 7.07, 14.1, 28.3~km) and destination "
           "attributes at their medians over all destination tracts of the 67 cities ($N_j=71$, $a_j=2.6$~km$^2$, $E_j=934$ "
           "jobs). Employment rows: the banded flexible model with $\\ln(1+E_j)$ as the banded mass (log employment removed "
           "from the covariates; net of $\\ln N$: log POI count added as a covariate). POI count and area only: the flexible model refitted with "
           "log area as its only other destination attribute, the information that the Reach model uses. Weekend visits: the model refitted on the visits of "
           "Saturdays and Sundays (local date). Thirds of origins: the model refitted on the inner, middle and outer third of "
           "each city's origins by distance to the CBD. Flexible-model elasticities are "
           "$\\gamma_b M/(M+1)$ at the reference mass $M$. Cities in which the elasticity rises with $\\ln d$ (OLS slope "
           "$>0$): " + "; ".join(rise) + ".")
    return table("S6", cap, ["Model and reference point", ""] + BANDS, parts, "llrrrrrr")


LADDER = [("grav_N", "grav_P"), ("str3_N", "grav_N"), ("str3a_N", "str3_N"),
          ("sk", "sep_N"), ("sk", "str3_N"), ("sk_plus", "sk"), ("sk_plus1", "sk"), ("sk_lna", "sk"), ("sk", "sk_frozen"),
          ("sk_frozen", "tp_frozen"), ("sk_frozen", "tr_frozen")]
EMP = [("grav_E", "grav_P"), ("grav_N", "grav_E"), ("sk_E", "str3_E"), ("sk_E", "sep_E"), ("sk_E1", "sep_E1")]


def _contrast_rows(line: str, pairs: list) -> list:
    p = RT / f"models_{line}_contrasts.csv"
    if not p.exists():
        return []
    c = pd.read_csv(p)
    rows = []
    for a, b in pairs:
        for lv in ("local", "uniform"):
            g = c[(c["a"] == a) & (c["b"] == b) & (c["level"] == lv)].set_index("cities")
            if not len(g):
                continue
            cells = []
            for name in ("target", "all"):
                if name in g.index:
                    r = g.loc[name]
                    cells.append(f"{num(r['dNLL median'], 4)} [{num(r['ci_lo'], 4)}, {num(r['ci_hi'], 4)}]")
                    cells.append(str(r["a better"]))
                else:
                    cells += ["--", "--"]
            rows.append([MODEL.get(a, a), MODEL.get(b, b), lv] + cells)
    return rows


def scored(line: str) -> pd.DataFrame:
    """rev_models' scored rows of a line; line "all" also carries the fixed-shift refits of line "all2"."""
    p = FD / "rev" / f"models_{line}_scored.parquet"
    if not p.exists():
        return None
    t = pd.read_parquet(p)
    q = FD / "rev" / "models_all2_scored.parquet"
    if line == "all" and q.exists():
        t = pd.concat([t, pd.read_parquet(q).query("model not in ('null_unif', 'grav_ref', 'M2_ref')")], ignore_index=True)
    t["city"] = t["city"].astype(str)
    return t


def _median_rows(line: str, models: list) -> tuple:
    """Medians over the target cities where every listed model has a finite NLL at both levels:
    (rows, number of cities)."""
    t = scored(line)
    if t is None:
        return [], 0
    t = t[(t["set"] == "target") & t["model"].isin(models)]
    ok = t.assign(f=np.isfinite(t["nll"].astype(float))).groupby("city")["f"].agg(["all", "size"])
    need = 2 * len([m for m in models if m in set(t["model"])])
    cs = ok.index[(ok["all"]) & (ok["size"] == need)]
    t = t[t["city"].isin(cs)]
    rows = []
    for k in models:
        cells = []
        for lv in ("local", "uniform"):
            g = t[(t["model"] == k) & (t["level"] == lv)]
            cells += [num(float(g["share"].median()), 3), num(float(g["nll"].median()), 4)] if len(g) else ["--", "--"]
        rows.append([MODEL.get(k, k)] + cells)
    return rows, len(cs)


HEAD_C = ["$a$", "$b$", "Parameters", "$\\Delta$NLL, 32 target [95\\% CI]", "$a$ better", "$\\Delta$NLL, 67 [95\\% CI]", "$a$ better"]
HEAD_M = ["Model", "Share, local", "NLL, local", "Share, uniform", "NLL, uniform"]


def s_r1(label: str, line: str, pairs: list, models: list, what: str) -> str:
    rc, (rm, n_common) = _contrast_rows(line, pairs), _median_rows(line, models)
    if not rc:
        return ""
    cap = (f"{what} Paired differences $\\Delta$NLL $=\\mathrm{{NLL}}_b-\\mathrm{{NLL}}_a$ in nats per visit (positive: $a$ "
           "fits better), medians over cities with city-bootstrap 95\\% intervals (5,000 resamples), on the test period and "
           "held-out block; local: fitted in the city; uniform: discovery-city medians (the city itself left out). A city in "
           "which a model's utility is undefined on the held-out block (a fitted shift below $-N_j$ of an unseen tract) counts "
           "as a loss for that model and a tie when both fail; the number of such cities is given in the source data. All "
           "models fitted with the same protocol (300 sampled origins, five starts) unless marked as main-analysis fits; "
           "``shift $+1$'' marks variants whose shift is held at $+1$.")
    out = table(label + "a", cap, HEAD_C, rc, "p{3.4cm}p{3.0cm}lrrrr", size="\\scriptsize", fit=True)
    if rm:
        cap2 = (f"Medians for the models of Table~{label}a over the {n_common} target cities in which every model listed has a "
                "finite NLL at both parameter levels: share of the gravity-to-flexible gap and NLL (nats per visit).")
        out += table(label + "b", cap2, HEAD_M, rm, "lrrrr", size="\\scriptsize")
    return out


QUANT = {"theta_turn_c0514c1": "Two-regime: turning distance $r$", "transfer_loss_c0355r1": "Reach model: transfer loss",
         "ed_5_10km_c0262": "Tempered: $e_d$ at 5--10~km", "slope_eN_logd_c0355r1": "Reach model: slope of $e_N$ on $\\ln d$",
         "ed_5_10km_c0355r1": "Reach model: $e_d$ at 5--10~km", "ed_10_20km": "Flexible: $e_d$ at 10--20~km",
         "ed_1_2km": "Flexible: $e_d$ at 1--2~km", "ed_20km": "Flexible: $e_d$ beyond 20~km", "log_population": "log population",
         "road_density": "main-road density", "compactness": "residential compactness", "subcentre_job_share": "employment-centre job share",
         "circuity": "road circuity", "pop_density_gini": "Gini of population density"}


def readable(s: str) -> str:
    a, b = [x.strip() for x in s.split("~")]
    return f"{QUANT.get(a, tex_escape(a))} $\\sim$ {QUANT.get(b, tex_escape(b))}"


def s12() -> str:
    t = pd.read_csv(RT / "form_power.csv")
    rows = [[r["id"], readable(r["outcome ~ covariate"]), num(r["discovery slope"], 3),
             num(r["held-out slope"], 3), num(r["se"], 3), num(r["MDE (80%, alpha .05)"], 3), num(r["MDE / |discovery|"], 2),
             num(r["largest effect not excluded / discovery"], 2)] for _, r in t.iterrows()]
    cap = ("Power of the ten held-out tests of the second pre-specified set (26 cities). SE: bootstrap standard error of the "
           "held-out slope (from its 95\\% interval). MDE: smallest true slope detectable with 80\\% power by a two-sided test "
           "at $\\alpha=0.05$ without multiplicity correction; under Holm's first step ($\\alpha=0.005$) it is 1.30 times larger. "
           "Last column: the end of the held-out interval in the pre-specified direction, as a multiple of the discovery slope "
           "(effects up to this size are not excluded).")
    return table("S12", cap, ["ID", "Outcome $\\sim$ characteristic", "Discovery", "Held-out", "SE", "MDE", "MDE/disc.", "Not excluded/disc."],
                 rows, "lp{5.2cm}rrrrrr", size="\\scriptsize")


def s8_label(o: str) -> str:
    """Outcome labels of Table S8 in mathematical notation."""
    return (tex_escape(o).replace("e\\_d at 5-10 km", "$e_d$ at 5--10~km").replace("turning distance r (km)", "turning distance $r$ (km)")
            .replace("ln r (log-log slope)", "$\\ln r$ (log--log slope)"))


def _size_checks() -> list:
    """Three checks of the held-out slope of the flexible model's 5-10 km distance elasticity on log population:
    elasticities at the common reference point, log device penetration as a covariate, and each city left out in turn."""
    c = cities()
    c = c[c["set"].astype(str).str.lower().str.startswith("h") | c["set"].astype(str).str.lower().str.contains("held|law")]
    assert len(c) == 26, len(c)
    c = c.assign(lp=np.log(c["population"].astype(float)), pen=np.log(c["home_devices"].astype(float) / c["population"].astype(float)))
    band = pd.read_csv(FD / "bands.csv", dtype={"city": str})
    fixed = pd.read_parquet(FD / "rev" / "banded_fixed.parquet")
    fixed = fixed[fixed["ref"] == "fixed"] if "ref" in fixed.columns else fixed
    pick = lambda e: e[(e["model"] == "Flexible") & (e["band"] == 3)].assign(city=lambda d: d["city"].astype(str))[["city", "ed"]]  # noqa: E731
    fb, ff = c.merge(pick(band), on="city"), c.merge(pick(fixed), on="city")
    assert len(fb) == 26 and len(ff) == 26

    def slope(X, y):
        return float(np.linalg.lstsq(np.column_stack([np.ones(len(y))] + list(X)), y, rcond=None)[0][1])

    def cell(X, y, tag):
        rng = np.random.default_rng([20261122, tag])
        n = len(y)
        bs = []
        for _ in range(5000):
            i = rng.integers(0, n, n)
            bs.append(slope([v[i] for v in X], y[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return f"{num(slope(X, y))} [{num(float(lo))}, {num(float(hi))}]"

    yb, yf = fb["ed"].to_numpy(float), ff["ed"].to_numpy(float)
    loo = [slope([np.delete(fb["lp"].to_numpy(), k)], np.delete(yb, k)) for k in range(len(yb))]
    q = "Flexible model: $e_d$ at 5--10~km, "
    rows = [[q + "common reference point", "Held-out (26)", cell([ff["lp"].to_numpy()], yf, 1), "--"],
            [q + "one city left out in turn", "Held-out (25)", f"{num(min(loo))} to {num(max(loo))}", "--"]]
    st = pd.read_parquet(FD / "rev" / "banded_strata.parquet")       # thirds of origins by distance to the CBD
    for k, name in enumerate(("inner", "middle", "outer")):
        g = c.merge(st[(st["stratum"] == name) & (st["model"] == "Flexible") & (st["band"] == 3) & (st["ref"] == "band")]
                    .assign(city=lambda d: d["city"].astype(str))[["city", "ed"]], on="city")
        assert len(g) == 26, (name, len(g))
        rows.append([q + f"{name} third of origins", "Held-out (26)", cell([g["lp"].to_numpy()], g["ed"].to_numpy(float), 30 + k), "--"])
    pen = " device penetration controlled"
    for k, m in enumerate(("Flexible", "Reach model", "Tempered power law", "Two-regime power law")):
        g = c.merge(band[(band["model"] == m) & (band["band"] == 3)].assign(city=lambda d: d["city"].astype(str))[["city", "ed"]], on="city")
        assert len(g) == 26, (m, len(g))
        name = "Flexible model" if m == "Flexible" else m
        rows.append([f"{name}: $e_d$ at 5--10~km," + pen, "Held-out (26)",
                     cell([g["lp"].to_numpy(), g["pen"].to_numpy()], g["ed"].to_numpy(float), 10 + k), "--"])
    tr = pd.read_csv(FD / "turning.csv", dtype={"city": str})
    g = c.merge(tr[["city", "theta_turn"]], on="city")
    assert len(g) == 26, len(g)
    rows.append(["Two-regime power law: turning distance $r$ (km)," + pen, "Held-out (26)",
                 cell([g["lp"].to_numpy(), g["pen"].to_numpy()], g["theta_turn"].to_numpy(float), 20), "--"])
    # the same spline knots in every city (rev_round5.py knots; knots at exp = 15.14, 26.42, 43.6 km,
    # ledger 2026-11-23), and two city-level covariates (figdata/rev/city_controls_r5.csv: ACS 2015-2019 county commute times
    # B08013 / B08303 summed over the city's counties; median over home tracts of the POIs within 5 km)
    kn = pd.read_parquet(FD / "rev" / "banded_knots.parquet")
    gk = c.merge(kn[(kn["model"] == "Flexible") & (kn["band"] == 3) & (kn["ref"] == "band")]
                 .assign(city=lambda d: d["city"].astype(str))[["city", "ed"]], on="city")
    assert len(gk) == 26
    rows.append([q + "same spline knots in every city", "Held-out (26)", cell([gk["lp"].to_numpy()], gk["ed"].to_numpy(float), 80), "--"])
    x = pd.read_csv(FD / "rev" / "city_controls_r5.csv", dtype={"city": str})
    cc = c.merge(x.assign(lc=np.log(x["commute_min"]), ld=np.log(x["poi_5km_median"]))[["city", "lc", "ld"]], on="city")
    assert len(cc) == 26

    def interval(X, y, j, tag):
        rng = np.random.default_rng([20261122, tag])
        n = len(y)
        fit = lambda idx: float(np.linalg.lstsq(np.column_stack([np.ones(len(idx))] + [v[idx] for v in X]), y[idx], rcond=None)[0][j])  # noqa: E731
        return np.percentile([fit(rng.integers(0, n, n)) for _ in range(5000)], [2.5, 97.5])

    ctl = (("lc", " mean commute time controlled", 50), ("ld", " POI density controlled", 60))
    for k, m in enumerate(("Flexible", "Reach model", "Tempered power law", "Two-regime power law")):
        gm = cc.merge(band[(band["model"] == m) & (band["band"] == 3)].assign(city=lambda d: d["city"].astype(str))[["city", "ed"]], on="city")
        assert len(gm) == 26, (m, len(gm))
        for col, lab, tag in ctl:
            X, y = [gm["lp"].to_numpy(), gm[col].to_numpy()], gm["ed"].to_numpy(float)
            lo, hi = interval(X, y, 1, tag + k)
            assert hi < 0, (m, col, lo, hi)                      # caption: the population slope stays negative, interval excludes zero
            lo2, hi2 = interval(X, y, 2, tag + 5 + k)
            assert lo2 < 0 < hi2, (m, col, lo2, hi2)             # caption: the covariate's own slope is not distinguishable from zero
            if m == "Flexible":
                rows.append([q[:-2] + "," + lab, "Held-out (26)", cell(X, y, tag + k), "--"])
    gt = cc.merge(tr[["city", "theta_turn"]], on="city")
    assert len(gt) == 26
    for col, lab, tag in ctl:
        rows.append(["Two-regime power law: turning distance $r$ (km)," + lab, "Held-out (26)",
                     cell([gt["lp"].to_numpy(), gt[col].to_numpy()], gt["theta_turn"].to_numpy(float), tag + 4), "--"])
    # a turning distance read from the flexible model alone (rev_round5.py rstar: last crossing of a threshold)
    rs = pd.read_parquet(FD / "rev" / "rstar.parquet").assign(city=lambda d: d["city"].astype(str))
    for k, thr in enumerate(("1.25", "1.5", "1.75")):
        col = f"r_{float(thr):.2f}"
        gr = c.merge(rs[["city", col]], on="city")
        assert len(gr) == 26 and gr[col].notna().all(), (thr, len(gr))
        rows.append([f"Flexible model: distance beyond which $e_d$ stays below $-${thr} (log--log slope)", "Held-out (26)",
                     cell([gr["lp"].to_numpy()], np.log(gr[col].to_numpy(float)), 90 + k), "--"])
    return rows


def s13() -> str:
    t = pd.read_csv(RT / "size_robust.csv")
    rows = []
    for (o, c), g in t.groupby(["outcome", "cities"], sort=False):
        g = g.set_index("control")
        a, b = g.loc["none"], g.loc["log mean tract area"]
        rows.append([s8_label(o), c, f"{num(a['slope'])} [{num(a['ci_lo'])}, {num(a['ci_hi'])}]",
                     f"{num(b['slope'])} [{num(b['ci_lo'])}, {num(b['ci_hi'])}]"])
    rows += [["\\pagebreak\n\\multicolumn{4}{l}{\\textit{Further checks in the held-out cities}}"]] + _size_checks()
    tr = pd.read_csv(FD / "turning.csv")                  # the prefactor of the log-log fit: r at one million inhabitants
    r1 = {}
    for k, g in (("discovery", tr[tr["set"] == "discovery"]), ("heldout", tr[tr["set"] == "heldout"]), ("all", tr)):
        slope, icpt = np.polyfit(g["log_population"] - np.log(1e6), np.log(g["theta_turn"]), 1)
        ref = t[(t["outcome"].str.startswith("Two-regime power law: ln r")) & (t["control"] == "none") & (t["n"] == len(g))]["slope"]
        assert len(ref) == 1 and abs(slope - float(ref.iloc[0])) < 1e-9, (k, slope)      # same fit as the ln r rows above
        r1[k] = float(np.exp(icpt))
    cap = ("City-size slopes with and without log mean tract area as a covariate: OLS slope on log population with "
           "city-bootstrap 95\\% intervals (5,000 resamples). Elasticities are evaluated at each city's band-specific "
           "reference point; the turning distance $r$ in km and, in the last rows, $\\ln r$ (log--log slope). "
           "The $\\ln r$ slope without the covariate is the exponent $-\\kappa$ of $r=r_1(P/10^{6})^{-\\kappa}$, with "
           f"$r_1={r1['discovery']:.1f}$~km in the discovery cities, ${r1['heldout']:.1f}$~km in the held-out cities and "
           f"${r1['all']:.1f}$~km across all 67. Correlation "
           "between log population and log mean tract area: $-0.63$ (67 cities). Intervals use an independent set of bootstrap resamples and "
           "may differ slightly from those in Table~S3 and in the main text. For the Reach model, the ratio in Fig.~3c uses the discovery slope on which "
           "the hypothesis was based ($-0.125$, from 34 of the 35 discovery cities), whereas the slope here uses all 35. "
           "Last eighteen rows, all for the held-out cities: the slope with elasticities evaluated at the common reference point of "
           "Table~S6; the range of the 26 slopes obtained by leaving out one city at a time; the slopes with the flexible model "
           "refitted on the inner, middle and outer third of each city's origins by distance to the CBD; the slopes with the logarithm of "
           "the number of resident devices per inhabitant as a covariate; and three further checks. "
           "Same spline knots: the banded flexible model refitted with the interior knots of its distance spline at 15, 26 and 44~km in "
           "every city, the medians of the cities' own knots. Mean commute time: log of the mean travel time to work in the city's "
           "counties (ACS 2015--2019), as a covariate. POI density: log of the median number of POIs within 5~km of a home tract, as a "
           "covariate. Neither covariate has a slope distinguishable from zero once population is included, and with either one the "
           "population slopes of the three formulas remain negative with intervals that exclude zero. "
           "Distance beyond which $e_d$ stays below a threshold: a turning distance read from the flexible model alone, namely the "
           "distance from which its distance elasticity remains steeper than the threshold.")
    return table("S8", cap, ["Quantity", "Cities", "Slope", "Slope, tract area controlled"], rows, "p{5.2cm}lrr", size="\\scriptsize",
                 long=True)                  # with the added checks the table no longer fits one page


def s18() -> str:
    b = pd.read_csv(FD / "rev" / "flows_baselines.csv", dtype={"city": str})
    o = pd.read_csv(TB / "tableS_od_cities.csv")
    short = lambda n: n.split(",")[0].split("-")[0]  # noqa: E731
    b["short"] = b["name"].map(short)
    order = ["Denver", "Philadelphia", "Phoenix", "Pittsburgh", "Sacramento", "San Diego"]
    lab = {"reach_model": "Reach model", "strexp3_N": "Relaxed stretched exp., POI (strongest benchmark)",
           "distance_only": "Distance only, $\\exp(-\\beta\\ln d)$", "uniform_dest": "Uniform destinations",
           "reach_model_pop": "Reach model, origin totals $\\propto$ population"}
    # The other five versions of Fig. 6 (od_maps, the same pipeline as the Fig. 6 source data), so that the Results claim
    # "the Reach model ranked first in each of the other five confirmation cities" can be read off the table (2026-11-23).
    o["short"] = o["name"].map(short)
    other = [("grav_power_N", "Power-law gravity, POI"), ("grav_power_P", "Power-law gravity, population"),
             ("rad_N", "Radiation, POI"), ("rad_P", "Radiation, population"),
             ("strexp1_P", "Stretched exp., Guo et al., population")]
    o = o.rename(columns={"in_observed_top": "top2000"})
    rows, versions = [], {}
    for metric, nd in (("top2000", 0), ("cpc", 3)):
        rows.append([f"\\multicolumn{{7}}{{l}}{{\\textit{{{'Top-2,000 flows recovered' if metric == 'top2000' else 'CPC'}}}}}"])
        for m in ("reach_model", "strexp3_N"):
            g = b[b["model"] == m].set_index("short")[metric]
            versions[(metric, m)] = g
            rows.append([lab[m]] + [num(float(g.get(c, np.nan)), nd) for c in order])
        for m, name in other:
            g = o[o["model"] == m].set_index("short")[metric]
            assert set(order) <= set(g.index), (m, metric)
            versions[(metric, m)] = g
            rows.append([name] + [num(float(g[c]), nd) for c in order])
        reach_od = o[o["model"] == "c0355r1"].set_index("short")[metric]
        assert all(abs(float(reach_od[c]) - float(versions[(metric, "reach_model")][c])) < 1e-6 for c in order), metric
        for c in order:      # Results 2.6: the Reach model ranks first among the seven versions in every city, on both measures
            assert all(versions[(metric, "reach_model")][c] > versions[(metric, m)][c] for (k, m) in versions if k == metric and m != "reach_model"), (metric, c)
    cap = ("Flow reconstruction in the six confirmation cities with uniform parameters (no local visits used to calibrate destination choice), test period, all "
           "destinations. Flows are the observed test-period origin totals times $P(j\\mid i)$. "
           "The models are the seven versions shown in Fig.~6.")
    body_rows = rows
    head = ["Model"] + order
    lines = []
    for r in body_rows:
        lines.append(r if len(r) > 1 else r)
    txt = table("S12", cap, head, [r if len(r) > 1 else [r[0]] for r in lines], "p{5.2cm}rrrrrr", size="\\scriptsize")
    return txt.replace(" & \\\\", " \\\\")


MASS_STEPS = [("grav_N", "grav_P", "POI count instead of residential population"),
              ("grav_E", "grav_P", "LODES employment instead of residential population"),
              ("grav_N", "grav_E", "POI count instead of LODES employment")]
COUPLING_STEPS = [("str3a_N", "str3_N", "Relaxed stretched exponential plus $\\theta\\ln a_j$ vs relaxed stretched exponential"),
                  ("sk", "sep_N", "Reach model vs its separable twin, POI count as mass"),
                  ("sk_E1", "sep_E1", "Reach model vs its separable twin, LODES employment as mass")]
FORMULA_STEPS = [("sk_frozen", "tp_frozen", "Reach model vs Tempered power law"),
                 ("sk_frozen", "tr_frozen", "Reach model vs Two-regime power law")]


def _contrasts(line):
    p = RT / f"models_{line}_contrasts.csv"
    return pd.read_csv(p) if p.exists() else None


def _cell(c, a, b, lv="local", name="target"):
    if c is None:
        return "--"
    g = c[(c["a"] == a) & (c["b"] == b) & (c["level"] == lv) & (c["cities"] == name)]
    if not len(g):
        return "--"
    r = g.iloc[0]
    return f"{num(r['dNLL median'], 4)} [{num(r['ci_lo'], 4)}, {num(r['ci_hi'], 4)}] {r['a better']}"


OUTSIDE = [("c0413c1r1", "Reach model vs variable-exponent control"),
           ("c0079c1", "Reach model vs shifted log-gravity control")]


def _paired_cell(a: str, la: str, b: str, lb: str, tag: int, B: int = 5000) -> str:
    """Paired NLL difference (model b minus model a, positive: a fits better) over the 32 target cities from the scored
    table of the revision models; city bootstrap of the median with a seed fixed per comparison."""
    t = scored("all")
    t = t[t["set"] == "target"]
    x = t[(t["model"] == a) & (t["level"] == la)].set_index("city")["nll"].astype(float)
    y = t[(t["model"] == b) & (t["level"] == lb)].set_index("city")["nll"].astype(float)
    d = (y - x).to_numpy()
    assert len(d) == 32 and np.isfinite(d).all(), (a, la, b, lb, len(d))
    rng = np.random.default_rng([20261122, tag])
    bs = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{num(float(np.median(d)), 4)} [{num(float(lo), 4)}, {num(float(hi), 4)}] {int((d > 0).sum())}/{len(d)}"


def _outside_cell(b: str, B: int = 5000) -> str:
    """Paired NLL difference (control minus Reach model) over the 32 target cities, local parameters, test period and
    held-out block; city bootstrap of the median with a seed fixed per comparison."""
    t = pd.read_csv(FD / "p9_target.csv", dtype={"city": str})
    t = t[t["level"] == "local"]
    x = t[t["model"] == "c0355r1"].set_index("city")["nll"].astype(float)
    y = t[t["model"] == b].set_index("city")["nll"].astype(float)
    d = (y - x).dropna().to_numpy()
    assert len(d) == 32, (b, len(d))
    rng = np.random.default_rng([20261122, int(b[1:5])])
    bs = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{num(float(np.median(d)), 4)} [{num(float(lo), 4)}, {num(float(hi), 4)}] {int((d > 0).sum())}/{len(d)}"


def s7() -> str:
    """Destination mass in power-law gravity, the comparison of the three formulas, three rows on what the coupling
    contributes, the two expressions kept outside the library and
    the employment-mass transfer."""
    c = _contrasts("all")
    t = scored("all")
    t = t[(t["set"] == "target") & (t["level"] == "local") & t["model"].isin(["grav_P", "grav_E", "grav_N"])]
    assert t.groupby("model")["city"].nunique().eq(32).all() and np.isfinite(t["nll"].astype(float)).all()
    sh = t.groupby("model")["share"].median()
    rows = [["\\multicolumn{2}{l}{\\textit{Destination mass in power-law gravity}}"]]
    rows += [[lab, _cell(c, x, y)] for x, y, lab in MASS_STEPS]
    rows += [["\\midrule\n\\multicolumn{2}{l}{\\textit{The three formulas}}"]]
    rows += [[lab, _cell(c, x, y)] for x, y, lab in FORMULA_STEPS]
    rows += [["\\midrule\n\\multicolumn{2}{l}{\\textit{What the coupling contributes}}"]]
    same = "Same comparison, uniform parameters"
    (xa, ya, la), (xb, yb, lb), (xc, yc, lc) = COUPLING_STEPS
    rows += [[la, _cell(c, xa, ya)], [same, _cell(c, xa, ya, lv="uniform")],
             [lb, _cell(c, xb, yb)], [same, _cell(c, xb, yb, lv="uniform")],
             [lc, _cell(c, xc, yc)],
             ["Nesting model vs separable twin, POI count as mass", _paired_cell("sk_plus1", "local", "sep_N", "local", 40)],
             ["Nesting model vs Reach model", _cell(c, "sk_plus1", "sk")], [same, _cell(c, "sk_plus1", "sk", lv="uniform")]]
    rows += [["\\midrule\n\\multicolumn{2}{l}{\\textit{Form of the area term}}"]]
    rows += [["Reach model with $\\theta_4\\ln a_j$ vs Reach model", _cell(c, "sk_lna", "sk")], [same, _cell(c, "sk_lna", "sk", lv="uniform")]]
    rows += [["\\midrule\n\\multicolumn{2}{l}{\\textit{Expressions kept outside the library (Note~S3)}}"]]
    rows += [[lab, _outside_cell(b)] for b, lab in OUTSIDE]
    te = scored("all")
    te = te[(te["set"] == "target") & (te["level"] == "uniform") & (te["model"] == "sk_E")]
    assert te["city"].nunique() == 32 and np.isfinite(te["nll"].astype(float)).all()
    rows += [["\\midrule\n\\multicolumn{2}{l}{\\textit{Reach model vs relaxed stretched exponential, both with LODES employment as mass}}"]]
    rows += [["Both with local parameters", _cell(c, "sk_E", "str3_E")],
             ["Both with uniform parameters", _cell(c, "sk_E", "str3_E", lv="uniform")],
             ["Reach model with uniform, relaxed stretched exponential with local parameters",
              _paired_cell("sk_E", "uniform", "str3_E", "local", 32)]]
    assert all("--" not in r[-1] for r in rows if len(r) == 2)
    rows = [[r[0], r[1].replace("$-$0.0000 [", "0.0000 [")] if len(r) == 2 else r for r in rows]      # a median that rounds to zero
    cap = ("Destination mass and comparison of the three formulas in the 32 target cities, with local parameters unless a row says otherwise. Each cell "
           "gives the paired difference $\\Delta$NLL $=\\mathrm{NLL}_b-\\mathrm{NLL}_a$ in nats per visit for the row's comparison "
           "of model $a$ with model $b$ (positive: $a$ fits better; median over cities, city-bootstrap 95\\% interval from 5,000 "
           "resamples) and the number of cities in which $a$ fits better. Mass rows: power-law gravity fitted with residential "
           "population, LODES workplace employment or POI count as destination mass; its median shares are "
           f"{num(abs(float(sh['grav_P'])), 2)}, {num(float(sh['grav_E']), 2)} and {num(float(sh['grav_N']), 2)}. "
           "Formula rows use the city fits of the main analysis, and the two expressions kept outside the library are compared with the Reach model in the same way. The separable twin, "
           "$\\eta_{ij}=a_j^{\\theta_4}+c\\ln(N_j+\\theta_2)-b\\,d^{\\alpha}$, has the inputs and the number of parameters of the Reach "
           "model but no coupling of distance and size; with employment as mass, both it and the Reach model have the shift fixed at 1. "
           "The nesting model adds $\\gamma\\ln(N_j+1)$ to the Reach model, so that POI count enters both through the distance scale and "
           "as a separate term. Rows marked uniform use the medians of the discovery-city fits for both models. With "
           "$\\theta_4\\ln a_j$ in place of $a_j^{\\theta_4}$ the utility no longer depends on the unit of area. "
           "Last three rows: both models with LODES employment in place of POI count. Uniform parameters are the medians of the fits "
           "in the 35 discovery cities, and with them the Reach model's median share is "
           f"{num(float(te['share'].astype(float).median()), 2)}.")
    return table("S7", cap, ["Comparison ($a$ vs $b$)", "$\\Delta$NLL, median [95\\% CI], cities"], rows, "p{7.2cm}c")


def s4() -> str:
    t = pd.read_csv(RT / "abs_nll.csv", dtype={"city": str})
    u = t["nll_uniform_choice"]
    gap = t["nll_gravity_pop"] - t["nll_flexible"]
    rows = []
    for k, lab in (("nll_uniform_choice", "Uniform choice"), ("nll_gravity_pop", "Power-law gravity, population (share 0)"),
                   ("nll_gravity_poi", "Power-law gravity, POI count"), ("nll_strexp3_poi", "Relaxed stretched exponential, POI"),
                   ("nll_two_regime", "Two-regime power law"), ("nll_tempered", "Tempered power law"),
                   ("nll_reach", "Reach model"), ("nll_flexible", "Flexible model (share 1)")):
        v = t[k]
        rows.append([lab, num(v.median(), 4), f"{num(v.quantile(.25), 3)}--{num(v.quantile(.75), 3)}",
                     num((1 - v / u).median(), 3), num(((t['nll_gravity_pop'] - v) / gap).median(), 3)])
    ch = t[t["city"] == "16700"].iloc[0]
    cap = ("Absolute fit on the test period and held-out block: medians over the 67 cities. NLL in nats per visit; "
           "pseudo-$R^2=1-\\mathrm{NLL}/\\mathrm{NLL}_{\\mathrm{uniform}}$ (McFadden, against uniform choice over the block's "
           f"destinations); share as in the main text. The gravity-to-flexible gap has a median of {num(gap.median(), 3)} nats "
           f"(range {num(gap.min(), 3)}--{num(gap.max(), 3)}). It is smallest in Charleston ({num(ch['gap_nats'], 3)}), where the "
           f"flexible model is {num(ch['nll_flexible'] - ch['nll_gravity_poi'], 3)} nats worse than POI gravity on the held-out "
           "block, which gives the shares above 2 in Fig.~2d. Re-anchored at POI gravity, the Reach model's median share is "
           f"{num(((t['nll_gravity_poi'] - t['nll_reach']) / (t['nll_gravity_poi'] - t['nll_flexible'])).median(), 2)}. "
           f"The Reach model's NLL lies a median of {num((t['nll_reach'] - t['nll_flexible']).median(), 3)} nats above the flexible "
           "model's when the difference is taken city by city; "
           "the medians in the table are taken separately for each model. Per-city values are in Additional file~2.")
    return table("S4", cap, ["Model", "NLL", "IQR", "Pseudo-$R^2$", "Share"], rows, "lrrrr")


def s5() -> str:
    sm = pd.read_csv(RT / "sk_params_summary.csv").set_index("parameter")
    t = pd.read_csv(RT / "sk_params.csv")
    lab = {"theta0": "$\\theta_0$ (amplitude)", "theta1": "$\\theta_1$ (distance exponent)", "theta2": "$\\theta_2$ (POI shift)",
           "theta3": "$\\theta_3$ (coupling exponent)", "theta4": "$\\theta_4$ (area exponent)",
           "theta3_over_theta1": "$\\theta_3/\\theta_1$ (exponent of the reach in $N_j+\\theta_2$)"}
    rows = []
    for k, l in lab.items():
        r = sm.loc[k]
        default = (sm.loc["theta3", "default (discovery median)"] / sm.loc["theta1", "default (discovery median)"]
                   if k == "theta3_over_theta1" else r["default (discovery median)"])
        rows.append([l, num(default, 3), num(r["median (67)"], 3),
                     f"{num(r['q25 (67)'], 3)}--{num(r['q75 (67)'], 3)}", f"{num(t[k].min(), 3)}--{num(t[k].max(), 3)}",
                     num(r["median se"], 3)])
    th = sm["default (discovery median)"].round(3)
    eta = 2.6 ** th["theta4"] - th["theta0"] * 5.0 ** th["theta1"] / (71.0 + th["theta2"]) ** th["theta3"]
    cap = ("Reach-model constants, $\\eta_{ij}=a_j^{\\theta_4}-\\theta_0 d^{\\theta_1}/(N_j+\\theta_2)^{\\theta_3}$ with $d$ in km, "
           "$a_j$ in km$^2$ and $N_j$ the count of places with at least one visit. Default: the uniform parameter vector used "
           "for transfer (median over the 35 discovery cities; for $\\theta_3/\\theta_1$, the ratio of the two default values). SE: median standard error of the city fits, clustered by 20 random groups of devices (for $\\theta_2$, over the 62 cities in which its constraint is not active). $\\theta_0$, "
           "$\\theta_1$ and $\\theta_3$ are positive in all 67 cities, so the size elasticity rises with distance in every city. "
           "The shift is constrained to $\\theta_2\\ge 1$, which is active in five cities. Worked example with the default: a destination 5~km "
           f"away with $N_j=71$ and $a_j=2.6$~km$^2$ has $\\eta={eta:.3f}$; choice probabilities are $\\exp(\\eta_{{ij}})$ "
           "normalised over the origin's destinations. Per-city constants are in Additional file~2.")
    return table("S5", cap, ["Parameter", "Default", "Median", "IQR", "Range", "SE"], rows, "lrrrrr")


# Plain-text labels for Additional file 2 (the workbook has no LaTeX).
PLAIN = {"null_unif": "Uniform choice", "grav_ref": "Power-law gravity, population (share 0)", "M2_ref": "Flexible model (share 1)",
         "grav_P": "Power-law gravity, population", "grav_N": "Power-law gravity, POI count",
         "str3_N": "Relaxed stretched exponential, POI count (strongest benchmark)",
         "str3a_N": "Relaxed stretched exponential, POI count, + ln a",
         "sep_N": "Separable twin of the Reach model", "sk": "Reach model", "sk_lna": "Reach model with theta4 ln a",
         "sk_plus": "Reach model + gamma ln(N+c)", "sk_plus1": "Reach model + gamma ln(N+1), shift +1",
         "grav_E": "Power-law gravity, employment", "str3_E": "Relaxed stretched exponential, employment",
         "sep_E": "Separable twin of the Reach model, employment", "sk_E": "Reach model, employment",
         "sep_E1": "Separable twin of the Reach model, employment, shift +1", "sk_E1": "Reach model, employment, shift +1",
         "sk_frozen": "Reach model (main-analysis fit)", "tp_frozen": "Tempered power law (main-analysis fit)",
         "tr_frozen": "Two-regime power law (main-analysis fit)"}
LINE_PLAIN = {"all": "all visits", "all2": "all visits, shifts held at +1", "poitrain": "POI count from March-April only",
              "nonwork": "non-work visits"}
SET_PLAIN = {"discovery": "discovery", "heldout": "held-out", "confirmation": "confirmation", "source": "discovery", "target": "target"}


def _plain_text(v):
    if not isinstance(v, str):
        return v
    return (v.replace("Scaled kernel", "Reach model").replace("(both refitted here)", "(both refitted)")
            .replace("refit here vs the frozen fit", "refitted vs the main-analysis fit").replace("frozen fit", "main-analysis fit"))


FORM_COV = {"log_population": "Population (log)", "road_density": "Main-road density", "compactness": "Residential compactness",
            "subcentre_job_share": "Employment-centre job share", "circuity": "Road circuity", "pop_density_gini": "Gini of tract population density",
            "cbd_job_share": "CBD job share", "jobs_housing": "Jobs-housing separation (job-resident dissimilarity index)", "n_subcentres": "Number of employment centres",
            "no_car_share": "Share of households without a vehicle", "orientation_entropy": "Street-orientation entropy",
            "poi_mix": "POI class entropy", "secondary_centres": "Secondary-centre job share",
            "transit_per_1k": "Transit stops per 1,000 inhabitants", "water_share": "Water share of the built-up footprint"}
_BAND = {"0_1km": "0-1 km", "1_2km": "1-2 km", "2_5km": "2-5 km", "5_10km": "5-10 km", "10_20km": "10-20 km", "20km": "beyond 20 km"}
_FORMULA = {"c0355r1": "Reach model", "c0262": "Tempered power law", "c0514c1": "Two-regime power law"}
_QUANTITY = {"theta_turn_c0514c1": "Two-regime power law: turning distance r (km)",
             "gain_grav_m2": "NLL gap between population gravity and the flexible model (nats per visit)",
             "share_s0_main": "Reach model: fit share with shifted gravity as the lower reference",
             "opp_near": "Variable-exponent control family fits near the best formula (0/1)",
             "ed_5_10km_resid": "Flexible model: e_d at 5-10 km minus the discovery-city trend with log population"}


def quantity_label(o: str) -> str:
    """Plain name of a city-level model quantity of the urban-form analyses (Fig. 4)."""
    if o in _QUANTITY:
        return _QUANTITY[o]
    stem, _, tag = o.rpartition("_")
    who, base = (_FORMULA[tag], stem) if tag in _FORMULA else ("Flexible model", o)
    if base.startswith(("ed_", "eN_")) and base[3:] in _BAND:
        band = _BAND[base[3:]]
        return f"{who}: e_{base[1]} {band if band.startswith('beyond') else 'at ' + band}"
    if base == "slope_eN_logd":
        return f"{who}: slope of e_N on ln d"
    if base == "transfer_loss":
        return f"{who}: transfer loss, extra NLL with discovery-median parameters (nats per visit)"
    if base == "transfer_frac":
        return f"{who}: transfer loss as a fraction of the gravity-to-flexible NLL gap"
    raise KeyError(f"no plain name for the model quantity {o!r}")


def form_relations() -> pd.DataFrame:
    """The ten tested relationships of Fig. 4a and the 69 further relationships of Fig. 4b: standardised slopes in the
    discovery and the held-out cities (figdata/rev/e1_tested.csv from e1_tested.py; figdata/e1_pairs.csv)."""
    t = pd.read_csv(FD / "rev" / "e1_tested.csv")
    o = pd.read_csv(FD / "e1_pairs.csv")
    t = t.assign(group=t["group"].map({"size": "pre-specified hypothesis, city size", "form": "pre-specified hypothesis, urban form"}), fig4a_row=t["row"].astype("Int64"))
    o = o.assign(group="further relationship", fig4a_row=pd.array([pd.NA] * len(o), dtype="Int64"))
    r = pd.concat([t, o], ignore_index=True)
    return pd.DataFrame({"group": r["group"], "fig4a_row": r["fig4a_row"], "city_characteristic": r["covariate"].map(FORM_COV),
                         "model_quantity": r["outcome"].map(quantity_label), "slope_discovery": r["d_std"], "slope_heldout": r["l_std"],
                         "same_sign": (np.sign(r["d_std"]) == np.sign(r["l_std"])).map({True: "yes", False: "no"})})


def additional_file_2(path: Path):
    """City-level results as a workbook (one sheet per table)."""
    c = cities()[["city", "name", "set", "role", "population", "n_units", "train_trips", "val_trips", "test_trips", "poi", "jobs",
                  "home_devices"]]
    sheets = {"README": pd.DataFrame({"sheet": ["cities", "fit_by_city", "reach_model_constants", "elasticities",
                                                "form_relations"],
                                      "content": ["the 67 metropolitan areas: group, population, tracts, visits by period, POIs, jobs, devices",
                                                  "NLL (nats per visit) per city and model, test period x held-out block",
                                                  "Reach-model constants and standard errors per city (local fits); theta2 has no standard error in the five cities where the constraint theta2 >= 1 is active",
                                                  "banded flexible model and the three formulas: e_d and e_N per city and distance band, "
                                                  "at band-specific and common reference points (ref), for four measurements of size, for weekend visits, for the flexible model with POI count and area only and for thirds of origins by distance to the CBD",
                                                  "urban form (Fig. 4a,b): the ten pre-specified hypotheses and the 69 further relationships "
                                                  "significant in the discovery cities, with standardised slopes in the discovery and the "
                                                  "held-out cities"]}),
              "cities": c, "fit_by_city": pd.read_csv(RT / "abs_nll.csv", dtype={"city": str}),
              "reach_model_constants": pd.read_csv(RT / "sk_params.csv", dtype={"city": str})}
    el = [pd.read_csv(FD / "bands.csv", dtype={"city": str}).assign(variant="POI count", ref="band")]
    for v, lab in (("fixed", "POI count"), ("emp", "employment"), ("empN", "employment, log N fixed"), ("nonwork", "POI count, non-work visits"),
                   ("weekend", "POI count, weekend visits"), ("poiarea", "POI count, flexible model with POI count and area only")):
        p = FD / "rev" / f"banded_{v}.parquet"
        if p.exists():
            el.append(pd.read_parquet(p).assign(variant=lab))
    p = FD / "rev" / "banded_strata.parquet"
    if p.exists():
        t3 = pd.read_parquet(p)
        el += [t3[t3["stratum"] == k].assign(variant=f"POI count, {k} third of origins") for k in ("inner", "middle", "outer")]
    sheets["elasticities"] = pd.concat(el, ignore_index=True)[["city", "variant", "ref", "model", "band", "d_ref", "ed", "eN"]]
    sheets["form_relations"] = form_relations()
    c2 = sheets["cities"]
    sheets["cities"] = c2.assign(set=c2["set"].map(SET_PLAIN), search_city=(c2["role"] == "search").map({True: "yes", False: "no"}))[
        ["city", "name", "set", "search_city", "population", "n_units", "train_trips", "val_trips", "test_trips", "poi", "jobs", "home_devices"]]
    for k in ("fit_by_city", "reach_model_constants"):
        sheets[k] = sheets[k].assign(set=sheets[k]["set"].map(SET_PLAIN))
    e = sheets["elasticities"]
    sheets["elasticities"] = e.assign(model=e["model"].map(_plain_text), ref=e["ref"].map({"band": "band-specific", "fixed": "common"}))
    notes = pd.DataFrame({"sheet": ["notes"], "content": [
        "NLL: negative log-likelihood in nats per visit; share = (NLL_gravity - NLL) / (NLL_gravity - NLL_flexible); "
        "level: local = parameters fitted in the city, uniform = medians of the 35 discovery-city fits; "
        "nll_strexp3_poi: relaxed stretched exponential with POI count (the strongest benchmark); "
        "form_relations: slopes are standardised (both variables z-scored within the city set) and, for urban-form "
        "characteristics, controlled for log population; the two city-size rows are slopes on log population itself"]})
    sheets["README"] = pd.concat([sheets["README"], notes], ignore_index=True)
    fr = sheets["form_relations"]
    assert len(fr) == 79 and not fr[["city_characteristic", "model_quantity"]].isna().any().any(), "form_relations: 10 + 69 named rows"
    with pd.ExcelWriter(path) as w:
        for k, df in sheets.items():
            df.to_excel(w, sheet_name=k, index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"T_S1": s1, "T_S4": s4, "T_S5": s5, "T_S6": s6, "T_S7": s7, "T_S8": s13, "T_S12": s18}
    for k, fn in files.items():
        (OUT / f"{k}.tex").write_text(fn())
        print("wrote", k)
    additional_file_2(OUT.parent / "additional_file_2.xlsx")
    print("wrote additional_file_2.xlsx")


if __name__ == "__main__":
    main()
