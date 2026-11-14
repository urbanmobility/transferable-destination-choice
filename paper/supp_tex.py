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
         "str3_N": "Relaxed stretched exp., POI", "str3a_N": "+ $\\ln a_j$", "int_N": "+ $\\ln d$, $\\ln d\\cdot\\ln(N_j+1)$",
         "sep_N": "Separable twin of the Scaled kernel", "sk": "Scaled kernel", "sk_lna": "Scaled kernel, $\\theta_4\\ln a_j$",
         "sk_plus": "Scaled kernel $+\\ \\gamma\\ln(N_j+c)$", "sk_plus1": "Scaled kernel $+\\ \\gamma\\ln(N_j+1)$, shift $+1$",
         "sep_E1": "Separable twin, employment, shift $+1$", "sk_E1": "Scaled kernel, employment, shift $+1$", "grav_E": "Power-law gravity, employment",
         "str3_E": "Relaxed stretched exp., employment", "sep_E": "Separable twin, employment", "sk_E": "Scaled kernel, employment",
         "sk_frozen": "Scaled kernel (frozen fit)", "tp_frozen": "Tempered power law (frozen fit)",
         "tr_frozen": "Two-regime power law (frozen fit)"}


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
    return (f"\\begin{{table}}[p]\n\\renewcommand{{\\thetable}}{{{label}}}\n\\centering\n{size}\n"
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
            ("nll_scaled", "Scaled"), ("nll_tempered", "Tempered"), ("nll_two_regime", "Two-reg."), ("nll_flexible", "Flexible"),
            ("gap_nats", "Gap"), ("pseudo_r2_scaled", "$R^2_{\\mathrm{McF}}$")]
    cols = [c for c in cols if c[0] in t]
    rows = [[tex_escape(r["name"].split(",")[0]), SET[r["set"]]] + [num(r[k], 3) for k, _ in cols] for _, r in t.iterrows()]
    cap = ("Absolute fit per city: NLL in nats per visit on the test period and reserved block. Uniform: uniform choice over "
           "the block's destinations. Grav. P and Grav. N: power-law gravity with residential population (the share-0 "
           "reference) and with POI count. Flexible: the 22-parameter reference (share 1). Gap: Grav. P minus Flexible. "
           "$R^2_{\\mathrm{McF}}$: 1 minus the Scaled kernel's NLL over the uniform-choice NLL. Group as in Table~S1.")
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
    cap = ("Scaled-kernel constants per city (standard errors in parentheses) in "
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


def s6() -> str:
    parts = []
    bands = pd.read_csv(FD / "bands.csv", dtype={"city": str})
    for ref, lab, med in (("band", "band medians (main figures)", bands.groupby(["model", "band"])[["ed", "eN"]].median().reset_index()),):
        for m in ("Flexible", "Scaled kernel", "Tempered power law", "Two-regime power law"):
            g = med[med["model"] == m].set_index("band")
            parts.append([f"{m}, {lab}", "$e_N$"] + [num(g["eN"].get(b), 2) for b in range(6)])
    for v, lab in (("fixed", "common point"), ("emp", "employment, common point"), ("empN", "employment, net of $\\ln N$"),
                   ("nonwork", "non-work visits, common point")):
        p = RT / f"banded_{v}_medians.csv"
        if not p.exists():
            continue
        g0 = pd.read_csv(p)
        g0 = g0[g0["ref"] == "fixed"]
        for m in (("Flexible", "Scaled kernel", "Tempered power law", "Two-regime power law") if v == "fixed" else ("Flexible",)):
            g = g0[g0["model"] == m].set_index("band")
            parts.append([f"{m}, {lab}", "$e_N$" if v not in ("emp", "empN") else "$e_E$"] + [num(g["eN"].get(b), 2) for b in range(6)])
    rise = []
    for v, lab in (("fixed", "POI count, common point"), ("emp", "employment"), ("empN", "employment, net of $\\ln N$"),
                   ("nonwork", "POI count, non-work visits")):
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
           "from the covariates; net of $\\ln N$: log POI count added as a covariate). Flexible-model elasticities are "
           "$\\gamma_b M/(M+1)$ at the reference mass $M$. Cities in which the elasticity rises with $\\ln d$ (OLS slope "
           "$>0$): " + "; ".join(rise) + ".")
    return table("S6", cap, ["Model and reference point", ""] + BANDS, parts, "llrrrrrr")


LADDER = [("grav_N", "grav_P"), ("str3_N", "grav_N"), ("str3a_N", "str3_N"), ("int_N", "str3a_N"), ("sk", "int_N"),
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
           "reserved block; local: fitted in the city; uniform: discovery-city medians (the city itself left out). A city in "
           "which a model's utility is undefined on the reserved block (a fitted shift below $-N_j$ of an unseen tract) counts "
           "as a loss for that model and a tie when both fail; the number of such cities is given in the source data. All "
           "models fitted with the same protocol (300 sampled origins, five starts) unless marked as frozen fits; "
           "``shift $+1$'' marks variants whose shift is held at $+1$.")
    out = table(label + "a", cap, HEAD_C, rc, "p{3.4cm}p{3.0cm}lrrrr", size="\\scriptsize", fit=True)
    if rm:
        cap2 = (f"Medians for the models of Table~{label}a over the {n_common} target cities in which every model listed has a "
                "finite NLL at both parameter levels: share of the gravity-to-flexible gap and NLL (nats per visit).")
        out += table(label + "b", cap2, HEAD_M, rm, "lrrrr", size="\\scriptsize")
    return out


QUANT = {"theta_turn_c0514c1": "Two-regime: turning distance $r$", "transfer_loss_c0355r1": "Scaled kernel: transfer loss",
         "ed_5_10km_c0262": "Tempered: $e_d$ at 5--10~km", "slope_eN_logd_c0355r1": "Scaled kernel: slope of $e_N$ on $\\ln d$",
         "ed_5_10km_c0355r1": "Scaled kernel: $e_d$ at 5--10~km", "ed_10_20km": "Flexible: $e_d$ at 10--20~km",
         "ed_1_2km": "Flexible: $e_d$ at 1--2~km", "ed_20km": "Flexible: $e_d$ beyond 20~km", "log_population": "log population",
         "road_density": "main-road density", "compactness": "residential compactness", "subcentre_job_share": "subcentre job share",
         "circuity": "road circuity", "pop_density_gini": "Gini of population density"}


def readable(s: str) -> str:
    a, b = [x.strip() for x in s.split("~")]
    return f"{QUANT.get(a, tex_escape(a))} $\\sim$ {QUANT.get(b, tex_escape(b))}"


def s12() -> str:
    t = pd.read_csv(RT / "form_power.csv")
    rows = [[r["id"], readable(r["outcome ~ covariate"]), num(r["discovery slope"], 3),
             num(r["held-out slope"], 3), num(r["se"], 3), num(r["MDE (80%, alpha .05)"], 3), num(r["MDE / |discovery|"], 2),
             num(r["largest effect not excluded / discovery"], 2)] for _, r in t.iterrows()]
    cap = ("Power of the ten held-out tests of the second registered set (26 cities). SE: bootstrap standard error of the "
           "held-out slope (from its 95\\% interval). MDE: smallest true slope detectable with 80\\% power by a two-sided test "
           "at $\\alpha=0.05$ without multiplicity correction; under Holm's first step ($\\alpha=0.005$) it is 1.30 times larger. "
           "Last column: the end of the held-out interval in the registered direction, as a multiple of the discovery slope "
           "(effects up to this size are not excluded).")
    return table("S12", cap, ["ID", "Outcome $\\sim$ characteristic", "Discovery", "Held-out", "SE", "MDE", "MDE/disc.", "Not excluded/disc."],
                 rows, "lp{5.2cm}rrrrrr", size="\\scriptsize")


def s13() -> str:
    t = pd.read_csv(RT / "size_robust.csv")
    rows = []
    for (o, c), g in t.groupby(["outcome", "cities"], sort=False):
        g = g.set_index("control")
        a, b = g.loc["none"], g.loc["log mean tract area"]
        rows.append([tex_escape(o), c, f"{num(a['slope'])} [{num(a['ci_lo'])}, {num(a['ci_hi'])}]",
                     f"{num(b['slope'])} [{num(b['ci_lo'])}, {num(b['ci_hi'])}]"])
    cap = ("City-size slopes with and without log mean tract area as a covariate: OLS slope on log population with "
           "city-bootstrap 95\\% intervals (5,000 resamples). Elasticities are evaluated at each city's band-specific "
           "reference point; the turning distance $r$ in km and, in the last rows, $\\ln r$ (log--log slope). Correlation "
           "between log population and log mean tract area: $-0.63$ (67 cities). Intervals come from a fresh bootstrap and can "
           "differ in the third decimal from those of the registered tests (Table~S3).")
    return table("S8", cap, ["Quantity", "Cities", "Slope", "Slope, tract area controlled"], rows, "p{5.2cm}lrr", size="\\scriptsize")


def s18() -> str:
    b = pd.read_csv(FD / "rev" / "flows_baselines.csv", dtype={"city": str})
    o = pd.read_csv(TB / "tableS_od_cities.csv")
    short = lambda n: n.split(",")[0].split("-")[0]  # noqa: E731
    b["short"] = b["name"].map(short)
    order = ["Denver", "Philadelphia", "Phoenix", "Pittsburgh", "Sacramento", "San Diego"]
    lab = {"scaled_kernel": "Scaled kernel", "strexp3_N": "Relaxed stretched exp., POI (strongest benchmark)",
           "distance_only": "Distance only, $\\exp(-\\beta\\ln d)$", "uniform_dest": "Uniform destinations",
           "scaled_kernel_pop": "Scaled kernel, origin totals $\\propto$ population"}
    rows = []
    for metric, nd in (("top2000", 0), ("cpc", 3)):
        rows.append([f"\\multicolumn{{7}}{{l}}{{\\textit{{{'Top-2,000 flows recovered' if metric == 'top2000' else 'CPC'}}}}}"])
        for m in ("scaled_kernel", "strexp3_N", "distance_only", "uniform_dest", "scaled_kernel_pop"):
            g = b[b["model"] == m].set_index("short")[metric]
            rows.append([lab[m]] + [num(float(g.get(c, np.nan)), nd) for c in order])
        if "model" in o.columns:
            pass
    rows = [r if len(r) > 1 else [r[0]] for r in rows]
    body_rows = [r for r in rows]
    cap = ("Flow reconstruction in the six confirmation cities with uniform parameters (no local visits), test period, all "
           "destinations. Flows are origin totals times $P(j\\mid i)$; except in the last row of each block, origin totals are "
           "the observed test-period totals. Uniform destinations and distance only show what the observed origin totals "
           "and distance alone recover. The other earlier models are compared in the source data of Fig.~6.")
    head = ["Model"] + order
    lines = []
    for r in body_rows:
        lines.append(r if len(r) > 1 else r)
    txt = table("S12", cap, head, [r if len(r) > 1 else [r[0]] for r in lines], "p{5.2cm}rrrrrr", size="\\scriptsize")
    return txt.replace(" & \\\\", " \\\\")


LINES3 = [("all", "All visits"), ("poitrain", "POI count from March--April"), ("nonwork", "Non-work visits")]
STEPS = [("grav_N", "grav_P", "POI count instead of population as mass"),
         ("str3_N", "grav_N", "Stretched-exponential instead of power-law distance"),
         ("str3a_N", "str3_N", "$+\ \ln a_j$ (the strongest benchmark plus area)"),
         ("int_N", "str3a_N", "$+\ \ln d$ and $\ln d\cdot\ln(N_j+1)$ (interaction logit)"),
         ("sk", "str3_N", "Scaled kernel vs the strongest benchmark"),
         ("sk", "sep_N", "Scaled kernel vs its separable twin (coupling)"),
         ("sk", "int_N", "Scaled kernel vs the interaction logit"),
         ("sk_plus1", "sk", "Scaled kernel $+\ \gamma\ln(N_j+1)$ vs Scaled kernel"),
         ("sk_lna", "sk", "Scaled kernel with $\\theta_4\\ln a_j$ vs with $a_j^{\\theta_4}$"),
         ("sk_frozen", "tp_frozen", "Scaled kernel vs Tempered power law"),
         ("sk_frozen", "tr_frozen", "Scaled kernel vs Two-regime power law")]
EMP_STEPS = [("grav_E", "grav_P", "Employment instead of population as mass"),
             ("grav_N", "grav_E", "POI count instead of employment as mass"),
             ("sk_E", "str3_E", "Scaled kernel vs stretched exponential, employment mass"),
             ("sk_E1", "sep_E1", "Scaled kernel vs separable twin, employment mass (shifts $+1$)")]


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


def s7() -> str:
    C = {l: _contrasts(l) for l, _ in LINES3}
    rows = [[lab] + [_cell(C[l], a, b) for l, _ in LINES3] for a, b, lab in STEPS]
    rows.append(["\\multicolumn{4}{l}{\\textit{Employment (LODES) as destination mass, all visits}}"])
    rows += [[lab, _cell(C["all"], a, b), "--", "--"] for a, b, lab in EMP_STEPS]
    cap = ("What the Scaled kernel's accuracy consists of: an ablation ladder in the 32 target cities with local parameters. "
           "Each cell gives the paired difference $\\Delta$NLL $=\\mathrm{NLL}_b-\\mathrm{NLL}_a$ in nats per visit for the row's "
           "comparison of model $a$ with model $b$ (positive: $a$ fits better; median over cities, city-bootstrap 95\\% interval "
           "from 5,000 resamples) and the number of cities in which $a$ fits better. A city in which a model's utility is "
           "undefined on the reserved block counts as a loss for that model. All models fitted with the same protocol; rows "
           "with the Tempered and Two-regime power laws use the frozen fits. Columns: visits of all purposes; the POI count "
           "rebuilt from the places observed in March and April only; visits to the inferred workplace removed. Results with "
           "uniform parameters, over all 67 cities, and per city are in Additional file~2.")
    out = table("S7a", cap, ["Comparison ($a$ vs $b$)"] + [h for _, h in LINES3], rows, "p{5.6cm}ccc", size="\\scriptsize", fit=True)
    out = out.replace("\\multicolumn{4}{l}{\\textit{Employment (LODES) as destination mass, all visits}} \\\\", "\\midrule\n\\multicolumn{4}{l}{\\textit{Employment (LODES) as destination mass, all visits}} \\\\")
    models = ["grav_P", "grav_E", "grav_N", "str3_N", "str3a_N", "int_N", "sep_N", "sk", "sk_plus1", "sk_lna", "sep_E1", "sk_E1"]
    rm, n_common = _median_rows("all", models)
    cap2 = (f"Median share of the gravity-to-flexible gap and NLL (nats per visit) of the models in Table~S7a, all visits, "
            f"over the {n_common} target cities in which every model listed has a finite NLL at both parameter levels.")
    return out + table("S7b", cap2, HEAD_M, rm, "lrrrr", size="\\scriptsize")


def s4() -> str:
    t = pd.read_csv(RT / "abs_nll.csv", dtype={"city": str})
    u = t["nll_uniform_choice"]
    gap = t["nll_gravity_pop"] - t["nll_flexible"]
    rows = []
    for k, lab in (("nll_uniform_choice", "Uniform choice"), ("nll_gravity_pop", "Power-law gravity, population (share 0)"),
                   ("nll_gravity_poi", "Power-law gravity, POI count"), ("nll_strexp3_poi", "Relaxed stretched exponential, POI"),
                   ("nll_two_regime", "Two-regime power law"), ("nll_tempered", "Tempered power law"),
                   ("nll_scaled", "Scaled kernel"), ("nll_flexible", "Flexible model (share 1)")):
        v = t[k]
        rows.append([lab, num(v.median(), 4), f"{num(v.quantile(.25), 3)}--{num(v.quantile(.75), 3)}",
                     num((1 - v / u).median(), 3), num(((t['nll_gravity_pop'] - v) / gap).median(), 3)])
    ch = t[t["city"] == "16700"].iloc[0]
    cap = ("Absolute fit on the test period and reserved block: medians over the 67 cities. NLL in nats per visit; "
           "pseudo-$R^2=1-\\mathrm{NLL}/\\mathrm{NLL}_{\\mathrm{uniform}}$ (McFadden, against uniform choice over the block's "
           f"destinations); share as in the main text. The gravity-to-flexible gap has a median of {num(gap.median(), 3)} nats "
           f"(range {num(gap.min(), 3)}--{num(gap.max(), 3)}). It is smallest in Charleston ({num(ch['gap_nats'], 3)}), where the "
           f"flexible model is {num(ch['nll_flexible'] - ch['nll_gravity_poi'], 3)} nats worse than POI gravity on the reserved "
           "block, which gives the shares above 2 in Fig.~2d. Re-anchored at POI gravity, the Scaled kernel's median share is "
           f"{num(((t['nll_gravity_poi'] - t['nll_scaled']) / (t['nll_gravity_poi'] - t['nll_flexible'])).median(), 2)}. "
           "Per-city values are in Additional file~2.")
    return table("S4", cap, ["Model", "NLL", "IQR", "Pseudo-$R^2$", "Share"], rows, "lrrrr")


def s5() -> str:
    sm = pd.read_csv(RT / "sk_params_summary.csv").set_index("parameter")
    t = pd.read_csv(RT / "sk_params.csv")
    lab = {"theta0": "$\\theta_0$ (amplitude)", "theta1": "$\\theta_1$ (distance exponent)", "theta2": "$\\theta_2$ (POI shift)",
           "theta3": "$\\theta_3$ (coupling exponent)", "theta4": "$\\theta_4$ (area exponent)",
           "theta3_over_theta1": "$\\theta_3/\\theta_1$ (growth of the distance scale with $N_j$)"}
    rows = []
    for k, l in lab.items():
        r = sm.loc[k]
        rows.append([l, num(r["default (discovery median)"], 3), num(r["median (67)"], 3),
                     f"{num(r['q25 (67)'], 3)}--{num(r['q75 (67)'], 3)}", f"{num(t[k].min(), 3)}--{num(t[k].max(), 3)}",
                     num(r["median se"], 3)])
    th = sm["default (discovery median)"].round(3)
    eta = 2.6 ** th["theta4"] - th["theta0"] * 5.0 ** th["theta1"] / (71.0 + th["theta2"]) ** th["theta3"]
    cap = ("Scaled-kernel constants, $\\eta_{ij}=a_j^{\\theta_4}-\\theta_0 d^{\\theta_1}/(N_j+\\theta_2)^{\\theta_3}$ with $d$ in km, "
           "$a_j$ in km$^2$ and $N_j$ the count of places with at least one visit. Default: the uniform parameter vector used "
           "for transfer (median over the 35 discovery cities). SE: median standard error of the city fits. $\\theta_0$, "
           "$\\theta_1$ and $\\theta_3$ are positive in all 67 cities, so the size elasticity rises with distance in every city; "
           "$\\theta_2$ is negative in four cities ($-1.53$ to $-0.05$). Worked example with the default: a destination 5~km "
           f"away with $N_j=71$ and $a_j=2.6$~km$^2$ has $\\eta={eta:.3f}$; choice probabilities are $\\exp(\\eta_{{ij}})$ "
           "normalised over the origin's destinations. Per-city constants are in Additional file~2.")
    return table("S5", cap, ["Parameter", "Default", "Median", "IQR", "Range", "SE"], rows, "lrrrrr")


def additional_file_2(path: Path):
    """City-level results as a workbook (one sheet per table)."""
    c = cities()[["city", "name", "set", "role", "population", "n_units", "train_trips", "val_trips", "test_trips", "poi", "jobs",
                  "home_devices"]]
    sheets = {"README": pd.DataFrame({"sheet": ["cities", "fit_by_city", "scaled_kernel_constants", "elasticities",
                                                "ablation_by_city", "ablation_contrasts"],
                                      "content": ["the 67 metropolitan areas: group, population, tracts, visits by period, POIs, jobs, devices",
                                                  "NLL (nats per visit) per city and model, test period x reserved block",
                                                  "Scaled-kernel constants and standard errors per city (local fits)",
                                                  "banded flexible model and library formulas: e_d and e_N per city and distance band, "
                                                  "at band-specific and common reference points, for four measurements",
                                                  "NLL and share per city, model, parameter level and input line of the ablation ladder",
                                                  "paired NLL differences (all contrasts, both parameter levels, 32 target and 67 cities)"]}),
              "cities": c, "fit_by_city": pd.read_csv(RT / "abs_nll.csv", dtype={"city": str}),
              "scaled_kernel_constants": pd.read_csv(RT / "sk_params.csv", dtype={"city": str})}
    el = [pd.read_csv(FD / "bands.csv", dtype={"city": str}).assign(variant="POI count", ref="band")]
    for v, lab in (("fixed", "POI count"), ("emp", "employment"), ("empN", "employment, log N fixed"), ("nonwork", "POI count, non-work visits")):
        p = FD / "rev" / f"banded_{v}.parquet"
        if p.exists():
            el.append(pd.read_parquet(p).assign(variant=lab))
    sheets["elasticities"] = pd.concat(el, ignore_index=True)[["city", "variant", "ref", "model", "band", "d_ref", "ed", "eN"]]
    ab = []
    for line in ("all", "all2", "poitrain", "nonwork"):
        p = FD / "rev" / f"models_{line}_scored.parquet"
        if p.exists():
            ab.append(pd.read_parquet(p)[["city", "set", "model", "level", "nll", "share"]].assign(line=line))
    sheets["ablation_by_city"] = pd.concat(ab, ignore_index=True)
    sheets["ablation_contrasts"] = pd.concat([pd.read_csv(RT / f"models_{l}_contrasts.csv").assign(line=l)
                                              for l in ("all", "poitrain", "nonwork") if (RT / f"models_{l}_contrasts.csv").exists()])
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
