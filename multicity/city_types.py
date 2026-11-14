"""Exploration E-3 / E-4 (form effects and city types): can the 67 cities be split
into stable types, and what do the types mean? Exploratory only.

Features: (main) skeleton-family support = per family the best gain share (grav -> M2) among its members on the test
block, minus the city's best family (<= 0; an unfittable family gets the global minimum lag); (control) behaviour
curves = the flexible model's e_d and e_N in five distance bands (cities with a missing band are left out).
Ward clustering of the standardized features for k = 2..6; a k is reported only when both gates pass - mean adjusted
Rand index over 200 80%-subsamples >= 0.6 and prediction strength (fit on the discovery cities, judged on the law-test
cities, standardized with the discovery statistics) >= 0.8 - and the largest such k is used; otherwise "no stable
types". Silhouette is descriptive. Behaviour types never stand in for skeleton types.

Usage (run_stage.sh, M line, after explore_form.py e2): city_types.py types | report
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import explore_form as ef  # noqa: E402


# --------------------------------------------------------------------------------------------- features
def family_profile(cv: pd.DataFrame, refs: dict, families: dict, floor: str = "global_min") -> pd.DataFrame:
    import scoring
    rows = {}
    for city, g in cv.groupby("city", sort=True):
        r = refs.get(city) or {}
        best = {}
        for fam, members in families.items():
            m = g[g["cid"].isin(members) & g["finite"].astype(bool)]
            sh = [scoring.share(float(v), r, "grav_power") for v in m["val_nll"]]
            sh = [s for s in sh if s is not None and np.isfinite(s)]
            best[fam] = max(sh) if sh else np.nan
        top = np.nanmax(list(best.values())) if any(np.isfinite(v) for v in best.values()) else np.nan
        if np.isfinite(top):
            rows[city] = {f: v - top for f, v in best.items()}
    prof = pd.DataFrame.from_dict(rows, orient="index", columns=list(families))
    if floor != "global_min":
        raise ValueError(f"floor rule {floor!r}: only 'global_min'")
    return prof.fillna(float(np.nanmin(prof.to_numpy())))


def behaviour_profile(out: pd.DataFrame, bands) -> pd.DataFrame:
    cols = [f"{q}_{b}" for b in bands for q in ("ed", "eN")]
    return out.set_index("city")[cols].astype(float).dropna()


def zscore(X: pd.DataFrame, ref: pd.DataFrame = None) -> pd.DataFrame:
    ref = X if ref is None else ref
    sd = ref.std(ddof=1).replace(0, 1.0)
    return (X - ref.mean()) / sd


# --------------------------------------------------------------------------------------------- clustering
def ward_labels(X, k: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    X = np.asarray(X, float)
    if len(X) < 2:
        return np.ones(len(X), int)
    return fcluster(linkage(X, "ward"), t=k, criterion="maxclust")


def stability(X, k: int, B: int, frac: float, seed: int) -> float:
    from sklearn.metrics import adjusted_rand_score
    X = np.asarray(X, float)
    full = ward_labels(X, k)
    rng = np.random.default_rng(seed)
    m = int(round(frac * len(X)))
    aris = []
    for _ in range(B):
        idx = rng.choice(len(X), m, replace=False)
        aris.append(adjusted_rand_score(full[idx], ward_labels(X[idx], k)))
    return float(np.mean(aris))


def prediction_strength(Xtr, Xte, k: int) -> float:
    Xtr, Xte = np.asarray(Xtr, float), np.asarray(Xte, float)
    ltr = ward_labels(Xtr, k)
    C = np.array([Xtr[ltr == j].mean(0) for j in np.unique(ltr)])
    assign = np.argmin(((Xte[:, None, :] - C[None, :, :]) ** 2).sum(-1), axis=1)
    lte = ward_labels(Xte, k)
    vals = []
    for j in np.unique(lte):
        a = assign[lte == j]
        if len(a) < 2:
            continue
        same = (a[:, None] == a[None, :]).sum() - len(a)
        vals.append(same / (len(a) * (len(a) - 1)))
    return float(min(vals)) if vals else float("nan")


def choose_k(X, Xtr, Xte, ks, min_ari: float, min_ps: float, B: int, frac: float, seed: int) -> dict:
    from sklearn.metrics import silhouette_score
    Xa = np.asarray(X, float)
    table, chosen = [], None
    for k in ks:
        lab = ward_labels(Xa, k)
        sil = float(silhouette_score(Xa, lab)) if 1 < len(np.unique(lab)) < len(Xa) else float("nan")
        st = stability(Xa, k, B, frac, seed)
        ps = prediction_strength(Xtr, Xte, k)
        table.append({"k": int(k), "silhouette": sil, "stability": st, "ps": ps})
        if st >= min_ari and np.isfinite(ps) and ps >= min_ps:
            chosen = int(k)
    return {"table": table, "k": chosen}


# --------------------------------------------------------------------------------------------- description
def describe(labels: pd.Series, out: pd.DataFrame, cov: pd.DataFrame, scores: pd.DataFrame, curves, covariates,
             size: str, B: int, seed: int, q: float, C: float) -> dict:
    """Per type: members, mean behaviour curves with city-bootstrap CIs, standardized covariate means; Kruskal-Wallis
    per covariate (BH across covariates); multinomial logistic type ~ size + form axes; crosstab with plan-7 classes."""
    from scipy.stats import kruskal
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    o = out.set_index("city").reindex(labels.index)
    c = cov.set_index("city").reindex(labels.index)
    zc = zscore(c[list(covariates)].astype(float))
    types = {}
    for t in sorted(labels.unique()):
        mem = labels.index[labels == t].tolist()
        cur = {}
        for col in curves:
            v = o.loc[mem, col].astype(float).dropna().to_numpy()
            bs = [rng.choice(v, len(v)).mean() for _ in range(B)] if len(v) > 1 else []
            cur[col] = {"mean": float(v.mean()) if len(v) else None,
                        "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None}
        types[int(t)] = {"n": len(mem), "members": mem, "curves": cur,
                         "covariates": {k: float(zc.loc[mem, k].mean()) for k in zc.columns}}
    kw = {}
    for k in zc.columns:
        groups = [zc.loc[labels.index[labels == t], k].dropna().to_numpy() for t in sorted(labels.unique())]
        groups = [g for g in groups if len(g)]
        kw[k] = float(kruskal(*groups).pvalue) if len(groups) > 1 and len({x for g in groups for x in g}) > 1 else float("nan")
    keys = list(kw)
    sel = ef.bh(np.array([kw[k] for k in keys]), q)
    X = pd.concat([zscore(c[[size]].astype(float)), scores.reindex(labels.index)], axis=1).dropna()
    logit = None
    if len(X) and labels.loc[X.index].nunique() > 1:
        m = LogisticRegression(C=C, max_iter=5000).fit(X.to_numpy(), labels.loc[X.index].to_numpy())
        logit = {"classes": [int(x) for x in m.classes_], "terms": list(X.columns),
                 "coef": np.atleast_2d(m.coef_).tolist(), "n": int(len(X))}
    cls = o["cls"] if "cls" in o else pd.Series(index=labels.index, dtype=object)
    cross = pd.crosstab(labels, cls.fillna("—")).to_dict()
    return {"types": types, "kruskal": {k: {"p": kw[k], "bh": bool(s)} for k, s in zip(keys, sel)}, "logit": logit,
            "crosstab_plan7_class": {str(a): {str(b): int(v) for b, v in d.items()} for a, d in cross.items()}}


def city_xy(cities) -> pd.DataFrame:
    """Population-weighted tract centroid (EPSG:5070) of each city, from the M line's units.parquet."""
    mi = config.line_constants("M")["MI"]
    rows = []
    for c in cities:
        u = pd.read_parquet(mi / str(c) / "tract" / "units.parquet", columns=["x", "y", "population"])
        w = u["population"].clip(lower=0).to_numpy(float)
        w = w if w.sum() > 0 else np.ones(len(u))
        rows.append({"city": str(c), "x": float(np.average(u["x"], weights=w)), "y": float(np.average(u["y"], weights=w))})
    return pd.DataFrame(rows).set_index("city")


def plot_map(xy: pd.DataFrame, labels: pd.Series, path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for t in sorted(labels.unique()):
        idx = labels.index[labels == t]
        ax.scatter(xy.loc[idx, "x"], xy.loc[idx, "y"], s=40, label=f"type {t} (n={len(idx)})")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------- stages
def _names() -> dict:
    c = pd.read_csv(config.CITIES_CSV, dtype=str)
    return {r.city: str(r.name).split(",")[0].split("-")[0] for r in c.itertuples()}


def stage_types() -> dict:
    import ledger
    import city_outcomes as co
    from sklearn.metrics import adjusted_rand_score
    cfg = ef.require_fixed()
    e3, e4, seed, size = cfg["e3"], cfg["e4"], cfg["seed"], cfg["size"]
    out, cov, group = ef.load_frame(tuple(cfg["groups"]))
    cities = out["city"].tolist()
    cv = co._cv("test_dest")
    cv = cv[cv["city"].astype(str).isin(cities)]
    fam = family_profile(cv, co._refs("test_dest", cities), e3["families"], e3["floor"])
    beh = behaviour_profile(out, e3["behaviour_bands"])
    scores = pd.read_csv(ef.OUT / "e2_axis_scores.csv", dtype={"city": str}).set_index("city")
    curves = [f"{q}_{b}" for b in e3["behaviour_bands"] for q in ("ed", "eN")]
    forms = [k for k in cfg["covariates"] if k not in ("log_jobs",)]
    res, lab = {"config_sha": ef._sha(ef.CONFIG), "features": {}}, {}
    for name, F in (("skeleton", fam), ("behaviour", beh)):
        tr = [c for c in F.index if group.get(c) == e3["prediction_strength"]["train"]]
        te = [c for c in F.index if group.get(c) == e3["prediction_strength"]["test"]]
        ch = choose_k(zscore(F), zscore(F.loc[tr], F.loc[tr]), zscore(F.loc[te], F.loc[tr]), e3["k"],
                      e3["stability"]["min_ari"], e3["prediction_strength"]["min"], e3["stability"]["B"],
                      e3["stability"]["frac"], seed)
        r = {"n": int(len(F)), "dropped": sorted(set(cities) - set(F.index)), "features": list(F.columns), **ch}
        if ch["k"]:
            labels = pd.Series(ward_labels(zscore(F), ch["k"]), index=F.index)
            lab[name] = labels
            r["describe"] = describe(labels, out, cov, scores, curves, forms, size, e4["bootstrap"], seed,
                                     e4["kw_fdr_q"], e4["logit_C"])
        res["features"][name] = r
    if len(lab) == 2:
        common = lab["skeleton"].index.intersection(lab["behaviour"].index)
        res["agreement_ari"] = float(adjusted_rand_score(lab["skeleton"][common], lab["behaviour"][common]))
    ef.OUT.mkdir(parents=True, exist_ok=True)
    p = ef.OUT / "types.json"
    save = lambda: p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))  # noqa: E731
    save()                                                  # every result is on disk before any plotting
    pd.DataFrame({n: s for n, s in lab.items()}).rename_axis("city").to_csv(ef.OUT / "types_members.csv")
    fam.rename_axis("city").to_csv(ef.OUT / "types_skeleton_features.csv")
    for name, labels in lab.items():                        # a failed map is recorded, never fatal
        r, png = res["features"][name], ef.OUT / f"types_{name}_map.png"
        try:
            plot_map(city_xy(labels.index), labels, png, f"{name} types (k={r['k']})")
            r["map"] = png.name
        except Exception as err:  # noqa: BLE001
            r["map"] = f"not drawn - {type(err).__name__}: {err}"
    if lab:
        save()
    ledger.add("Exploration E-3: " + "; ".join(
        f"{n} types: {'k ' + str(r['k']) if r['k'] else 'no stable types'} (n {r['n']}, best stability "
        f"{max(t['stability'] for t in r['table']):.2f}, best PS {np.nanmax([t['ps'] for t in r['table']]):.2f})"
        for n, r in res["features"].items()) + f"; {p.name} sha {ledger.sha(p)}")
    return res


def render_types(res: dict) -> str:
    f = ef._f
    names = _names() if Path(config.CITIES_CSV).exists() else {}
    L = ["# Exploration: city types (E-3, E-4)", "",
         "Exploratory. Types must pass two gates: stability (mean adjusted Rand index over 200 80% subsamples >= 0.6) and "
         "prediction strength (discovery cities -> law-test cities >= 0.8); k is the largest value that passes both. "
         "Silhouette is descriptive only. Behaviour types do not stand in for skeleton types.", ""]
    title = {"skeleton": "Skeleton types (by skeleton-family support)", "behaviour": "Behaviour types (by distance-band elasticity curves)"}
    for name, r in res["features"].items():
        L += [f"## {title.get(name, name)}", "", f"Cities: {r['n']}" + (f" ({len(r['dropped'])} cities dropped: missing data)" if r["dropped"] else ""), "",
              "| k | Silhouette | Stability | Prediction strength |", "|---|---|---|---|"]
        L += [f"| {t['k']} | {f(t['silhouette'], 2)} | {f(t['stability'], 2)} | {f(t['ps'], 2)} |" for t in r["table"]]
        if not r.get("k"):
            L += ["", "**No stable types**: no k passes both gates. The differences between cities are continuous, not categorical.", ""]
            continue
        d = r["describe"]
        L += ["", f"**k = {r['k']}**", ""]
        for t, v in d["types"].items():
            L.append(f"- Type {t} ({v['n']} cities): " + ", ".join(names.get(c, c) for c in v["members"]))
        curves = list(next(iter(d["types"].values()))["curves"])
        L += ["", "| Type | " + " | ".join(curves) + " |", "|---|" + "---|" * len(curves)]
        for t, v in d["types"].items():
            L.append(f"| {t} | " + " | ".join(f(v["curves"][c]["mean"]) for c in curves) + " |")
        L += ["", "| Covariate | " + " | ".join(f"Type {t}" for t in d["types"]) + " | Kruskal–Wallis p | FDR |",
              "|---|" + "---|" * (len(d["types"]) + 2)]
        for k, kw in d["kruskal"].items():
            L.append(f"| {k} | " + " | ".join(f(v["covariates"][k], 2) for v in d["types"].values())
                     + f" | {f(kw['p'], 2)} | {'yes' if kw['bh'] else 'no'} |")
        L += ["", "Cross-tabulation with the plan 7 skeleton classes (kernel / opp / both / neither): " + json.dumps(d["crosstab_plan7_class"], ensure_ascii=False), ""]
    if "agreement_ari" in res:
        L += [f"Agreement between skeleton types and behaviour types (adjusted Rand index): {f(res['agreement_ari'], 2)}", ""]
    return "\n".join(L) + "\n"


def stage_report() -> Path:
    ef.require_fixed()
    p = ef.OUT / "report_types.md"
    p.write_text(render_types(json.loads((ef.OUT / "types.json").read_text())))
    return p


STAGES = {"types": stage_types, "report": stage_report}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        sys.exit(__doc__)
    STAGES[sys.argv[1]]()
