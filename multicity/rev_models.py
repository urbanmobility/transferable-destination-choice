"""Revision 2026-11-13 (revision plan, Task R1): reviewer-requested model comparisons.

Post-freeze, supplementary. Nothing frozen and nothing of plan 9 is rewritten; outputs go to results/multicity/rev/.

What is fitted (REV_EQ; #1 d, #2 ln d, #4 P_j, #5 E_j (LODES jobs), #6 N_j, #7 a_j; every numeral is a constant):
  an ablation ladder with POI count as the mass (gravity -> stretched exponential -> + area -> + a parametric
  distance x size interaction -> Reach model -> + a separate log N term); the separable twin of the Reach model
  (same inputs, same five constants, no coupling); the Reach model with ln a_j instead of a_j^theta4 (invariant to
  the area unit); the same comparisons with LODES employment as the mass (independent of the POI catalogue); a
  uniform-choice null model (pseudo-R2).
Protocol = plan 9's: fitted on train x search blocks (300 sampled origins, 5 starts, seed 0), scored on the test
  period x destination fold 0 with every origin; "uniform" constants = element-wise median of the 35 source cities'
  local constants (the city itself left out).
Lines: "all" = the counts of the paper (n_dev_5); "nonwork" = the same inputs with work visits removed
  (n_nonwork_5: a device-day-tract with a visit to the device's inferred workplace is dropped); "poitrain" = the
  paper's counts with N_j (and the class counts) rebuilt from the places seen in the training months only
  (March-April; poi.build into derived/multicity_poitrain), so that no test-period visit enters the POI count. For
  the lines other than "all" the gravity and M2 references are refitted on the same inputs.
Usage (run_stage.sh, M line): rev_models.py fit|transfer|report all|nonwork|poitrain|all2
"""
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401  (puts scripts/sr on the path)

_T, _F = True, False
_SK = "-4.755*#1^0.2994/(#6 + 1.0)^0.1131"
REV_EQ = {
    "null_unif": ("f = 0.0*#1", (_F,)),
    "grav_P": ("f = 1.0*log(#4 + 1.0) - 1.5*#2", (_T, _F, _T)),
    "grav_N": ("f = 1.0*log(#6 + 1.0) - 1.5*#2", (_T, _F, _T)),
    "str3_N": ("f = 1.0*log(#6 + 1.0) - 1.0*#1^0.45", (_T, _F, _T, _T)),
    "str3a_N": ("f = 1.0*log(#6 + 1.0) - 1.0*#1^0.45 + 0.1*log(#7)", (_T, _F, _T, _T, _T)),
    "int_N": ("f = 1.0*log(#6 + 1.0) - 1.0*#1^0.45 + 0.1*log(#7) - 0.5*#2 + 0.05*#2*log(#6 + 1.0)",
              (_T, _F, _T, _T, _T, _T, _T, _F)),
    "sep_N": ("f = #7^0.1299 + 0.8*log(#6 + 1.0) - 1.0*#1^0.45", (_T,) * 5),
    "sk": (f"f = {_SK} + #7^0.1299", (_T,) * 5),
    "sk_lna": (f"f = {_SK} + 0.13*log(#7)", (_T,) * 5),
    "sk_plus": (f"f = {_SK} + #7^0.1299 + 0.1*log(#6 + 1.0)", (_T,) * 7),
    "grav_E": ("f = 1.0*log(#5 + 1.0) - 1.5*#2", (_T, _F, _T)),
    "str3_E": ("f = 1.0*log(#5 + 1.0) - 1.0*#1^0.45", (_T, _F, _T, _T)),
    "sep_E": ("f = #7^0.1299 + 0.8*log(#5 + 20.0) - 1.0*#1^0.45", (_T,) * 5),
    "sk_E": ("f = -4.755*#1^0.2994/(#5 + 20.0)^0.1131 + #7^0.1299", (_T,) * 5),
    # code review 2026-11-13: a free shift can drift below -min(N or E) of an unseen tract and make the utility undefined
    # there; these variants hold the shift at +1 (line "all2").
    "sep_E1": ("f = #7^0.1299 + 0.8*log(#5 + 1.0) - 1.0*#1^0.45", (_T, _T, _F, _T, _T)),
    "sk_E1": ("f = -4.755*#1^0.2994/(#5 + 1.0)^0.1131 + #7^0.1299", (_T, _T, _F, _T, _T)),
    "sk_plus1": (f"f = {_SK} + #7^0.1299 + 0.1*log(#6 + 1.0)", (_T,) * 6 + (_F,)),
}
LINES = {"all": ("n_dev_5", False), "nonwork": ("n_nonwork_5", False), "poitrain": ("n_dev_5", True),
         "all2": ("n_dev_5", False)}
POI_TRAIN = config.ROOT / "derived/multicity_poitrain/poi_tracts.parquet"     # places seen in March-April only
LINE_MODELS = {"all": list(REV_EQ),
               "nonwork": ["null_unif", "grav_P", "grav_N", "str3_N", "str3a_N", "int_N", "sep_N", "sk"],
               "poitrain": ["null_unif", "grav_N", "str3_N", "str3a_N", "int_N", "sep_N", "sk"],
               "all2": ["null_unif", "sep_E1", "sk_E1", "sk_plus1"]}
# (a, b, label): NLL_b - NLL_a > 0 means a fits better. "_frozen" = the frozen library fits (line "all" only).
CONTRASTS = [
    ("grav_N", "grav_P", "POI count instead of population as the mass"),
    ("str3_N", "grav_N", "stretched-exponential distance instead of a power law"),
    ("str3a_N", "str3_N", "+ log area"),
    ("int_N", "str3a_N", "+ parametric distance x size interaction"),
    ("sk", "sep_N", "coupling: Reach model vs its separable twin (same inputs, 5 constants each)"),
    ("sk", "int_N", "Reach model vs the interaction logit"),
    ("sk", "str3_N", "Reach model vs the strongest benchmark (both refitted here)"),
    ("sk_plus", "sk", "+ a separate log N term"),
    ("sk_lna", "sk", "ln a instead of a^theta4"),
    ("sk", "sk_frozen", "refit here vs the frozen fit"),
    ("sk_frozen", "tp_frozen", "Reach model vs Tempered power law (frozen fits)"),
    ("sk_frozen", "tr_frozen", "Reach model vs Two-regime power law (frozen fits)"),
    ("grav_E", "grav_P", "employment instead of population as the mass"),
    ("grav_N", "grav_E", "POI count instead of employment as the mass"),
    ("sk_E", "str3_E", "employment mass: Reach model vs stretched exponential"),
    ("sk_E", "sep_E", "employment mass: coupling (Reach model vs separable twin)"),
    ("sk_E1", "sep_E1", "employment mass, shift held at +1: coupling"),
]
FROZEN = {"sk_frozen": "c0355r1", "tp_frozen": "c0262", "tr_frozen": "c0514c1"}
B, SEED = 5000, 20261113
_G = {}


def out_dir() -> Path:
    return config.results_dir("M") / "rev"


# --------------------------------------------------------------------------------------------- pure parts
def _patch_load(transform):
    """Context: sr_data.load and every module that imported it by name return transform(d) of the loaded city."""
    import sr_data
    import val_refs  # noqa: F401  (imports load by name; make sure it is switched too)
    orig = sr_data.load

    def load(region, scale):
        return transform(orig(region, scale))

    mods = []
    for m in list(sys.modules.values()):
        try:
            if m is not None and getattr(m, "load", None) is orig:
                mods.append(m)
        except Exception:  # noqa: BLE001 — lazy modules may refuse getattr
            continue
    for m in mods:
        m.load = load
    return orig, mods, load


@contextmanager
def use_counts(col: str):
    """Every loaded city carries the counts of column `col` of T_<split>.npz instead of n_dev_5."""
    import sr_data
    if col == "n_dev_5":
        yield
        return

    def tf(d):
        d["T"] = {sp: np.load(d["base"] / f"T_{sp}.npz")[col].astype(np.float64) for sp in sr_data.SPLITS}
        return d

    orig, mods, _ = _patch_load(tf)
    try:
        yield
    finally:
        for m in mods:
            m.load = orig


@contextmanager
def use_poi(table: Path):
    """Every loaded city carries the POI counts (N and the class counts) of `table` (tract, N, n_<cls>, n_other);
    a tract missing from the table gets 0. The choice set (destination mask) is the one of the loaded city."""
    tab = pd.read_parquet(table)
    tab = tab.assign(tract=tab["tract"].astype(str)).set_index("tract")
    cols = [c for c in tab.columns if c == "N" or c.startswith("n_")]

    def tf(d):
        u = d["u"].copy()
        key = u["unit"].astype(str)
        for c in cols:
            if c in u.columns:
                u[c] = key.map(tab[c]).fillna(0).astype(float).to_numpy()
        d["u"] = u
        return d

    orig, mods, _ = _patch_load(tf)
    try:
        yield
    finally:
        for m in mods:
            m.load = orig


@contextmanager
def use_line(line: str):
    col, poi = LINES[line]
    with use_counts(col):
        if poi:
            with use_poi(POI_TRAIN):
                yield
        else:
            yield


def uniform_consts(K: dict, city: str) -> np.ndarray:
    """Element-wise median of the donors' constants, the city itself left out."""
    Y = np.array([K[c] for c in K if c != city], float)
    if not len(Y):
        raise ValueError(f"no donor for {city}")
    return np.median(Y, axis=0)


def paired_dnll(t: pd.DataFrame, a: tuple, b: tuple, cities: list, B: int = B, seed: int = SEED) -> dict:
    """NLL of b minus NLL of a (nats per visit; > 0: a fits better), city-bootstrap median, 95% interval and wins.
    A city where one side has no finite NLL counts as a loss for that side (-inf / +inf), a tie when both fail
    (the p9_stats convention); n_failed = cities where either side failed."""
    import p9_stats as st

    def side(k):
        g = t[(t["model"] == k[0]) & (t["level"] == k[1])]
        return g.set_index("city")["nll"].astype(float)

    A, Bs = side(a), side(b)
    cs = [c for c in cities if c in A.index and c in Bs.index]
    va, vb = A.loc[cs].to_numpy(float), Bs.loc[cs].to_numpy(float)
    fa, fb = ~np.isfinite(va), ~np.isfinite(vb)
    with np.errstate(invalid="ignore"):
        d = np.where(fa & fb, 0.0, np.where(fa, -np.inf, np.where(fb, np.inf, vb - va)))
    r = st.contrast(pd.Series(d, index=cs), B, seed)
    r["n_failed"] = int((fa | fb).sum())
    return r


def consts_of(equation: str) -> list:
    from refit_select import parse_equation
    return [float(v) for v in parse_equation(equation, None, "A", 9).consts]


# --------------------------------------------------------------------------------------------- per city
def _data(city: str):
    import frozen
    import sr_data
    frozen.require()
    cfg = _G["cfg"]
    return (sr_data.build([city], "tract", "train", "search", cfg["n_orig"], 0),
            sr_data.build([city], "tract", "test", "fold:0", None, 0))


def _refs(city: str, line: str) -> dict:
    import p9
    if LINES[line] == ("n_dev_5", False):          # the paper's inputs: the references already scored in plan 6
        r = p9.refs(city, "test_dest")
        return {"grav_ref": r["grav"], "M2_ref": r["m2"]}
    import refs
    g = refs.glm_refs(city, "tract", transit_col=refs.TRANSIT_COL, train_block="search", eval_split="test",
                      eval_block="fold:0", models=("grav_power", "M2_glm"))
    return {"grav_ref": g["grav_power"]["nll_per_visit"], "M2_ref": g["M2_glm"]["nll_per_visit"]}


def fit_city(city: str) -> list:
    import prior_models as pm
    cfg, line = _G["cfg"], _G["line"]
    tr, ev = _data(city)
    rows = []
    for name in LINE_MODELS[line]:
        eq, free = REV_EQ[name]
        r = {"city": city, "model": name, "level": "local", "consts": None, "train_nll": np.nan, "nll": np.nan,
             "cpc": np.nan}
        if not any(free):
            k = consts_of(eq)
            r.update(consts=json.dumps(k), **pm.score_eq(eq, k, ev))
        else:
            b = pm.fit_eq(eq, free, tr, cfg["starts"], cfg["fit_seed"])
            if b is not None:
                r.update(consts=json.dumps(b[1]), train_nll=b[0], **pm.score_eq(eq, b[1], ev))
        rows.append(r)
    for k, v in _refs(city, line).items():
        rows.append({"city": city, "model": k, "level": "local", "consts": None, "train_nll": np.nan,
                     "nll": float(v), "cpc": np.nan})
    print(city, line, "fit done", flush=True)
    return rows


def transfer_city(city: str) -> list:
    import prior_models as pm
    _, ev = _data(city)
    rows = []
    for name, K in _G["K"].items():
        k = uniform_consts(K, city)
        rows.append({"city": city, "model": name, "level": "uniform", "consts": json.dumps([float(v) for v in k]),
                     "train_nll": np.nan, **pm.score_eq(REV_EQ[name][0], k, ev)})
    print(city, _G["line"], "transfer done", flush=True)
    return rows


# --------------------------------------------------------------------------------------------- stages
def _setup(line: str):
    import p9
    if line not in LINES:
        raise ValueError(f"line {line!r}: one of {list(LINES)}")
    _G.update(cfg=p9.require_fixed(), line=line)
    src, tgt = p9.cities("source"), p9.cities("target")
    return src, tgt, {**{c: "source" for c in src}, **{c: "target" for c in tgt}}


def _path(line: str, part: str) -> Path:
    return out_dir() / f"models_{line}_{part}.parquet"


def stage_fit(line: str):
    import ledger
    import p9
    src, tgt, role = _setup(line)
    with use_line(line):
        rows = [r for rs in p9._pool_map(fit_city, src + tgt) for r in rs]
    t = pd.DataFrame(rows)
    t["set"] = t["city"].map(role)
    out_dir().mkdir(parents=True, exist_ok=True)
    t.to_parquet(_path(line, "local"), index=False)
    ledger.add(f"Revision R1 (post hoc, supplementary): rev_models fit {line}: {t['city'].nunique()} cities x "
               f"{len(LINE_MODELS[line])} forms; rev/{_path(line, 'local').name} sha {ledger.sha(_path(line, 'local'))}")


def stage_transfer(line: str):
    import ledger
    import p9
    src, tgt, role = _setup(line)
    loc = pd.read_parquet(_path(line, "local"))
    K = {}
    for name in LINE_MODELS[line]:
        if not any(REV_EQ[name][1]):
            continue
        g = loc[(loc["model"] == name) & loc["city"].isin(src) & np.isfinite(loc["nll"].astype(float))]
        K[name] = {r.city: json.loads(r.consts) for r in g.itertuples() if r.consts}
    _G["K"] = K
    with use_line(line):
        rows = [r for rs in p9._pool_map(transfer_city, src + tgt) for r in rs]
    t = pd.DataFrame(rows)
    t["set"] = t["city"].map(role)
    t.to_parquet(_path(line, "uniform"), index=False)
    ledger.add(f"Revision R1: rev_models transfer {line}: uniform constants (source median, city left out); "
               f"rev/{_path(line, 'uniform').name} sha {ledger.sha(_path(line, 'uniform'))}")


def _frozen_rows(src: list, tgt: list) -> pd.DataFrame:
    """The frozen library fits on the same block (local) and plan 9's uniform constants of the primary formula."""
    import p9
    ours = p9.ours_fits("test_dest")
    rows = [{"city": c, "model": m, "level": "local", "nll": v["nll"]}
            for m, cid in FROZEN.items() for c, v in ours.get(cid, {}).items() if c in set(src + tgt)]
    uni = pd.concat([pd.read_parquet(p9.OUT / "target.parquet"), pd.read_parquet(p9.OUT / "explore" / "source_test.parquet")])
    uni = uni[(uni["level"] == "uniform") & (uni["model"] == "c0355r1")]
    rows += [{"city": r.city, "model": "sk_frozen", "level": "uniform", "nll": float(r.nll)} for r in uni.itertuples()]
    return pd.DataFrame(rows)


def _f(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def stage_report(line: str):
    import alldest_check as ac
    src, tgt, role = _setup(line)
    t = pd.concat([pd.read_parquet(_path(line, "local")), pd.read_parquet(_path(line, "uniform"))], ignore_index=True)
    if line == "all":
        t = pd.concat([t, _frozen_rows(src, tgt)], ignore_index=True)
    t["nll"] = t["nll"].astype(float)
    t = ac.add_share(t)
    null = t[t["model"] == "null_unif"].set_index("city")["nll"]
    t["pseudo_r2"] = 1.0 - t["nll"] / t["city"].map(null)
    t["set"] = t["city"].map(role)
    t.to_parquet(out_dir() / f"models_{line}_scored.parquet", index=False)
    res = {"line": line, "n_cities": int(t["city"].nunique()), "models": {}, "contrasts": {}}
    for (m, lv), g in t.groupby(["model", "level"]):
        res["models"][f"{m}|{lv}"] = {
            k: {"n": int(np.isfinite(gg["nll"]).sum()), "share": float(np.nanmedian(gg["share"])),
                "nll": float(np.nanmedian(gg["nll"])), "pseudo_r2": float(np.nanmedian(gg["pseudo_r2"]))}
            for k, gg in (("target", g[g["city"].isin(tgt)]), ("all", g)) if len(gg)}
    have = set(t["model"])
    for a, b, lab in CONTRASTS:
        if a not in have or b not in have:
            continue
        for lv in ("local", "uniform"):
            if not ((t["model"] == a) & (t["level"] == lv)).any() or not ((t["model"] == b) & (t["level"] == lv)).any():
                continue
            for name, cs in (("target", tgt), ("all", src + tgt)):
                res["contrasts"][f"{a}|{b}|{lv}|{name}"] = {"label": lab, **paired_dnll(t, (a, lv), (b, lv), cs)}
    (out_dir() / f"summary_models_{line}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    L = [f"# Revision R1: model comparisons, line {line} (test period x fold 0)", "",
         "Post hoc, supplementary. NLL in nats per visit; share = (gravity - NLL) / (gravity - M2) with the references "
         "of the same counts; pseudo-R2 = 1 - NLL / NLL(uniform choice).", "",
         "## Medians", "", "| model | level | n (32 target) | share | NLL | pseudo-R2 | share (67) |", "|---|---|---|---|---|---|---|"]
    for k in sorted(res["models"]):
        m, lv = k.split("|")
        r = res["models"][k]
        a, b = r.get("target", {}), r.get("all", {})
        L.append(f"| {m} | {lv} | {a.get('n', 0)} | {_f(a.get('share'))} | {_f(a.get('nll'), 4)} | "
                 f"{_f(a.get('pseudo_r2'))} | {_f(b.get('share'))} |")
    L += ["", "## Paired differences: NLL(b) - NLL(a), nats per visit (> 0: a fits better)", "",
          "| a | b | level | cities | median [95% CI] | a better | failed | what |", "|---|---|---|---|---|---|---|---|"]
    for k, r in res["contrasts"].items():
        a, b, lv, name = k.split("|")
        L.append(f"| {a} | {b} | {lv} | {name} | {_f(r['median'], 4)} [{_f(r['ci_lo'], 4)}, {_f(r['ci_hi'], 4)}] | "
                 f"{r['wins']}/{r['n']} | {r['n_failed']} | {r['label']} |")
    (out_dir() / f"report_models_{line}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


STAGES = {"fit": stage_fit, "transfer": stage_transfer, "report": stage_report}

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in STAGES:
        sys.exit(__doc__)
    STAGES[sys.argv[1]](sys.argv[2])
