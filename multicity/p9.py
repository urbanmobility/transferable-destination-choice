"""Plan 9: the frozen skeletons against the earlier models — fit and transfer to a city without trip data
(plan-9 design of 2026-11-11: earlier models and transfer; approved 2026-11-11 with 32 target cities).

Source cities (35 discovery) supply the constants and the "size -> constant" relations; results are reported on the
target cities (26 law-test + 6 confirmation) x the test block (test period x destination fold 0). Every model is
fitted on train x search block (300 sampled origins, 5 starts) and gets four parameter sources (param_transfer.LEVELS):
local refit, uniform (median of the source cities), predicted from size, predicted from size + form.

Order, enforced here:
  fix        the method config (config/p9.json) is fixed before anything is computed;
  source     local fits of every model on the source cities, scored on the validation block;
  prerun     leave-one-city-out transfer on the source cities (validation block);
  select     the opponents of the primary contrasts by the config's rule from the pre-run — written ONCE;
  target     local fits and transfer on the target cities x test block — refused until the opponents exist;
  contrasts  the four primary and three descriptive contrasts — written ONCE;
  report     results/multicity/p9/report_p9.md.
Nothing is tuned on a target city: donors are the source cities only.
The stages are bound to each other (review before the target run): the pre-run records the source fits, the config
and the code it used; select requires exactly those and no target file; target and contrasts require the config, the
code, the source fits and the pre-run that the opponents were selected from; target runs once, on every target city,
and is ledgered; the report requires the target file the contrasts were computed from.

Usage (run_stage.sh, M line): p9.py fix [--reason TEXT] | source | prerun | select | target | contrasts | report
P9_CITIES=a,b limits source / prerun to those cities (trial runs); select, target and contrasts then refuse.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import sr_data  # noqa: E402
import p9_stats as st  # noqa: E402
import param_transfer as pt  # noqa: E402
import prior_models as pm  # noqa: E402
from explore_form import group_cities  # noqa: E402

CONFIG = Path(__file__).resolve().parent / "config" / "p9.json"
OUT = config.results_dir("M") / "p9"
EVAL = {"val": ("val", "val"), "test_dest": ("test", "fold:0")}        # block -> (period, destination block)
NLL_TOL = 1e-6
PATH_TOL = 1e-5      # equation path (float32 rows) vs matrix path (float64 arrays), nats per visit
CONSTS_TOL = 1e-9    # skeleton donor constants: test-block file vs validation file (plan 6 enforced the same)
CODE = ("prior_models.py", "param_transfer.py", "p9_stats.py", "p9.py")
COLS = ["city", "model", "kind", "level", "consts", "nll", "cpc"]
LEVEL_EN = {"local": "Local refit", "uniform": "Uniform constants", "size": "Size-predicted", "form": "Size + form"}
_G = {}      # shared with forked workers: cfg, block, ours, cands, K, cov


# --------------------------------------------------------------------------------------------- methods fixed first
def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _entries() -> list:
    p = OUT / "CONFIG_FIXED"
    return json.loads(p.read_text()) if p.exists() else []


def _check_registry(cfg: dict) -> None:
    want = list(pm.EQ_MODELS) + list(pm.MAT_MODELS)
    if sorted(cfg["prior"]) != sorted(want):
        raise ValueError(f"{CONFIG.name}: the prior list differs from the model registry "
                         f"({sorted(set(cfg['prior']) ^ set(want))})")
    if list(cfg["levels"]) != list(pt.LEVELS):
        raise ValueError(f"{CONFIG.name}: levels {cfg['levels']} differ from the code's {list(pt.LEVELS)}")
    if cfg["blocks"] != {"source": "val", "target": "test_dest"}:
        raise ValueError(f"{CONFIG.name}: blocks {cfg['blocks']} differ from the code's (source val, target test_dest)")


def code_sha() -> dict:
    """sha (12 hex) of the modules that carry the models, the levels, the opponent rule and the stages."""
    import ledger
    return {f: ledger.sha(Path(__file__).resolve().parent / f) for f in CODE}


def _stamp() -> dict:
    return {"config_sha": _sha(CONFIG), "code_sha": code_sha()}


def _state() -> dict:
    """What the opponents are bound to: config, code, the source fits and the pre-run (None for a missing file)."""
    import ledger
    f = {k: OUT / f"{k}.parquet" for k in ("fits_source", "prerun")}
    return {**_stamp(), **{f"{k}_sha": ledger.sha(p) if p.exists() else None for k, p in f.items()}}


def fix_config(reason: str = None, log=print) -> str:
    """Record the config's sha (first time, or a change with a reason); the same sha again is a no-op."""
    _check_registry(json.loads(CONFIG.read_text()))
    sha, ent = _sha(CONFIG), _entries()
    if ent and ent[-1]["sha"] == sha:
        return sha
    if ent and not reason:
        raise ValueError("the plan-9 config changed after it was fixed: give a reason (fix --reason TEXT)")
    ent.append({"sha": sha, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reason": reason,
                "code_sha": code_sha()})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CONFIG_FIXED").write_text(json.dumps(ent, ensure_ascii=False, indent=1))
    log(f"Plan 9 config {'fixed' if len(ent) == 1 else 'amended'}: {CONFIG.name} sha {sha}"
        + (f"; reason: {reason}" if reason else ""))
    return sha


def require_fixed() -> dict:
    ent = _entries()
    if not ent or ent[-1]["sha"] != _sha(CONFIG):
        raise RuntimeError(f"{CONFIG.name} is not fixed (or changed after fixing): run p9.py fix first")
    cfg = json.loads(CONFIG.read_text())
    _check_registry(cfg)
    return cfg


# --------------------------------------------------------------------------------------------- cities and guards
def cities(role: str) -> list:
    """role "source" or "target": the cities of the config's groups, in order."""
    return [str(c) for g in json.loads(CONFIG.read_text())[f"{role}_groups"] for c in group_cities(g)]


def _subset(cs: list) -> list:
    want = os.environ.get("P9_CITIES")
    if not want:
        return cs
    pick = [c for c in want.split(",") if c]
    bad = sorted(set(pick) - set(cs))
    if bad:
        raise ValueError(f"P9_CITIES: not cities of this stage: {bad}")
    return pick


def guard(block: str, cs) -> None:
    """The validation block is for source cities; the test block for target cities, and only once the opponents are
    selected. Either way the config must be fixed."""
    require_fixed()
    if block == "val":
        bad = sorted(set(cs) - set(cities("source")))
        if bad:
            raise RuntimeError(f"the validation block is for the source cities only; not source: {bad[:5]}")
    elif block == "test_dest":
        if not (OUT / "opponents.json").exists():
            raise RuntimeError("the opponents are not selected yet (p9.py select): no target-city result before that")
        _require_bound()
        bad = sorted(set(cs) - set(cities("target")))
        if bad:
            raise RuntimeError(f"the test block is for the target cities only; not target: {bad[:5]}")
    else:
        raise ValueError(f"block {block!r}")


def _require_bound() -> dict:
    """The opponents file must come from this config, this code, these source fits and this pre-run."""
    o = json.loads((OUT / "opponents.json").read_text())
    now = _state()
    bad = [k.replace("_sha", "") for k in now if o.get(k) != now[k]]
    if bad:
        raise RuntimeError(f"opponents.json was selected under a different {', '.join(bad)}: the target stages need "
                           "the config, code, fits_source and prerun the opponents came from")
    return o


# --------------------------------------------------------------------------------------------- inputs
def city_inputs(city: str, block: str) -> dict:
    """{"tr", "ev", "A"}: training rows (train x search, sampled origins), scoring rows (the block, every origin) and
    the matrix-path arrays of the same blocks."""
    import p7seal
    p7seal.guard_cities([city])
    cfg = _G.get("cfg") or require_fixed()
    split, eb = EVAL[block]
    if block == "test_dest":
        import frozen
        frozen.require()
    d = sr_data.load(city, "tract")
    A = pm.city_arrays(d, sr_data.folds(d), split, eb, np.load(d["base"] / "S.npy").astype(np.float64),
                       n_orig=cfg["n_orig"], seed=0)
    return {"tr": sr_data.build([city], "tract", "train", "search", cfg["n_orig"], 0),
            "ev": sr_data.build([city], "tract", split, eb, None, 0), "A": A}


def ours_fits(block: str) -> dict:
    """{cid: {city: {"consts", "nll"}}} of the skeletons' existing fits (train x search), scored on the block."""
    p = config.RESULTS / ("confirm/test_dest/cities_val.parquet" if block == "test_dest"
                          else "prefreeze_mv/discovery/cities_val.parquet")
    cv = pd.read_parquet(p)
    cv = cv[cv["finite"].astype(bool)]
    out = {}
    for r in cv.itertuples():
        out.setdefault(r.cid, {})[str(r.city)] = {"consts": [float(x) for x in json.loads(r.consts)],
                                                  "nll": float(r.val_nll)}
    return out


def ours_candidates() -> dict:
    import library
    cfg = _G.get("cfg") or require_fixed()
    pool = {x["cid"]: x for x in json.loads((config.RESULTS / "prefreeze_mv/loop/pool.json").read_text())}
    return {cid: library.candidate({**pool[cid], "consts_refit": None}) for cid in cfg["ours"]["reported"]}


def refs(city: str, block: str) -> dict:
    """Gravity and M2 NLL per visit (and M2's CPC) of the city on the block, from the existing reference files."""
    if block == "test_dest":
        m = json.loads((config.RESULTS / "refs_test" / f"{city}.json").read_text())["test_dest"]
    else:
        m = json.loads((config.RESULTS / "refs" / f"{city}.json").read_text())["models"]
    return {"grav": float(m["grav_power"]["nll_per_visit"]), "m2": float(m["M2_glm"]["nll_per_visit"]),
            "m2_cpc": float(m["M2_glm"]["cpc"])}


def covariates() -> pd.DataFrame:
    import explore_form as ef
    return ef.load_frame()[1].set_index("city")


def prepare(block: str, K: dict = None) -> None:
    """Fill the state the per-city workers read (before forking)."""
    _G.clear()
    _G["cfg"] = require_fixed()
    _G.update(block=block, ours=ours_fits(block), cands=ours_candidates(), K=K,
              cov=covariates() if K is not None else None)


# --------------------------------------------------------------------------------------------- one city
def _n_free() -> dict:
    return {**{k: int(np.sum(s.free)) for k, s in pm.EQ_MODELS.items()},
            **{k: len(s.start) for k, s in pm.MAT_MODELS.items()}}


def fit_city(job) -> list:
    """Local fit of every model on the city, scored on the block: one row per model."""
    city, block = job
    guard(block, [city])
    cfg, D = _G["cfg"], city_inputs(job[0], job[1])
    tr, ev, A = D["tr"], D["ev"], D["A"]
    check_paths(city, ev, A)
    nan = {"nll": np.nan, "cpc": np.nan}
    rows = []
    for name, s in pm.EQ_MODELS.items():
        b = pm.fit_eq(s.equation, s.free, tr, cfg["starts"], cfg["fit_seed"])
        r = {"city": city, "model": name, "kind": "eq", "n_free": int(np.sum(s.free)), "consts": None,
             "train_nll": None, **nan}
        if b is not None:
            r.update(consts=json.dumps(b[1]), train_nll=b[0], **pm.score_eq(s.equation, b[1], ev))
        rows.append(r)
    for name, s in pm.MAT_MODELS.items():
        th = pm.fit_mat(name, A)
        rows.append({"city": city, "model": name, "kind": "mat", "n_free": len(th), "consts": json.dumps(th),
                     "train_nll": pm.score_mat(pm.mat_eta(name, th, A, A["io"]["tr"]), A["T"]["tr"], A["av"]["tr"])["nll"],
                     **pm.score_mat(pm.mat_eta(name, th, A, A["io"]["ev"]), A["T"]["ev"], A["av"]["ev"])})
    for cid in cfg["ours"]["reported"]:
        f = _G["ours"].get(cid, {}).get(city)
        r = {"city": city, "model": cid, "kind": "ours", "n_free": len(_G["cands"][cid].consts), "consts": None,
             "train_nll": None, **nan}
        if f is not None:
            s = pm.score_cand(_G["cands"][cid], f["consts"], ev)
            if not abs(s["nll"] - f["nll"]) <= NLL_TOL:
                raise RuntimeError(f"{cid} in {city} ({block}): recomputed NLL {s['nll']} differs from the stored "
                                   f"fit's {f['nll']} — the data rows are not those of the frozen fits")
            r.update(consts=json.dumps(f["consts"]), **s)
        rows.append(r)
    print(city, block, "fit done", flush=True)
    return rows


def check_paths(city: str, ev, A: dict) -> None:
    """The matrix arrays and the data rows must describe the same scoring block: one fixed gravity, both paths."""
    e = np.log(A["mass"]["N"])[None, :] - 1.5 * A["logd"][A["io"]["ev"]]
    a = pm.score_mat(e, A["T"]["ev"], A["av"]["ev"])
    b = pm.score_eq("f = log(#6 + 1.0) - 1.5*#2", [1.0, 1.5], ev)
    if not (abs(a["nll"] - b["nll"]) <= PATH_TOL and abs(a["cpc"] - b["cpc"]) <= PATH_TOL):
        raise RuntimeError(f"{city}: the matrix arrays and the data rows disagree on the scoring block "
                           f"(NLL {a['nll']} vs {b['nll']}, CPC {a['cpc']} vs {b['cpc']})")


def transfer_city(job) -> list:
    """The three levels without the city's own trip data: one row per model with parameters x level. A model with
    too few donors, or constants outside the formula's domain, gets nan (a failure)."""
    city, block = job
    guard(block, [city])
    cfg, D = _G["cfg"], city_inputs(job[0], job[1])
    ev, A = D["ev"], D["A"]
    ra = cfg["ridge_alphas"]
    rows = []
    for (model, kind), Km in _G["K"].items():
        lv = pt.predict(Km, _G["cov"], city, cfg["size_covariate"], cfg["form_covariates"], cfg["min_donors"],
                        np.logspace(ra["log10_lo"], ra["log10_hi"], ra["n"]), tuple(cfg["clip_quantiles"]))
        for level in pt.LEVELS[1:]:
            k = lv.get(level)
            if k is None:
                s = {"nll": np.nan, "cpc": np.nan}
            elif kind == "mat":
                s = pm.score_mat(pm.mat_eta(model, list(k), A, A["io"]["ev"]), A["T"]["ev"], A["av"]["ev"])
            elif kind == "eq":
                s = pm.score_eq(pm.EQ_MODELS[model].equation, k, ev)
            else:
                s = pm.score_cand(_G["cands"][model], k, ev)
            rows.append({"city": city, "model": model, "kind": kind, "level": level,
                         "consts": None if k is None else json.dumps([float(v) for v in k]), **s})
    print(city, block, "transfer done", flush=True)
    return rows


def scored(rows: pd.DataFrame, block: str) -> pd.DataFrame:
    """Add failed (no finite NLL) and share = (gravity - NLL) / (gravity - M2), -inf for a failure."""
    t = rows.copy()
    r = {c: refs(c, block) for c in t["city"].unique()}
    nll = t["nll"].astype(float)
    bad = sorted(c for c, v in r.items() if not (np.isfinite(v["grav"] - v["m2"]) and v["grav"] - v["m2"] > 0))
    if bad:
        raise RuntimeError(f"the gravity -> M2 gap is not positive and finite in {bad[:5]}: no share can be defined")
    g, m2 = t["city"].map(lambda c: r[c]["grav"]).astype(float), t["city"].map(lambda c: r[c]["m2"]).astype(float)
    t["nll"] = nll
    t["failed"] = ~np.isfinite(nll)
    t["share"] = np.where(t["failed"], -np.inf, (g - nll) / (g - m2))
    return t


def donors(fits: pd.DataFrame, ours: dict, source: list) -> dict:
    """{(model, kind): {source city: constants}} for every model with parameters: the earlier models' local fits
    with a finite score, and the skeletons' existing fits. A model without any donor keeps an empty entry."""
    nf = _n_free()
    K = {(m, "eq" if m in pm.EQ_MODELS else "mat"): {} for m in nf if nf[m] > 0}
    src = set(source)
    for r in fits.itertuples():
        if (r.model, r.kind) in K and r.city in src and r.consts and np.isfinite(r.nll):
            K[(r.model, r.kind)][str(r.city)] = json.loads(r.consts)
    for cid in _G["cfg"]["ours"]["reported"] if _G.get("cfg") else require_fixed()["ours"]["reported"]:
        K[(cid, "ours")] = {c: f["consts"] for c, f in ours.get(cid, {}).items() if c in src}
    return K


def assemble(fits: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    """Local rows + transfer rows; a model without parameters has the same prediction at every level."""
    loc = fits.assign(level="local")
    same = [loc[loc["n_free"] == 0].assign(level=lv) for lv in pt.LEVELS[1:]]
    t = pd.concat([loc[COLS], transfer[COLS]] + [s[COLS] for s in same], ignore_index=True)
    if t.duplicated(["city", "model", "level"]).any():
        raise RuntimeError("several rows for one city x model x level")
    return t


def _pool_map(fn, items):
    import multiprocessing as mp
    procs = min(int(os.environ.get("LOOP_OUTER_PROCS", "8")), len(items))
    if procs <= 1:
        return [fn(x) for x in items]
    with mp.get_context("fork").Pool(procs, maxtasksperchild=1) as P:
        return list(P.imap_unordered(fn, items, chunksize=1))


def _flat(fn, cs, block) -> pd.DataFrame:
    rows = [r for rs in _pool_map(fn, [(c, block) for c in cs]) for r in rs]
    return pd.DataFrame(rows).sort_values(["model", "city"]).reset_index(drop=True)


def _need_all(t: pd.DataFrame, role: str, what: str) -> None:
    want = set(cities(role))
    if set(t["city"]) != want:
        raise RuntimeError(f"{what} does not cover the {role} cities ({len(set(t['city']))} of {len(want)})")


def _no_partial(stage: str) -> None:
    if os.environ.get("P9_CITIES"):
        raise RuntimeError(f"P9_CITIES is set: {stage} needs the complete run")


# --------------------------------------------------------------------------------------------- stages
def stage_source():
    require_fixed()
    cs = _subset(cities("source"))
    guard("val", cs)
    prepare("val")
    OUT.mkdir(parents=True, exist_ok=True)
    _flat(fit_city, cs, "val").to_parquet(OUT / "fits_source.parquet", index=False)


def stage_prerun():
    require_fixed()
    fits = pd.read_parquet(OUT / "fits_source.parquet")
    cs = sorted(fits["city"].unique())
    guard("val", cs)
    prepare("val")
    prepare("val", K=donors(fits, ours_fits("val"), cities("source")))
    t = scored(assemble(fits, _flat(transfer_city, cs, "val")), "val")
    t.to_parquet(OUT / "prerun.parquet", index=False)
    (OUT / "prerun_meta.json").write_text(json.dumps(_state(), indent=1))
    s = st.summary(t)
    s.to_csv(OUT / "prerun_summary.csv", index=False)
    (OUT / "prerun_summary.md").write_text("\n".join(
        ["# Plan 9 source pre-run: discovery cities x validation block (leave one city out)", ""] + _table(s, fits, "share", 3) + [""]) + "\n")
    print((OUT / "prerun_summary.md").read_text())


def _meta() -> dict:
    nf = _n_free()
    return {k: {"original": bool(s.original), "needs_trips": bool(getattr(s, "needs_trips", False)), "n_free": nf[k]}
            for k, s in {**pm.EQ_MODELS, **pm.MAT_MODELS}.items()}


def regression_check(s: pd.DataFrame, cfg: dict) -> None:
    """The pre-run must reproduce the spike's median shares (config `regression`) within the tolerance."""
    reg = cfg["regression"]
    got = s.set_index(["model", "level"])["share_med"]
    bad = [(m, lv, v, float(got.get((m, lv), np.nan))) for m, lv, v in reg["median_share"]
           if not abs(float(got.get((m, lv), np.nan)) - v) <= reg["tolerance"]]
    if bad:
        raise RuntimeError(f"regression against the spike failed (model, level, spike, pre-run): {bad}")


def stage_select():
    import ledger
    cfg = require_fixed()
    _no_partial("select")
    p = OUT / "opponents.json"
    if p.exists():
        raise FileExistsError(f"{p}: the opponents are selected once")
    late = [f for f in ("fits_target.parquet", "target.parquet", "contrasts.json") if (OUT / f).exists()]
    if late:
        raise FileExistsError(f"{late} exist: the opponents are selected before any target-city result")
    meta = json.loads((OUT / "prerun_meta.json").read_text())
    now = _state()
    bad = [k.replace("_sha", "") for k in now if meta.get(k) != now[k]]
    if bad:
        raise RuntimeError(f"the pre-run was made under a different {', '.join(bad)}: run p9.py prerun again")
    t = pd.read_parquet(OUT / "prerun.parquet")
    _need_all(t, "source", "the pre-run")
    s = st.summary(t)
    regression_check(s, cfg)
    opp = st.select_opponents(s, _meta())
    p.write_text(json.dumps({"opponents": opp, "rule": cfg["opponent_rule"], **now,
                             "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, ensure_ascii=False, indent=1))
    ledger.add("Plan 9 opponents (selected once, by the config's rule, from the source pre-run; no target-city result "
               "exists yet): " + "; ".join(f"{k} = {v['model']} ({v['share_med']:.3f})" for k, v in opp.items())
               + f"; opponents.json sha {ledger.sha(p)}")
    print(json.dumps(opp, indent=1))


def stage_target():
    import ledger
    cfg = require_fixed()
    _no_partial("target")
    done = [f for f in ("target.parquet", "contrasts.json") if (OUT / f).exists()]
    if done:
        raise FileExistsError(f"{done} exist: the target cities are computed once")
    cs = cities("target")
    guard("test_dest", cs)
    src = pd.read_parquet(OUT / "fits_source.parquet")
    _need_all(src, "source", "fits_source.parquet")
    source = cities("source")
    ot, ov = ours_fits("test_dest"), ours_fits("val")
    P = cfg["ours"]["primary"]                              # pre-flight, before anything is computed
    miss = sorted(set(source) - set(ot.get(P, {})))
    if miss:
        raise RuntimeError(f"{P}: no test-block fit row for the source cities {miss[:5]} (donors of the primary)")
    gap = max((float(np.max(np.abs(np.subtract(f["consts"], ov[cid][c]["consts"]))))
               for cid in cfg["ours"]["reported"] for c, f in ot.get(cid, {}).items()
               if c in source and c in ov.get(cid, {})), default=float("nan"))
    if not gap <= CONSTS_TOL:
        raise RuntimeError(f"skeleton donor constants: the test-block file differs from the validation file by {gap}")
    print(f"skeleton donor constants: test-block file vs validation file, largest difference {gap:.3g}; donors "
          + ", ".join(f"{cid} {len(set(source) & set(ot.get(cid, {})))}" for cid in cfg["ours"]["reported"]), flush=True)
    prepare("test_dest")
    fits = _flat(fit_city, cs, "test_dest")
    fits.to_parquet(OUT / "fits_target.parquet", index=False)
    prepare("test_dest", K=donors(src, ot, source))
    t = scored(assemble(fits, _flat(transfer_city, cs, "test_dest")), "test_dest")
    grp = {str(c): g for g in _G["cfg"]["target_groups"] for c in group_cities(g)}
    t["group"] = t["city"].map(grp)
    t.to_parquet(OUT / "target.parquet", index=False)
    st.summary(t).to_csv(OUT / "target_summary.csv", index=False)
    ledger.add(f"Plan 9 target (computed once): {t['city'].nunique()} target cities x test block, {t['model'].nunique()} "
               f"models x {t['level'].nunique()} levels; fits_target.parquet sha "
               f"{ledger.sha(OUT / 'fits_target.parquet')}, target.parquet sha {ledger.sha(OUT / 'target.parquet')}")


def stage_contrasts():
    import ledger
    cfg = require_fixed()
    _no_partial("contrasts")
    p = OUT / "contrasts.json"
    if p.exists():
        raise FileExistsError(f"{p}: the contrasts are computed once")
    guard("test_dest", cities("target"))
    t = pd.read_parquet(OUT / "target.parquet")
    _need_all(t, "target", "target.parquet")
    opp = json.loads((OUT / "opponents.json").read_text())["opponents"]
    loc = t[(t["model"] == cfg["ours"]["primary"]) & (t["level"] == "local")]
    if loc["failed"].any():                 # the frozen fits exist for every city: a missing one is a data problem
        raise RuntimeError(f"{cfg['ours']['primary']} has no local fit in {sorted(loc.loc[loc['failed'], 'city'])[:5]}")

    def side(x):
        return (cfg["ours"]["primary"] if x[0] == "primary" else opp[x[0]]["model"], x[1])
    spec = {**cfg["contrasts"]["primary"], **cfg["contrasts"]["descriptive"]}
    res, diffs = {}, {}
    for i, (cid, (a, b)) in enumerate(spec.items()):
        d = st.paired(t, side(a), side(b))
        res[cid] = dict(st.contrast(d, cfg["bootstrap"], [cfg["seed"], i]), a=list(side(a)), b=list(side(b)))
        diffs[cid] = d
    out = st.judge(res, list(cfg["contrasts"]["primary"]), cfg["alpha"])
    pd.DataFrame(diffs).rename_axis("city").reset_index().to_csv(OUT / "contrast_diffs.csv", index=False)
    p.write_text(json.dumps({"contrasts": out, "opponents": opp, "n_cities": int(t["city"].nunique()),
                             **_stamp(), "target_sha": ledger.sha(OUT / "target.parquet"),
                             "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, ensure_ascii=False, indent=1))
    ledger.add("Plan 9 contrasts (computed once, target cities x test block): "
               + "; ".join(f"{k} {r['median']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                           f"{r['wins']}/{r['n']}" + (f" {r['verdict']}" if r["verdict"] else "")
                           for k, r in out.items()) + f"; contrasts.json sha {ledger.sha(p)}")
    print(json.dumps(out, ensure_ascii=False, indent=1))


# --------------------------------------------------------------------------------------------- report
def _f(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if np.isinf(v):
        return "−∞" if v < 0 else "+∞"
    return f"{v:.{nd}f}"


def _kind(model: str, cfg: dict) -> str:
    if model in cfg["ours"]["reported"]:
        return "ours (primary)" if model == cfg["ours"]["primary"] else "ours"
    s = {**pm.EQ_MODELS, **pm.MAT_MODELS}[model]
    if getattr(s, "needs_trips", False):
        return "earlier, as published (needs the city's own trips)"
    return "earlier, as published" if s.original else ("earlier, POI count" if s.mass == "N" and model.rsplit("_", 1)[0] not in ("S0", "strexp3")
                                       else "earlier, relaxed or reference")


def _table(s: pd.DataFrame, fits: pd.DataFrame, col: str, nd: int, cfg: dict = None) -> list:
    cfg = cfg or require_fixed()
    nf = fits.drop_duplicates("model").set_index("model")["n_free"]
    order = [m for m in cfg["ours"]["reported"] + list(pm.EQ_MODELS) + list(pm.MAT_MODELS) if m in set(s["model"])]
    s = s.set_index(["model", "level"])
    L = ["| Model | Kind | Free parameters | " + " | ".join(LEVEL_EN[lv] for lv in pt.LEVELS) + " | Failed cities (four levels) |",
         "|---|---|---|" + "---|" * (len(pt.LEVELS) + 1)]
    for m in order:
        cells = [f"{_f(s.loc[(m, lv), f'{col}_med'], nd)} [{_f(s.loc[(m, lv), f'{col}_q1'], nd)}, "
                 f"{_f(s.loc[(m, lv), f'{col}_q3'], nd)}]" for lv in pt.LEVELS]
        fails = "/".join(str(int(s.loc[(m, lv), "n_failed"])) for lv in pt.LEVELS)
        L.append(f"| {m} | {_kind(m, cfg)} | {int(nf.get(m, 0))} | " + " | ".join(cells) + f" | {fails} |")
    return L


def render(cfg: dict, pre: pd.DataFrame, tg: pd.DataFrame, fits_s: pd.DataFrame, fits_t: pd.DataFrame, opp: dict,
           con: dict, m2_cpc: float) -> str:
    P = cfg["ours"]["primary"]
    ng = tg.drop_duplicates("city")["group"].value_counts().to_dict()
    s = st.summary(tg)
    L = ["# Plan 9: earlier models and transfer without trip data (target cities x test block)", "",
         f"- {tg['city'].nunique()} target cities ({ng.get('lawtest', 0)} law-test + {ng.get('confirm', 0)} confirmation); "
         "scored block = test period x destination fold 0; source cities = discovery cities, which supply only the "
         "constants and the parameter-prediction relations.",
         "- Each cell = median over target cities [lower quartile, upper quartile]; failed = number of cities without a "
         "finite NLL. Share quantiles count failures as −∞; NLL and CPC quantiles use only the cities without failure.",
         "- Share = (gravity NLL − model NLL) / (gravity NLL − M2 NLL); gravity and M2 are the existing reference values "
         "of the city on the same block.",
         f"- Reference: median CPC of M2 (flexible GLM, fitted in each city) {_f(m2_cpc)}.", "",
         "## 1. Opponents (chosen from the source pre-run by the fixed rule, before any target-city result)", "",
         "| Opponent | Model | Median share, source validation block |", "|---|---|---|"]
    names = {"fit": "Strongest earlier model (fit)", "zero_shot": "Strongest earlier model (no data)",
             "fit_original": "Strongest as-published version (fit)",
             "zero_shot_original": "Strongest as-published version (no data)"}
    L += [f"| {names[k]} | {v['model']} | {_f(v['share_med'])} |" for k, v in opp.items()]
    L += ["", "## 2. Contrasts (share difference paired by city; C1–C4 are the primary contrasts, with Holm; D are "
          "descriptive)", "",
          "| ID | Contrast | Median share difference [95% interval] | p | Wins/losses/ties | Holm | Verdict |",
          "|---|---|---|---|---|---|---|"]
    for k, r in con.items():
        L.append(f"| {k} | {r['a'][0]} ({LEVEL_EN[r['a'][1]]}) − {r['b'][0]} ({LEVEL_EN[r['b'][1]]}) | "
                 f"{_f(r['median'])} [{_f(r['ci_lo'])}, {_f(r['ci_hi'])}] | {_f(r['p'], 4)} | "
                 f"{r['wins']}/{r['losses']}/{r['ties']} | "
                 f"{'—' if r['holm'] is None else 'pass' if r['holm'] else 'fail'} | {r['verdict'] or '(descriptive)'} |")
    for title, col, nd in (("3. Share (higher is better)", "share", 3), ("4. NLL (nats/visit, lower is better)", "nll", 4),
                           ("5. CPC (higher is better)", "cpc", 3)):
        L += ["", f"## {title}", ""] + _table(s, fits_t, col, nd, cfg)
    L += ["", f"## 6. Primary skeleton {P} against each earlier model, city by city (share difference = {P} − other)", "",
          "| Earlier model | Local refit: wins/losses/ties, median share difference | "
          "Uniform constants: wins/losses/ties, median share difference |", "|---|---|---|"]
    for m in list(pm.EQ_MODELS) + list(pm.MAT_MODELS):
        cells = []
        for lv in ("local", "uniform"):
            d = st.paired(tg, (P, lv), (m, lv)).to_numpy()
            cells.append(f"{int((d > 0).sum())}/{int((d < 0).sum())}/{int((d == 0).sum())}, {_f(float(st._medians(d[None, :])[0]))}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## 7. By city group (median share)", "", "| Model | Level | Law-test cities | Confirmation cities |",
          "|---|---|---|---|"]
    for m in dict.fromkeys([P] + [v["model"] for v in opp.values()]):
        for lv in pt.LEVELS:
            g = tg[(tg["model"] == m) & (tg["level"] == lv)]
            L.append(f"| {m} | {LEVEL_EN[lv]} | " + " | ".join(
                _f(st._quantile(g.loc[g["group"] == k, "share"], 0.5)) for k in ("lawtest", "confirm")) + " |")
    L += ["", "## 8. Source pre-run (discovery cities x validation block, leave one city out)", "",
          "The skeletons were selected on these cities, so this table favours them; it is used only to choose the "
          "opponents, not as a comparison result.", ""]
    L += _table(st.summary(pre), fits_s, "share", 3, cfg)
    L += ["", "Kwon's grouping and masses use the city's own training-period trips, so its last three levels are not "
          "transfer without trip data: only the exponents and the intercept are taken from the source cities.", "",
          "All masses are incremented by 1; the intervening opportunities of radiation, extended radiation, intervening "
          "opportunities and OPS are the sums of the raw masses of strictly closer units (both ends excluded); "
          "the S_ji of PWO is the sum of the incremented masses (both ends included).", "",
          "p is the two-sided city-bootstrap p, computed as (k + 1) / (B + 1), at least about 2 / (B + 1)."]
    return "\n".join(L) + "\n"


def stage_report() -> Path:
    import ledger
    cfg = require_fixed()
    tg = pd.read_parquet(OUT / "target.parquet")
    con = json.loads((OUT / "contrasts.json").read_text())
    if ledger.sha(OUT / "target.parquet") != con["target_sha"]:
        raise RuntimeError("target.parquet is not the file the contrasts were computed from")
    _need_all(tg, "target", "target.parquet")
    m2 = float(np.median([refs(c, "test_dest")["m2_cpc"] for c in tg["city"].unique()]))
    md = render(cfg, pd.read_parquet(OUT / "prerun.parquet"), tg, pd.read_parquet(OUT / "fits_source.parquet"),
                pd.read_parquet(OUT / "fits_target.parquet"), con["opponents"], con["contrasts"], m2)
    p = OUT / "report_p9.md"
    p.write_text(md)
    print(md)
    return p


STAGES = {"source": stage_source, "prerun": stage_prerun, "select": stage_select, "target": stage_target,
          "contrasts": stage_contrasts, "report": stage_report}


def main(argv: list):
    if argv[:1] == ["fix"] and (len(argv) == 1 or (len(argv) == 3 and argv[1] == "--reason")):
        import ledger
        return fix_config(argv[2] if len(argv) == 3 else None, log=ledger.add)
    if len(argv) == 1 and argv[0] in STAGES:
        return STAGES[argv[0]]()
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
