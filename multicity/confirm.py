"""Freeze and confirmation (plan 6; design decision of 2026-11-10).

Stages (compute server, loop env; every stage but freeze needs frozen.require()):
  freeze    frozen.freeze(config/frozen_library.json, config/confirmation_criteria.md, the merge-variant pool)
  refs      every eligible city: reference models on the test blocks (refs.test_refs); a confirmation city first
            gets its validation references (refs.run_city, unsealed) in the same job
  fit       the 26 evaluation candidates (EVAL_SET = prefreeze_mv's discovery candidates) on all 67 cities, constants on
            train x search as before, scored on test x destination fold 0 ("test_dest"); the confirmation cities also
            on the validation block; constants must equal the pre-freeze ones (discovery cities) -> else STOP
  rescore   the same constants scored on test x fold 1 ("test_time"), no refit
  banded    the flexible distance-band model on the confirmation cities
  analyze   P1-P5 of the protocol (Task 4)
  report    markdown (Task 4)
Outputs: results/multicity/confirm/. No LLM call.
Usage: python -I confirm.py freeze | refs | fit | rescore | banded | analyze | report
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import config  # noqa: E402
import frozen  # noqa: E402

ROOT = config.RESULTS / "confirm"
MV = config.RESULTS / "prefreeze_mv"
HERE = Path(__file__).resolve().parent
LIBRARY_CFG = HERE / "config" / "frozen_library.json"
PROTOCOL = HERE / "config" / "confirmation_criteria.md"
CONSTS_TOL = 1e-9
PF_BANDED = config.RESULTS / "prefreeze" / "banded"
PF_LAW_BANDED = config.RESULTS / "prefreeze" / "lawtest_banded"


def _confirm_ids():
    return set(config.CONFIRM) | set(config.cities_with_role("confirm"))


def all_cities() -> list:
    """The 67 eligible cities: search, discovery, lawtest and confirmation."""
    return [c for r in ("search", "discovery", "lawtest", "confirm") for c in config.cities_with_role(r)]


def eval_set() -> list:
    """The pool entries of the merge-variant run's discovery candidates (V3 gate 24 + plan 5's 12 = 26), in order."""
    cids = list(dict.fromkeys(pd.read_parquet(MV / "discovery" / "cities_val.parquet")["cid"]))
    pool = {x["cid"]: x for x in json.loads((MV / "loop" / "pool.json").read_text())}
    return [pool[c] for c in cids]


def refs_job(city: str) -> str:
    """One writer per city: a confirmation city's run_city (unsealed) then test_refs; others test_refs only."""
    import refs
    try:
        if city in _confirm_ids():
            os.environ.update(frozen.unseal_env())
            refs.run_city(city)
        refs.test_refs(city)
        return city
    except Exception as e:  # noqa: BLE001 — reported by the stage
        return f"{city}: {e!r}"[:300]


def _none(v):
    """A parquet cell without constants: None, NaN or the JSON string "null"."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "null"):
        return None
    return v


def consts_match(new_cv, old_cv, cids) -> float:
    """Largest absolute difference of the constants of cids between two fits (inf when a (cid, city) of old_cv is
    missing from new_cv or one side has constants and the other has none)."""
    new = {(r.cid, r.city): r.consts for r in new_cv.itertuples()}
    worst = 0.0
    for r in old_cv[old_cv["cid"].isin(cids)].itertuples():
        if (r.cid, r.city) not in new:
            return np.inf
        a, b = _none(r.consts), _none(new[(r.cid, r.city)])
        if (a is None) != (b is None):
            return np.inf
        if a is None:
            continue
        a, b = np.asarray(json.loads(a), float), np.asarray(json.loads(b), float)
        if a.shape != b.shape:
            return np.inf
        d = np.abs(a - b)
        if not np.all(np.isfinite(d)):
            if not np.array_equal(a, b, equal_nan=True):
                return np.inf
            d = np.where(np.isfinite(d), d, 0.0)
        worst = max(worst, float(np.max(d)) if d.size else 0.0)
    return worst


def _rescore_city(args):
    import fit_library
    import library
    from refit_select import PENALTY, boot_nll, eta, loss
    city, rows, cands, mode, B, n_orig = args
    by = {x["cid"]: x for x in cands}
    D = library.city_data(city, n_orig, mode)
    W = library._boot_weights(B)
    flags = fit_library._qc_flags(city)
    out, cols = [], []
    for r in rows:
        r = dict(r)
        if _none(r.get("consts")) is not None:                    # scored on this block whatever the source block said
            c = dataclasses.replace(library.candidate({**by[r["cid"]], "consts_refit": None}),
                                    consts=np.asarray(json.loads(r["consts"]), float))
            lv = loss(c, D["va"])
            r["finite"] = bool(lv < PENALTY)
            r["val_nll"] = float(lv) if r["finite"] else None
            cols.append(boot_nll(eta(c, D["va"]), D["va"], D["Yg"], W) if r["finite"] else np.full(len(W), PENALTY))
            r["trust"] = bool(r["finite"] and all(fit_library.zero_safe(c, D["va"], v) for v in flags))
        else:
            r.update(finite=False, val_nll=None, trust=False)
            cols.append(np.full(len(W), PENALTY))
        out.append(r)
    return city, out, np.column_stack(cols)


def rescore(cv, cands, mode: str, B: int = 200, procs: int = 1, n_orig=None) -> pd.DataFrame:
    """The rows of cv (one fit_library output) re-scored on another block with their own constants: val_nll, the
    bootstrap columns, trust, then fit_library._finish (1-SE, near-best, shares, primary) with that block's refs."""
    import fit_library
    cx = {x["cid"]: int(x["complexity"]) for x in cands}
    jobs = [(c, g.to_dict("records"), cands, mode, B, n_orig) for c, g in cv.groupby("city", sort=False)]
    if procs <= 1 or len(jobs) <= 1:
        res = list(map(_rescore_city, jobs))
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(procs, len(jobs)), maxtasksperchild=1) as P:
            res = list(P.imap_unordered(_rescore_city, jobs, chunksize=1))
    rows = []
    for city, rs, boot in res:
        done, _ = fit_library._finish(city, rs, boot, fit_library._refs(city, mode), fit_library._qc_flags(city), cx)
        rows += done
    return pd.DataFrame(rows)


# ================================================================================================ CLI stages
def _procs(default="10"):
    return int(os.environ.get("LOOP_OUTER_PROCS", default))


def stage_freeze():
    import ledger
    pool = {x["cid"]: x for x in json.loads((MV / "loop" / "pool.json").read_text())}
    lib = [e["cid"] for e in json.loads(LIBRARY_CFG.read_text())["entries"]]
    why = freeze_checks([x["cid"] for x in eval_set()], lib, passed_laws())
    if why:
        ledger.add(f"Plan 6 freeze: REFUSED — {'; '.join(why)}")
        return 3
    pins = {"eval_set": MV / "discovery" / "cities_val.parquet", "pool": MV / "loop" / "pool.json",
            "gates": MV / "gates.json", "library_cfg": LIBRARY_CFG}
    pins.update({f"laws_{Path(f).stem}": LAWS / f for pair in LAW_FILES for f in pair})
    sha = frozen.freeze(LIBRARY_CFG, PROTOCOL, pool, log=ledger.add, pinned=pins)
    print("frozen", sha)
    return 0


def stage_refs():
    import multiprocessing as mp
    import fit_library
    import ledger
    frozen.require()
    cities = sorted(all_cities(), key=lambda c: -fit_library._size(c))
    with mp.get_context("fork").Pool(min(_procs(), len(cities)), maxtasksperchild=1) as P:
        res = list(P.imap_unordered(refs_job, cities, chunksize=1))
    errs = [r for r in res if r not in set(cities)]
    conf = _confirm_ids()
    miss = {}
    for c in cities:
        r = {**_read(config.RESULTS / f"refs/{c}.json", {}), **_read(config.RESULTS / f"refs_test/{c}.json", {})}
        m = refs_missing(r, c in conf)
        if m:
            miss[c] = m
    ledger.add(f"Plan 6 refs: test-block references (test_dest = test x fold 0, test_time = test x fold 1) in "
               f"refs_test/ for {len(cities) - len(errs)} of {len(cities)} cities (pre-freeze refs/ files untouched); "
               f"confirmation cities' validation references computed fresh (the sealed copy stays unread)"
               + (f"; job errors {errs[:5]}" if errs else "")
               + (f"; INCOMPLETE {len(miss)} cities, e.g. {dict(list(miss.items())[:3])}" if miss else "; all complete"))
    return 4 if errs or miss else 0


def stage_fit():
    import fit_library
    import ledger
    frozen.require()
    cands = eval_set()
    cids = [x["cid"] for x in cands]
    conf = sorted(_confirm_ids())
    fit_library.main(all_cities(), procs=_procs(), cands=cands, out_dir=ROOT / "test_dest", mode="test_dest",
                     allow_confirm=True)
    fit_library.main(conf, procs=_procs(), cands=cands, out_dir=ROOT / "val_confirm", mode="val", allow_confirm=True)
    td = pd.read_parquet(ROOT / "test_dest" / "cities_val.parquet")
    old = pd.read_parquet(MV / "discovery" / "cities_val.parquet")
    d_disc = consts_match(td, old, cids)
    d_conf = consts_match(td, pd.read_parquet(ROOT / "val_confirm" / "cities_val.parquet"), cids)
    if not (d_disc <= CONSTS_TOL and d_conf <= CONSTS_TOL):
        ledger.add(f"Plan 6 fit: STOP — constants differ from the pre-freeze fits (discovery max diff {d_disc:.2e}) or "
                   f"between the confirmation cities' two runs ({d_conf:.2e})")
        return 3
    (ROOT / "fit_ok").write_text(f"{d_disc!r} {d_conf!r}\n")
    ledger.add(f"Plan 6 fit: {len(cids)} candidates on {td['city'].nunique()} cities scored on test x fold 0; "
               f"confirmation cities also on the validation block; constants equal the pre-freeze discovery fits "
               f"(max diff {d_disc:.1e}) and across the two runs ({d_conf:.1e}); cities_val.parquet sha "
               f"{ledger.sha(ROOT / 'test_dest' / 'cities_val.parquet')}")
    return 0


def stage_rescore():
    import ledger
    frozen.require()
    _require_fit_ok()
    cv = pd.read_parquet(ROOT / "test_dest" / "cities_val.parquet")
    rs = rescore(cv, eval_set(), "test_time", procs=_procs())
    out = ROOT / "test_time"
    out.mkdir(parents=True, exist_ok=True)
    rs.to_parquet(out / "cities_val.parquet", index=False)
    ledger.add(f"Plan 6 rescore: the same constants on test x fold 1: {rs['city'].nunique()} cities, finite rows "
               f"{int(rs['finite'].astype(bool).sum())} of {len(rs)}; cities_val.parquet sha {ledger.sha(out / 'cities_val.parquet')}")
    return 0


def stage_banded():
    import banded
    import ledger
    frozen.require()
    conf = sorted(_confirm_ids())
    errs = {}
    for c in conf:
        try:
            banded.city_banded(c, out_dir=ROOT / "banded", allowed=conf, allow_confirm=True)
        except Exception as e:  # noqa: BLE001
            errs[c] = repr(e)[:200]
    ledger.add(f"Plan 6 banded: flexible distance-band model on {len(conf) - len(errs)} of {len(conf)} confirmation cities"
               + ("" if not errs else f"; errors {errs}"))
    return 4 if errs else 0


PASS_MIN = 5          # P4: at least 5 of the 6 confirmation cities
LAWS = config.RESULTS / "laws"
LAW_FILES = (("registered.json", "results.json"), ("registered_extra.json", "results_extra.json"),
             ("registered_v3.json", "results_v3.json"))
BAND_QUANT = ("ed_5_10km", "eN_5_10km", "slope_eN_logd")


def _read(p, default):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


# ------------------------------------------------------------------------------------------ pure analysis helpers
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def prediction_band(x, y, x_new, alpha: float = 0.05):
    """OLS of y on x; the (1 - alpha) prediction interval of a new observation at x_new."""
    from scipy.stats import t
    x, y, x0 = np.asarray(x, float), np.asarray(y, float), np.atleast_1d(np.asarray(x_new, float))
    n = len(x)
    b, a = np.polyfit(x, y, 1)
    s = np.sqrt(np.sum((y - (a + b * x)) ** 2) / (n - 2))
    sxx = np.sum((x - x.mean()) ** 2)
    half = t.ppf(1 - alpha / 2, n - 2) * s * np.sqrt(1 + 1 / n + (x0 - x.mean()) ** 2 / sxx)
    return a + b * x0 - half, a + b * x0 + half


def transfer_consts(cv_disc, cid) -> list:
    """Element-wise median of the constants of cid over the finite rows of cv_disc (direct transfer)."""
    g = cv_disc[(cv_disc["cid"] == cid) & cv_disc["finite"].astype(bool)]
    m = [json.loads(v) for v in g["consts"] if _none(v) is not None]
    if not m:
        return None
    return [float(v) for v in np.median(np.array(m, float), axis=0)]


def coverage(cv, refs: dict, clean: list, margin: float) -> dict:
    """{city: covered}: some clean library member near-best (sensitivity.recompute_near at margin, relative to the
    best row of cv in that city) with an NLL below the floor (scoring.passes_floor)."""
    import scoring
    import sensitivity as sv
    nn = sv.recompute_near(cv, refs, margin)
    out = {}
    for city, g in nn.groupby("city", sort=True):
        ok = g[g["cid"].isin(clean) & g["near_best"].astype(bool)]
        out[city] = bool(any(scoring.passes_floor(v, refs.get(city)) for v in ok["val_nll"]))
    return out


def law_checks(fit_xy: dict, confirm_xy: dict, alpha: float = 0.05) -> dict:
    """P4: OLS on the fit cities' (x, y); per confirmation city inside its (1 - alpha) prediction interval (None
    without a value); pass = at least PASS_MIN inside; the slope among the confirmation cities (>= 3 values)."""
    if len(fit_xy) < 3:
        return {"n_fit": len(fit_xy), "slope_fit": None, "intercept_fit": None, "inside": {c: None for c in confirm_xy},
                "n_inside": 0, "n_confirm": len(confirm_xy), "pass": False, "slope_confirm": None,
                "note": "too few fit cities"}
    x = np.array([v[0] for v in fit_xy.values()], float)
    y = np.array([v[1] for v in fit_xy.values()], float)
    b, a = np.polyfit(x, y, 1)
    inside = {}
    for c, v in confirm_xy.items():
        if v is None or not np.all(np.isfinite(v)):
            inside[c] = None
            continue
        lo, hi = prediction_band(x, y, v[0], alpha)
        inside[c] = bool(lo[0] <= v[1] <= hi[0])
    avail = [v for v in confirm_xy.values() if v is not None and np.all(np.isfinite(v))]
    sc = float(np.polyfit([v[0] for v in avail], [v[1] for v in avail], 1)[0]) if len(avail) >= 3 else None
    n_in = sum(1 for v in inside.values() if v)
    return {"n_fit": int(len(x)), "slope_fit": float(b), "intercept_fit": float(a), "inside": inside, "n_inside": n_in,
            "n_confirm": len(confirm_xy), "pass": bool(n_in >= PASS_MIN), "slope_confirm": sc}


def sign_check(values: dict, direction: str) -> dict:
    sgn = 1 if direction == "+" else -1
    n_ok = sum(1 for v in values.values() if v is not None and np.isfinite(v) and sgn * v > 0)
    return {"n_ok": n_ok, "n": len(values), "values": values, "pass": bool(n_ok >= PASS_MIN)}


def p2_check(disc: list, conf: list) -> dict:
    d = np.array([v for v in disc if v is not None and np.isfinite(v)], float)
    c = np.array([v for v in conf if v is not None and np.isfinite(v)], float)
    q10, q90 = (float(np.percentile(d, 10)), float(np.percentile(d, 90))) if len(d) else (None, None)
    med = float(np.median(c)) if len(c) else None
    return {"disc_q10": q10, "disc_q90": q90, "conf_median": med, "n_conf": int(len(c)),
            "pass": bool(med is not None and q10 is not None and q10 <= med <= q90)}


EXPECTED_LAWS = ("H1-01", "H2-03", "H3-01", "H3-02", "H4-01", "H4-02", "H4-03")     # the protocol's 7 passed laws
REF_NEED = ("grav_power", "M2_glm", "S0")
FLOOR = ("grav_power", "grav_exp", "ext_rad")


def _has(m, k):
    return isinstance((m or {}).get(k), dict) and m[k].get("nll_per_visit") is not None


def refs_missing(r: dict, is_confirm: bool) -> list:
    """The scores a city's references lack: per test block grav_power, M2_glm and S0 and at least one floor model;
    a confirmation city also its validation "models" (same needs)."""
    out = []
    keys = ["test_dest", "test_time"] + (["models"] if is_confirm else [])
    for key in keys:
        m = r.get(key)
        if not isinstance(m, dict):
            out.append(key)
            continue
        out += [f"{key}:{k}" for k in REF_NEED if not _has(m, k)]
        if not any(_has(m, k) for k in FLOOR):
            out.append(f"{key}:floor")
    return out


def freeze_checks(eval_cids: list, library_cids: list, laws_: list) -> list:
    """What must hold before the freeze: 26 evaluation candidates, every library member and law skeleton among them,
    the passed laws = the protocol's 7."""
    out = []
    if len(eval_cids) != 26:
        out.append(f"evaluation set has {len(eval_cids)} candidates, not 26")
    out += [f"library member {c} not in the evaluation set" for c in library_cids if c not in eval_cids]
    out += [f"law {h['id']} skeleton {h['skeleton']} not in the evaluation set" for h in laws_
            if h["skeleton"] != "flex_banded" and h["skeleton"] not in eval_cids]
    ids = tuple(sorted(h["id"] for h in laws_))
    if ids != tuple(sorted(EXPECTED_LAWS)):
        out.append(f"passed laws {list(ids)} differ from the protocol's {list(EXPECTED_LAWS)}")
    return out


def p5_cids() -> list:
    """P5 uses the 24 V3 gate members."""
    return [g["cid"] for g in json.loads((MV / "gates.json").read_text())["variants"]["V3"]["gate"]]


def _require_fit_ok():
    if not (ROOT / "fit_ok").exists():
        raise RuntimeError("the fit stage has not passed its reproduction check (confirm/fit_ok missing)")


def passed_laws() -> list:
    out = []
    for rf, sf in LAW_FILES:
        reg = {h["id"]: h for h in _read(LAWS / rf, [])}
        out += [reg[r["id"]] for r in _read(LAWS / sf, []) if r.get("pass") and r["id"] in reg]
    return out


# ------------------------------------------------------------------------------------------ analyze stage
def _group(cov: dict, groups: dict, list_cities=("confirm",)) -> dict:
    out = {}
    for g, cities in groups.items():
        v = [cov[c] for c in cities if c in cov]
        k, n = int(sum(v)), len(v)
        out[g] = {"k": k, "n": n, "ci": list(clopper_pearson(k, n)) if n else None}
        if g in list_cities:
            out[g]["cities"] = {c: cov.get(c) for c in cities}
    return out


def _bands(cities_by_dir: dict):
    import prefreeze
    bd = {}
    for d, cities in cities_by_dir.items():
        for c in cities:
            p = Path(d) / f"{c}.json"
            if p.exists():
                bd[c] = json.loads(p.read_text())
    refs_b = {c: prefreeze._intkeys(b["refs"]) for c, b in bd.items()}
    return bd, refs_b, {c: prefreeze._intkeys(b["visits"]) for c, b in bd.items()}


def _ref_point(city):
    import laws
    import loop
    return city, laws.reference_points(loop._small_datas([city]))[city]


def stage_analyze():
    import multiprocessing as mp
    import laws
    import ledger
    import library
    import loop
    import prefreeze
    import scoring
    import v3_library as v3
    from refit_select import PENALTY, loss
    entries = frozen.require()
    _require_fit_ok()
    clean = frozen.clean_library()
    lib = [e["cid"] for e in entries]
    pool = {x["cid"]: x for x in json.loads((MV / "loop" / "pool.json").read_text())}
    roles = {r: config.cities_with_role(r) for r in ("search", "discovery", "lawtest", "confirm")}
    disc, law, conf = roles["search"] + roles["discovery"], roles["lawtest"], roles["confirm"]
    groups = {"discovery": disc, "lawtest": law, "confirm": conf}
    td = pd.read_parquet(ROOT / "test_dest" / "cities_val.parquet")
    tt = pd.read_parquet(ROOT / "test_time" / "cities_val.parquet")
    R = {c: {**_read(config.RESULTS / f"refs/{c}.json", {}), **_read(config.RESULTS / f"refs_test/{c}.json", {})}
         for c in all_cities()}
    out = {"library": {"clean": clean, "entries": [[e["cid"], e["role"]] for e in entries]}}
    # P1 coverage
    P1 = {}
    for name, cv in (("test_dest", td), ("test_time", tt)):
        rf = {c: R[c].get(name) for c in R}
        P1[name] = {f"{m:g}": _group(coverage(cv, rf, clean, m), groups) for m in (0.03, 0.05)}
    vc = pd.concat([pd.read_parquet(ROOT / "val_confirm" / "cities_val.parquet"),
                    pd.read_parquet(MV / "discovery" / "cities_val.parquet")], ignore_index=True)
    rf = {c: R[c].get("models") for c in R}
    P1["val"] = {f"{m:g}": _group(coverage(vc, rf, clean, m), {"discovery": disc, "confirm": conf}) for m in (0.03, 0.05)}
    out["P1"] = P1
    # P2 test-period share_S0 (double holdout)
    ok = td[td["finite"].astype(bool)]
    out["P2"] = {c: p2_check(ok[(ok.cid == c) & ok.city.isin(disc)]["share_s0"].tolist(),
                             ok[(ok.cid == c) & ok.city.isin(conf)]["share_s0"].tolist()) for c in lib}
    # P3 transfer vs local refit on the confirmation cities (double holdout)
    mv = pd.read_parquet(MV / "discovery" / "cities_val.parquet")            # the pre-freeze discovery fits
    tc = {c: transfer_consts(mv[mv.city.isin(disc)], c) for c in lib}
    P3 = {c: {} for c in lib}
    for city in conf:
        D = library.city_data(city, None, "test_dest")
        rt = R[city].get("test_dest")
        for c in lib:
            row = td[(td.cid == c) & (td.city == city)]
            refit = row["share_s0"].iloc[0] if len(row) and bool(row["finite"].iloc[0]) else None
            if tc[c] is None:
                P3[c][city] = {"refit": None if refit is None or refit != refit else float(refit), "transfer": None,
                               "transfer_finite": False}
                continue
            cand = dataclasses.replace(library.candidate({**pool[c], "consts_refit": None}), consts=np.asarray(tc[c], float))
            lv = loss(cand, D["va"])
            P3[c][city] = {"refit": None if refit is None or refit != refit else float(refit),
                           "transfer": scoring.share(float(lv), rt, "S0") if lv < PENALTY else None,
                           "transfer_finite": bool(lv < PENALTY)}
    out["P3"] = P3
    # P4 laws
    bd, refs_b, visits = _bands({PF_BANDED: disc, PF_LAW_BANDED: law, ROOT / "banded": conf})
    flex = {c: prefreeze.flex_quantities(b) for c, b in bd.items()}
    laws_ = passed_laws()
    cov = laws.covariates(disc + law + conf, allow_confirm=True).set_index("city")
    skel_cids = sorted({h["skeleton"] for h in laws_ if h["skeleton"] in pool})
    sub = td[td.cid.isin(skel_cids)]
    pt = laws.param_table(sub, list(pool.values()))
    if any(str(h["parameter"]).startswith("elasticity_") for h in laws_):
        with mp.get_context("fork").Pool(min(_procs(), len(disc + law + conf)), maxtasksperchild=1) as P:
            refp = dict(P.imap_unordered(_ref_point, disc + law + conf, chunksize=1))
        pt = loop._with_composites(pt, sub, list(pool.values()),
                                   [pool[h["skeleton"]] for h in laws_ if str(h["parameter"]).startswith("elasticity_")], refp)
    prof, _ = prefreeze.skeleton_profiles(sub, pool, refs_b)
    bq = {(cid, city): v3.skel_quantities(p, refs_b[city], visits.get(city, {})) for (cid, city), p in prof.items()}
    P4 = []
    for h in laws_:
        k, q = h["skeleton"], h["parameter"]
        if k == "flex_banded":
            val = {c: (flex.get(c) or {}).get(q) for c in disc + law + conf}
        elif q in BAND_QUANT:
            val = {c: (bq.get((k, c)) or {}).get(q) for c in disc + law + conf}
        else:
            g = pt[(pt.cid == k) & (pt.param == q) & pt.identifiable.astype(bool)]
            val = dict(zip(g.city, g.value.astype(float)))
        base = {"id": h["id"], "skeleton": k, "parameter": q, "covariate": h["covariate"], "direction": h["direction"]}
        if h["covariate"] == "none":
            P4.append({**base, "kind": "sign", **sign_check({c: val.get(c) for c in conf}, h["direction"])})
            continue
        xy = lambda cs: {c: (float(cov.loc[c, h["covariate"]]), float(val[c])) for c in cs  # noqa: E731
                         if val.get(c) is not None and np.isfinite(val[c]) and np.isfinite(cov.loc[c, h["covariate"]])}
        cxy = xy(conf)
        P4.append({**base, "kind": "band", **law_checks(xy(disc + law), {c: cxy.get(c) for c in conf})})
    out["P4"] = P4
    # P5 elasticity shapes on the confirmation cities
    cprof, failed = prefreeze.skeleton_profiles(td[td.city.isin(conf) & td.cid.isin(p5_cids())], pool,
                                                {c: refs_b[c] for c in conf if c in refs_b})
    dref = {c: {kk: r["d"] for kk, r in refs_b[c].items() if r} for c in conf if c in refs_b}
    cmp_ = prefreeze.compare(cprof, {c: prefreeze._intkeys(bd[c]["profile"]) for c in conf if c in bd}, visits, dref)
    out["P5"] = {**prefreeze.closest(cmp_), "n_failed": failed}
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / "analysis.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    p1 = out["P1"]["test_dest"]["0.03"]["confirm"]
    ledger.add(f"Plan 6 analyze: P1 confirmation cities covered by the clean library (test x fold 0, 0.03) {p1['k']}/{p1['n']}; "
               f"P2 {sum(v['pass'] for v in out['P2'].values())}/{len(out['P2'])} library members within the discovery "
               f"10-90% range; P4 {sum(r['pass'] for r in P4)}/{len(P4)} passed laws confirmed "
               f"({', '.join(r['id'] + (' ok' if r['pass'] else ' no') for r in P4)}); analysis.json sha {ledger.sha(p)}")
    return 0


def _yn(v):
    return "—" if v is None else ("✔" if v else "✘")


def _f(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def build_report(out_md: Path, root: Path = None) -> dict:
    import prefreeze
    import report
    root = Path(root) if root else ROOT
    a = json.loads((root / "analysis.json").read_text())
    L = ["# Confirmation results (plan 6: confirmation cities and test period opened once, after the freeze)", "",
         "Frozen skeleton library: " + ", ".join(f"{c} ({r})" for c, r in a["library"]["entries"])
         + f"; clean library = {', '.join(a['library']['clean'])}. Criteria and definitions: the plan 6 confirmation scheme.", "",
         "## P1 Skeleton-library coverage (descriptive)", ""]
    rows = []
    for blk, by_m in a["P1"].items():
        for m, by_g in by_m.items():
            for g, v in by_g.items():
                rows.append([blk, m, g, f"{v['k']}/{v['n']}", "—" if not v.get("ci") else f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]"])
    L += prefreeze._md_table(["Scoring block", "Margin", "City group", "Coverage", "95% interval"], rows)
    cc = a["P1"].get("test_dest", {}).get("0.03", {}).get("confirm", {}).get("cities", {})
    if cc:
        L += ["", "Per confirmation city (double holdout, 0.03): " + ", ".join(f"{c} {_yn(v)}" for c, v in cc.items())]
    L += ["", "## P2 Test-period performance (criterion: confirmation-city median share_S0 within the discovery-city 10%–90% range)", ""]
    L += prefreeze._md_table(["Skeleton", "Confirmation median", "Discovery 10%", "Discovery 90%", "Met"],
                             [[c, _f(v["conf_median"]), _f(v["disc_q10"]), _f(v["disc_q90"]), _yn(v["pass"])]
                              for c, v in a["P2"].items()])
    L += ["", "## P3 Transfer vs local refit (share_S0, double holdout; descriptive)", ""]
    rows = [[c, city, _f(v.get("refit")), _f(v.get("transfer")),
             _f(None if v.get("refit") is None or v.get("transfer") is None else v["transfer"] - v["refit"])]
            for c, by in a["P3"].items() for city, v in by.items()]
    L += prefreeze._md_table(["Skeleton", "Confirmation city", "Local refit", "Direct transfer", "Difference"], rows)
    L += ["", "Direct transfer not fittable: " + ", ".join(f"{c} {sum(1 for v in by.values() if v.get('transfer_finite') is False)}/{len(by)}"
                                         for c, by in a["P3"].items())]
    L += ["", f"## P4 Previously passed laws (criterion: at least {PASS_MIN}/6 confirmation cities agree)", ""]
    rows = []
    for r in a["P4"]:
        if r["kind"] == "sign":
            rows.append([r["id"], f"{r.get('skeleton', '')} {r.get('parameter', '')}".strip(), "sign", f"{r['n_ok']}/{r['n']}", "—", _yn(r["pass"])])
        else:
            rows.append([r["id"], f"{r['skeleton']} {r['parameter']} ~ {r['covariate']}", "prediction interval",
                         f"{r['n_inside']}/{r['n_confirm']}", _f(r.get("slope_confirm")), _yn(r["pass"])])
    L += prefreeze._md_table(["ID", "Quantity", "Test", "Confirmation cities agreeing", "Slope within confirmation cities", "Met"], rows)
    L += ["", "## P5 Elasticity profiles (confirmation cities; number of cities where closest to the flexible model)", ""]
    p5 = a["P5"]
    cids = sorted({c for k in ("rmse_d", "rmse_N_shape") for c in p5.get(k, {})})
    L += prefreeze._md_table(["cid", "Closest e_d", "Closest e_N shape"],
                             [[c, p5.get("rmse_d", {}).get(c, 0), p5.get("rmse_N_shape", {}).get(c, 0)] for c in cids])
    text = "\n".join(L) + "\n"
    try:
        report._scan(text, allow_confirm=True)
    except ValueError as e:
        raise RuntimeError(f"confirmation report refused: {e}") from e
    Path(out_md).write_text(text)
    checks = [v["pass"] for v in a["P2"].values()] + [r["pass"] for r in a["P4"]]
    return {"pass": int(sum(checks)), "checks": len(checks)}


def stage_report():
    import ledger
    frozen.require()
    p = ROOT / "report.md"
    s = build_report(p)
    ledger.add(f"Plan 6 report: {s['pass']}/{s['checks']} registered checks met; report.md sha {ledger.sha(p)}")
    return 0


STAGES = {"freeze": stage_freeze, "refs": stage_refs, "fit": stage_fit, "rescore": stage_rescore,
          "banded": stage_banded, "analyze": stage_analyze, "report": stage_report}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: confirm.py {{{'|'.join(STAGES)}}}")
    sys.exit(STAGES[sys.argv[1]]())
