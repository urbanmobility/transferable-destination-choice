"""Pre-freeze consolidation (plan 5).

Hand-written reforms of the loop's skeletons (config/prefreeze_reforms.json: dimensionally consistent, zero-POI safe,
without s_j) judged with the loop's own rules, which stay frozen:
  reforms    acceptance exactly as an agent cleaning (review.verify_cleaned: 8 search cities, folds 2-3 -> fold 4,
             mean loss increase <= delta_city, no new non-fittable city)
Every output goes to results/multicity/prefreeze/; results/multicity/loop/ is only read. Data boundary: search and
discovery cities only (cities()); no lawtest / confirmation city, no test period. No LLM call.

Usage (compute server, same environment as the loop): python -I prefreeze.py reforms
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import config  # noqa: E402

ROOT = config.RESULTS / "prefreeze"
LOOP = config.RESULTS / "loop"
REFORMS = Path(__file__).resolve().parent / "config" / "prefreeze_reforms.json"
ROUND = 8                       # the loop's last round is 7
SOURCE = "prefreeze_reform"


def cities() -> list[str]:
    """The only cities this plan reads: search, then discovery (config/cities.csv roles)."""
    return config.cities_with_role("search") + config.cities_with_role("discovery")


def load_reforms(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def reform_entries(reforms: list[dict], pool: list[dict], verify) -> tuple[list[dict], list[dict]]:
    """verify(orig_entry, sympy_str) -> review.verify_cleaned's dict. cid = <parent>r<k>, k = the reform's position
    among the parent's reforms (rejected ones keep their number). Returns (accepted pool entries, every result)."""
    by = {x["cid"]: x for x in pool}
    k, acc, res = {}, [], []
    for r in reforms:
        orig = by[r["parent"]]
        k[r["parent"]] = k.get(r["parent"], 0) + 1
        cid = f"{r['parent']}r{k[r['parent']]}"
        v = verify(orig, r["sympy"])
        res.append({"id": r["id"], "cid": cid, "parent": r["parent"], "sympy": r["sympy"],
                    "accepted": bool(v.get("accepted")), "reason": v.get("reason"),
                    "mean_norm_delta": v.get("mean_norm_delta"), "new_nonfittable": v.get("new_nonfittable", []),
                    "per_city": v.get("per_city", {}), "equation": v.get("equation"), "complexity": v.get("complexity")})
        if v.get("accepted"):
            cs = v.get("consts")
            acc.append({"cid": cid, "equation": v["equation"], "template": None, "variant": "A",
                        "complexity": int(v["complexity"]), "consts_refit": None if cs is None else [float(c) for c in cs],
                        "source": [SOURCE], "parent": r["parent"], "round": ROUND, "insight": r["rationale"]})
    return acc, res


COPY_FILES = ("pool.json", "inner_fits.parquet", "flags.json")


def prepare_copy(src: Path, dst: Path) -> None:
    """Copy of the loop state the ranking needs (pool, inner fits, bootstrap columns, flags); never overwrites."""
    import shutil
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    dst.mkdir(parents=True)
    for f in COPY_FILES:
        shutil.copy2(src / f, dst / f)
    shutil.copytree(src / "inner_boot", dst / "inner_boot")


def rank_with_reforms(steps, entries: list[dict]) -> list[dict]:
    """The loop's own steps on the copy: add the reforms, fit them on the 8 search cities (fold 4), rank (frozen
    lexicographic rule, equivalence merge) -> ranked_r8.json in the copy. Plan 5 ran under the registered rule (one
    multiplier 0.03, chain merge); scoring.registered_rule keeps it so after the V3 rule change of 2026-11-10."""
    import scoring
    steps.save_pool(steps.pool() + list(entries))
    steps.fit_new(ROUND)
    with scoring.registered_rule():
        return steps.rank(ROUND)


def reform_fate(ranked: list[dict], entries: list[dict]) -> list[dict]:
    """Where each reform ended: its representative (itself, the one it was merged into, or None when unranked), the
    1-based position of that representative and its ranking fields."""
    out = []
    for x in entries:
        hit = next(((k, r) for k, r in enumerate(ranked, 1)
                    if r["cid"] == x["cid"] or x["cid"] in r.get("merged_from", [])), None)
        k, r = hit if hit else (None, {})
        out.append({"cid": x["cid"], "rep": r.get("cid"), "position": k, "n_near": r.get("n_near"),
                    "n_flags": r.get("n_flags"), "complexity": r.get("complexity"), "gate1": r.get("gate1")})
    return out


REPRO_TOL = 1e-6                # nats per visit: the loop's six skeletons refitted here vs loop/c2_discovery


def check_reproduction(new_cv, old_cv, cids: list[str]) -> dict[str, float]:
    """Max |val_nll difference| per cid over every city of the old fit; inf when a city of the old fit is missing
    from the new one (fit_library drops a city's rows on any error there), when one side is finite and the other not,
    or when the old fit has no row of the cid (nothing verified)."""
    out = {}
    for cid in cids:
        a = new_cv[new_cv["cid"] == cid].set_index("city")
        b = old_cv[old_cv["cid"] == cid].set_index("city")
        worst = float("inf") if not len(b) or set(b.index) - set(a.index) else 0.0
        for c in sorted(set(a.index) & set(b.index)):
            fa, fb = bool(a.at[c, "finite"]), bool(b.at[c, "finite"])
            if fa != fb:
                worst = float("inf")
            elif fa:
                worst = max(worst, abs(float(a.at[c, "val_nll"]) - float(b.at[c, "val_nll"])))
        out[cid] = worst
    return out


def reproduction_gate(rep: dict, summary) -> list[str]:
    """Reasons to stop the discovery stage (empty = go on): nothing compared, a loop skeleton above REPRO_TOL, or a
    city whose fit_library summary carries an error (its rows are missing for every candidate)."""
    why = [] if rep else ["no loop skeleton was compared with loop/c2_discovery"]
    why += [f"{c}: max |val_nll diff| {v:.3g} > {REPRO_TOL}" for c, v in sorted(rep.items()) if not v <= REPRO_TOL]
    if "error" in summary.columns:
        why += [f"city {r.city}: fit error {r.error}" for r in summary.itertuples()
                if isinstance(r.error, str) and r.error]
    return why


def coverage_tables(cv, refs: dict, pool: dict, flags: dict) -> dict[str, list]:
    """Greedy coverage (report.coverage_curve) over every candidate and over the clean ones only (scoring.n_flags
    == 0); a flagged candidate's rows stay in the table as not near-best, so the city count is unchanged."""
    import report
    import scoring
    cx = {cid: int(x["complexity"]) for cid, x in pool.items()}
    flagged = {cid for cid in set(cv["cid"]) if scoring.n_flags(flags.get(cid)) > 0}
    clean = cv.copy()
    clean["near_best"] = clean["near_best"].astype(bool) & ~clean["cid"].isin(flagged)
    return {"all": report.coverage_curve(cv, refs, cx), "clean": report.coverage_curve(clean, refs, cx)}


# ================================================================================================ comparison, report
MIN_BAND_VISITS = 1000          # training visits a band needs in a city to enter the comparison
MIN_BANDS = 3
BOOT_B, BOOT_SEED = 2000, 20261109
BAND_NAMES = ("0–1 km", "1–2 km", "2–5 km", "5–10 km", "10–20 km", "≥ 20 km")


def _usable(p, b):
    v = None if p is None else p.get(b)
    return v is not None and all(x is not None and math.isfinite(x) for x in v)


def compare(skel: dict, flex: dict, visits: dict, dref: dict):
    """skel {(cid, city): {band: (e_d, e_N) | None}}, flex {city: same}, visits {city: {band: training visits}},
    dref {city: {band: reference distance km}} -> one row per (cid, city): RMSE of e_d and e_N against the flexible
    model over the bands both have (finite) with >= MIN_BAND_VISITS visits; rmse_N_shape = the same for e_N after
    removing each profile's mean over those bands (the flexible model's e_N is conditional on M2's destination
    covariates, the skeletons' is not, so only the shape over distance is like for like); the OLS slope of e_N on
    log d_ref of each side. NaN with fewer than MIN_BANDS bands."""
    import numpy as np
    import pandas as pd
    rows = []
    for (cid, city), sp in skel.items():
        fp, vb, db = flex.get(city) or {}, visits.get(city) or {}, dref.get(city) or {}
        bands = sorted(b for b in sp if _usable(sp, b) and _usable(fp, b) and (vb.get(b) or 0) >= MIN_BAND_VISITS
                       and db.get(b))
        r = {"cid": cid, "city": city, "n_bands": len(bands), "rmse_d": math.nan, "rmse_N": math.nan,
             "rmse_N_shape": math.nan, "slope_N_skel": math.nan, "slope_N_flex": math.nan}
        if len(bands) >= MIN_BANDS:
            ld = np.log([db[b] for b in bands])
            ds = np.array([sp[b][0] - fp[b][0] for b in bands])
            ns, nf = np.array([sp[b][1] for b in bands]), np.array([fp[b][1] for b in bands])
            r.update(rmse_d=float(np.sqrt(np.mean(ds ** 2))), rmse_N=float(np.sqrt(np.mean((ns - nf) ** 2))),
                     rmse_N_shape=float(np.sqrt(np.mean(((ns - ns.mean()) - (nf - nf.mean())) ** 2))),
                     slope_N_skel=float(np.polyfit(ld, ns, 1)[0]), slope_N_flex=float(np.polyfit(ld, nf, 1)[0]))
        rows.append(r)
    return pd.DataFrame(rows, columns=["cid", "city", "n_bands", "rmse_d", "rmse_N", "rmse_N_shape", "slope_N_skel",
                                       "slope_N_flex"])


def closest(df) -> dict:
    """{"rmse_N" | "rmse_N_shape" | "rmse_d": {cid: number of cities where it is the closest}}, over cities with
    >= 2 candidates that have the metric."""
    import numpy as np
    out = {"rmse_N": {}, "rmse_N_shape": {}, "rmse_d": {}}
    for m, cnt in out.items():
        if m not in df.columns:
            continue
        ok = df[np.isfinite(df[m].astype(float))]
        for _, g in ok.groupby("city"):
            if len(g) >= 2:
                w = g.loc[g[m].astype(float).idxmin(), "cid"]
                cnt[w] = cnt.get(w, 0) + 1
    return out


def _unit_mixing(sympy_str):
    """True / False: checks.unit_mixing flags the reform as written (constants as printed); None if it does not
    parse."""
    import checks
    import review
    from library import to_single
    from refit_select import parse_equation
    e, why = review._parse(sympy_str)
    if e is None:
        return None
    eq, _ = review._new_entry_fields({"equation": "f = log(#6 + 1.0); p1 = []; p2 = []", "template": None,
                                      "variant": "A", "complexity": 10}, e, None)
    return bool(checks.unit_mixing(to_single(parse_equation(eq, None, "A", 1))))


def skeleton_profiles(cv, pool: dict, refs_b: dict) -> tuple[dict, int]:
    """{(cid, city): banded.skeleton_profile} for the finite rows of cv whose cid is in pool and whose city has band
    references; also the number of rows whose profile could not be evaluated (left out, counted)."""
    import banded
    out, failed = {}, 0
    for r in cv[cv["finite"].astype(bool)].itertuples():
        if r.cid in pool and r.city in refs_b:
            try:
                out[(r.cid, r.city)] = banded.skeleton_profile(pool[r.cid], json.loads(r.consts), refs_b[r.city])
            except Exception:  # noqa: BLE001 — counted and reported by the caller
                failed += 1
    return out, failed


def flex_quantities(b: dict) -> dict:
    """Two city-level quantities of the flexible distance-band model (banded/<city>.json, string band keys), as in
    plan 5 §7 / the 11-09 diagnostic: ed_5_10km = e_d of band 3 (5-10 km) when it has >= MIN_BAND_VISITS training
    visits; slope_eN_logd = OLS slope of e_N on log d_ref over the bands with >= MIN_BAND_VISITS visits, a finite
    profile and a reference point (None with fewer than MIN_BANDS such bands)."""
    import numpy as np
    pr, rf, vi = _intkeys(b.get("profile")), _intkeys(b.get("refs")), _intkeys(b.get("visits"))
    ok = lambda k: _usable(pr, k) and (vi.get(k) or 0) >= MIN_BAND_VISITS and bool(rf.get(k))  # noqa: E731
    use = [k for k in sorted(pr) if ok(k)]
    slope = None
    if len(use) >= MIN_BANDS:
        slope = float(np.polyfit(np.log([rf[k]["d"] for k in use]), [pr[k][1] for k in use], 1)[0])
    return {"ed_5_10km": float(pr[3][0]) if ok(3) else None, "slope_eN_logd": slope}


def _intkeys(d):
    return {int(k): (tuple(v) if isinstance(v, list) else v) for k, v in (d or {}).items()}


def _median(xs):
    import numpy as np
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.median(xs)) if xs else None


def _md_table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def build_report(out_md: Path, root: Path = None) -> dict:
    """Markdown tables of plan 5 (sections 1-7) from root's outputs; refuses (RuntimeError, nothing written) a text
    that names a lawtest or confirmation city."""
    import numpy as np
    import pandas as pd
    import banded
    import report
    import scoring
    root = Path(root) if root else ROOT
    rf = json.loads((root / "reforms.json").read_text())
    fate = json.loads((root / "fate.json").read_text())
    cv = pd.read_parquet(root / "discovery/cities_val.parquet")
    cov = json.loads((root / "discovery/coverage.json").read_text())
    pool = {x["cid"]: x for x in json.loads((root / "loop/pool.json").read_text())}
    flags = json.loads((root / "loop/flags.json").read_text())
    bd = {p.stem: json.loads(p.read_text()) for p in sorted((root / "banded").glob("*.json"))}
    refs_b = {c: _intkeys(b["refs"]) for c, b in bd.items()}
    flex = {c: _intkeys(b["profile"]) for c, b in bd.items()}
    visits = {c: _intkeys(b["visits"]) for c, b in bd.items()}
    dref = {c: {k: r["d"] for k, r in rb.items() if r} for c, rb in refs_b.items()}
    cids = list(dict.fromkeys(cv["cid"]))
    cities_ = sorted(set(cv["city"]))
    fin = cv[cv["finite"].astype(bool)]
    skel, n_failed = skeleton_profiles(fin, pool, refs_b)
    df = compare(skel, flex, visits, dref)
    cl = closest(df)
    nf = lambda cid: scoring.n_flags(flags[cid]) if cid in flags else "—"  # noqa: E731
    f = report._f
    L = [f"# Pre-freeze consolidation: result tables (plan 5, {len(cities_)} cities)", "",
         "Data: the 8 search cities (training-period folds 2–4) and the discovery cities (training and validation periods); "
         "law-test cities, confirmation cities and the test period are not read. Judging rules as in the loop (δ_city, "
         "near-best, floor, lexicographic order, cleaning acceptance).", ""]
    L += ["## 1. Acceptance of the reform candidates (8 search cities, training-period fold 4; same rule as for agent cleanings)", ""]
    yn = {True: "yes", False: "no", None: "—"}
    L += _md_table(["id", "cid", "parent", "equation", "8-city mean loss increase (× δ_city)", "accepted", "reason",
                    "cleanliness flags (n)", "unit mixing (current check)"],
                   [[r["id"], r["cid"], r["parent"], f"`{r['sympy']}`", f(r.get("mean_norm_delta"), 2),
                     "yes" if r["accepted"] else "no", r.get("reason") or "", nf(r["cid"]), yn[_unit_mixing(r["sympy"])]]
                    for r in rf["results"]])
    L += ["", "## 2. Fate under the frozen rule (ranked on a copy of the loop state; representative ≠ itself = merged as equivalent)", ""]
    L += _md_table(["cid", "representative", "rank position", "near-best search cities", "flags (n)", "complexity", "in top 8"],
                   [[x["cid"], x["rep"] or "not ranked (no city above the floor)", x["position"] or "—", x["n_near"] if x["n_near"] is not None else "—",
                     x["n_flags"] if x["n_flags"] is not None else "—", x["complexity"] or "—",
                     "yes" if x.get("gate1") else "no"] for x in fate])
    L += ["", f"## 3. Discovery-city fits ({len(cities_)} cities, validation period)", ""]
    rows = []
    for cid in cids:
        g = cv[cv["cid"] == cid]
        src = "reform" if SOURCE in (pool.get(cid, {}).get("source") or []) else "loop"
        rows.append([cid, src, pool.get(cid, {}).get("complexity", "—"), int(g["finite"].astype(bool).sum()),
                     int(g["near_best"].astype(bool).sum()), int(g["primary"].astype(bool).sum()),
                     f(_median(g["share_s0"].tolist())), f(_median(g["share_grav"].tolist()))])
    L += _md_table(["cid", "source", "complexity", "fittable cities", "near-best cities", "primary-skeleton cities",
                    "share_S0 median", "share_grav median"], rows)
    bad = sorted({r.city for r in cv.itertuples() if not bool(r.finite)})
    if bad:
        L += ["", "Cities where some candidate cannot be fitted (✓ fittable, ✗ not fittable):", ""]
        L += _md_table(["city"] + cids, [[c] + ["✓" if bool(cv[(cv.cid == k) & (cv.city == c)]["finite"].any()) else "✗"
                                              for k in cids] for c in bad])
    vrefs = {}
    for c in cities_:
        p = config.RESULTS / f"refs/{c}.json"
        vrefs[c] = json.loads(p.read_text())["models"] if p.exists() else None
    clean_ok = {r.city for r in fin.itertuples() if bool(r.near_best) and scoring.n_flags(flags.get(r.cid)) == 0
                and scoring.passes_floor(r.val_nll, vrefs.get(r.city))}
    L += ["", "## 4. Coverage curves (greedy; covered = near-best and above the floor)", ""]
    for k, name in (("all", "all candidates"), ("clean", "clean candidates only"), ("c2_all", "first version c2, all"),
                    ("c2_clean", "first version c2, clean only")):
        t = cov.get(k) or []
        L.append(f"- {name}: " + (" → ".join(f"{r['cid']} (+{r['new']}, cumulative {r['covered']}/{r['n_cities']})" for r in t)
                                 if t else "none"))
    L.append(f"- Cities not covered with clean candidates only: {', '.join(c for c in cities_ if c not in clean_ok) or 'none'}")
    L += ["", f"## 5. Elasticities by distance band (median over {len(bd)} cities; only bands with ≥ {MIN_BAND_VISITS} "
          "training-period trips)", ""]
    nb = len(banded.BANDS)
    ok_b = lambda c, b: (visits.get(c, {}).get(b) or 0) >= MIN_BAND_VISITS  # noqa: E731
    for k, nm in ((0, "e_d = ∂η/∂ln d (along S_ij(d))"), (1, "e_N = ∂η/∂ln N_j")):
        rows = [[cid] + [f(_median([skel[(cid, c)][b][k] for c in bd if (cid, c) in skel and _usable(skel[(cid, c)], b)
                                    and ok_b(c, b)]), 2) for b in range(nb)] for cid in cids]
        rows.append(["flexible model"] + [f(_median([flex[c][b][k] for c in bd if _usable(flex[c], b) and ok_b(c, b)]), 2)
                                 for b in range(nb)])
        L += [f"{nm}:", ""] + _md_table(["candidate"] + list(BAND_NAMES), rows) + [""]
    L += [f"## 6. Distance to the flexible model (at least {MIN_BANDS} comparable bands per city)", ""]
    rows = []
    for cid in cids:
        g = df[(df.cid == cid) & np.isfinite(df.rmse_N.astype(float))]
        rows.append([cid, len(g), f(_median(g.rmse_d.tolist())), f(_median(g.rmse_N_shape.tolist())),
                     f(_median(g.rmse_N.tolist())), f(_median(g.slope_N_skel.tolist())),
                     cl["rmse_d"].get(cid, 0), cl["rmse_N_shape"].get(cid, 0), cl["rmse_N"].get(cid, 0)])
    L += _md_table(["cid", "comparable cities", "rmse_d median", "rmse_N shape median", "rmse_N level median",
                    "slope of e_N on ln d (median)", "cities where e_d is closest", "cities where the e_N shape is closest",
                    "cities where the e_N level is closest"], rows)
    L += ["", "Note: the flexible model's e_N is conditional on M2's destination covariates (population, jobs, road network, "
          "transit, composition, etc.) and the skeletons' e_N is not, so their levels are not directly comparable; trends are "
          "judged by the \"shape\" (each minus its mean over the bands used). The skeletons' e_d is converted with one "
          "citywide log S–log d slope and is only approximate in the far bands.", "",
          f"(candidate, city) pairs omitted because the elasticity profile could not be computed: {n_failed}."]
    slopes = []
    for c in bd:
        bs = [b for b in range(nb) if _usable(flex[c], b) and ok_b(c, b) and dref.get(c, {}).get(b)]
        if len(bs) >= MIN_BANDS:
            slopes.append(float(np.polyfit(np.log([dref[c][b] for b in bs]), [flex[c][b][1] for b in bs], 1)[0]))
    L += ["", "## 7. Does the size elasticity rise with distance? (flexible model, slope of e_N on ln d)", ""]
    med, ci = _median(slopes), (None, None)
    if slopes:
        rng = np.random.default_rng(BOOT_SEED)
        a = np.asarray(slopes)
        bm = np.median(a[rng.integers(0, len(a), (BOOT_B, len(a)))], axis=1)
        ci = (float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975)))
    dv = [b["val_nll"] - b["m2_val_nll"] for b in bd.values() if b.get("m2_val_nll") is not None]
    L += [f"- Cities {len(slopes)}; median slope {f(med)}, city-bootstrap 95% interval [{f(ci[0])}, {f(ci[1])}] (B = {BOOT_B}); "
          f"cities with slope > 0: {sum(s > 0 for s in slopes)}",
          f"- Median of the flexible model's validation NLL − M2: {f(_median(dv), 4)} nats/visit ({len(dv)} cities)", ""]
    text = "\n".join(L) + "\n"
    try:
        report._scan(text)
    except ValueError as e:
        raise RuntimeError(f"plan 5 report refused: {e}") from e
    Path(out_md).write_text(text)
    last = lambda t: (t[-1]["covered"], t[-1]["n_cities"]) if t else None  # noqa: E731
    return {"n_cities": len(cities_), "n_candidates": len(cids), "accepted": [x["cid"] for x in rf["entries"]],
            "coverage_all": last(cov.get("all")), "coverage_clean": last(cov.get("clean")),
            "slope_N_flex_median": med, "slope_N_flex_ci": ci, "closest": cl}


# ================================================================================================ CLI stages
_W = {}


def _verify_job(args):
    import review
    orig, s = args
    return review.verify_cleaned(orig, s, _W["cities"], _W["datas"], _W["refs"], inner_fits=_W["fits"])


def stage_reforms():
    import multiprocessing as mp
    import pandas as pd
    import ledger
    import library
    search = config.cities_with_role("search")
    pool = json.loads((LOOP / "pool.json").read_text())
    reforms = load_reforms(REFORMS)
    _W.update(cities=search, datas={c: library.city_data(c, None, mode="inner") for c in search},
              refs={c: json.loads((config.RESULTS / f"refs/{c}.json").read_text())["inner"] for c in search},
              fits=pd.read_parquet(LOOP / "inner_fits.parquet"))
    by = {x["cid"]: x for x in pool}
    jobs = [(by[r["parent"]], r["sympy"]) for r in reforms]
    with mp.get_context("fork").Pool(len(jobs)) as P:
        out = P.map(_verify_job, jobs, chunksize=1)
    cache = {(o["cid"], s): v for (o, s), v in zip(jobs, out)}
    acc, res = reform_entries(reforms, pool, lambda o, s: cache[(o["cid"], s)])
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / "reforms.json"
    p.write_text(json.dumps({"entries": acc, "results": res}, ensure_ascii=False, indent=1, default=float))
    ledger.add(f"Plan 5 reforms: {len(res)} verified as agent cleanings (8 search cities, fold 4); accepted "
               f"{[r['cid'] + '=' + r['id'] for r in res if r['accepted']]}, rejected "
               f"{[(r['cid'] + '=' + r['id'], r['reason']) for r in res if not r['accepted']]}; reforms.json sha {ledger.sha(p)}")
    return 0


def stage_rank():
    import ledger
    import loop
    entries = json.loads((ROOT / "reforms.json").read_text())["entries"]
    prepare_copy(LOOP, ROOT / "loop")
    steps = loop.Steps(json.loads(loop.CONFIG.read_text()), root=ROOT / "loop")
    steps.log = ledger.add
    ranked = rank_with_reforms(steps, entries)
    fate = reform_fate(ranked, entries)
    p = ROOT / "fate.json"
    p.write_text(json.dumps(fate, ensure_ascii=False, indent=1))
    g1 = [r["cid"] for r in ranked if r.get("gate1")]
    ledger.add(f"Plan 5 rank: {len(entries)} accepted reforms fitted on the 8 search cities (fold 4) in a copy of the "
               f"loop state and ranked by the frozen rule; gate1 now {g1}; fate "
               f"{[(f['cid'], f['rep'], f['position'], f['n_near'], f['n_flags']) for f in fate]}; fate.json sha "
               f"{ledger.sha(p)}, ranked_r{ROUND}.json sha {ledger.sha(ROOT / 'loop' / f'ranked_r{ROUND}.json')}")
    return 0


def _val_refs(cs):
    return {c: json.loads((config.RESULTS / f"refs/{c}.json").read_text())["models"] for c in cs}


def stage_discovery():
    """The loop's final six + the accepted reforms on the 35 discovery cities (fit_library defaults = the outer
    cycle's), reproduction check of the six against loop/c2_discovery, coverage curves."""
    import os
    import pandas as pd
    import fit_library
    import ledger
    top = json.loads((LOOP / "state.json").read_text())["top8_history"][-1]
    pool = {x["cid"]: x for x in json.loads((ROOT / "loop" / "pool.json").read_text())}
    entries = json.loads((ROOT / "reforms.json").read_text())["entries"]
    cands = [pool[c] for c in top] + [pool[e["cid"]] for e in entries]
    out = ROOT / "discovery"
    fit_library.main(cities(), procs=int(os.environ.get("LOOP_OUTER_PROCS", "24")), cands=cands, out_dir=out)
    cv = pd.read_parquet(out / "cities_val.parquet")
    old = pd.read_parquet(LOOP / "c2_discovery" / "cities_val.parquet")
    rep = check_reproduction(cv, old, list(top))
    why = reproduction_gate(rep, pd.read_csv(out / "cities_val_summary.csv", dtype={"city": str}))
    if why:
        ledger.add(f"Plan 5 discovery: STOP — {'; '.join(why)[:1500]}; cities_val.parquet sha "
                   f"{ledger.sha(out / 'cities_val.parquet')}")
        return 3
    flags = json.loads((ROOT / "loop" / "flags.json").read_text())
    refs = _val_refs(sorted(set(cv["city"])))
    cov = coverage_tables(cv, refs, pool, flags)
    c2 = coverage_tables(old, refs, pool, flags)
    p = out / "coverage.json"
    p.write_text(json.dumps({"all": cov["all"], "clean": cov["clean"], "c2_all": c2["all"], "c2_clean": c2["clean"],
                             "reproduction": rep}, ensure_ascii=False, indent=1))
    last = lambda t: (t[-1]["covered"], t[-1]["n_cities"]) if t else (0, len(refs))  # noqa: E731
    ledger.add(f"Plan 5 discovery: {len(cands)} candidates ({len(top)} loop + {len(entries)} reforms) on "
               f"{len(set(cv['city']))} cities; reproduction of the loop's six max diff {max(rep.values()):.2e}; "
               f"coverage all {[r['cid'] for r in cov['all']]} -> {last(cov['all'])}, clean "
               f"{[r['cid'] for r in cov['clean']]} -> {last(cov['clean'])} (c2: all {last(c2['all'])}, clean "
               f"{last(c2['clean'])}); cities_val.parquet sha {ledger.sha(out / 'cities_val.parquet')}, coverage.json "
               f"sha {ledger.sha(p)}")
    return 0


def _banded_one(city):
    import banded
    try:
        banded.city_banded(city, out_dir=ROOT / "banded")
        return city, None
    except Exception as e:  # noqa: BLE001 — reported in the ledger line; the other cities go on
        return city, repr(e)[:300]


def stage_banded(*only):
    """The distance-band flexible model of every search / discovery city (or the given ones); a city whose output
    exists is skipped; BANDED_PROCS cities at a time (default 4), largest first, one city per worker process."""
    import multiprocessing as mp
    import os
    import fit_library
    import ledger
    out = ROOT / "banded"
    out.mkdir(parents=True, exist_ok=True)
    todo = sorted((c for c in (list(only) or cities()) if not (out / f"{c}.json").exists()),
                  key=lambda c: -fit_library._size(c))
    if not todo:
        return 0
    with mp.get_context("fork").Pool(min(int(os.environ.get("BANDED_PROCS", "4")), len(todo)), maxtasksperchild=1) as P:
        res = list(P.imap_unordered(_banded_one, todo, chunksize=1))
    errs = {c: e for c, e in res if e}
    ledger.add(f"Plan 5 banded: flexible distance-band model fitted in {len(todo) - len(errs)} of {len(todo)} cities"
               f"{'' if not errs else f'; errors {errs}'}; outputs results/multicity/prefreeze/banded/<city>.json "
               f"({len(list(out.glob('*.json')))} present)")
    return 4 if errs else 0


def stage_report():
    import ledger
    p = ROOT / "report.md"
    s = build_report(p)
    ledger.add(f"Plan 5 report: {s['n_candidates']} candidates on {s['n_cities']} cities; accepted reforms {s['accepted']}; "
               f"coverage all {s['coverage_all']}, clean {s['coverage_clean']}; flexible-model slope of e_N on log d "
               f"median {s['slope_N_flex_median']} CI {s['slope_N_flex_ci']}; closest {s['closest']}; report.md sha "
               f"{ledger.sha(p)}")
    return 0


STAGES = {"reforms": stage_reforms, "rank": stage_rank, "discovery": stage_discovery, "banded": stage_banded,
          "report": stage_report}


def main(argv):
    if not argv or argv[0] not in STAGES:
        sys.exit(f"usage: prefreeze.py {{{'|'.join(STAGES)}}} [args]")
    return STAGES[argv[0]](*argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
