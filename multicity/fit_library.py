"""Per-city library fits on the validation set (plan 2 Task 7; plan 4 Task 7).

fit_city(): every candidate refitted on the city's training block (library.refit_city: 5 starts, convergence and
  dispersion recorded) and scored on its validation block with the device-group bootstrap; near_best = within
  max(1 SE, delta_city) of the city's best (scoring.near_best); primary = the simplest near-best candidate (ties:
  lower validation NLL); covered = the primary passes the floor (scoring.passes_floor); share_s0 / share_grav = gain
  shares on the corrected-gravity / classic zero point; trust = False for a candidate that is not fittable in the city
  or whose eta is not finite when a support-mismatch variable of the city (qc.json) is set to 0.
main(): the eligible non-confirmation cities (or the given ones), parallel over (city, candidate) pairs, largest
  cities first, -> <out_dir>/cities_val.parquet (row = city x cid), cities_val_summary.csv (row = city) and
  uncovered.txt (primary above the floor, or no candidate). Candidates: `cands` (agent loop) or the GATE 1 ones of
  candidates_gate1.json; none -> empty outputs and no process pool. Confirmation cities are never fitted here.

Usage: ODSR_ROOT=... ODSR_LINE=M ODSR_CITIES=... python -I fit_library.py [CITY ...]
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import config  # noqa: E402
import sr_data  # noqa: E402
from library import _G as lib_G, _boot_weights, _fit_one, _one_se, candidate, city_data, lib_dir  # noqa: E402
from refit_select import eta  # noqa: E402
import scoring  # noqa: E402

STARTS = 5
SUMMARY_COLUMNS = ["city", "primary", "n_near", "n_1se", "n_finite", "share_s0", "share_grav", "covered", "trust",
                   "grav", "m2", "support_flags"]


def share(val, refs):
    if val is None or not np.isfinite(val):
        return None
    g, m2 = refs["grav_power"]["nll_per_visit"], refs["M2_glm"]["nll_per_visit"]
    return float((g - val) / (g - m2)) if g > m2 else None


def zero_safe(c, data, var):
    """Is eta finite on data's rows when the variable is 0 everywhere?"""
    k = sr_data.VARS.index(var)
    X = data.X.copy()
    X[:, k] = 0.0
    e = eta(c, dataclasses.replace(data, X=X))
    return bool(np.isfinite(e).all())


def _qc_flags(city):
    p = sr_data.ROOT / f"derived/model_inputs/{sr_data.LINE}/{city}/tract/qc.json"
    return list(json.loads(p.read_text()).get("support_mismatch", [])) if p.exists() else []


_CACHE = {}


def _city_data(city, n_orig, mode="val"):
    """The last city's data per worker: jobs are (city, candidate) pairs, so a worker rebuilds only on a city change."""
    if _CACHE.get("key") != (city, n_orig, mode):
        _CACHE.clear()
        _CACHE.update(key=(city, n_orig, mode), D=city_data(city, n_orig, mode))
    return _CACHE["D"]


def _eligible(cities, confirm, allow_confirm):
    """The cities to fit: confirmation cities only with allow_confirm (after the freeze, plan 6)."""
    return [c for c in cities if allow_confirm or c not in confirm]


def _pair(job):
    """Refit of one candidate in one city: (city, k, row with trust, bootstrap column, error)."""
    city, k, cands, B, starts, seed, n_orig, flags = job[:8]
    mode = job[8] if len(job) > 8 else "val"
    try:
        D = _city_data(city, n_orig, mode)
        lib_G.update(data={city: D}, cands=cands, starts=starts, seed=seed, W=_boot_weights(B), se=True)
        _, _, row, col = _fit_one((city, k))
        if row["finite"]:
            x = cands[k]
            c = dataclasses.replace(candidate({**x, "consts_refit": None}), consts=np.asarray(json.loads(row["consts"]), float))
            row["trust"] = all(zero_safe(c, D["va"], v) for v in flags)
        else:
            row["trust"] = False
        return city, k, row, col, None
    except Exception as e:  # noqa: BLE001 — reported in the city's summary, not fatal
        return city, k, None, None, repr(e)


def _finish(city, rows, boot, refs, flags, cx=None):
    """1-SE and near-best flags, gain shares, primary skeleton and the city summary from its candidates' rows.
    cx: {cid: complexity} (default: the rows' own "complexity")."""
    rows = _one_se(rows, boot)
    for r in rows:
        r["city"] = city
        if cx is not None:
            r["complexity"] = int(cx[r["cid"]])
        v = r["val_nll"] if r["finite"] else None
        r["share_grav"] = scoring.share(v, refs, "grav_power")
        r["share_s0"] = scoring.share(v, refs, "S0")
    near = scoring.near_best(pd.DataFrame(rows, columns=["cid", "city", "val_nll", "finite", "se_vs_best", "within_1se"]),
                             {city: refs}) if rows else set()
    for r in rows:
        r["near_best"] = (r["cid"], city) in near
    cand = [r for r in rows if r["near_best"]]
    prim = min(cand, key=lambda r: (r["complexity"], r["val_nll"], r["cid"])) if cand else None
    for r in rows:
        r["primary"] = prim is not None and r is prim
    summ = {"city": city, "primary": prim["cid"] if prim else None, "n_near": len(cand),
            "n_1se": int(sum(r["within_1se"] for r in rows)), "n_finite": int(sum(r["finite"] for r in rows)),
            "share_s0": prim["share_s0"] if prim else None, "share_grav": prim["share_grav"] if prim else None,
            "covered": bool(prim is not None and scoring.passes_floor(prim["val_nll"], refs)),
            "trust": bool(prim is not None and prim["trust"]),
            "grav": refs["grav_power"]["nll_per_visit"], "m2": refs["M2_glm"]["nll_per_visit"],
            "support_flags": flags}
    return rows, summ


def _refs(city, mode="val"):
    """The city's reference models of the scoring block: refs/<city>.json["models"] (validation) or
    refs_test/<city>.json[mode] (plan 6 test blocks)."""
    if mode == "val":
        return json.loads((config.RESULTS / f"refs/{city}.json").read_text())["models"]
    return json.loads((config.RESULTS / f"refs_test/{city}.json").read_text())[mode]


def fit_city(city, cands, B=200, starts=STARTS, seed=0, refs=None, qc=None, n_orig=None):
    refs = _refs(city) if refs is None else refs
    flags = _qc_flags(city) if qc is None else list(qc.get("support_mismatch", []))
    res = [_pair((city, k, cands, B, starts, seed, n_orig, flags)) for k in range(len(cands))]
    err = [e for *_, e in res if e]
    if err:
        raise RuntimeError(err[0])
    rows = [r for _, _, r, _, _ in res]
    boot = np.column_stack([c for _, _, _, c, _ in res]) if res else np.zeros((B, 0))
    return _finish(city, rows, boot, refs, flags, {x["cid"]: x["complexity"] for x in cands})


def _size(city):
    p = sr_data.ROOT / f"derived/model_inputs/{sr_data.LINE}/{city}/tract/qc.json"
    return json.loads(p.read_text()).get("n_origins", 0) if p.exists() else 0


def main(cities=None, procs=8, confirm=None, B=200, starts=STARTS, seed=0, n_orig=None, cands=None, out_dir=None,
         mode="val", allow_confirm=False):
    """mode: the scoring block (library.city_data); "test_dest" / "test_time" and allow_confirm need the freeze."""
    import p7seal
    p7seal.guard_line(p7seal.current_line())         # plan 7: sealed data lines need the signed registration
    if mode != "val" or allow_confirm:
        import frozen
        frozen.require()
    out_dir = Path(out_dir) if out_dir else config.RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    if cands is None:
        lib = lib_dir()
        pool = {x["cid"]: x for x in json.loads((lib / "pool.json").read_text())}
        g1 = json.loads((lib / "candidates_gate1.json").read_text())
        cands = [pool[c["cid"]] for c in g1["candidates"] if c.get("gate1")]
    confirm = set(config.CONFIRM) if confirm is None else set(confirm)
    try:
        cc = pd.read_csv(config.CITIES_CSV, dtype=str)
        if "role" in cc.columns:
            confirm |= set(cc[cc["role"] == "confirm"]["city"])
    except OSError:
        pass
    if cities is None:
        q = pd.read_csv(config.RESULTS / "qc/cities_qc.csv", dtype={"city": str})
        cities = list(q[q["eligible"]]["city"])
    cities = _eligible(cities, confirm, allow_confirm)
    if not cands:                                   # no candidate -> no Pool(0), empty outputs
        from library import FIT_COLUMNS
        pd.DataFrame(columns=FIT_COLUMNS + ["consts_se", "consts_cov", "trust", "near_best", "primary", "share_s0",
                                            "share_grav", "complexity"]).to_parquet(out_dir / "cities_val.parquet", index=False)
        out = pd.DataFrame(columns=SUMMARY_COLUMNS)
        out.to_csv(out_dir / "cities_val_summary.csv", index=False)
        (out_dir / "uncovered.txt").write_text("".join(c + "\n" for c in cities))
        return out
    flags = {c: _qc_flags(c) for c in cities}
    order = sorted(cities, key=lambda c: -_size(c))          # largest cities first: no long tail
    jobs = [(c, k, cands, B, starts, seed, n_orig, flags[c], mode) for c in order for k in range(len(cands))]
    if procs == 1 or len(jobs) == 1:
        res = list(map(_pair, jobs))
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(procs, len(jobs))) as P:
            res = list(P.imap_unordered(_pair, jobs, chunksize=1))
    rows_all, summs = [], []
    for c in cities:
        got = sorted((k, row, col, err) for cc_, k, row, col, err in res if cc_ == c)
        errs = [e for *_, e in got if e]
        if errs:
            summs.append({"city": c, "error": errs[0], "covered": False})
            continue
        rows, summ = _finish(c, [g[1] for g in got], np.column_stack([g[2] for g in got]), _refs(c, mode), flags[c],
                             {x["cid"]: x["complexity"] for x in cands})
        rows_all += [dict(r, city=c) for r in rows]
        summs.append(summ)
        print(c, json.dumps({k: summ[k] for k in ("primary", "n_near", "share_s0", "covered", "trust")}), flush=True)
    pd.DataFrame(rows_all).to_parquet(out_dir / "cities_val.parquet", index=False)
    out = pd.DataFrame(summs)
    out.to_csv(out_dir / "cities_val_summary.csv", index=False)
    (out_dir / "uncovered.txt").write_text("".join(s["city"] + "\n" for s in summs if not s.get("covered")))
    return out


if __name__ == "__main__":
    import os
    import ledger  # noqa: E402
    out = main(sys.argv[1:] or None, procs=int(os.environ.get("FIT_PROCS", "8")))
    ledger.add(f"Task 7: library fits on {len(out)} non-confirmation cities: covered {int(out['covered'].sum())}, "
               f"uncovered {len(out) - int(out['covered'].sum())}, trust False "
               f"{int((~out['trust'].fillna(False).astype(bool)).sum()) if 'trust' in out else 'n/a'}; "
               f"cities_val.parquet sha {ledger.sha(config.RESULTS / 'cities_val.parquet')}, summary sha "
               f"{ledger.sha(config.RESULTS / 'cities_val_summary.csv')}")
    print(out.to_string())
