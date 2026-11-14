"""Candidate pool and per-search-city refits (plan 2 Task 3; design §5.4.1).

pool(): the union of the pooled and single-city selections (results/sr/M/select_*.json), deduplicated by structure
(the sympy form of f, g, h with literals as symbols — the key refit_select.select uses, template ignored), in
first-seen order with ids c0001, c0002, ...
to_single(): a city-template candidate as a one-category candidate (p1, p2 = the mean city parameters), the start
of every per-city refit.
refit_city(): L-BFGS refit on one city's train x search block from the given constants plus starts - 1 log-normal
perturbations of them (sd 0.3; round-1 ZCTA lesson: a refit can stall at its start), keeping the lowest training
loss; a candidate whose eta is not finite at any start (log / negative power of a variable with zeros in that city)
is "not fittable": finite = False, val_nll = None, and the city's 1-SE comparison ignores it.
search_city_fits(): every pool candidate on every search city (refits on LIB_N_ORIG = 300 sampled origins per city,
validation on every origin; parallel over (city, candidate)), with the device-group bootstrap (B = 200, seed
20261101) of the paired validation difference to the city's best candidate -> se_vs_best, within_1se; the bootstrap
loss matrices are kept per city for the equivalence step (Task 6).

Output: results/multicity/library/{pool.json, search_city_fits.parquet, boot_<city>.npy}
Usage: ODSR_ROOT=... ODSR_LINE=M ODSR_CITIES=... python -I library.py SELECT_JSON [SELECT_JSON ...]
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import sr_data  # noqa: E402
from refit_select import PENALTY, _device_group_y, boot_nll, eta, loss, parse_equation, refit  # noqa: E402

B_BOOT, BOOT_SEED, N_GROUPS = 200, 20261101, 20
LIB_N_ORIG = int(__import__("os").environ.get("LIB_N_ORIG", "300"))   # origins per city for the refits
PERTURB_SD = 0.3
CONV_TOL = 1e-4          # a start "converged" when its training loss is within this of the best start (nats / visit)
FIT_COLUMNS = ["cid", "city", "train_nll", "val_nll", "finite", "consts", "se_vs_best", "within_1se",
               "n_converged", "loss_sd_starts", "consts_sd"]


def lib_dir():
    d = sr_data.ROOT / "results/multicity/library"
    d.mkdir(parents=True, exist_ok=True)
    return d


def structure_key(c):
    return tuple(sorted((k, str(v)) for k, v in c.comps.items()))


def pool(select_paths):
    """Union of selection files; one entry per structure, first-seen order, cid c0001.. ; writes pool.json."""
    cands = {}
    for p in map(Path, select_paths):
        s = json.loads(p.read_text())
        variant = s.get("variant", "A")
        for x in s["candidates"]:
            c = parse_equation(x["equation"], x.get("template"), variant, int(x["complexity"]))
            key = structure_key(c)
            if key in cands:
                if p.name not in cands[key]["source"]:
                    cands[key]["source"].append(p.name)
                continue
            consts = x.get("consts_refit")
            cands[key] = {"cid": None, "equation": x["equation"], "template": x.get("template"), "variant": variant,
                          "complexity": int(x["complexity"]),
                          "consts_refit": [float(v) for v in (consts if consts is not None else c.consts)],
                          "source": [p.name]}
    out = list(cands.values())
    for k, x in enumerate(out):
        x["cid"] = f"c{k + 1:04d}"
    (lib_dir() / "pool.json").write_text(json.dumps(out, indent=1))
    return out


def to_single(c):
    """One-category form of a candidate: p1, p2 = means of the city parameters; literals unchanged."""
    if c.n_cat == 1 and c.template is None:
        return c
    lit = np.asarray(c.consts[:c.n_lit], float)
    if "g" in c.comps:
        p1 = c.consts[c.n_lit:c.n_lit + c.n_cat]
        p2 = c.consts[c.n_lit + c.n_cat:c.n_lit + 2 * c.n_cat]
        consts = np.r_[lit, float(np.mean(p1)), float(np.mean(p2))]
    else:
        consts = lit
    return dataclasses.replace(c, template=None, n_cat=1, consts=consts)


def candidate(x):
    """Pool entry -> single-category Candidate starting from its refitted constants."""
    c = parse_equation(x["equation"], x.get("template"), x.get("variant", "A"), int(x["complexity"]))
    cr = x.get("consts_refit")
    if cr is not None and len(cr) == len(c.consts):
        c = dataclasses.replace(c, consts=np.asarray(cr, float))
    return to_single(c)


def _multistart(c, tr, starts, seed):
    """(best loss, best candidate, n_converged, loss_sd, consts_sd) over the starts, or None if no start is finite.
    n_converged: starts whose refitted loss is within CONV_TOL of the best; loss_sd: SD of the finite start losses;
    consts_sd: per-constant SD over the converged starts (0 with one) — a flat direction shows up here (plan 4)."""
    rng = np.random.default_rng(seed)
    done = []
    for k in range(starts):
        s0 = c.consts if k == 0 else c.consts * np.exp(rng.normal(0.0, PERTURB_SD, len(c.consts)))
        cs = dataclasses.replace(c, consts=np.asarray(s0, float))
        if loss(cs, tr) >= PENALTY:      # eta not finite at this start: nothing to optimise from here
            continue
        r = refit(cs, tr)
        lt = loss(r, tr)
        if lt < PENALTY:
            done.append((float(lt), r))
    if not done:
        return None
    lt, r = min(done, key=lambda t: t[0])
    conv = [x for x in done if x[0] - lt <= CONV_TOL]
    cs = np.array([x[1].consts for x in conv], float)
    return (lt, r, len(conv), float(np.std([x[0] for x in done])),
            [float(v) for v in (cs.std(0) if len(conv) > 1 else np.zeros(len(r.consts)))])


def refit_city(c, city, starts=5, seed=0, tr=None, va=None, se=False, Yg=None):
    """Multi-start refit of a (single-category) candidate on one city; validation loss on val x validation block
    (or on the given training / scoring data). Plan 4: 5 starts by default, convergence and dispersion recorded;
    se=True adds consts_se / consts_cov of the best start on the training rows (Yg: the training rows' device-group
    counts -> cluster-robust sandwich; refit_select.const_cov)."""
    tr = sr_data.build([city], "tract", "train", "search", None, 0) if tr is None else tr
    va = sr_data.build([city], "tract", "val", "val", None, 0) if va is None else va
    best = _multistart(to_single(c), tr, starts, seed)
    if best is None:
        return {"consts": None, "train_nll": None, "val_nll": None, "finite": False, "candidate": None,
                "n_starts": starts, "n_converged": 0, "loss_sd_starts": None, "consts_sd": None}
    lt, r, nconv, lsd, csd = best
    lv = loss(r, va)
    out = {"consts": [float(v) for v in r.consts], "train_nll": lt, "val_nll": float(lv) if lv < PENALTY else None,
           "finite": bool(lv < PENALTY), "candidate": r, "n_starts": starts, "n_converged": nconv,
           "loss_sd_starts": lsd, "consts_sd": csd}
    if se:
        from refit_select import const_cov
        cov = const_cov(r, tr, Yg)
        d = np.diag(cov)
        out["consts_se"] = [float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else None for v in d]
        out["consts_cov"] = [[float(v) if np.isfinite(v) else None for v in row] for row in cov]
    return out


def _boot_weights(B):
    return np.random.default_rng(BOOT_SEED).multinomial(N_GROUPS, np.full(N_GROUPS, 1.0 / N_GROUPS), size=B).astype(float)


TEST_BLOCKS = {"test_dest": "fold:0", "test_time": "val"}      # plan 6: test period x destination fold 0 / fold 1


def city_data(city, n_orig=None, mode="val"):
    """Training block on n_orig sampled origins (default LIB_N_ORIG = 300, seed 0, the selection's sampling: a
    full-origin refit takes > 25 min per start in New York), scoring block on every origin, device-group counts.
    mode "val": refit on train x search (folds 2-4), score on val x validation block (outer loop, plans 1-2).
    mode "inner": training period only — refit on folds 2, 3 (search23), score on fold 4 (plan 4 inner loop; the
    validation period is never touched).
    mode "test_dest" / "test_time" (plan 6, after the freeze only): refit as "val", score on the test period x
    destination fold 0 (destinations and period never seen) / x fold 1 (the validation destinations)."""
    n_orig = LIB_N_ORIG if n_orig is None else n_orig
    if mode == "inner":
        tr = sr_data.build([city], "tract", "train", "search23", n_orig, 0)
        va = sr_data.build([city], "tract", "train", "fold:4", None, 0)
        return {"tr": tr, "va": va, "Yg": _device_group_y(va, [city], "tract", "train")}
    if mode in TEST_BLOCKS:                     # plan 6: the test period, only after the freeze
        import frozen
        frozen.require()
        tr = sr_data.build([city], "tract", "train", "search", n_orig, 0)
        va = sr_data.build([city], "tract", "test", TEST_BLOCKS[mode], None, 0)
        return {"tr": tr, "va": va, "Yg": _device_group_y(va, [city], "tract", "test")}
    if mode != "val":
        raise ValueError(f"city_data mode {mode!r}")
    tr = sr_data.build([city], "tract", "train", "search", n_orig, 0)
    va = sr_data.build([city], "tract", "val", "val", None, 0)
    return {"tr": tr, "va": va, "Yg": _device_group_y(va, [city], "tract", "val")}


_G = {}      # data shared with forked workers: {"data": {city: city_data}, "cands": [...], "starts", "seed", "W"}


def _fit_one(job):
    """One (city, candidate) refit and its bootstrap column."""
    city, k = job
    D, x = _G["data"][city], _G["cands"][k]
    if _G.get("se") and "Yg_tr" not in D:            # training-row device-group counts, once per city and worker
        D["Yg_tr"] = _device_group_y(D["tr"], [city], "tract", "train")
    r = refit_city(candidate(x), city, _G["starts"], _G["seed"], tr=D["tr"], va=D["va"], se=bool(_G.get("se")),
                   Yg=D.get("Yg_tr") if _G.get("se") else None)
    cand = r.pop("candidate")
    col = boot_nll(eta(cand, D["va"]), D["va"], D["Yg"], _G["W"]) if r["finite"] else np.full(len(_G["W"]), PENALTY)
    row = {"cid": x["cid"], "city": city, "train_nll": r["train_nll"], "val_nll": r["val_nll"],
           "finite": r["finite"], "consts": json.dumps(r["consts"]), "n_converged": r["n_converged"],
           "loss_sd_starts": r["loss_sd_starts"], "consts_sd": json.dumps(r["consts_sd"])}
    if _G.get("se"):
        row["consts_se"], row["consts_cov"] = json.dumps(r.get("consts_se")), json.dumps(r.get("consts_cov"))
    print(f"{city} {x['cid']} train {r['train_nll']} val {r['val_nll']}", flush=True)
    return city, k, row, col


def _one_se(rows, boot):
    """se_vs_best and within_1se of one city's rows from its (B, n_cand) bootstrap matrix."""
    vals = np.array([PENALTY if r["val_nll"] is None else r["val_nll"] for r in rows])
    ib = int(np.argmin(vals))
    diff = boot - boot[:, [ib]]
    with np.errstate(all="ignore"):   # a replicate whose resampled device groups carry no visit is NaN: left out
        sd = np.nanstd(np.where(np.isfinite(diff), diff, np.nan), 0)
    sd = np.where(np.isfinite(sd), sd, 0.0)
    for k, r in enumerate(rows):
        r["se_vs_best"] = float(sd[k]) if r["finite"] else None
        r["within_1se"] = bool(r["finite"] and vals[k] - vals[ib] <= sd[k])
    return rows


def fit_cities(cands, cities, procs=8, starts=3, seed=0, B=B_BOOT, n_orig=None, mode="val", data=None):
    """{city: (rows, boot)} for every (city, candidate), parallel over the pairs (fork pool; the data are built once
    in the parent and shared copy-on-write). mode: city_data mode (plan 4 inner loop: "inner"); data: prebuilt."""
    _G.update(data=data if data is not None else {c: city_data(c, n_orig, mode) for c in cities}, cands=list(cands),
              starts=starts, seed=seed, W=_boot_weights(B))
    jobs = [(c, k) for c in cities for k in range(len(cands))]
    if procs == 1 or len(jobs) == 1:
        res = list(map(_fit_one, jobs))
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(procs, len(jobs))) as P:
            res = list(P.imap_unordered(_fit_one, jobs, chunksize=1))
    out = {}
    for c in cities:
        got = sorted((k, row, col) for cc, k, row, col in res if cc == c)
        rows = [row for _, row, _ in got]
        boot = np.column_stack([col for _, _, col in got]) if got else np.zeros((B, 0))
        out[c] = (_one_se(rows, boot), boot)
    _G.clear()
    return out


def _city_job(args):
    """Sequential fits of one city (Task 7's per-city jobs): (city, rows with 1-SE flags, bootstrap matrix)."""
    city, cands, starts, seed, B = args[:5]
    n_orig = args[5] if len(args) > 5 else None
    rows, boot = fit_cities(cands, [city], procs=1, starts=starts, seed=seed, B=B, n_orig=n_orig)[city]
    return city, rows, boot


def search_city_fits(pool, cities, procs=8, starts=3, seed=0, B=B_BOOT, n_orig=None):
    """(table of (cid, city) fits, {city: (B, n_cand) bootstrap loss matrix}); both written under library/."""
    res = fit_cities(pool, cities, procs, starts, seed, B, n_orig)
    rows, boot = [], {}
    for city in cities:
        r, bm = res[city]
        rows += r
        boot[city] = bm
        np.save(lib_dir() / f"boot_{city}.npy", bm)
    df = pd.DataFrame(rows, columns=FIT_COLUMNS)
    df.to_parquet(lib_dir() / "search_city_fits.parquet", index=False)
    return df, boot


if __name__ == "__main__":
    import os
    from config import CITIES_CSV  # noqa: E402
    import ledger  # noqa: E402
    cities = pd.read_csv(CITIES_CSV, dtype=str)
    search = list(cities[cities["role"] == "search"]["city"])
    p = pool(sys.argv[1:])
    df, _ = search_city_fits(p, search, procs=int(os.environ.get("LIB_PROCS", "8")))
    n1 = df[df.within_1se].groupby("cid").size()
    ledger.add(f"Task 3: library pool {len(p)} candidates from {len(sys.argv[1:])} selections (pool.json sha "
               f"{ledger.sha(lib_dir() / 'pool.json')}); search_city_fits.parquet sha "
               f"{ledger.sha(lib_dir() / 'search_city_fits.parquet')}: {int(df.finite.sum())}/{len(df)} fits finite, "
               f"candidates in >= 4 city 1-SE sets: {int((n1 >= 4).sum())}")
    print(len(p), "candidates;", int(df.finite.sum()), "of", len(df), "fits finite")
