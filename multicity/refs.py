"""Validation-set references per city: power-law and exponential gravity, extended radiation (alpha
fitted), M2 and M3 GLM — fitted on train x search block, scored on val x validation block with the SR per-visit NLL
(scripts/sr/val_refs.py). M2 / M3 use the OSM stop count (n_transit_osm) for log_transit. Confirmation cities are
skipped (sealed until the freeze). gain_m2 = NLL(grav_power) - NLL(M2) is the denominator of the gain share.
S0 (plan 4 Task 5): corrected gravity log(N_j + c) - beta log(d + c0), refitted per
city with library.refit_city (5 starts; train x search block, val x validation block) — the zero point of share_S0
(reporting only; the tolerance delta_city stays on grav_power).

Output: results/multicity/refs/<city>.json
Usage: ODSR_ROOT=... ODSR_LINE=M ODSR_CITIES=... python -I refs.py CITY [CITY ...]
       ... python -I refs.py --s0-only [CITY ...]   (adds models.S0 to existing refs/<city>.json in place; no CITY =
       every existing file; confirmation cities are skipped before any data is read; S0_PROCS parallel cities)
       ... python -I refs.py --inner [CITY ...]     (plan 4 inner-loop references -> refs/<city>.json["inner"]; search
       cities only, default all of them)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from config import CITIES_CSV, CONFIRM  # noqa: E402
import sr_data  # noqa: E402
from baselines_l import ext_radiation_eta, score_eta  # noqa: E402
from sr_data import dest_mask, folds, load  # noqa: E402
from val_refs import refs as glm_refs  # noqa: E402

TRANSIT_COL = "n_transit_osm"
ALPHA_BOUNDS = (0.01, 5.0)
S0_EQUATION = "f = log(#6 + 10.0) - 1.5 * log(#1 + 1.0); p1 = []; p2 = []"
S0_STARTS = 5


def s0_val(city, tr=None, va=None):
    """Corrected gravity S0 refitted on the city's train x search block (every origin), validation NLL (or on the
    given training / scoring data)."""
    import library
    from refit_select import parse_equation
    r = library.refit_city(parse_equation(S0_EQUATION, None, "A", 9), city, starts=S0_STARTS, tr=tr, va=va)
    if not r["finite"]:
        return {"error": "S0 not fittable", "consts": r["consts"]}
    return {"nll_per_visit": float(r["val_nll"]), "train_nll": float(r["train_nll"]), "consts": r["consts"]}


def ext_rad_val(city, train_block="search", eval_split="val", eval_block="val"):
    d = load(city, "tract")
    D, T, orig, dest, N = d["D"], d["T"], d["orig"], d["dest"], d["N"]
    f = folds(d)
    ds, dv = dest_mask(d, f, train_block), dest_mask(d, f, eval_block)
    pop1 = d["u"]["population"].to_numpy(float) + 1.0
    S = np.load(d["base"] / "S.npy").astype(np.float64)[orig]
    eye = np.eye(N, dtype=bool)
    av_tr, av_va = (ds[None, :] & ~eye)[orig], (dv[None, :] & ~eye)[orig]
    eta = lambda a: ext_radiation_eta(pop1[orig], pop1, S, a)  # noqa: E731
    r = minimize_scalar(lambda a: score_eta(eta(a), T["train"][orig], av_tr)["nll_per_visit"],
                        bounds=ALPHA_BOUNDS, method="bounded")
    return {**score_eta(eta(r.x), T[eval_split][orig], av_va), "alpha": float(r.x)}


def run_city(city):
    if city in CONFIRM and os.environ.get("ODSR_UNSEAL") != "1":   # the seal does not depend on any file
        raise RuntimeError(f"{city} is a sealed confirmation city: references are computed only after the freeze "
                           "(ODSR_UNSEAL=1 with a ledger entry)")
    if city in CONFIRM:                                           # plan 6: and only after the freeze
        import frozen
        frozen.require()
    out = {"city": city, "models": {}}
    try:
        for k, v in glm_refs(city, "tract", transit_col=TRANSIT_COL).items():
            out["models"][k] = v
    except Exception as e:  # noqa: BLE001 — a city whose GLM cannot be fitted is reported, not fatal
        for k in ("grav_power", "grav_exp", "M2_glm", "M3_glm"):
            out["models"].setdefault(k, {"error": repr(e)})
    try:
        out["models"]["ext_rad"] = ext_rad_val(city)
    except Exception as e:  # noqa: BLE001
        out["models"]["ext_rad"] = {"error": repr(e)}
    try:
        out["models"]["S0"] = s0_val(city)
    except Exception as e:  # noqa: BLE001
        out["models"]["S0"] = {"error": repr(e)}
    g, m2 = out["models"]["grav_power"], out["models"]["M2_glm"]
    out["gain_m2"] = (g["nll_per_visit"] - m2["nll_per_visit"]) if "nll_per_visit" in g and "nll_per_visit" in m2 else None
    d = config.results_dir() / "refs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{city}.json").write_text(json.dumps(out, indent=1))
    return out


TEST_BLOCKS = {"test_dest": "fold:0", "test_time": "val"}      # plan 6 (library.TEST_BLOCKS)


def test_refs(city):
    """Plan 6, after the freeze only: the reference models fitted on train x search (as for validation) and scored on
    the test period x destination fold 0 ("test_dest") and x fold 1 ("test_time"): grav_power, grav_exp, M2_glm
    (M3_glm when val_refs gives it), ext_rad, S0 -> refs_test/<city>.json under those keys, written atomically; the
    pre-freeze refs/<city>.json is never touched."""
    import frozen
    frozen.require()
    out = {}
    for key, block in TEST_BLOCKS.items():
        m = {}
        try:
            m.update(glm_refs(city, "tract", transit_col=TRANSIT_COL, train_block="search", eval_split="test",
                              eval_block=block))
        except Exception as e:  # noqa: BLE001 — reported, not fatal
            for k in ("grav_power", "grav_exp", "M2_glm"):
                m.setdefault(k, {"error": repr(e)})
        try:
            m["ext_rad"] = ext_rad_val(city, "search", "test", block)
        except Exception as e:  # noqa: BLE001
            m["ext_rad"] = {"error": repr(e)}
        try:
            m["S0"] = s0_val(city, tr=sr_data.build([city], "tract", "train", "search", None, 0),
                             va=sr_data.build([city], "tract", "test", block, None, 0))
        except Exception as e:  # noqa: BLE001
            m["S0"] = {"error": repr(e)}
        out[key] = m
    p = config.results_dir() / "refs_test" / f"{city}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps({"city": city, **out}, indent=1))
    os.replace(tmp, p)
    return out


INNER = {"train": "search23", "split": "train", "eval": "fold:4"}


def inner_refs(city):
    """Plan 4 inner-loop references (Task 6): the inner loop scores candidates on fold 4 of the training period
    after refits on folds 2, 3, so its floor (grav_power, grav_exp, ext_rad), tolerance (grav_power - M2_glm) and
    share zero point (S0) are fitted and scored on the same blocks; written to refs/<city>.json["inner"]."""
    if city in CONFIRM and os.environ.get("ODSR_UNSEAL") != "1":
        raise RuntimeError(f"{city} is a sealed confirmation city")
    if city in CONFIRM:
        import frozen
        frozen.require()
    out = {}
    try:
        out.update(glm_refs(city, "tract", transit_col=TRANSIT_COL, train_block=INNER["train"],
                            eval_split=INNER["split"], eval_block=INNER["eval"],
                            models=("grav_power", "grav_exp", "M2_glm")))
    except Exception as e:  # noqa: BLE001
        for k in ("grav_power", "grav_exp", "M2_glm"):
            out.setdefault(k, {"error": repr(e)})
    try:
        out["ext_rad"] = ext_rad_val(city, INNER["train"], INNER["split"], INNER["eval"])
    except Exception as e:  # noqa: BLE001
        out["ext_rad"] = {"error": repr(e)}
    try:
        tr = sr_data.build([city], "tract", "train", INNER["train"], None, 0)
        va = sr_data.build([city], "tract", INNER["split"], INNER["eval"], None, 0)
        out["S0"] = s0_val(city, tr=tr, va=va)
    except Exception as e:  # noqa: BLE001
        out["S0"] = {"error": repr(e)}
    out["blocks"] = dict(INNER)
    d = config.results_dir() / "refs"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{city}.json"
    r = json.loads(p.read_text()) if p.exists() else {"city": city, "models": {}}
    r["inner"] = out
    p.write_text(json.dumps(r, indent=1))
    return out


def _s0_update(city):
    p = config.results_dir() / f"refs/{city}.json"
    r = json.loads(p.read_text())
    try:
        r["models"]["S0"] = s0_val(city)
    except Exception as e:  # noqa: BLE001 — reported in the file, not fatal for the other cities
        r["models"]["S0"] = {"error": repr(e)}
    p.write_text(json.dumps(r, indent=1))
    return city, r["models"]["S0"]


def s0_only(cities, roles):
    d = config.results_dir() / "refs"
    cities = list(cities) or sorted(p.stem for p in d.glob("*.json"))
    todo = []
    for c in cities:
        if roles.get(c) == "confirm" or c in CONFIRM:
            print(c, "skipped (confirmation city, sealed)", flush=True)
        elif not (d / f"{c}.json").exists():
            print(c, "skipped (no refs file)", flush=True)
        else:
            todo.append(c)
    procs = int(os.environ.get("S0_PROCS", "1"))
    out = {}

    def report(res):
        for c, s0 in res:
            out[c] = s0
            print(c, "S0", json.dumps(s0), flush=True)
    if procs == 1 or len(todo) <= 1:
        report(map(_s0_update, todo))
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(procs, len(todo))) as P:
            report(P.imap_unordered(_s0_update, todo, chunksize=1))
    return out


def _line():
    return config.LINE


def _roles():
    cfg = pd.read_csv(CITIES_CSV, dtype=str)       # config.CITIES_CSV of the line (module name kept: tests patch it)
    if "role" not in cfg.columns:   # the roles are written by the tiers stage; without them nothing is sealed
        raise RuntimeError(f"{CITIES_CSV} has no role column: run the tiers stage first (confirmation cities stay sealed)")
    return cfg.set_index("city")["role"]


def _complete(r: dict) -> bool:
    return all(isinstance(v, dict) and "nll_per_visit" in v for v in r.get("models", {}).values())


def summary_line(results: dict) -> str:
    """Counts only (plan 7): on a sealed line no score may leave the run before the registration is signed off."""
    bad = sorted(c for c, r in results.items() if not _complete(r))
    tail = (": " + ", ".join(bad)) if bad else ""
    return f"references ({_line()}): {len(results) - len(bad)} complete, {len(bad)} with errors{tail}"


def main(cities):
    import p7seal
    roles = _roles()
    sealed = _line() in p7seal.SEALED_LINES and not p7seal.is_signed()
    if sealed and cities and cities[0] in ("--s0-only", "--inner"):     # these print scores per city
        raise RuntimeError(f"{cities[0]} prints scores: not on the sealed line {_line()} before sign-off")
    if cities and cities[0] == "--s0-only":
        return s0_only(cities[1:], roles)
    if cities and cities[0] == "--inner":           # plan 4 inner-loop references: search cities only
        todo = list(cities[1:]) or [c for c, r in roles.items() if r == "search"]
        for c in todo:
            if roles.get(c) != "search":
                print(c, "skipped (inner references are for search cities only)", flush=True)
                continue
            r = inner_refs(c)
            print(c, "inner", json.dumps({k: round(v["nll_per_visit"], 5) if "nll_per_visit" in v else v.get("error")
                                          for k, v in r.items() if k != "blocks"}), flush=True)
        return None
    done = {}
    for c in cities:
        if roles.get(c) == "confirm":
            print(c, "skipped (confirmation city, sealed)", flush=True)
            continue
        r = run_city(c)
        done[c] = r
        if sealed:
            print(c, "complete" if _complete(r) else "with errors", flush=True)
            continue
        print(c, json.dumps({k: round(v["nll_per_visit"], 4) if "nll_per_visit" in v else v.get("error", "?")[:60]
                             for k, v in r["models"].items()}), "gain_m2", r["gain_m2"], flush=True)
    if sealed:
        print(summary_line(done), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
