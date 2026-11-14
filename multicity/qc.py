"""Per-city quality gate: variable support per destination block (needed because a
variable with zeros only outside the search block makes log / negative-power candidates unscorable), representativeness
of the device panel, eligibility thresholds.

Output: derived/model_inputs/M/<city>/tract/qc.json ; results/multicity/qc/cities_qc.csv (one row per city)
Usage: ODSR_ROOT=... ODSR_LINE=M ODSR_CITIES=... python -I qc.py CITY [CITY ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIRM, MIN_ORIGIN_TRACTS, MIN_TRAIN_TRIPS, RESULTS, SEARCH  # noqa: E402
import sr_data  # noqa: E402

BLOCKS = ("search", "val", "test")
MAX_COVERAGE_OUT = 0.3
MIN_INCOME_RATIO = 0.5
LOG_OR_POWER_VARS = None   # every variable is reported; the candidate check happens at selection time (plan 2)


def _support(city):
    out = {}
    for block in BLOCKS:
        s = sr_data.build([city], "tract", "train", block, None, 0)
        X = s.X.astype(float)
        for k, v in enumerate(sr_data.VARS):
            col = X[:, k]
            fin = np.isfinite(col)
            out.setdefault(v, {})[block] = {"min": float(col[fin].min()) if fin.any() else float("nan"),
                                            "zeros": int((col == 0).sum()), "rows": int(len(col)),
                                            "nonfinite": int((~fin).sum())}
    mismatch = [v for v in sr_data.VARS
                if out[v]["search"]["zeros"] == 0 and (out[v]["val"]["zeros"] > 0 or out[v]["test"]["zeros"] > 0)]
    nonfinite = [v for v in sr_data.VARS if any(out[v][b]["nonfinite"] > 0 for b in BLOCKS)]
    return out, mismatch, nonfinite


def _representativeness(u):
    ok = u["population"] > 0
    x, y = np.log1p(u.loc[ok, "population"].to_numpy(float)), np.log1p(u.loc[ok, "home_devices"].to_numpy(float))
    if len(x) > 2 and x.std() > 0:
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    else:
        r2 = float("nan")
    ratio = {}
    for g in (1, 2, 3):
        m = u["ses"] == g
        pop = u.loc[m, "population"].sum()
        ratio[str(g)] = float(u.loc[m, "home_devices"].sum() / pop) if pop > 0 else float("nan")
    vals = [v for v in ratio.values() if np.isfinite(v) and v > 0]
    imbalance = bool(vals and min(vals) / max(vals) < MIN_INCOME_RATIO)
    return {"r2": r2, "coverage_ratio_by_ses": ratio, "income_imbalance": imbalance}


def run_city(city):
    base = sr_data.ROOT / f"derived/model_inputs/{sr_data.LINE}/{city}/tract"
    rep = json.loads((base / "prepare_report.json").read_text())
    u = pd.read_parquet(base / "units.parquet")
    support, mismatch, nonfinite = _support(city)
    rp = _representativeness(u)
    d = sr_data.load(city, "tract")
    q = {"city": city, "n_units": int(d["N"]), "n_origins": int(d["orig"].sum()), "population": float(u["population"].sum()),
         "trips": {sp: v["visits"] for sp, v in rep["splits"].items()},
         "coverage_out_of_city": rep["splits"]["train"]["coverage_out_of_city"],
         "o_unmatched_trips": rep["o_unmatched_trips"],
         "support": support, "support_mismatch": mismatch, "nonfinite": nonfinite, "representativeness": rp}
    q["eligible"] = bool(q["n_origins"] >= MIN_ORIGIN_TRACTS and q["trips"]["train"] >= MIN_TRAIN_TRIPS)
    flags = []
    if not q["eligible"]:
        flags.append("ineligible")
    if mismatch:
        flags.append("support_mismatch")
    if nonfinite:
        flags.append("nonfinite")
    if rp["income_imbalance"]:
        flags.append("income_imbalance")
    if q["coverage_out_of_city"] > MAX_COVERAGE_OUT:
        flags.append("coverage_out_of_city")
    q["flags"] = flags
    (base / "qc.json").write_text(json.dumps(q, indent=1))
    return q


def main(cities):
    rows, failed = [], []
    for c in cities:
        try:
            q = run_city(c)
        except Exception as e:  # noqa: BLE001
            print(c, "FAILED", repr(e), flush=True)
            failed.append(c)
            continue
        rows.append({"city": c, "eligible": q["eligible"], "n_origins": q["n_origins"], "train_trips": q["trips"]["train"],
                     "population": q["population"],
                     "val_trips": q["trips"]["val"], "test_trips": q["trips"]["test"],
                     "r2": q["representativeness"]["r2"],
                     "income_ratio": (min(v for v in q["representativeness"]["coverage_ratio_by_ses"].values() if v == v)
                                      / max(v for v in q["representativeness"]["coverage_ratio_by_ses"].values() if v == v)
                                      if any(v == v for v in q["representativeness"]["coverage_ratio_by_ses"].values()) else float("nan")),
                     "coverage": q["coverage_out_of_city"], "support_mismatch": ";".join(q["support_mismatch"]),
                     "nonfinite": ";".join(q["nonfinite"]), "flags": ";".join(q["flags"])})
        print(c, json.dumps({k: rows[-1][k] for k in ("eligible", "n_origins", "train_trips", "flags")}), flush=True)
    out = pd.DataFrame(rows).set_index("city")
    import config
    RESULTS_QC = config.results_dir() / "qc"
    RESULTS_QC.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_QC / "cities_qc.csv")
    (RESULTS_QC / "qc_failed.txt").write_text("".join(c + "\n" for c in failed))
    listed = [c for c in failed if c in SEARCH or c in CONFIRM]
    if listed:   # a search / confirmation city without a QC row would silently lose its role in tiers
        raise RuntimeError(f"QC failed for search / confirmation cities {listed} (results/multicity/qc/qc_failed.txt)")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
