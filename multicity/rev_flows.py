"""Revision 2026-11-13 (response to review, Task R4): baselines for the flow reconstruction.

The paper's flows are T_i P(j|i) with T_i the origin's observed test-period trips (od_maps.py). To show how much of
the corridor recovery comes from the destination model rather than from the observed origin totals, the same six
confirmation cities (test period x all destinations) are scored with
  uniform_dest   T_i spread evenly over the destinations (no destination model at all);
  distance_only  T_i exp(-beta ln d) normalised, beta = the uniform (discovery-median) distance exponent of POI gravity;
  scaled_kernel  the Scaled kernel with uniform constants (as Fig. 6);
  strexp3_N      the strongest benchmark with uniform constants;
  scaled_kernel_pop  the Scaled kernel's P(j|i) times origin totals proportional to residential population, scaled
                 to the city's observed total (no origin total is read; the total only fixes the scale of the CPC).
Metrics: how many of the 2,000 largest observed flows are among the model's 2,000 largest, and the CPC.
Post hoc, supplementary. Output: results/multicity/rev/flows_baselines.csv. Usage: rev_flows.py run
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401
from od_maps import cpc, flows, top  # noqa: E402,F401

K_TOP = 2000


def uniform_flows(y, offsets) -> np.ndarray:
    st, rep = offsets[:-1], np.diff(offsets)
    return np.repeat(np.add.reduceat(np.asarray(y, float), st) / rep, rep)


def population_totals(p, offsets, pop, total: float) -> np.ndarray:
    """p = P(j|i) per row; each origin's total proportional to its population, all origins summing to `total`."""
    pop = np.asarray(pop, float)
    return np.repeat(total * pop / pop.sum(), np.diff(offsets)) * np.asarray(p, float)


def overlap_k(y, mu, k: int = K_TOP) -> int:
    return int(len(np.intersect1d(top(y, k), top(mu, k))))


def city_rows(city: str, name: str) -> list:
    import frozen
    import od_maps
    import p9
    import prior_models as pm
    import sr_data
    from refit_select import eta as eq_eta, parse_equation
    frozen.require()
    cfg = p9.require_fixed()
    K, cands = od_maps.constants(cfg, city), p9.ours_candidates()
    d = sr_data.load(city, "tract")
    ev = sr_data.build([city], "tract", "test", "all", None, 0)
    y, off = np.asarray(ev.y, float), ev.offsets
    o = ev.origin_unit[off[:-1]]
    logd = np.log(np.maximum(d["D"][ev.origin_unit, ev.dest_unit], 0.05))
    beta = float(K[("grav_power_N", "uniform")][1][2])
    mus = {"uniform_dest": uniform_flows(y, off), "distance_only": flows(-beta * logd, y, off)}
    for m, kind in (("c0355r1", "ours"), ("strexp3_N", "eq")):
        c = cands[m] if kind == "ours" else parse_equation(pm.EQ_MODELS[m].equation, None, "A", 9)
        with np.errstate(all="ignore"):
            e = eq_eta(c, ev, consts=np.asarray(K[(m, "uniform")][1], float))
        mus["scaled_kernel" if m == "c0355r1" else m] = flows(e, y, off)
    p = mus["scaled_kernel"] / np.repeat(np.add.reduceat(y, off[:-1]), np.diff(off))
    mus["scaled_kernel_pop"] = population_totals(p, off, d["u"]["population"].to_numpy(float)[o], y.sum())
    rows = [{"city": city, "name": name, "model": m, "beta": beta if m == "distance_only" else np.nan,
             "top2000": overlap_k(y, mu), "cpc": cpc(y, mu)} for m, mu in mus.items()]
    print(city, rows, flush=True)
    return rows


def stage_run():
    import ledger
    import od_maps
    cc = od_maps.confirmation_cities()
    names = cc["name"] if "name" in cc else pd.Series(cc.index, index=cc.index)
    rows = [r for c in cc.index.astype(str) for r in city_rows(c, str(names.get(c, c)))]
    out = config.results_dir("M") / "rev" / "flows_baselines.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    ledger.add(f"Revision R4 (post hoc, supplementary): flow-reconstruction baselines in the six confirmation cities; "
               f"rev/flows_baselines.csv sha {ledger.sha(out)}")


if __name__ == "__main__":
    if sys.argv[1:] != ["run"]:
        sys.exit(__doc__)
    stage_run()
