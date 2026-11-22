"""Revision 2026-11-22 (response to review; post hoc, supplementary): the banded flexible model on weekend visits.

A visit is attributed to the device's home tract whatever the preceding stop was, so a commuter's purchases near the
workplace count as long home-based visits to POI-rich tracts. The check: does the size elasticity e_N still rise with
distance on Saturdays and Sundays (local dates), when few visits are anchored at a workplace?

  build   results/multicity/rev/weekend/<city>/T_<split>.npz, key n_weekend_5: the device-day-destination counts of the
          city's T_<split>.npz restricted to Saturdays and Sundays. Built from the line's trips table exactly as
          units.build_city builds n_dev_5; the all-day counts rebuilt here must equal n_dev_5 (checked per split).
          Nothing in derived/model_inputs is written.
  run     rev_banded.variant_city (the paper's banded model: train x search blocks, all origins) with those counts
          -> results/multicity/rev/banded_weekend.parquet, one ledger line.
  summary medians by band and the number of cities whose e_N rises with ln d, next to the all-day fits.

Usage (run_stage.sh, M line): rev_weekend.py build|run|summary [CITY,CITY,...]
A city list is a trial: its outputs are written beside the full ones with the suffix _trial and are not ledgered.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import rev_banded  # noqa: E402

KEY = "n_weekend_5"


def wk_dir() -> Path:
    return rev_banded.out_dir() / "weekend"


def is_weekend(dates) -> np.ndarray:
    """Saturday or Sunday of the local date (trips.py stores the local calendar date)."""
    return pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy() >= 5


def weekend_counts(t: pd.DataFrame, units: list, splits: dict) -> dict:
    """{split: (all-day N x N, weekend N x N)} from trips (date, o, d, n), as units.build_city aggregates them."""
    idx = pd.Index(units)
    N = len(units)
    date = t["date"].astype(str).to_numpy()
    wk = is_weekend(date)
    inside = t["d"].isin(idx).to_numpy()
    oi_all, di_all = idx.get_indexer(t["o"]), idx.get_indexer(t["d"])
    n_all = t["n"].to_numpy(float)
    out = {}
    for sp, (a0, a1) in splits.items():
        m = (date >= a0) & (date <= a1) & inside
        A, W = np.zeros((N, N), dtype=np.float32), np.zeros((N, N), dtype=np.float32)
        np.add.at(A, (oi_all[m], di_all[m]), n_all[m])
        mw = m & wk
        np.add.at(W, (oi_all[mw], di_all[mw]), n_all[mw])
        out[sp] = (A, W)
    return out


def build_city(city: str) -> dict:
    import sr_data
    import units as U
    from cities import counties_of
    d = sr_data.load(city, "tract")
    unit_list = [str(x) for x in d["u"]["unit"]]
    counties = counties_of(city, config.CITIES_CSV)
    states = sorted({c[:2] for c in counties})
    t, _, _ = U._trips(config.ROOT, city, unit_list, counties, states, derived=config.DERIVED)
    if not config.INCLUDE_WORK:
        t = t[~t["work"].astype(bool)]
    out = wk_dir() / city
    out.mkdir(parents=True, exist_ok=True)
    rep = {"city": city}
    for sp, (A, W) in weekend_counts(t, unit_list, dict(config.SPLITS)).items():
        ref = d["T"][sp]
        if not np.allclose(A, ref, rtol=0, atol=1e-3):
            raise RuntimeError(f"{city} {sp}: the all-day counts rebuilt here differ from n_dev_5 "
                               f"({float(A.sum()):.0f} vs {float(ref.sum()):.0f})")
        np.savez_compressed(out / f"T_{sp}.npz", **{KEY: W})
        rep[sp] = {"visits": float(A.sum()), "weekend": float(W.sum())}
    print(json.dumps(rep), flush=True)
    return rep


def _subset(arg) -> list:
    cs = rev_banded._cities()
    if not arg:
        return cs
    want = arg.split(",")
    bad = [c for c in want if c not in cs]
    if bad:
        raise ValueError(f"not study cities: {bad}")
    return want


def stage_build(arg=None):
    cs = _subset(arg)
    procs = max(1, min(int(os.environ.get("REV_WEEKEND_PROCS", "4")), len(cs)))
    if procs == 1:
        reps = [build_city(c) for c in cs]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(procs, maxtasksperchild=1) as P:
            reps = list(P.imap_unordered(build_city, cs, chunksize=1))
    tot = {sp: sum(r[sp]["visits"] for r in reps) for sp in config.SPLITS}
    wk = {sp: sum(r[sp]["weekend"] for r in reps) for sp in config.SPLITS}
    name = "counts_trial.json" if arg else "counts.json"
    (wk_dir() / name).write_text(json.dumps({"cities": reps, "visits": tot, "weekend": wk}, indent=1))
    print(f"{len(reps)} cities; weekend share of visits " + ", ".join(f"{sp} {wk[sp] / tot[sp]:.3f}" for sp in tot), flush=True)


def stage_run(arg=None):
    import ledger
    import p9
    import rev_models
    import sr_data
    cs = _subset(arg)
    missing = [c for c in cs if not all((wk_dir() / c / f"T_{sp}.npz").exists() for sp in sr_data.SPLITS)]
    if missing:
        raise RuntimeError(f"no weekend counts for {missing[:5]} (rev_weekend.py build)")
    pooled = rev_banded.pooled_reference(rev_banded._cities())            # the paper's common reference point
    rev_banded._G.update(variant="weekend", F=rev_banded.fixed_refs(pooled))

    def tf(d):
        city = d["base"].parent.name
        d["T"] = {sp: np.load(wk_dir() / city / f"T_{sp}.npz")[KEY].astype(np.float64) for sp in sr_data.SPLITS}
        return d

    orig, mods, _ = rev_models._patch_load(tf)
    try:
        rows = [r for rs in p9._pool_map(rev_banded.variant_city, cs) for r in rs]
    finally:
        for m in mods:
            m.load = orig
    t = pd.DataFrame(rows)
    t["variant"] = "weekend"
    p = rev_banded.out_dir() / ("banded_weekend_trial.parquet" if arg else "banded_weekend.parquet")
    t.to_parquet(p, index=False)
    if not arg:
        ledger.add(f"Revision 2026-11-22 (post hoc, supplementary): rev_weekend run: banded flexible model on weekend "
                   f"visits (Saturdays and Sundays, local date), {t['city'].nunique()} cities; rev/{p.name} sha {ledger.sha(p)}")
    print(f"wrote {p} ({t['city'].nunique()} cities)", flush=True)


def rising(t: pd.DataFrame, min_visits: float) -> pd.DataFrame:
    """Per city: OLS slope of e_N on ln d over the bands with enough training visits (at least three bands)."""
    rows = []
    for city, g in t.groupby("city"):
        g = g[(g["visits"] >= min_visits) & np.isfinite(g["eN"])].sort_values("band")
        if len(g) >= 3:
            rows.append({"city": city, "slope": float(np.polyfit(np.log(g["d_ref"]), g["eN"], 1)[0]), "bands": len(g)})
    return pd.DataFrame(rows)


def stage_summary(arg=None):
    import prefreeze
    mv = prefreeze.MIN_BAND_VISITS
    for name in (["banded_weekend_trial.parquet"] if arg else ["banded_weekend.parquet", "banded_nonwork.parquet", "banded_fixed.parquet"]):
        p = rev_banded.out_dir() / name
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        t = t[t["model"] == "Flexible"] if "model" in t else t
        for ref in sorted(t["ref"].unique()):
            g = t[(t["ref"] == ref) & (t["visits"] >= mv)]
            med = g.groupby("band")[["eN", "ed"]].median().round(3)
            r = rising(t[t["ref"] == ref], mv)
            print(f"{name} ref={ref}: e_N by band {med['eN'].tolist()}; e_d {med['ed'].tolist()}; cities per band "
                  f"{g.groupby('band')['city'].nunique().tolist()}; e_N rises with ln d in {int((r['slope'] > 0).sum())} of {len(r)}",
                  flush=True)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in ("build", "run", "summary"):
        sys.exit(__doc__)
    {"build": stage_build, "run": stage_run, "summary": stage_summary}[sys.argv[1]](sys.argv[2] if len(sys.argv) == 3 else None)
