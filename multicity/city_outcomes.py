"""Plan 7 Task 4 (plan-7 design, outcome set P1): one row per city of the quantities the city-form hypotheses
explain. Everything comes from existing fits (frozen library, 26-candidate evaluation set, flexible banded model):

- membership: is a kernel-family / opportunity-pressure (c0413 family) candidate near-best (and above the floor)?
- performance: gravity -> M2 gain, best clean-library share of it, the main skeleton's share_S0;
- parameters: flexible-model elasticities per distance band, slope of e_N on log d, the clean skeletons' 5-10 km
  e_d, c0514c1's turning distance;
- size residual: 5-10 km e_d minus the discovery cities' line on log population;
- transfer loss: test-block NLL with the other discovery cities' median constants minus the local refit.

Blocks: test_dest (test period x destination fold 0, primary) and val (validation block, secondary). Law-test cities
are refused until the plan-7 registration is signed (p7seal).

Plan 8 (design decision A1, 2026-11-10: the primary test uses the law-test cities only): the law-test cities (tag lawtest) get their own files, test_dest block only, so the
plan-7 discovery outputs are never overwritten.

Every outcome / transfer file is written once (an existing file is refused, so plan 7's discovery files stay as they are).

Usage (run_stage.sh): city_outcomes.py outcomes TAG | transfer TAG | meta      (TAG: discovery | lawtest | confirm)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

KERNEL = ("c0353", "c0353c1", "c0353c1r1", "c0354", "c0355", "c0355r1")
OPP = ("c0413", "c0413c1", "c0413c1r1", "c0413c1r2", "c0413c1r3", "c0413c1r4")
SKELETONS = ("c0355r1", "c0262", "c0514c1")
TRANSFER_CIDS = ("c0355r1", "c0262")
MAIN = "c0355r1"
MARGIN = 0.03
BLOCKS = ("test_dest", "val")
TAGS = ("discovery", "lawtest", "confirm")
REPORT_KEYS = ("cls", "gain_grav_m2", "ed_5_10km", "ed_5_10km_c0355r1", "theta_turn_c0514c1", "transfer_loss_c0355r1",
               "ed_5_10km_c0262", "slope_eN_logd_c0355r1", "ed_1_2km", "ed_10_20km", "ed_20km")
BAND_KEYS = ("0_1km", "1_2km", "2_5km", "5_10km", "10_20km", "20km")
OUT = config.RESULTS / "p7"

# The OUTCOME_DEFS texts are written to the ledgered outcome-definition file and enter the statements of the
# registered plan-7 hypotheses (results/registry/registered_p7.json), so they are kept verbatim in the original
# Chinese. English glosses:
#   cls                    skeleton membership: kernel = only the kernel family is near-best, opp = only the c0413
#                          family is near-best, both = both are, neither = neither is (tolerance 0.03, and better than
#                          the floor model)
#   kernel_near            one of the kernel family (c0353, c0353c1, c0353c1r1, c0354, c0355, c0355r1) is near-best and
#                          better than the floor model
#   opp_near               one of the c0413 family (c0413, c0413c1, c0413c1r1 to r4) is near-best and better than the
#                          floor model
#   gain_grav_m2           NLL difference from the (power-law) gravity model to M2 (nats/visit)
#   share_grav_best_clean  share of the gravity -> M2 gap covered by the best of the clean library (c0355r1, c0262,
#                          c0514c1)
#   share_s0_main          share_S0 of the main skeleton c0355r1: (NLL_S0 - NLL) / (NLL_S0 - NLL_M2)
#   ed_5_10km              flexible-model distance elasticity e_d in the 5-10 km band (bands with >= 1000 training trips)
#   slope_eN_logd          OLS slope of the flexible model's size elasticity e_N on ln d_ref (>= 3 usable bands)
#   theta_turn_c0514c1     turning distance theta1 of c0514c1 (km): e_d = -1 - d / (d + theta1)
#   ed_5_10km_resid        flexible-model 5-10 km e_d minus the discovery-city regression line of e_d on ln population
#                          (negative = steeper than size predicts)
#   ed_<band>, eN_<band>   flexible-model distance elasticity e_d / size elasticity e_N in the <band> km band
#   ed_5_10km_<cid>        distance elasticity e_d of <cid> at the 5-10 km band reference point
#   slope_eN_logd_<cid>    slope of <cid>'s e_N on ln d_ref
#   transfer_loss_<cid>    extra test-block NLL (nats/visit) of <cid> with the median constants of the other discovery
#                          cities applied directly, compared with the city's own refit
#   transfer_frac_<cid>    the previous quantity divided by the city's gravity -> M2 gap
OUTCOME_DEFS = {
    "cls": "骨架归属：kernel = 只有核家族接近最优，opp = 只有 c0413 族接近最优，both = 两者都是，neither = 都不是（容差 0.03，且优于门槛模型）",
    "kernel_near": "核家族（c0353、c0353c1、c0353c1r1、c0354、c0355、c0355r1）中有一条接近最优且优于门槛模型",
    "opp_near": "c0413 族（c0413、c0413c1、c0413c1r1 至 r4）中有一条接近最优且优于门槛模型",
    "gain_grav_m2": "重力模型（幂律）到 M2 的 NLL 差（nats/visit）",
    "share_grav_best_clean": "干净库（c0355r1、c0262、c0514c1）中最好的一条走完重力→M2 差距的比例",
    "share_s0_main": "主骨架 c0355r1 的 share_S0：(NLL_S0 − NLL)/(NLL_S0 − NLL_M2)",
    "ed_5_10km": "灵活模型 5–10 km 段的距离弹性 e_d（训练出行 ≥ 1000 的段）",
    "slope_eN_logd": "灵活模型的规模弹性 e_N 对 ln d_ref 的 OLS 斜率（≥ 3 个可用段）",
    "theta_turn_c0514c1": "c0514c1 的转折距离 θ1（km）：e_d = −1 − d/(d + θ1)",
    "ed_5_10km_resid": "灵活模型 5–10 km e_d 减去发现城上 e_d ~ ln 人口 的回归线（负 = 比规模预测更陡）",
}
for _k, _b in zip(BAND_KEYS, ("0–1", "1–2", "2–5", "5–10", "10–20", "≥ 20")):
    OUTCOME_DEFS.setdefault(f"ed_{_k}", f"灵活模型 {_b} km 段的距离弹性 e_d")
    OUTCOME_DEFS[f"eN_{_k}"] = f"灵活模型 {_b} km 段的规模弹性 e_N"
for _c in SKELETONS:
    OUTCOME_DEFS[f"ed_5_10km_{_c}"] = f"{_c} 在 5–10 km 段参考点上的距离弹性 e_d"
    OUTCOME_DEFS[f"slope_eN_logd_{_c}"] = f"{_c} 的 e_N 对 ln d_ref 的斜率"
for _c in TRANSFER_CIDS:
    OUTCOME_DEFS[f"transfer_loss_{_c}"] = f"{_c} 用其余发现城常数中位数直接套用，比本城重拟合多出的测试块 NLL（nats/visit）"
    OUTCOME_DEFS[f"transfer_frac_{_c}"] = f"上一项除以该城的重力→M2 差距"


# ------------------------------------------------------------------------------------------------ pure functions
def _nll(refs_city, m):
    import scoring
    return scoring._nll(refs_city, m)


def membership(cv: pd.DataFrame, refs: dict, margin: float = MARGIN) -> pd.DataFrame:
    import scoring
    import sensitivity as sv
    nn = sv.recompute_near(cv, refs, margin)
    rows = []
    for city, g in nn.groupby("city", sort=True):
        ok = g[g["near_best"].astype(bool)]
        ok = ok[[scoring.passes_floor(v, refs.get(city)) for v in ok["val_nll"]]]
        k, o = bool(ok["cid"].isin(KERNEL).any()), bool(ok["cid"].isin(OPP).any())
        rows.append({"city": city, "kernel_near": k, "opp_near": o,
                     "cls": "both" if k and o else "kernel" if k else "opp" if o else "neither"})
    return pd.DataFrame(rows)


def performance(cv: pd.DataFrame, refs: dict, clean=None) -> pd.DataFrame:
    import scoring
    if clean is None:
        import frozen
        clean = frozen.clean_library()
    rows = []
    for city, g in cv[cv["finite"].astype(bool)].groupby("city", sort=True):
        r = refs.get(city) or {}
        a, b = _nll(r, "grav_power"), _nll(r, "M2_glm")
        sh = [scoring.share(v, r, "grav_power") for v in g.loc[g["cid"].isin(clean), "val_nll"]]
        sh = [s for s in sh if s is not None]
        main = g.loc[g["cid"] == MAIN, "val_nll"]
        rows.append({"city": city, "gain_grav_m2": (a - b) if a is not None and b is not None else None,
                     "share_grav_best_clean": max(sh) if sh else None,
                     "share_s0_main": scoring.share(float(main.iloc[0]), r, "S0") if len(main) else None})
    return pd.DataFrame(rows)


def size_residual(df: pd.DataFrame, quantity: str, fit_cities, fit_df: pd.DataFrame = None) -> pd.Series:
    """quantity minus the OLS line quantity ~ log_population fitted on the fit_cities rows of fit_df (default df) only;
    NaN where either is missing, and everywhere when fewer than 3 fit points exist."""
    src = df if fit_df is None else fit_df
    f = src[src["city"].isin(set(fit_cities))][["log_population", quantity]].astype(float).dropna()
    if len(f) < 3:
        return pd.Series(np.nan, index=df.index)
    b, a = np.polyfit(f["log_population"], f[quantity], 1)
    return df[quantity].astype(float) - (a + b * df["log_population"].astype(float))


def _consts(v):
    v = json.loads(v) if isinstance(v, str) else v
    return None if v is None else [float(x) for x in v]


def loo_transfer_consts(cv: pd.DataFrame, cid: str, city: str) -> list:
    """Element-wise median of cid's constants over the other cities' finite rows of cv (leave this city out)."""
    g = cv[(cv["cid"] == cid) & (cv["city"] != city) & cv["finite"].astype(bool)]
    m = [c for c in (_consts(v) for v in g["consts"]) if c is not None]
    return [float(x) for x in np.median(np.array(m, float), axis=0)] if m else None


def turning_distance(cv: pd.DataFrame) -> dict:
    g = cv[(cv["cid"] == "c0514c1") & cv["finite"].astype(bool)]
    return {c: _consts(v)[1] for c, v in zip(g["city"], g["consts"]) if _consts(v) is not None}


def band_columns(b: dict) -> dict:
    """Flexible-model e_d / e_N per band (bands with >= MIN_BAND_VISITS training visits and a reference point)."""
    import prefreeze
    pr, rf, vi = (prefreeze._intkeys(b.get(k)) for k in ("profile", "refs", "visits"))
    out = {}
    for i, k in enumerate(BAND_KEYS):
        ok = prefreeze._usable(pr, i) and (vi.get(i) or 0) >= prefreeze.MIN_BAND_VISITS and bool(rf.get(i))
        out[f"ed_{k}"] = float(pr[i][0]) if ok else None
        out[f"eN_{k}"] = float(pr[i][1]) if ok else None
    q = prefreeze.flex_quantities(b)
    out["ed_5_10km"], out["slope_eN_logd"] = q["ed_5_10km"], q["slope_eN_logd"]
    return out


# ---------------------------------------------------------------------------------------------------- inputs
def _cv(block):
    return pd.read_parquet(config.RESULTS / ("confirm/test_dest/cities_val.parquet" if block == "test_dest"
                                             else "prefreeze_mv/discovery/cities_val.parquet"))


def _refs(block, cities):
    out = {}
    for c in cities:
        p = config.RESULTS / ("refs_test" if block == "test_dest" else "refs") / f"{c}.json"
        if p.exists():
            r = json.loads(p.read_text())
            out[c] = r.get("test_dest") if block == "test_dest" else r.get("models")
    return out


def _pool():
    return {x["cid"]: x for x in json.loads((config.RESULTS / "prefreeze_mv/loop/pool.json").read_text())}


def _banded(cities):
    bd = {}
    for d in ("prefreeze/banded", "prefreeze/lawtest_banded", "confirm/banded"):
        for c in cities:
            p = config.RESULTS / d / f"{c}.json"
            if c not in bd and p.exists():
                bd[c] = json.loads(p.read_text())
    return bd


def _log_population(cities):
    fm = pd.read_csv(config.RESULTS / "form_metrics.csv", dtype={"city": str}).set_index("city")
    return {c: float(np.log(fm.loc[c, "population"])) if c in fm.index else None for c in cities}


def outcome_table(cities, block: str, transfer_path=None) -> pd.DataFrame:
    import prefreeze
    import v3_library as v3
    import p7seal
    if block not in BLOCKS:
        raise ValueError(f"block {block!r}: one of {BLOCKS}")
    cities = [str(c) for c in cities]
    p7seal.guard_cities(cities)
    cv = _cv(block)
    cv = cv[cv["city"].isin(cities)]
    refs = _refs(block, cities)
    df = pd.DataFrame({"city": cities})
    df = df.merge(membership(cv, refs), on="city", how="left").merge(performance(cv, refs), on="city", how="left")
    bd = _banded(cities)
    df = df.merge(pd.DataFrame([{"city": c, **band_columns(b)} for c, b in bd.items()] or [{"city": None}]),
                  on="city", how="left")
    refs_b = {c: prefreeze._intkeys(b["refs"]) for c, b in bd.items()}
    visits = {c: prefreeze._intkeys(b["visits"]) for c, b in bd.items()}
    pool = _pool()
    prof, _ = prefreeze.skeleton_profiles(cv[cv["cid"].isin(SKELETONS)], pool, refs_b)
    rows = {}
    for (cid, c), p in prof.items():
        q = v3.skel_quantities(p, refs_b[c], visits.get(c, {}))
        rows.setdefault(c, {"city": c}).update({f"ed_5_10km_{cid}": q["ed_5_10km"], f"slope_eN_logd_{cid}": q["slope_eN_logd"]})
    if rows:
        df = df.merge(pd.DataFrame(list(rows.values())), on="city", how="left")
    df["theta_turn_c0514c1"] = df["city"].map(turning_distance(cv))
    lp = _log_population(cities)
    df["log_population"] = df["city"].map(lp)
    tl = Path(transfer_path) if transfer_path is not None else OUT / "transfer_loss.parquet"
    if tl.exists():
        df = df.merge(pd.read_parquet(tl), on="city", how="left")
    return df


# ------------------------------------------------------------------------------------------------------ stages
def discovery() -> list:
    return config.cities_with_role("search") + config.cities_with_role("discovery")


def cities_for(tag: str) -> list:
    """The cities of a city group: discovery (search + discovery), lawtest (the 26 law-test cities) or confirm (the 6
    confirmation cities, descriptive use after the freeze)."""
    if tag == "discovery":
        return discovery()
    if tag in ("lawtest", "confirm"):
        return config.cities_with_role(tag)
    raise ValueError(f"city group {tag!r}: one of {TAGS}")


def paths_for(tag: str) -> dict:
    """The outcome and transfer files of a city group (discovery keeps the plan-7 names)."""
    if tag not in TAGS:
        raise ValueError(f"city group {tag!r}: one of {TAGS}")
    suf = "" if tag == "discovery" else f"_{tag}"
    return {"outcomes": OUT / f"city_outcomes{suf}.parquet", "transfer": OUT / f"transfer_loss{suf}.parquet"}


def _discovery_fit_table() -> pd.DataFrame:
    """The discovery cities' test_dest rows (the size line of ed_5_10km_resid for other city groups)."""
    t = pd.read_parquet(paths_for("discovery")["outcomes"])
    return t[t["block"] == "test_dest"]


def transfer_loss(city: str, cids=TRANSFER_CIDS) -> dict:
    """Test-block NLL with the other discovery cities' median constants minus the local refit (same constants as the
    frozen test_dest fits), in nats per visit and as a fraction of the city's gravity -> M2 gap."""
    import dataclasses
    import library
    import p7seal
    from refit_select import PENALTY, loss
    p7seal.guard_cities([city])
    cv = _cv("test_dest")
    mv = pd.read_parquet(config.RESULTS / "prefreeze_mv/discovery/cities_val.parquet")
    mv = mv[mv["city"].isin(discovery())]
    rt = _refs("test_dest", [city]).get(city) or {}
    gap = (_nll(rt, "grav_power") or np.nan) - (_nll(rt, "M2_glm") or np.nan)
    D = library.city_data(city, None, "test_dest")
    pool = _pool()
    out = {"city": city}
    for cid in cids:
        row = cv[(cv["cid"] == cid) & (cv["city"] == city) & cv["finite"].astype(bool)]
        tc = loo_transfer_consts(mv, cid, city)
        if not len(row) or tc is None:
            out[f"transfer_loss_{cid}"] = out[f"transfer_frac_{cid}"] = None
            continue
        base = library.candidate({**pool[cid], "consts_refit": None})
        lt = loss(dataclasses.replace(base, consts=np.asarray(tc, float)), D["va"])
        ll = loss(dataclasses.replace(base, consts=np.asarray(_consts(row["consts"].iloc[0]), float)), D["va"])
        d = float(lt - ll) if lt < PENALTY and ll < PENALTY else None
        out[f"transfer_loss_{cid}"] = d
        out[f"transfer_frac_{cid}"] = d / gap if d is not None and gap == gap and gap > 0 else None
    return out


def stage_outcomes(tag: str = "discovery"):
    """The outcome table of a city group; needs that group's transfer file first (P7-02's outcome lives there).
    Discovery: both blocks, size line on its own rows; lawtest: test_dest only, size line from the discovery table."""
    import ledger
    paths = paths_for(tag)
    if paths["outcomes"].exists():
        raise FileExistsError(f"{paths['outcomes']} exists: outcome tables are written once")
    if not paths["transfer"].exists():
        raise FileNotFoundError(f"{paths['transfer']} missing: run `city_outcomes.py transfer {tag}` first")
    cities = cities_for(tag)
    blocks = BLOCKS if tag == "discovery" else ("test_dest",)
    fit_df = None if tag == "discovery" else _discovery_fit_table()
    disc = discovery()
    frames = []
    for b in blocks:
        t = outcome_table(cities, b, transfer_path=paths["transfer"])
        t["ed_5_10km_resid"] = size_residual(t, "ed_5_10km", disc, fit_df=fit_df)
        t["block"] = b
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    p = paths["outcomes"]
    df.to_parquet(p, index=False)
    n = {k: int(df.loc[df["block"] == "test_dest", k].notna().sum()) if k in df else 0 for k in REPORT_KEYS}
    ledger.add(f"P1 outcomes ({tag}): {len(cities)} cities x blocks {list(blocks)}; non-missing (test_dest) {n}; "
               f"{p.name} sha {ledger.sha(p)}")


def stage_transfer(tag: str = "discovery"):
    """Direct-transfer losses of a city group (donors: the discovery cities' median constants, leave this city out)."""
    import multiprocessing as mp
    import ledger
    import os
    import frozen
    p = paths_for(tag)["transfer"]
    if p.exists():
        raise FileExistsError(f"{p} exists: transfer files are written once")
    frozen.require()
    cities = cities_for(tag)
    procs = min(int(os.environ.get("LOOP_OUTER_PROCS", "8")), len(cities))
    if procs <= 1:
        rows = [transfer_loss(c) for c in cities]
    else:
        with mp.get_context("fork").Pool(procs, maxtasksperchild=1) as P:
            rows = list(P.imap_unordered(transfer_loss, cities, chunksize=1))
    df = pd.DataFrame(rows).sort_values("city")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    ledger.add(f"P1 transfer ({tag}): leave-one-out median discovery constants on {len(df)} cities (test_dest); "
               f"finite {int(df['transfer_loss_c0355r1'].notna().sum())}; {p.name} sha {ledger.sha(p)}")


def stage_meta():
    import ledger
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "outcomes_meta.json"
    p.write_text(json.dumps(OUTCOME_DEFS, ensure_ascii=False, indent=1))
    ledger.add(f"Plan 7 P1 outcome definitions: {len(OUTCOME_DEFS)} quantities; {p.name} sha {ledger.sha(p)}")


STAGES = {"outcomes": stage_outcomes, "transfer": stage_transfer, "meta": stage_meta}

def main(argv: list):
    """CLI: outcomes TAG | transfer TAG | meta - the city group is never implied."""
    if argv[:1] in (["outcomes"], ["transfer"]) and len(argv) == 2 and argv[1] in TAGS:
        return STAGES[argv[0]](argv[1])
    if argv == ["meta"]:
        return STAGES["meta"]()
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
