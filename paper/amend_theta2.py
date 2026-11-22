"""Revision 2026-11-22: theta_2 >= 1 becomes part of the specification of the Reach model.

The bound is active in five cities (figdata/rev/theta2_full.json, from multicity/rev_theta2_full.py). This script
writes their bounded local fits into the exported tables that the figures and the supplement read, and recomputes
the summaries that depend on them with the routines and seeds of the original stages (each stored statistic is first
reproduced from the unamended files; a statistic that cannot be reproduced is not touched and is reported).

Not amended (analyses of record on the earlier fits): the registered hypothesis tests (size_laws.csv, Table S3), the
exploratory urban-form analyses (e1 / e2 / p7 files) and the revision refits of rev_models (summary_models_*.json,
except the contrasts of the main-analysis fits sk_frozen).

Run once on the unamended files (keep a copy of them first): python amend_theta2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "multicity"))
import p9_stats as st  # noqa: E402

FD = HERE.parents[1] / "paper" / "figdata"
RV = FD / "rev"
CID, REACH = "c0355r1", "Reach model"
MARK = RV / "theta2_bound_applied.json"
LOG = []


def note(msg):
    LOG.append(msg)
    print(msg)


def close(a, b, tol=1e-9):
    return (a is None and b is None) or (a is not None and b is not None and abs(float(a) - float(b)) <= tol)


def same_stat(r, s):
    return all(close(r[k], s[k]) for k in ("median", "ci_lo", "ci_hi")) and r["wins"] == s["wins"] and r["losses"] == s["losses"]


def main():
    if MARK.exists():
        sys.exit(f"{MARK.name} exists: the bound has already been applied")
    full = {r["city"]: r for r in json.loads((RV / "theta2_full.json").read_text())}
    cities = sorted(full)
    new, old = {c: full[c]["bounded"] for c in cities}, {c: full[c]["main"] for c in cities}
    dshare = {c: new[c]["block_share"] - old[c]["block_share"] for c in cities}

    # ---- constants
    k = pd.read_csv(FD / "consts.csv", dtype={"city": str})
    for c in cities:
        row = k["city"] == c
        assert row.sum() == 1 and np.allclose(k.loc[row, [f"theta{i}" for i in range(5)]].to_numpy(float)[0], old[c]["consts"], atol=1e-9), c
        k.loc[row, [f"theta{i}" for i in range(5)]] = new[c]["consts"]
    k.to_csv(FD / "consts.csv", index=False)
    cv = pd.read_parquet(RV / "cities_val.parquet")
    for c in cities:
        row = (cv["cid"] == CID) & (cv["city"].astype(str) == c)
        assert row.sum() == 1 and close(cv.loc[row, "val_nll"].iloc[0], old[c]["block_nll"], 1e-8), c
        cv.loc[row, "consts"] = json.dumps(new[c]["consts"])
        cv.loc[row, "val_nll"] = new[c]["block_nll"]
    cv.to_parquet(RV / "cities_val.parquet", index=False)
    note(f"constants: consts.csv and rev/cities_val.parquet, {len(cities)} cities (standard errors left as fitted without the bound)")

    # ---- block scores
    sh = pd.read_csv(FD / "shares.csv", dtype={"city": str})
    for c in cities:
        row = (sh["city"] == c) & (sh["model"] == REACH)
        assert row.sum() == 1 and close(sh.loc[row, "nll"].iloc[0], old[c]["block_nll"], 1e-8), c
        sh.loc[row, ["nll", "share"]] = [new[c]["block_nll"], new[c]["block_share"]]
    sh.to_csv(FD / "shares.csv", index=False)

    # ---- band profiles
    b = pd.read_csv(FD / "bands.csv", dtype={"city": str})
    fx = pd.read_parquet(RV / "banded_fixed.parquet")
    n_b = n_f = 0
    for c in cities:
        for band, v in new[c]["profiles"]["band"].items():
            row = (b["city"] == c) & (b["model"] == REACH) & (b["band"] == int(band))
            if row.sum() and v is not None:
                assert close(b.loc[row, "ed"].iloc[0], old[c]["profiles"]["band"][band][0], 1e-8), (c, band)
                b.loc[row, ["ed", "eN"]] = v
                n_b += 1
        for band, v in new[c]["profiles"]["fixed"].items():
            row = (fx["city"].astype(str) == c) & (fx["model"] == REACH) & (fx["band"] == int(band)) & (fx["ref"] == "fixed")
            if row.sum() and v is not None:
                assert close(fx.loc[row, "ed"].iloc[0], old[c]["profiles"]["fixed"][band][0], 1e-8), (c, band)
                fx.loc[row, ["ed", "eN"]] = v
                n_f += 1
    b.to_csv(FD / "bands.csv", index=False)
    fx.to_parquet(RV / "banded_fixed.parquet", index=False)
    note(f"elasticities: bands.csv ({n_b} rows) and rev/banded_fixed.parquet ({n_f} rows)")
    for f in ("city_outcomes.parquet", "city_outcomes_lawtest.parquet", "city_outcomes_confirm.parquet"):
        o = pd.read_parquet(RV / f)
        hit = 0
        for c in cities:
            row = o["city"].astype(str) == c
            if "block" in o.columns:                        # the validation-block rows hold the earlier fits of record
                row &= o["block"] == "test_dest"
            if row.sum():
                assert np.allclose(o.loc[row, f"ed_5_10km_{CID}"].to_numpy(float), old[c]["profiles"]["band"]["3"][0], atol=1e-8), (f, c)
                o.loc[row, f"ed_5_10km_{CID}"] = new[c]["profiles"]["band"]["3"][0]
                hit += int(row.sum())
        o.to_parquet(RV / f, index=False)
        note(f"rev/{f}: ed_5_10km_{CID} in {hit} rows (the other Reach columns feed analyses of record and stay)")

    # ---- the comparisons in the 32 target cities
    t = pd.read_csv(FD / "p9_target.csv", dtype={"city": str})
    d = pd.read_csv(FD / "p9_diffs.csv", dtype={"city": str})
    con_all = json.loads((FD / "p9_contrasts.json").read_text())
    con = con_all["contrasts"]
    order = list(con)
    for cid in order:                                           # the stored statistics must come out of the stored differences
        assert same_stat(st.contrast(d[cid], 5000, [20261111, order.index(cid)]), con[cid]), cid
    d0_c1 = d["C1"].to_numpy(float).copy()                      # before the amendment, for the check of the all-destination file
    touched = []
    for c in cities:
        row = (t["city"] == c) & (t["model"] == CID) & (t["level"] == "local")
        if row.sum():
            assert close(t.loc[row, "nll"].iloc[0], old[c]["block_nll"], 1e-8), c
            t.loc[row, ["nll", "cpc", "share"]] = [new[c]["block_nll"], new[c]["block_cpc"], new[c]["block_share"]]
            for cid in order:
                if con[cid]["a"] == [CID, "local"]:
                    d.loc[d["city"] == c, cid] += dshare[c]
                elif con[cid]["b"] == [CID, "local"]:
                    d.loc[d["city"] == c, cid] -= dshare[c]
            touched.append(c)
    for cid in order:
        if [CID, "local"] in (con[cid]["a"], con[cid]["b"]):
            r = st.contrast(d[cid], 5000, [20261111, order.index(cid)])
            note(f"{cid}: median {con[cid]['median']:.4f} [{con[cid]['ci_lo']:.4f}, {con[cid]['ci_hi']:.4f}] {con[cid]['wins']}/{con[cid]['n']}"
                 f" -> {r['median']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] {r['wins']}/{r['n']}")
            con[cid].update(r)
    t.to_csv(FD / "p9_target.csv", index=False)
    d.to_csv(FD / "p9_diffs.csv", index=False)
    (FD / "p9_contrasts.json").write_text(json.dumps(con_all, ensure_ascii=False, indent=1))
    ps = pd.read_csv(FD / "p9_summary.csv")
    g = t[(t["model"] == CID) & (t["level"] == "local")]
    row = (ps["model"] == CID) & (ps["level"] == "local")
    for col in ("share", "nll", "cpc"):
        for tag, q in (("med", 0.5), ("q1", 0.25), ("q3", 0.75)):
            ps.loc[row, f"{col}_{tag}"] = st._quantile(g[col].to_numpy(float), q)
    ps.to_csv(FD / "p9_summary.csv", index=False)
    note(f"target cities amended: {touched}")

    # ---- all destinations
    al = pd.read_csv(FD / "alldest_scores.csv", dtype={"city": str})
    ref = al[al["model"].isin(("grav_ref", "M2_ref"))].pivot(index="city", columns="model", values="nll")
    summ = json.loads((FD / "alldest_summary.json").read_text())
    tg = d["city"].tolist()
    ab = pd.read_csv(FD.parent / "tables" / "rev" / "abs_nll.csv", dtype={"city": str}).set_index("city")   # block NLL of all 67 cities
    gap = ab["nll_gravity_pop"] - ab["nll_flexible"]
    blk_r = (ab["nll_gravity_pop"] - ab["nll_reach"]) / gap
    blk_b = (ab["nll_gravity_pop"] - ab["nll_strexp3_poi"]) / gap
    c3o = pd.read_csv(FD / "c3_cities.csv", dtype={"city": str})["city"].tolist()
    orders = {"alldest file": al[(al["model"] == CID) & (al["level"] == "local")]["city"].tolist(), "c3 file": c3o,
              "sorted": sorted(ab.index), "source then target": [c for c in c3o if c not in tg] + tg,
              "sorted source then target": sorted(c for c in ab.index if c not in tg) + tg}

    def col(m, lv):
        return al[(al["model"] == m) & (al["level"] == lv)].set_index("city")["share"]

    def all_stat(cs):
        return st.contrast(col(CID, "local").loc[cs] - col("strexp3_N", "local").loc[cs], 5000, 20261112)

    def blk_stat(cs, reach):
        return st.contrast(reach.loc[cs] - blk_b.loc[cs], 5000, 20261112)

    P = summ["paired"]
    assert same_stat(all_stat(tg), P["local|target"]["all_destinations"])
    ok_tb = same_stat(st.contrast(pd.Series(d0_c1), 5000, 20261112), P["local|target"]["block"])
    o67a = next((n for n, cs in orders.items() if same_stat(all_stat(cs), P["local|all"]["all_destinations"])), None)
    o67b = next((n for n, cs in orders.items() if same_stat(blk_stat(cs, blk_r), P["local|all"]["block"])), None)
    for c in cities:
        row = (al["city"] == c) & (al["model"] == CID) & (al["level"] == "local")
        assert row.sum() == 1
        al.loc[row, ["nll", "cpc"]] = [new[c]["all_nll"], new[c]["all_cpc"]]
        al.loc[row, "share"] = (ref.loc[c, "grav_ref"] - new[c]["all_nll"]) / (ref.loc[c, "grav_ref"] - ref.loc[c, "M2_ref"])
        blk_r.loc[c] = new[c]["block_share"]
    al.to_csv(FD / "alldest_scores.csv", index=False)

    def put(slot, r, label):
        note(f"{label}: {slot['median']:.4f} [{slot['ci_lo']:.4f}, {slot['ci_hi']:.4f}] {slot['wins']}/{slot['losses']}"
             f" -> {r['median']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] {r['wins']}/{r['losses']}")
        slot.update(r)

    put(P["local|target"]["all_destinations"], all_stat(tg), "all destinations, local, 32 target cities")
    if ok_tb:
        put(P["local|target"]["block"], st.contrast(d["C1"], 5000, 20261112), "block, local, 32 target cities (as stored beside the all-destination check)")
    else:
        note("NOT REPRODUCED, left as stored: block statistic of local|target in alldest_summary.json")
    if o67a:
        put(P["local|all"]["all_destinations"], all_stat(orders[o67a]), f"all destinations, local, all 67 cities (city order: {o67a})")
    else:
        note("NOT REPRODUCED, left as stored: all-destination statistic of local|all in alldest_summary.json")
    if o67b:
        put(P["local|all"]["block"], blk_stat(orders[o67b], blk_r), f"block, local, all 67 cities (city order: {o67b})")
    else:
        note("NOT REPRODUCED, left as stored: block statistic of local|all in alldest_summary.json")
    g = al[(al["model"] == CID) & (al["level"] == "local")]
    m = summ["models"][f"{CID}|local"]
    note(f"all destinations, local Reach, medians over the cities scored: n {m['n']} -> {int(g['nll'].notna().sum())}, share "
         f"{m['share_all']:.4f} -> {float(g['share'].median()):.4f}, CPC {m['cpc_all']:.4f} -> {float(g['cpc'].median()):.4f}")
    m.update(n=int(g["nll"].notna().sum()), share_all=float(g["share"].median()), cpc_all=float(g["cpc"].median()), nll_all=float(g["nll"].median()),
             share_block=float(blk_r.median()))
    (FD / "alldest_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))

    # ---- transfer loss against the local fit (Fig. 7b)
    c3 = pd.read_csv(FD / "c3_cities.csv", dtype={"city": str})
    for c in cities:
        row = c3["city"] == c
        assert row.sum() == 1
        c3.loc[row, ["loss_uniform", "loss_size"]] += dshare[c]
    c3.to_csv(FD / "c3_cities.csv", index=False)
    note("c3_cities.csv: loss_uniform and loss_size of the five cities (their local share changed)")

    MARK.write_text(json.dumps({"cities": cities, "bound": 1.0, "log": LOG}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
