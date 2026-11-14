"""The V3 skeleton library and cycle-4 hypotheses (design decision of 2026-11-10: switch to the V3 rule; item 5).

Inputs: the merge-variant run (results/multicity/prefreeze_mv: V3 gate, discovery fits of every V3 gate member,
coverage), the flexible distance-band fits (prefreeze/banded: discovery; prefreeze/lawtest_banded: lawtest, H3).
Stages:
  shape     the V3 gate's elasticity profiles against the flexible model on the discovery cities (as in plan 5)
  params    the library (clean greedy picks of the V3 gate at 0.03 and 0.05, in order) and its parameter regularities
            on the 35 discovery cities: identifiable constants (laws.param_table) and three profile quantities
            (ed_5_10km, eN_5_10km, slope_eN_logd; banded reference points, bands with >= 1000 training visits);
            per quantity the city distribution, and laws.cross_city_summary against every covariate (B = 2000) + rho
  register  cycle 4, before any lawtest data is read: for each of H4_SKELETONS (the skeletons that enter the library
            only under V3), the strongest discovery association (n >= 20, bootstrap p < 0.05, Spearman rho and OLS
            slope of one sign; max |rho|) becomes one slope_ci hypothesis -> laws/registered_v3.json
  lawtest   refuses before registration; fits the registered skeletons on the lawtest cities (fit_library)
  test      laws.run_test on the lawtest cities (B = 5000); Holm over every registered hypothesis (loop 10, H3 2, H4)
  report    markdown, redaction-checked (no lawtest / confirmation city id; lawtest results are aggregates)
No confirmation city, no test period, no LLM call.
Usage (compute server, loop env): python -I v3_library.py shape | params | register | lawtest | test | report
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import config  # noqa: E402
import laws  # noqa: E402
import prefreeze  # noqa: E402
import scoring  # noqa: E402

ROOT = config.RESULTS / "prefreeze_v3"
MV = config.RESULTS / "prefreeze_mv"
PF = prefreeze.ROOT
LAWS = config.RESULTS / "laws"
VARIANT = "V3"
H4_SKELETONS = ("c0262", "c0514c1", "c0355r1")
CYCLE, SOURCE = 4, "executor (design decision 2026-11-10)"   # provenance label stored in registered_v3.json
# QUANT_MEANING, QUANT_DEF and the statement / criterion texts built in hypothesis() are part of the registered
# hypotheses and enter their sha (laws._hsha), so they are kept verbatim in the original Chinese. English glosses:
#   ed_5_10km      distance elasticity e_d in the 5-10 km band (banded reference point, dη/dln d, band 3,
#                  >= 1000 training-period trips)
#   eN_5_10km      destination-size elasticity e_N in the 5-10 km band (N_j dη/dN_j, band 3, >= 1000 trips)
#   slope_eN_logd  OLS slope of e_N on ln d_ref across distance bands (>= 3 bands with >= 1000 trips)
#   statement      "the <quantity> of skeleton <cid> increases / decreases as <covariate> increases (discovery
#                  evidence: OLS slope, Spearman rho, n)"
#   criterion      "OLS with <quantity> as response and <covariate> as sole predictor; city bootstrap 95% CI on the
#                  test cities excludes 0 with the stated sign; Holm over all registered hypotheses, alpha = 0.05"
MIN_N, ALPHA_SEL, EVID_B = 20, 0.05, 2000
QUANT_MEANING = {"ed_5_10km": "5–10 km 段的距离弹性 e_d", "eN_5_10km": "5–10 km 段的目的地规模弹性 e_N",
                 "slope_eN_logd": "规模弹性 e_N 对 ln d 的斜率"}
QUANT_DEF = {"ed_5_10km": "banded 参考点上 dη/dln d，第 3 段 5–10 km，训练期出行 ≥ 1000",
             "eN_5_10km": "banded 参考点上 N_j dη/dN_j，第 3 段 5–10 km，训练期出行 ≥ 1000",
             "slope_eN_logd": "各距离段 e_N 对 ln d_ref 的 OLS 斜率，训练期出行 ≥ 1000 的段至少 3 个"}
COLS = ["cid", "param", "city", "value", "se", "sd_starts", "identifiable"]


def _read(p, default):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


# ================================================================================================ pure helpers
def skel_quantities(profile: dict, refs_b: dict, visits: dict) -> dict:
    """ed_5_10km / eN_5_10km (band 3 when usable with >= MIN_BAND_VISITS visits) and slope_eN_logd (>= MIN_BANDS
    usable bands) of one skeleton profile {band: (e_d, e_N) | None} — prefreeze.flex_quantities' rules."""
    fq = prefreeze.flex_quantities({"profile": profile, "refs": refs_b, "visits": visits})
    pr, vi, rf = prefreeze._intkeys(profile), prefreeze._intkeys(visits), prefreeze._intkeys(refs_b)
    ok3 = prefreeze._usable(pr, 3) and (vi.get(3) or 0) >= prefreeze.MIN_BAND_VISITS and bool(rf.get(3))
    return {"ed_5_10km": fq["ed_5_10km"], "eN_5_10km": float(pr[3][1]) if ok3 else None,
            "slope_eN_logd": fq["slope_eN_logd"]}


def library_from_coverage(cov: dict, variant: str = VARIANT, cand_set: str = "gate") -> list:
    """The clean greedy picks of the variant's own gate at 0.03, then at 0.05, first occurrence kept."""
    t = cov[variant][cand_set]
    return list(dict.fromkeys([r["cid"] for m in ("0.03", "0.05") for r in t[m]["clean"]]))


def band_rows(cid: str, q_by_city: dict) -> pd.DataFrame:
    rows = [{"cid": cid, "param": k, "city": city, "value": float(v), "se": None, "sd_starts": 0.0, "identifiable": True}
            for city, q in q_by_city.items() for k, v in q.items() if v is not None and np.isfinite(v)]
    return pd.DataFrame(rows, columns=COLS)


def select(evidence: list, skeletons, min_n: int = MIN_N, alpha: float = ALPHA_SEL) -> list:
    """Per skeleton at most one association: n >= min_n, bootstrap p < alpha, rho and slope of one sign; max |rho|
    (ties: param, covariate)."""
    out = []
    for s in skeletons:
        ok = [e for e in evidence if e["cid"] == s and e["n"] >= min_n and e["p"] < alpha and e.get("rho") is not None
              and np.isfinite(e["rho"]) and e["rho"] != 0 and np.sign(e["rho"]) == np.sign(e["slope"])]
        if not ok:
            continue
        b = min(ok, key=lambda e: (-abs(e["rho"]), e["param"], e["covariate"]))
        out.append({"cid": s, "param": b["param"], "covariate": b["covariate"], "direction": "+" if b["slope"] > 0 else "-",
                    "evidence": {k: b[k] for k in ("slope", "p", "n", "rho")}})
    return out


def hypothesis(sel: dict) -> dict:
    up, k, p = sel["direction"] == "+", sel["covariate"], sel["param"]
    ev = sel["evidence"]
    what = QUANT_MEANING.get(p, f"常数 {p}")
    return {"statement": f"骨架 {sel['cid']} 的{what}随{laws.COV_MEANING[k]}增大而{'增大' if up else '减小'}"
                         f"（发现城证据：OLS 斜率 {ev['slope']:.3g}，Spearman ρ {ev['rho']:.2f}，n = {ev['n']}）。",
            "skeleton": sel["cid"], "parameter": p, "covariate": k, "direction": sel["direction"], "test": "slope_ci",
            "criterion": f"以 {p}（{QUANT_DEF.get(p, '拟合常数，只用可识别的城：|值| > 2 SE 且多起点 SD < SE')}）为响应、"
                         f"{k} 为唯一预测变量做 OLS；检验城的城市 bootstrap（B = {laws.TEST_B}）95% CI 不含 0 且斜率"
                         f"{'> 0' if up else '< 0'}；Holm 按累计登记的全部假设（H1、H2、H3 与本轮）校正，α = 0.05。"}


def register_cycle4(hyps: list, existing: list, log) -> list:
    """The new cycle-4 entries (id H4-<k>, sha laws._hsha, source, time); a (skeleton, parameter, covariate) already
    registered -> ValueError."""
    have = {(h["skeleton"], h["parameter"], h["covariate"]) for h in existing}
    k0 = sum(1 for h in existing if str(h.get("id", "")).startswith(f"H{CYCLE}-"))
    new = []
    for h in hyps:
        key = (h["skeleton"], h["parameter"], h["covariate"])
        if key in have:
            raise ValueError(f"already registered: {key}")
        have.add(key)
        new.append({**{f: h[f] for f in laws.FIELDS}, "id": f"H{CYCLE}-{k0 + len(new) + 1:02d}", "cycle": CYCLE,
                    "sha": laws._hsha(h), "source": SOURCE, "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"Cycle-{CYCLE} hypotheses (2026-11-10 decision) registered before any lawtest data was read: "
        + "; ".join(f"{h['id']} sha {h['sha']} [{h['skeleton']} {h['parameter']} ~ {h['covariate']} ({h['direction']}) "
                    f"by {h['test']}]" for h in new))
    return new


def require_registered() -> list:
    reg = _read(LAWS / "registered_v3.json", [])
    if not reg:
        sys.exit("register the cycle-4 hypotheses first: no lawtest data is read before registration")
    return reg


def uncovered_clean(cv, ids, refs, flags, m) -> list:
    import sensitivity as sv
    nn = sv.recompute_near(cv[cv["cid"].isin(ids)], refs, m)
    ok = {r.city for r in nn[nn.near_best].itertuples() if scoring.n_flags(flags.get(r.cid)) == 0
          and scoring.passes_floor(r.val_nll, refs.get(r.city))}
    return [c for c in sorted(set(cv["city"])) if c not in ok]


def _band_inputs(directory: Path, cities):
    bd = {c: json.loads(p.read_text()) for c in cities if (p := directory / f"{c}.json").exists()}
    refs_b = {c: prefreeze._intkeys(b["refs"]) for c, b in bd.items()}
    return bd, refs_b, {c: prefreeze._intkeys(b["visits"]) for c, b in bd.items()}


def _quantity_table(cv, pool, cids, band_dir, cities):
    """Constants (laws.param_table) + profile quantities of cids on the given cities -> (pt, profile failures)."""
    sub = cv[cv["cid"].isin(cids)]
    pt_c = laws.param_table(sub, list(pool.values()))
    _, refs_b, visits = _band_inputs(band_dir, cities)
    skel, failed = prefreeze.skeleton_profiles(sub, pool, refs_b)
    q = {}
    for (cid, city), prof in skel.items():
        q.setdefault(cid, {})[city] = skel_quantities(prof, refs_b[city], visits.get(city, {}))
    parts = [pt_c] + [band_rows(c, q[c]) for c in q]
    return pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pt_c, failed


def _rho(pt, cov, e):
    from scipy.stats import spearmanr
    g = pt[(pt["cid"] == e["cid"]) & (pt["param"] == e["param"]) & pt["identifiable"].astype(bool)]
    ci = cov.set_index("city")
    g = g[g["city"].isin(ci.index)]
    z, y = ci.loc[g["city"], e["covariate"]].to_numpy(float), g["value"].to_numpy(float)
    ok = np.isfinite(z) & np.isfinite(y)
    return float(spearmanr(z[ok], y[ok])[0]) if ok.sum() >= 3 else None


def _summary(pt):
    out = {}
    for (cid, p), g in pt[pt["identifiable"].astype(bool)].groupby(["cid", "param"]):
        v = g["value"].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v):
            m = float(np.mean(v))
            out.setdefault(cid, {})[p] = {"n": int(len(v)), "median": float(np.median(v)), "q10": float(np.percentile(v, 10)),
                                          "q90": float(np.percentile(v, 90)),
                                          "cv": float(np.std(v) / abs(m)) if m != 0 else None}
    return out


# ================================================================================================ CLI stages
def _inputs():
    gate = [g["cid"] for g in json.loads((MV / "gates.json").read_text())["variants"][VARIANT]["gate"]]
    cv = pd.read_parquet(MV / "discovery" / "cities_val.parquet")
    pool = {x["cid"]: x for x in json.loads((MV / "loop" / "pool.json").read_text())}
    flags = json.loads((MV / "loop" / "flags.json").read_text())
    return gate, cv, pool, flags


def stage_shape():
    import ledger
    gate, cv, pool, _ = _inputs()
    bd, refs_b, visits = _band_inputs(PF / "banded", prefreeze.cities())
    flex = {c: prefreeze._intkeys(b["profile"]) for c, b in bd.items()}
    dref = {c: {k: r["d"] for k, r in rb.items() if r} for c, rb in refs_b.items()}
    skel, failed = prefreeze.skeleton_profiles(cv[cv["cid"].isin(gate)], pool, refs_b)
    df = prefreeze.compare(skel, flex, visits, dref)
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / "shape.parquet"
    df.to_parquet(p, index=False)
    cl = prefreeze.closest(df)
    ledger.add(f"V3 library (2026-11-10): shape comparison of the {len(gate)} V3 gate members on "
               f"{df['city'].nunique()} discovery cities ({failed} profiles not evaluable); closest e_d "
               f"{dict(sorted(cl['rmse_d'].items(), key=lambda t: -t[1])[:3])}, closest e_N shape "
               f"{dict(sorted(cl['rmse_N_shape'].items(), key=lambda t: -t[1])[:3])}; shape.parquet sha {ledger.sha(p)}")
    return 0


def stage_params():
    import ledger
    gate, cv, pool, flags = _inputs()
    cov_mv = json.loads((MV / "discovery" / "coverage.json").read_text())["variants"]
    library = library_from_coverage(cov_mv)
    cids = list(dict.fromkeys(library + list(H4_SKELETONS)))
    cities = prefreeze.cities()
    pt, failed = _quantity_table(cv, pool, cids, PF / "banded", cities)
    covdf = laws.covariates(cities)
    ev = laws.cross_city_summary(pt, covdf, B=EVID_B)
    for e in ev:
        e["rho"] = _rho(pt, covdf, e)
    refs = prefreeze._val_refs(sorted(set(cv["city"])))
    unc = {f"{m:g}": uncovered_clean(cv, gate, refs, flags, m) for m in (0.03, 0.05)}
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / "params.json"
    p.write_text(json.dumps({"library": library, "gate": gate, "h4_skeletons": list(H4_SKELETONS),
                             "summary": _summary(pt), "evidence": ev, "uncovered": unc, "n_profile_failed": failed},
                            ensure_ascii=False, indent=1, default=float))
    ledger.add(f"V3 library (2026-11-10) {library} (clean greedy picks of the V3 gate at 0.03 then 0.05); "
               f"parameter regularities on {len(cities)} discovery cities: {len(pt)} quantity rows, {len(ev)} "
               f"quantity x covariate associations (B {EVID_B}); uncovered clean cities 0.03 {len(unc['0.03'])}, 0.05 "
               f"{len(unc['0.05'])}; params.json sha {ledger.sha(p)}")
    return 0


def stage_register():
    import ledger
    if (ROOT / "lawtest").exists() or (LAWS / "results_v3.json").exists():
        sys.exit("lawtest outputs of cycle 4 exist: registration is closed")
    par = json.loads((ROOT / "params.json").read_text())
    sel = select(par["evidence"], H4_SKELETONS)
    hyps = [hypothesis(s) for s in sel]
    p = LAWS / "registered_v3.json"
    existing = _read(LAWS / "registered.json", []) + _read(LAWS / "registered_extra.json", []) + _read(p, [])
    new = register_cycle4(hyps, existing, log=ledger.add)
    p.write_text(json.dumps(_read(p, []) + new, ensure_ascii=False, indent=1))
    none = [s for s in H4_SKELETONS if s not in {x["cid"] for x in sel}]
    ledger.add(f"V3 rule (2026-11-10), cycle 4: {len(new)} hypotheses registered from discovery evidence (params.json sha "
               f"{ledger.sha(ROOT / 'params.json')}; rule n >= {MIN_N}, p < {ALPHA_SEL}, max |rho|); skeletons without a "
               f"qualifying association: {none}; registered_v3.json sha {ledger.sha(p)}")
    return 0


def stage_lawtest():
    import os
    import fit_library
    import ledger
    reg = require_registered()
    _, _, pool, _ = _inputs()
    law = config.cities_with_role("lawtest")
    cids = list(dict.fromkeys(h["skeleton"] for h in reg))
    out = ROOT / "lawtest"
    fit_library.main(law, procs=int(os.environ.get("LOOP_OUTER_PROCS", "10")), cands=[pool[c] for c in cids], out_dir=out)
    cv = pd.read_parquet(out / "cities_val.parquet")
    ledger.add(f"V3 rule (2026-11-10), cycle 4: {cids} fitted on {cv['city'].nunique()} lawtest cities "
               f"(finite rows {int(cv['finite'].astype(bool).sum())} of {len(cv)}); cities_val.parquet sha "
               f"{ledger.sha(out / 'cities_val.parquet')}")
    return 0


def stage_test():
    import ledger
    import lawtest_extra as lx
    reg = require_registered()
    _, _, pool, _ = _inputs()
    law = config.cities_with_role("lawtest")
    cv = pd.read_parquet(ROOT / "lawtest" / "cities_val.parquet")
    pt, failed = _quantity_table(cv, pool, list(dict.fromkeys(h["skeleton"] for h in reg)), PF / "lawtest_banded", law)
    cov = laws.covariates(law)
    res = [{"id": h["id"], "skeleton": h["skeleton"], "parameter": h["parameter"], "covariate": h["covariate"],
            "direction": h["direction"], "test": h["test"], **laws.run_test(h, pt, cov, None)} for h in reg]
    prev = _read(LAWS / "results.json", []) + _read(LAWS / "results_extra.json", [])
    res = lx.holm_cumulative(prev, res)
    p = LAWS / "results_v3.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    ledger.add(f"V3 rule (2026-11-10), cycle-4 hypotheses on the lawtest cities (Holm over m = {res[0]['m'] if res else 0}; "
               f"{failed} profiles not evaluable): "
               + "; ".join(f"{r['id']} {'PASS' if r['pass'] else 'fail'} (stat {r['stat']}, CI [{r['ci_lo']}, {r['ci_hi']}], "
                           f"p {r['p']:.4f}, n {r['n']})" for r in res)
               + f"; earlier laws still passing at the larger m: {res[0]['prev_still_pass'] if res else {}}"
               + f"; results_v3.json sha {ledger.sha(p)}")
    return 0


def _f(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def build_report(out_md: Path, root: Path = None, laws_dir: Path = None) -> dict:
    import report
    root, laws_dir = Path(root) if root else ROOT, Path(laws_dir) if laws_dir else LAWS
    shape = pd.read_parquet(root / "shape.parquet")
    par = json.loads((root / "params.json").read_text())
    reg, res = _read(laws_dir / "registered_v3.json", []), {r["id"]: r for r in _read(laws_dir / "results_v3.json", [])}
    lib = par["library"]
    L = ["# Skeleton library and cycle-4 regularity hypotheses under the V3 rule", "",
         "V3 = near-best margin 0.05 for the search-city gate, equivalence-merge margin 0.03, no chaining (decision of 2026-11-10). "
         "The library and the parameter regularities use search and discovery cities only; "
         "the cycle-4 hypotheses were registered before the test cities were read, and test results are reported as aggregates only.", "",
         "## 1. Skeleton library", "",
         f"- V3 gate: {len(par['gate'])} candidates; library (greedy picks among clean candidates in the gate, margin 0.03 then 0.05, in order of first appearance): {', '.join(lib)}"]
    for m in ("0.03", "0.05"):
        u = par["uncovered"].get(m, [])
        L.append(f"- Margin {m}, cities not covered by clean candidates alone ({len(u)}): {', '.join(u) or 'none'}")
    L += ["", "## 2. Elasticity profiles against the flexible distance-band model", ""]
    cl = prefreeze.closest(shape)
    rows = []
    for c, g in shape.groupby("cid"):
        g = g[np.isfinite(g["rmse_N"].astype(float))]
        rows.append([c, len(g), _f(prefreeze._median(g["rmse_d"].tolist())), _f(prefreeze._median(g["rmse_N_shape"].tolist())),
                     _f(prefreeze._median(g["slope_N_skel"].tolist())), cl["rmse_d"].get(c, 0), cl["rmse_N_shape"].get(c, 0)])
    rows.sort(key=lambda r: -(r[5] + r[6]))
    L += prefreeze._md_table(["cid", "Comparable cities", "Median rmse_d", "Median rmse_N shape", "Median slope of e_N on ln d",
                              "Cities with closest e_d", "Cities with closest e_N shape"], rows)
    L += ["", "## 3. Parameter regularities of the library (discovery cities)", ""]
    ev = par["evidence"]
    for c in lib + [s for s in par.get("h4_skeletons", []) if s not in lib]:
        L += [f"### {c}", ""]
        s = par["summary"].get(c, {})
        L += prefreeze._md_table(["Quantity", "Usable cities", "Median", "10%–90%", "Coefficient of variation"],
                                 [[p, v["n"], _f(v["median"]), f"{_f(v['q10'])} – {_f(v['q90'])}", _f(v["cv"], 2)]
                                  for p, v in s.items()]) if s else ["(no identifiable quantity)"]
        top = sorted([e for e in ev if e["cid"] == c and e["p"] < ALPHA_SEL and e["n"] >= MIN_N and e.get("rho") is not None],
                     key=lambda e: -abs(e["rho"]))[:3]
        L += [""] + ([f"- {e['param']} ~ {e['covariate']}: slope {_f(e['slope'])}, ρ {_f(e['rho'], 2)}, bootstrap p {_f(e['p'], 4)}, "
                      f"n {e['n']}" for e in top] or ["- no association with n ≥ 20 and p < 0.05"]) + [""]
    L += ["## 4. Cycle-4 regularity hypotheses (registered on discovery cities, tested on test cities)", ""]
    if reg:
        L += prefreeze._md_table(["ID", "Skeleton", "Quantity", "City covariate", "Direction", "Statement"],
                                 [[h["id"], h["skeleton"], h["parameter"], h["covariate"], h["direction"], h["statement"]] for h in reg])
        L += [""]
        if res:
            L += prefreeze._md_table(["ID", "Test-city slope", "95% CI", "p", "Cities", "Cumulative hypotheses m", "Pass after Holm"],
                                     [[i, _f(r.get("stat")), f"[{_f(r.get('ci_lo'))}, {_f(r.get('ci_hi'))}]", _f(r.get("p"), 4),
                                       r.get("n"), r.get("m"), "✔" if r.get("pass") else "✘"] for i, r in res.items()])
            prev = next(iter(res.values())).get("prev_still_pass", {})
            L += ["", f"- Earlier passed laws still passing at the larger m: {', '.join(f'{k} {chr(10004) if v else chr(10008)}' for k, v in prev.items()) or '—'}"]
        else:
            L.append("- not yet tested")
    else:
        L.append("- no registered hypotheses")
    text = "\n".join(L) + "\n"
    try:
        report._scan(text)
    except ValueError as e:
        raise RuntimeError(f"V3 library report refused: {e}") from e
    Path(out_md).write_text(text)
    return {"library": len(lib), "registered": len(reg), "passed": sum(1 for r in res.values() if r.get("pass"))}


def stage_report():
    import ledger
    p = ROOT / "report.md"
    s = build_report(p)
    ledger.add(f"V3 library (2026-11-10) report: {s}; report.md sha {ledger.sha(p)}")
    return 0


STAGES = {"shape": stage_shape, "params": stage_params, "register": stage_register, "lawtest": stage_lawtest,
          "test": stage_test, "report": stage_report}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: v3_library.py {{{'|'.join(STAGES)}}}")
    sys.exit(STAGES[sys.argv[1]]())
