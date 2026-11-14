"""Plan 9, POST-HOC EXPLORATION (added 2026-11-11, after the contrasts were written): where do size-predicted
constants help? Not a test: the question came from the results (size-predicted constants beat uniform ones in 29/35
source cities on the validation block but 18/32 target cities on the test block), and the source cities are the ones
the skeletons were selected on.

  score     the 35 source cities re-scored on the TEST block (test period x destination fold 0), leave-one-city-out:
            the same four parameter sources as plan 9. Nothing is refitted — the constants are the training-block
            fits of fits_source.parquet and of the frozen skeleton fits; only the scoring block changes. This puts
            all 67 cities on one block and separates "another block" from "other cities".
  analyze   for the primary skeleton, gain = share(size-predicted) - share(uniform) per city:
            (a) the source cities' gain on the validation block (pre-run) against the test block, paired;
            (b) on the 67 cities x test block, gain against dist = |log population - median log population of the
                source cities|: Spearman with a city-bootstrap interval, Theil-Sen slope, terciles of dist, city roles;
            (c) for context, the same win counts for the other skeletons and the opponents.
            These definitions were written here before the source cities' test-block scores were computed.

The pinned plan-9 modules (p9.CODE) and outputs are not touched: this module only reads them and writes
results/multicity/p9/explore/. It refuses to run before contrasts.json exists.

Usage (run_stage.sh, M line): p9_explore.py score | analyze
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import p9  # noqa: E402
import p9_stats as st  # noqa: E402
import param_transfer as pt  # noqa: E402
import prior_models as pm  # noqa: E402

B, SEED = 2000, 20261111
BINS = ("near", "mid", "far")


def out_dir() -> Path:
    return p9.OUT / "explore"


def search_cities() -> list:
    return [str(c) for c in config.cities_with_role("search")]


def _require_done() -> dict:
    cfg = p9.require_fixed()
    if not (p9.OUT / "contrasts.json").exists():
        raise RuntimeError("the plan-9 contrasts are not written yet: the exploration comes after them")
    return cfg


# --------------------------------------------------------------------------------------------- stage: score
def _score(model, kind, consts, ev, A) -> dict:
    if consts is None:
        return {"nll": np.nan, "cpc": np.nan}
    if kind == "mat":
        return pm.score_mat(pm.mat_eta(model, list(consts), A, A["io"]["ev"]), A["T"]["ev"], A["av"]["ev"])
    if kind == "eq":
        return pm.score_eq(pm.EQ_MODELS[model].equation, consts, ev)
    return pm.score_cand(p9._G["cands"][model], consts, ev)


def score_city(city: str) -> tuple:
    """(local rows, transfer rows) of a SOURCE city on the test block; the city itself is never among its donors."""
    G = p9._G
    cfg = G["cfg"]
    if city not in set(p9.cities("source")):
        raise RuntimeError(f"{city} is not a source city")
    D = p9.city_inputs(city, "test_dest")
    ev, A = D["ev"], D["A"]
    p9.check_paths(city, ev, A)
    ra = cfg["ridge_alphas"]
    local, transfer = [], []
    for model, kind, n_free, consts in G["local"][city]:
        local.append({"city": city, "model": model, "kind": kind, "n_free": n_free,
                      "consts": None if consts is None else json.dumps(consts), **_score(model, kind, consts, ev, A)})
    for (model, kind), Km in G["K"].items():
        lv = pt.predict(Km, G["cov"], city, cfg["size_covariate"], cfg["form_covariates"], cfg["min_donors"],
                        np.logspace(ra["log10_lo"], ra["log10_hi"], ra["n"]), tuple(cfg["clip_quantiles"]))
        for level in pt.LEVELS[1:]:
            k = lv.get(level)
            transfer.append({"city": city, "model": model, "kind": kind, "level": level,
                             "consts": None if k is None else json.dumps([float(v) for v in k]),
                             **_score(model, kind, k, ev, A)})
    print(city, "test block scored", flush=True)
    return local, transfer


def stage_score():
    import ledger
    cfg = _require_done()
    source = p9.cities("source")
    src = pd.read_parquet(p9.OUT / "fits_source.parquet")
    ours = p9.ours_fits("test_dest")
    p9.prepare("test_dest")
    p9.prepare("test_dest", K=p9.donors(src, ours, source))
    loc = {c: [] for c in source}
    for r in src.itertuples():
        if r.kind != "ours" and r.city in loc:
            loc[r.city].append((r.model, r.kind, int(r.n_free), json.loads(r.consts) if r.consts else None))
    for cid in cfg["ours"]["reported"]:
        for c in source:
            f = ours.get(cid, {}).get(c)
            loc[c].append((cid, "ours", len(p9._G["cands"][cid].consts), f["consts"] if f else None))
    p9._G["local"] = loc
    res = p9._pool_map(score_city, source)
    local = pd.DataFrame([r for a, _ in res for r in a])
    t = p9.scored(p9.assemble(local, pd.DataFrame([r for _, b in res for r in b])), "test_dest")
    t = t.sort_values(["model", "city", "level"]).reset_index(drop=True)
    out_dir().mkdir(parents=True, exist_ok=True)
    t.to_parquet(out_dir() / "source_test.parquet", index=False)
    ledger.add(f"Plan 9 exploration (post hoc, 2026-11-11): {t['city'].nunique()} source cities re-scored on the test "
               f"block, leave-one-city-out, no refit; explore/source_test.parquet sha "
               f"{ledger.sha(out_dir() / 'source_test.parquet')}")


# --------------------------------------------------------------------------------------------- analysis
def _wide(t: pd.DataFrame, model: str, col: str) -> pd.DataFrame:
    return t[t["model"] == model].pivot(index="city", columns="level", values=col)


def gain_table(frames: dict, cov: pd.DataFrame, model: str, source: list, roles: dict) -> pd.DataFrame:
    """One row per city of the frames {set: scored rows}: set, role (roles[city] or the set), logpop, dist = distance
    of log population from the source cities' median, gain = share(size) - share(uniform), gain_nll, loss_uniform =
    share(local) - share(uniform), loss_size. Cities where the uniform or the size level failed are dropped and
    listed in .attrs["dropped"]."""
    med = float(np.median(cov.loc[list(source), "log_population"].astype(float)))
    parts, dropped = [], {}
    for name, t in frames.items():
        sh, nl = _wide(t, model, "share"), _wide(t, model, "nll")
        ok = np.isfinite(sh["uniform"]) & np.isfinite(sh["size"])
        if (~ok).any():
            dropped[name] = sorted(sh.index[~ok])
        sh, nl = sh[ok], nl[ok]
        lp = cov.loc[sh.index, "log_population"].astype(float).to_numpy()
        with np.errstate(invalid="ignore"):
            loc = np.where(np.isfinite(sh["local"]), sh["local"], np.nan)
        parts.append(pd.DataFrame({"set": name, "role": [roles.get(c, name) for c in sh.index], "logpop": lp,
                                   "dist": np.abs(lp - med), "gain": (sh["size"] - sh["uniform"]).to_numpy(),
                                   "gain_nll": (nl["uniform"] - nl["size"]).to_numpy(),
                                   "loss_uniform": loc - sh["uniform"].to_numpy(),
                                   "loss_size": loc - sh["size"].to_numpy()}, index=sh.index))
    g = pd.concat(parts)
    g.attrs["dropped"] = dropped
    return g


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y)[0])


def dose_response(g: pd.DataFrame, B: int = B, seed: int = SEED) -> dict:
    """gain against dist over the cities of g: Spearman with a city-bootstrap 95% interval, Theil-Sen slope, and the
    terciles of dist (near / mid / far) with the win counts and medians."""
    from scipy.stats import theilslopes
    x, y = g["dist"].to_numpy(float), g["gain"].to_numpy(float)
    n = len(g)
    rng = np.random.default_rng(seed)
    bs = np.array([_spearman(x[i], y[i]) for i in rng.integers(0, n, (B, n))])
    bs = bs[np.isfinite(bs)]
    bins = pd.qcut(g["dist"].rank(method="first"), 3, labels=list(BINS))
    ter = []
    for b in BINS:
        d = g[bins == b]
        ter.append({"bin": b, "n": int(len(d)), "dist_max": float(d["dist"].max()), "wins": int((d["gain"] > 0).sum()),
                    "gain_med": float(d["gain"].median()), "gain_nll_med": float(d["gain_nll"].median()),
                    "loss_uniform_med": float(d["loss_uniform"].median()), "loss_size_med": float(d["loss_size"].median()),
                    "by_set": {s: {"n": int(len(v)), "wins": int((v["gain"] > 0).sum()), "gain_med": float(v["gain"].median())}
                               for s, v in d.groupby("set")}})
    return {"n": int(n), "spearman": _spearman(x, y), "ci_lo": float(np.quantile(bs, 0.025)),
            "ci_hi": float(np.quantile(bs, 0.975)), "slope": float(theilslopes(y, x)[0]), "terciles": ter}


def block_comparison(val: pd.DataFrame, test: pd.DataFrame, model: str) -> dict:
    """The same cities on two blocks: gain = share(size) - share(uniform), paired over the cities finite in both."""
    a, b = _wide(val, model, "share"), _wide(test, model, "share")
    ga, gb = (a["size"] - a["uniform"]), (b["size"] - b["uniform"])
    cs = [c for c in ga.index if c in gb.index and np.isfinite(ga[c]) and np.isfinite(gb[c])]
    ga, gb = ga[cs], gb[cs]
    return {"n": len(cs), "wins_val": int((ga > 0).sum()), "wins_test": int((gb > 0).sum()),
            "both": int(((ga > 0) & (gb > 0)).sum()), "gain_val_med": float(ga.median()), "gain_test_med": float(gb.median()),
            "spearman": _spearman(ga.to_numpy(), gb.to_numpy()) if len(cs) > 2 else None}


def _roles_table(g: pd.DataFrame) -> list:
    rows = []
    for (s, r), d in g.groupby(["set", "role"], sort=False):
        rows.append({"set": s, "role": r, "n": int(len(d)), "logpop_med": float(d["logpop"].median()),
                     "wins": int((d["gain"] > 0).sum()), "gain_med": float(d["gain"].median()),
                     "gain_nll_med": float(d["gain_nll"].median()), "loss_uniform_med": float(d["loss_uniform"].median()),
                     "loss_size_med": float(d["loss_size"].median())})
    return rows


def _f(v, nd=3) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:+.{nd}f}"


def stage_analyze():
    cfg = _require_done()
    P = cfg["ours"]["primary"]
    src_test = pd.read_parquet(out_dir() / "source_test.parquet")
    target = pd.read_parquet(p9.OUT / "target.parquet")
    prerun = pd.read_parquet(p9.OUT / "prerun.parquet")
    cov, source = p9.covariates(), p9.cities("source")
    roles = {c: "search" for c in search_cities()}
    roles.update(target.drop_duplicates("city").set_index("city")["group"].to_dict())
    g = gain_table({"source": src_test, "target": target}, cov, P, source, roles)
    opp = json.loads((p9.OUT / "opponents.json").read_text())["opponents"]
    ctx = {}
    for m in list(dict.fromkeys(cfg["ours"]["reported"] + [v["model"] for v in opp.values()])):
        try:
            gm = gain_table({"source": src_test, "target": target}, cov, m, source, roles)
            ctx[m] = {"n": int(len(gm)), "wins": int((gm["gain"] > 0).sum()), "gain_med": float(gm["gain"].median()),
                      "spearman": _spearman(gm["dist"], gm["gain"]),
                      "by_set": {s: {"n": int(len(v)), "wins": int((v["gain"] > 0).sum())} for s, v in gm.groupby("set")}}
        except KeyError:
            continue
    both = pd.concat([src_test.assign(set="source"), target.assign(set="target")])
    lev = {s: {lv: st._quantile(d.loc[(d["model"] == P) & (d["level"] == lv), "share"], 0.5) for lv in pt.LEVELS}
           for s, d in both.groupby("set")}
    res = {"model": P, "dropped": g.attrs["dropped"], "block": block_comparison(prerun, src_test, P),
           "roles": _roles_table(g), "dose_all": dose_response(g),
           "dose_source": dose_response(g[g["set"] == "source"]), "dose_target": dose_response(g[g["set"] == "target"]),
           "context": ctx, "share_median": lev, "cities": g.reset_index().to_dict("records")}
    (out_dir() / "c3_explore.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    b, d = res["block"], res["dose_all"]
    L = [f"# Plan 9 post-hoc exploration: in which cities do size-predicted constants help ({P})", "",
         "This is exploration after seeing the results, not a test: the question comes from the results themselves, "
         "and the source cities are the cities the skeletons were selected on.", "",
         "Gain = share (size-predicted) − share (uniform constants); distance = |log population − median log population "
         "of the source cities|.", "",
         "## 1. The same source cities, another scoring block", "",
         f"- Validation block (pre-run): better in {b['wins_val']}/{b['n']} cities, median gain "
         f"{_f(b['gain_val_med'], 4)};",
         f"- Test block (this run): better in {b['wins_test']}/{b['n']} cities, median gain "
         f"{_f(b['gain_test_med'], 4)};",
         f"- Better on both blocks: {b['both']} cities; Spearman correlation between the gains on the two blocks "
         f"{_f(b['spearman'], 2)}.", "",
         "## 2. 67 cities on one scoring block (test block), by city group", "",
         "| Set | Group | Cities | Median log population | Cities better | Median gain | Gain (nats) | Loss of uniform "
         "constants | Loss of size-predicted |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res["roles"]:
        L.append(f"| {r['set']} | {r['role']} | {r['n']} | {r['logpop_med']:.2f} | {r['wins']}/{r['n']} | {_f(r['gain_med'], 4)} | "
                 f"{_f(r['gain_nll_med'], 4)} | {_f(r['loss_uniform_med'], 4)} | {_f(r['loss_size_med'], 4)} |")
    L += ["", "## 3. Gain and distance", "",
          "| Cities | n | Spearman [95% CI] | Theil–Sen slope |", "|---|---|---|---|"]
    for name, k in (("67 cities", "dose_all"), ("Source cities", "dose_source"), ("Target cities", "dose_target")):
        r = res[k]
        L.append(f"| {name} | {r['n']} | {_f(r['spearman'], 2)} [{_f(r['ci_lo'], 2)}, {_f(r['ci_hi'], 2)}] | {_f(r['slope'], 4)} |")
    L += ["", "Terciles of distance (67 cities):", "",
          "| Tercile | Cities | Cities better | Median gain | Loss of uniform constants | Loss of size-predicted | "
          "Source cities: better / n, gain | Target cities: better / n, gain |", "|---|---|---|---|---|---|---|---|"]
    for t in d["terciles"]:
        cells = [f"{v['wins']}/{v['n']}, {_f(v['gain_med'], 4)}" if v else "—" for v in (t["by_set"].get("source"), t["by_set"].get("target"))]
        L.append(f"| {t['bin']} | {t['n']} | {t['wins']}/{t['n']} | {_f(t['gain_med'], 4)} | {_f(t['loss_uniform_med'], 4)} | "
                 f"{_f(t['loss_size_med'], 4)} | " + " | ".join(cells) + " |")
    L += ["", "## 4. Other models (67 cities, test block)", "", "| Model | Cities | Cities where size-predicted is better | "
          "Median gain | Gain ~ distance (Spearman) | Source cities | Target cities |", "|---|---|---|---|---|---|---|"]
    for m, r in ctx.items():
        bs = r["by_set"]
        L.append(f"| {m} | {r['n']} | {r['wins']}/{r['n']} | {_f(r['gain_med'], 4)} | {_f(r['spearman'], 2)} | "
                 + " | ".join(f"{bs[s]['wins']}/{bs[s]['n']}" if s in bs else "—" for s in ("source", "target")) + " |")
    level_en = {"local": "Local refit", "uniform": "Uniform constants", "size": "Size-predicted", "form": "Size + form"}
    L += ["", f"## 5. Median share of {P} (test block)", "", "| Set | " + " | ".join(level_en[lv] for lv in pt.LEVELS) + " |",
          "|---|" + "---|" * len(pt.LEVELS)]
    L += [f"| {s} | " + " | ".join(f"{v[lv]:.3f}" for lv in pt.LEVELS) + " |" for s, v in lev.items()]
    if res["dropped"]:
        L += ["", f"Cities not in the gain table (the uniform or the size-predicted level failed): {res['dropped']}"]
    (out_dir() / "c3_explore.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


STAGES = {"score": stage_score, "analyze": stage_analyze}


def main(argv: list):
    if len(argv) == 1 and argv[0] in STAGES:
        return STAGES[argv[0]]()
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
