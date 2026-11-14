"""LLM stages of the orchestrator (plan 2 Task 5/8), in two phases so the slow parts run in parallel:

propose(point, ...): one call point (simplify or readability) over the whole pool, in `threads` threads (the local
  LLM allows 2 concurrent requests), resumable — a candidate whose output file exists is not called again; a failed
  call writes {"result": null, "error": ...} and never raises. Simplification proposals are written with
  accepted = null.
verify_pending(...): every simplification proposal still unverified is checked by clean.verify_simplification on the
  first search city where the candidate is fittable (n_orig sampled training origins, every validation origin), in a
  fork pool; the file gains accepted, verify (without the candidate object) and verify_city.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import llm  # noqa: E402
import llm_points  # noqa: E402
from library import LIB_N_ORIG, candidate  # noqa: E402
from refit_select import XS  # noqa: E402


def payload(x, clean_json, fits, point):
    import clean as cl
    c = candidate(x)
    eq = (clean_json.get(x["cid"]) or {}).get("canonical_eta") or cl.eta_string(c)
    used = sorted({f"x{k + 1}" for k in range(len(XS)) if any(XS[k] in e.free_symbols for e in c.comps.values())},
                  key=lambda v: int(v[1:]))
    p = {"cid": x["cid"], "equation": eq,
         "variables": {v: f"{llm_points.XVAR[v]}：{llm_points.VAR_MEANING[llm_points.XVAR[v]]}" for v in used}}
    if point == "simplify":
        f = fits[(fits["cid"] == x["cid"]) & fits["finite"].astype(bool)]
        p["fits"] = [{"city": r.city, "val_nll": round(float(r.val_nll), 4), "within_1se": bool(r.within_1se)}
                     for r in f.itertuples()]
    return p


def _propose_one(args):
    point, x, clean_json, fits, out_dir = args
    target = out_dir / f"{point}_{x['cid']}.json"
    if target.exists():
        return x["cid"], "exists"
    inp = out_dir / f"{point}_{x['cid']}.in.json"
    inp.write_text(json.dumps(payload(x, clean_json, fits, point), ensure_ascii=False, indent=1))
    fn = llm_points.simplify if point == "simplify" else llm_points.readability
    try:
        llm.run_point(point, inp, target, fn)
        return x["cid"], "ok"
    except llm.LLMError as e:
        target.write_text(json.dumps({"point": point, "input": inp.name, "result": None, "error": str(e)}, ensure_ascii=False))
        return x["cid"], "error"


def propose(point, pool, clean_json, fits, out_dir, threads=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(point, x, clean_json, fits, out_dir) for x in pool]
    if threads == 1:
        res = list(map(_propose_one, jobs))
    else:
        with ThreadPoolExecutor(threads) as ex:
            res = list(ex.map(_propose_one, jobs))
    for cid, status in res:
        print(point, cid, status, flush=True)
    return dict(res)


_V = {}


def _verify_one(args):
    import clean as cl
    from library import city_data
    path, x, city, n_orig = args
    if _V.get("key") != (city, n_orig):
        _V.clear()
        _V.update(key=(city, n_orig), D=city_data(city, n_orig))
    d = json.loads(Path(path).read_text())
    r = d["result"]
    v = cl.verify_simplification(candidate(x), r["sympy"], _V["D"]["tr"], _V["D"]["va"])
    r["accepted"] = bool(v["accepted"])
    r["verify"] = {k: val for k, val in v.items() if k != "candidate"}
    r["verify_city"] = city
    Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print("verify", x["cid"], city, r["accepted"], v["reason"], flush=True)
    return x["cid"], r["accepted"]


def verify_pending(pool, fits, cities, out_dir, procs=1, n_orig=LIB_N_ORIG):
    out_dir = Path(out_dir)
    jobs, no_city = [], []
    for x in pool:
        p = out_dir / f"simplify_{x['cid']}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        r = d.get("result")
        if not r or r.get("accepted") is not None:
            continue
        fin = set(fits[(fits["cid"] == x["cid"]) & fits["finite"].astype(bool)]["city"])
        city = next((c for c in cities if c in fin), None)
        if city is None:
            r.update(accepted=False, verify={"reason": "not fittable in any search city"}, verify_city=None)
            p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
            no_city.append(x["cid"])
            continue
        jobs.append((str(p), x, city, n_orig))
    jobs.sort(key=lambda j: j[2])                       # one city after another: a worker rebuilds only on a change
    if procs == 1 or len(jobs) <= 1:
        res = list(map(_verify_one, jobs))
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(procs, len(jobs))) as P:
            res = list(P.imap_unordered(_verify_one, jobs, chunksize=1))
    return dict(res, **{c: False for c in no_city})
