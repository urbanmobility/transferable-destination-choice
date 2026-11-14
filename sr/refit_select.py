"""Export, constant refit and 1-SE selection of PySR candidates (plan Task 4).

A PySR template equation string ("f = ...; g = ...; h = ...; p1 = [..]; p2 = [..]", #k = k-th
variable) is parsed into sympy; every numeric literal and every template parameter becomes a free
constant. eta = f + p1[cat] g + p2[cat] h (+ offset_B for variant B). Constants are refitted by
L-BFGS on the grouped-softmax per-visit NLL with analytic gradients; non-finite predictions give a
1e9 penalty.
select(): pools the Pareto fronts of several runs, removes algebraic duplicates, keeps at most two
per complexity and 20 in total (lowest training loss first), refits on train x search destinations
x all origins, scores on val x validation destinations, and applies the 1-SE rule: the simplest
candidate whose validation per-visit NLL exceeds the best by no more than the device-group bootstrap
SD (B = 200) of that paired difference.
Usage: python -I refit_select.py SCALE VARIANT RUN_DIR [RUN_DIR ...]
Env: SELECT_OUT (output name), SELECT_PROCS, SELECT_NOCAP=1 (no 20 / per-complexity caps: every distinct
front member is refitted), SELECT_SNAPSHOTS=1 (also pool the saved earlier fronts runs/<tag>/snapshots/),
SELECT_REGIONS (default LA,SFBAY), SELECT_CACHE (comma list of earlier select outputs whose refits are reused),
SR_TRAIN_BLOCK / SR_EVAL_SPLIT / SR_EVAL_BLOCK (default search / val / val; agent loop: search23 / train / fold:4).
"""
import dataclasses
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import sympy
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pysr_setup import PENALTY, design_matrix, n_categories  # noqa: E402
from sr_data import LINE, ROOT, VARS, build  # noqa: E402

XS = sympy.symbols([f"x{k + 1}" for k in range(len(VARS))])
NUM = re.compile(r"(?<![\w.])(\d+\.\d*(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)")
COMPS = ("f", "g", "h")


@dataclass
class Candidate:
    expr: str                       # raw PySR equation string
    complexity: int
    consts: np.ndarray              # literals of f, g, h (in order) followed by p1, p2
    template: str
    variant: str
    comps: dict = field(default_factory=dict)      # name -> sympy expression in x*, c*
    n_lit: int = 0
    n_cat: int = 1


def _component(rhs, start):
    vals = []

    def sub(mt):
        vals.append(float(mt.group(1)))
        return f"c{start + len(vals) - 1}"
    s = re.sub(r"#(\d+)", r"x\1", rhs.replace("^", "**"))     # variables first, so their digits are not literals
    s = NUM.sub(sub, s)
    loc = {f"x{k + 1}": XS[k] for k in range(len(VARS))}
    loc.update({f"c{start + i}": sympy.Symbol(f"c{start + i}") for i in range(len(vals))})
    loc["nexp"] = lambda z: sympy.exp(-z)
    loc["log"] = sympy.log
    return sympy.sympify(s, locals=loc), vals


def absolute_equation(eq, active):
    """PySR prints #k as the k-th argument of a template sub-expression; the arguments are the run's active
    variables (config.json active_vars), so with a dropped variable #k is not x_k. Returns the equation with every
    #k renumbered to its position in VARS (+1); unchanged when active is None or the full VARS."""
    if not active or list(active) == list(VARS):
        return eq
    idx = [VARS.index(v) + 1 for v in active]
    return re.sub(r"#(\d+)", lambda m: f"#{idx[int(m.group(1)) - 1]}", eq)


def run_equations(rd, cfg, hof):
    """(absolute equation, PySR equation) of every row of a run's front, in order."""
    return [(absolute_equation(e, cfg.get("active_vars")), e) for e in hof["equation"]]


def parse_equation(eq, template, variant, complexity):
    parts = dict(p.split(" = ", 1) for p in eq.strip().split("; "))
    comps, consts = {}, []
    for name in COMPS:
        if name in parts:
            comps[name], vals = _component(parts[name], len(consts))
            consts += vals
    n_lit = len(consts)
    n_cat = n_categories(template)
    if "g" in comps:
        for p in ("p1", "p2"):
            consts += [float(v) for v in parts[p].strip("[] ").split(",")]
    return Candidate(expr=eq, complexity=complexity, consts=np.array(consts, float), template=template,
                     variant=variant, comps=comps, n_lit=n_lit, n_cat=n_cat)


def _compiled(c):
    key = id(c.comps)
    if getattr(c, "_key", None) != key:
        cs = sympy.symbols([f"c{i}" for i in range(max(c.n_lit, 1))])
        c._fun, c._grad = {}, {}
        for name, e in c.comps.items():
            c._fun[name] = sympy.lambdify(list(XS) + list(cs), e, "numpy")
            used = sorted(int(str(s)[1:]) for s in e.free_symbols if str(s).startswith("c"))
            c._grad[name] = [(i, sympy.lambdify(list(XS) + list(cs), sympy.diff(e, sympy.Symbol(f"c{i}")), "numpy"))
                             for i in used]
        c._key = key
    return c


def _parts(c, data, consts):
    c = _compiled(c)
    cols = [data.X[:, k].astype(np.float64) for k in range(len(VARS))]
    lit = list(consts[:c.n_lit]) if c.n_lit else [0.0]
    n = len(data.y)
    with np.errstate(all="ignore"):
        val = {k: np.broadcast_to(np.asarray(f(*cols, *lit), float), (n,)) for k, f in c._fun.items()}
    return cols, lit, val


def _cat(c, data):
    """Template category per row: the position of the row's city in the data's regions (data.cat), never the
    global CITY code (129 cities, 8 parameters)."""
    if c.template == "cityses2":
        return data.cat * 3 + data.ses
    return data.cat if re.fullmatch(r"city\d+", c.template or "") else np.zeros(len(data.y), int)


def eta(c, data, template=None, consts=None):
    consts = c.consts if consts is None else consts
    _, _, val = _parts(c, data, consts)
    e = val["f"].copy()
    if "g" in val:
        cat = _cat(c, data)
        p1 = consts[c.n_lit:c.n_lit + c.n_cat]
        p2 = consts[c.n_lit + c.n_cat:c.n_lit + 2 * c.n_cat]
        e = e + p1[cat] * val["g"] + p2[cat] * val["h"]
    if c.variant == "B":
        e = e + data.offset_B
    return e


def _grouped(e, data):
    """Per-visit NLL and the residual y - O p (both grouped by origin)."""
    st = data.offsets[:-1]
    m = np.maximum.reduceat(e, st)
    rep = np.diff(data.offsets)
    ce = e - np.repeat(m, rep)
    z = np.add.reduceat(np.exp(ce), st)
    lp = ce - np.repeat(np.log(z), rep)
    O = np.add.reduceat(data.y, st)
    V = O.sum()
    return -(data.y * lp).sum() / V, data.y - np.repeat(O, rep) * np.exp(lp), V


def boot_nll(e, data, Yg, W):
    """Per-visit NLL of eta under each device-group bootstrap replicate y_b = W[b] @ Yg, exactly and without a loop:
    the log-probabilities do not depend on the counts, so NLL_b = -(W[b] @ (Yg @ lp)) / (W[b] @ Yg.sum(1)).
    (B,) array; NaN where a replicate carries no visit; PENALTY everywhere if eta is not finite."""
    if not np.isfinite(e).all():
        return np.full(len(W), PENALTY)
    st = data.offsets[:-1]
    rep = np.diff(data.offsets)
    ce = e - np.repeat(np.maximum.reduceat(e, st), rep)
    lp = ce - np.repeat(np.log(np.add.reduceat(np.exp(ce), st)), rep)
    with np.errstate(all="ignore"):
        return -(W @ (Yg @ lp)) / (W @ Yg.sum(1))


def loss(c, data, template=None, consts=None):
    e = eta(c, data, consts=consts)
    if not np.isfinite(e).all():
        return PENALTY
    return float(_grouped(e, data)[0])


def _obj(theta, c, data):
    cols, lit, val = _parts(c, data, theta)
    e = eta(c, data, consts=theta)
    if not np.isfinite(e).all():
        return PENALTY, np.zeros_like(theta)
    nll, R, V = _grouped(e, data)
    g = np.zeros_like(theta)
    w = {"f": np.ones(len(e))}
    if "g" in val:
        cat = _cat(c, data)
        p1 = theta[c.n_lit:c.n_lit + c.n_cat]
        p2 = theta[c.n_lit + c.n_cat:c.n_lit + 2 * c.n_cat]
        w["g"], w["h"] = p1[cat], p2[cat]
        for k in range(c.n_cat):
            sel = cat == k
            g[c.n_lit + k] = -(R[sel] * val["g"][sel]).sum() / V
            g[c.n_lit + c.n_cat + k] = -(R[sel] * val["h"][sel]).sum() / V
    with np.errstate(all="ignore"):
        for name, lst in c._grad.items():
            for i, df in lst:
                d = np.broadcast_to(np.asarray(df(*cols, *lit), float), (len(e),))
                # eta is finite here, so a NaN can only be d(x^c)/dc = x^c log x at x = 0 (0 * -inf); its limit is 0
                d = np.where(np.isnan(d), 0.0, d)
                g[i] -= (R * w[name] * d).sum() / V
    if not np.isfinite(g).all():
        return PENALTY, np.zeros_like(theta)
    return nll, g


def const_cov(c, data, Yg=None, h=1e-5):
    """Covariance of the constants at c.consts (plan 4 Task 11): the inverse of the numerical Hessian of the total NLL
    (central differences of the analytic gradient); with Yg ((G, rows) device-group counts) the cluster-robust
    sandwich H^-1 (sum_g s_g s_g') H^-1 x G / (G - 1), s_g = the group's score — the linearised device-group
    bootstrap. NaN where the Hessian is singular."""
    th = np.asarray(c.consts, float)
    P = len(th)
    V = float(data.y.sum())
    steps = h * np.maximum(1.0, np.abs(th))
    H = np.zeros((P, P))
    for j in range(P):
        e = np.zeros(P)
        e[j] = steps[j]
        H[:, j] = (_obj(th + e, c, data)[1] - _obj(th - e, c, data)[1]) / (2 * steps[j])
    H = 0.5 * (H + H.T) * V
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full((P, P), np.nan)
    if Yg is None:
        return Hi
    e0 = eta(c, data, consts=th)
    J = np.zeros((len(e0), P))
    for j in range(P):
        e = np.zeros(P)
        e[j] = steps[j]
        J[:, j] = (eta(c, data, consts=th + e) - eta(c, data, consts=th - e)) / (2 * steps[j])
    st = data.offsets[:-1]
    rep = np.diff(data.offsets)
    ce = e0 - np.repeat(np.maximum.reduceat(e0, st), rep)
    p = np.exp(ce)
    p /= np.repeat(np.add.reduceat(p, st), rep)
    dl = J - np.repeat(np.add.reduceat(p[:, None] * J, st, axis=0), rep, axis=0)     # d log p / d theta per row
    S = -(np.asarray(Yg, float) @ dl)                                                 # (G, P) group scores
    G = S.shape[0]
    return Hi @ (S.T @ S) @ Hi * (G / max(G - 1, 1))


def refit(c, data, template=None, maxiter=500, free=None, return_se=False, Yg=None):
    """free: optional boolean mask of the constants to optimise (the others stay fixed).
    return_se: also return (SE of every constant, covariance) from const_cov (Yg: device-group counts -> sandwich)."""
    if len(c.consts) == 0:
        return (c, np.zeros(0), np.zeros((0, 0))) if return_se else c
    free = np.ones(len(c.consts), bool) if free is None else np.asarray(free, bool)
    base = c.consts.copy()

    def obj(th):
        full = base.copy()
        full[free] = th
        f, g = _obj(full, c, data)
        return f, g[free]
    r = minimize(obj, base[free], jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "gtol": 1e-9, "ftol": 1e-13})
    cand = base.copy()
    cand[free] = r.x
    best = cand if loss(c, data, consts=cand) <= loss(c, data) else c.consts.copy()
    out = dataclasses.replace(c, consts=best)
    if not return_se:
        return out
    cov = const_cov(out, data, Yg)
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.where(np.diag(cov) >= 0, np.diag(cov), np.nan))
    return out, se, cov


def _device_group_y(data, regions, scale, split):
    """(20, rows) counts of each device group for the rows of data."""
    from sr_data import CITY
    inv = {v: k for k, v in CITY.items()}
    Y = np.zeros((20, len(data.y)))
    for code in np.unique(data.city):
        region = inv[int(code)]
        Tg = np.load(ROOT / f"derived/model_inputs/{LINE}/{region}/{scale}/T_groups_{split}.npz")["T"]
        sel = data.city == code
        Y[:, sel] = Tg[:, data.origin_unit[sel], data.dest_unit[sel]]
    return Y


def _hof(rd):
    """Final front if the run finished, otherwise PySR's latest checkpoint of the front."""
    f = rd / "hall_of_fame.csv"
    if f.exists():
        return pd.read_csv(f), True
    ck = rd / "pysr" / "hall_of_fame.csv"
    if ck.exists():
        return pd.read_csv(ck).rename(columns=str.lower), False
    return pd.DataFrame(columns=["complexity", "loss", "equation"]), False      # stopped before its first front


_W = {}


def _refit_one(k):
    c = _W["kept"][k][1]
    r = refit(c, _W["tr"])
    e = eta(r, _W["va"])
    return k, r.consts, loss(r, _W["tr"]), loss(r, _W["va"]), e


def _cached(cache, scale, variant, regions):
    """(template, equation) -> candidate record of earlier select outputs on the same data (same refit, deterministic)."""
    out = {}
    for name in cache or ():
        s = json.loads((ROOT / f"results/sr/{LINE}/{name}.json").read_text())
        if (s["scale"], s["variant"], s.get("regions", ["LA", "SFBAY"])) != (scale, variant, list(regions)):
            raise ValueError(f"cache {name}: different scale / variant / regions")
        out.update({(x["template"], x["equation"]): x for x in s["candidates"]})
    return out


def select(run_dirs, scale, variant, regions=("LA", "SFBAY"), max_total=20, per_cx=2, B=200, out_name=None, procs=8,
           snapshots=False, cache=None, n_orig=None, train_block="search", eval_split="val", eval_block="val"):
    """max_total / per_cx = None: no cap. snapshots: also pool runs/<tag>/snapshots/hof_*.csv (earlier fronts).
    cache: names of earlier select outputs (same scale, variant, regions; refit code unchanged since) whose refitted
    constants are reused for identical equations; validation scores are recomputed.
    n_orig: origins per region sampled for the refits (seed 0; multi-city pooled selection uses 300); validation
    scoring always uses every origin. train_block / eval_split / eval_block: the refit block (training period) and
    the scored split x block; the agent loop ranks on search23 -> train x fold:4 (plan 4), the defaults are the
    validation scoring of plans 1-2. Output: results/sr/<LINE>/<out_name>.json."""
    hit = _cached(cache, scale, variant, regions)
    pool, finished = [], {}
    for rd in map(Path, run_dirs):
        cfg = json.loads((rd / "config.json").read_text())
        hof, finished[rd.name] = _hof(rd)
        if snapshots:
            hof = pd.concat([hof, *[pd.read_csv(f).rename(columns=str.lower)
                                    for f in sorted((rd / "snapshots").glob("hof_*.csv"))]])
        for (_, r), (eq, _) in zip(hof.iterrows(), run_equations(rd, cfg, hof)):
            if not np.isfinite(r["loss"]) or r["loss"] >= PENALTY:
                continue
            pool.append((r["loss"], parse_equation(eq, cfg["template"], variant, int(r["complexity"])), rd.name))
    pool.sort(key=lambda t: t[0])
    seen, kept, by_cx = set(), [], {}
    for lo, c, rd in pool:
        key = (c.template, tuple(sorted((k, str(v)) for k, v in c.comps.items())))   # literals are symbols: structural key
        if key in seen or (per_cx is not None and by_cx.get(c.complexity, 0) >= per_cx):
            continue
        seen.add(key)
        by_cx[c.complexity] = by_cx.get(c.complexity, 0) + 1
        kept.append((lo, c, rd))
        if max_total is not None and len(kept) >= max_total:
            break
    tr = build(list(regions), scale, "train", train_block, n_orig, 0, offset_B=(variant == "B"))
    va = build(list(regions), scale, eval_split, eval_block, None, 0, offset_B=(variant == "B"))
    Yg = _device_group_y(va, regions, scale, eval_split)
    rng = np.random.default_rng(20261101)
    W = rng.multinomial(20, np.full(20, 0.05), size=B).astype(float)
    import multiprocessing as mp
    _W.update(kept=kept, tr=tr, va=va)
    todo = [k for k, (_, c, _) in enumerate(kept) if (c.template, c.expr) not in hit]
    if procs == 1:                       # no fork (a process with a running Julia runtime must not fork)
        done = dict((k, rest) for k, *rest in map(_refit_one, todo))
    else:
        with mp.get_context("fork").Pool(procs) as P:
            done = dict((k, rest) for k, *rest in P.imap_unordered(_refit_one, todo))
    for k in sorted(set(range(len(kept))) - set(todo)):
        x = hit[(kept[k][1].template, kept[k][1].expr)]
        r = dataclasses.replace(kept[k][1], consts=np.array(x["consts_refit"], float))
        done[k] = (r.consts, x["train_loss_refit"], loss(r, va), eta(r, va))
    res, etas = [], []
    for k, (lo, c, rd) in enumerate(kept):
        consts, ltr, lva, e = done[k]
        etas.append(e)
        res.append({"run": rd, "template": c.template, "complexity": c.complexity, "train_loss_pysr": float(lo),
                    "train_loss_refit": ltr, "val_nll": lva, "equation": c.expr, "consts_refit": list(map(float, consts)),
                    "cached": k not in todo})
        print(f"cx {c.complexity:3d} train {ltr:.5f} val {lva:.5f} | {c.expr[:110]}", flush=True)
    vals = np.array([x["val_nll"] for x in res])
    ib = int(np.argmin(vals))
    boot = np.column_stack([boot_nll(e, va, Yg, W) for e in etas]) if etas else np.zeros((B, 0))
    sd = (boot - boot[:, [ib]]).std(0)
    within = [k for k in range(len(res)) if vals[k] - vals[ib] <= sd[k]]
    chosen = min(within, key=lambda k: (res[k]["complexity"], vals[k]))
    for k, x in enumerate(res):
        x["se_diff_vs_best"] = float(sd[k])
        x["within_1se"] = k in within
    out = {"scale": scale, "variant": variant, "regions": list(regions), "runs": [str(r) for r in run_dirs],
           "runs_finished": finished, "cache": list(cache or []), "n_orig": n_orig,
           "eval": {"train_block": train_block, "split": eval_split, "block": eval_block},
           "candidates": res, "best_val": ib, "chosen": chosen}
    (ROOT / f"results/sr/{LINE}/{out_name or f'select_{scale}_{variant}'}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import os
    cap = os.environ.get("SELECT_NOCAP") != "1"           # SELECT_NOCAP=1: refit every distinct front member
    env = os.environ.get
    select(sys.argv[3:], sys.argv[1], sys.argv[2], out_name=env("SELECT_OUT"),
           procs=int(env("SELECT_PROCS", "8")), max_total=20 if cap else None, per_cx=2 if cap else None,
           snapshots=env("SELECT_SNAPSHOTS") == "1", regions=tuple(env("SELECT_REGIONS", "LA,SFBAY").split(",")),
           cache=env("SELECT_CACHE").split(",") if env("SELECT_CACHE") else None,
           n_orig=int(env("SELECT_N_ORIG")) if env("SELECT_N_ORIG") else None,
           train_block=env("SR_TRAIN_BLOCK", "search"), eval_split=env("SR_EVAL_SPLIT", "val"),
           eval_block=env("SR_EVAL_BLOCK", "val"))
