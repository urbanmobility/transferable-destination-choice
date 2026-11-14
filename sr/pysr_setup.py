"""PySR configuration for the SR round (plan Task 3): design matrix, grouped-softmax loss, templates.

The search fits a utility eta_ij inside the fixed-budget conditional logit. Rows of one origin are
contiguous and carry the same gid; the Julia loss computes, per origin, -sum y (eta - logsumexp eta)
and returns the per-visit total (1e9 if any prediction is non-finite). Variables enter PySR as
anonymous x1..xK (columns sr_data.VARS); three extra columns follow: ob (variant-B offset), cat
(1-based category: city or city x SES) and gid (origin id); sub-expressions only see x1..xK.
Templates:
  single / shared : eta = f(x)
  city2           : eta = f(x) + p1[cat] g(x) + p2[cat] h(x), cat = city (2 levels)
  cityses2        : the same with cat = city x SES (6 levels)
  variant B adds the fixed offset ob.
"""
import os

os.environ.setdefault("PYTHON_JULIACALL_THREADS", os.environ.get("SR_THREADS", "16"))
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

import numpy as np  # noqa: E402
import sympy  # noqa: E402

from sr_data import VARS  # noqa: E402

# SR_DROP_VARS: comma-separated variables left out of the search (multi-city: s_health is 0 everywhere in Veraset).
# The x-numbering follows the position in VARS, so x10 simply disappears and x11.. keep their names: equations,
# parse_equation and the refit code are unchanged.
DROP = tuple(filter(None, os.environ.get("SR_DROP_VARS", "").split(",")))
ACTIVE = [v for v in VARS if v not in DROP]
ACTIVE_IDX = [VARS.index(v) for v in ACTIVE]
K = len(ACTIVE)
XNAMES = [f"x{i + 1}" for i in ACTIVE_IDX]
EXTRA = ["ob", "cat", "gid"]
GID_ROW = K + 3                       # 1-based row of gid in the Julia (features x rows) matrix
PENALTY = 1e9

LOSS_CORE = """
function _odsr_gnll(pred::AbstractVector, y::AbstractVector, gid::AbstractVector)
    n = length(y)
    tot = 0.0
    vis = 0.0
    s = 1
    @inbounds while s <= n
        e = s
        g = gid[s]
        while e < n && gid[e + 1] == g
            e += 1
        end
        m = -Inf
        for k in s:e
            p = pred[k]
            isfinite(p) || return %(pen)s
            m = max(m, p)
        end
        z = 0.0
        for k in s:e
            z += exp(pred[k] - m)
        end
        lz = log(z)
        for k in s:e
            tot -= y[k] * ((pred[k] - m) - lz)    # centre first: m + log(z) would swallow log(z)
            vis += y[k]
        end
        s = e + 1
    end
    return tot / vis
end

# Float64 before adding log(w): with precision=32 a Float32 sum drops the correction once |eta| is large
_odsr_gnll_weighted(pred, y, gid, w) = _odsr_gnll(w === nothing ? pred : Float64.(pred) .+ log.(Float64.(w)), y, gid)
""" % {"pen": repr(PENALTY)}

LOSS = "begin\n" + LOSS_CORE + """
function _odsr_loss(ex, dataset::Dataset{T,L}, options) where {T,L}
    pred, ok = eval_tree_array(ex, dataset.X, options)
    ok || return L(%(pen)s)
    return L(_odsr_gnll_weighted(pred, dataset.y, view(dataset.X, %(row)d, :), dataset.weights))
end
end
""" % {"pen": repr(PENALTY), "row": GID_ROW}


def n_categories(template):
    """city2 -> 2, cityses2 -> 6, city<N> -> N (multi-city pooled search), anything else -> 1."""
    import re
    if template == "cityses2":
        return 6
    m = re.fullmatch(r"city(\d+)", template or "")
    return int(m.group(1)) if m else 1


def design_matrix(data, template):
    """(rows, K + 3): the active variables, then ob, cat (1-based position of the row's city in the run's regions,
    or city x SES) and gid."""
    cat = data.cat * 3 + data.ses + 1 if template == "cityses2" else data.cat + 1
    if template in ("single", "shared"):
        cat = np.ones(len(data.y))
    gid = np.repeat(np.arange(len(data.offsets) - 1), np.diff(data.offsets))
    return np.column_stack([data.X[:, ACTIVE_IDX].astype(np.float64), data.offset_B, cat, gid])


def numpy_nll(eta, data):
    eta = eta + data.logw
    tot, vis = 0.0, 0.0
    for a, b in zip(data.offsets[:-1], data.offsets[1:]):
        e = eta[a:b]
        if not np.isfinite(e).all():
            return PENALTY
        c = e - e.max()
        tot -= (data.y[a:b] * (c - np.log(np.exp(c).sum()))).sum()
        vis += data.y[a:b].sum()
    return tot / vis


def grouped_nll_julia(eta, y, X, w=None):
    """The Julia loss core as PySR calls it (w = PySR sample weights = 1 / inclusion probability)."""
    from pysr import jl
    if not hasattr(grouped_nll_julia, "_ready"):
        jl.seval(LOSS_CORE)
        grouped_nll_julia._ready = True
    ww = None if w is None else np.asarray(w, float)
    return float(jl._odsr_gnll_weighted(np.asarray(eta, float), np.asarray(y, float), np.asarray(X[:, K + 2], float), ww))


def template_spec(variant, template):
    from pysr import TemplateExpressionSpec
    args = ", ".join(XNAMES)
    if template in ("single", "shared"):
        exprs, comb, params = ["f"], f"f({args})", None
    else:
        C = n_categories(template)
        exprs = ["f", "g", "h"]
        comb = f"f({args}) + p1[cat] * g({args}) + p2[cat] * h({args})"
        params = {"p1": C, "p2": C}
    if variant == "B":
        comb += " + ob"
    return TemplateExpressionSpec(combine=comb, expressions=exprs, variable_names=XNAMES + EXTRA, parameters=params)


def _positional(expr, active):
    """Variable names (sr_data.VARS) -> PySR's positional #k over the run's active variables."""
    import re
    pos = {v: k + 1 for k, v in enumerate(active)}
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, pos), key=len, reverse=True)) + r")\b")
    return pat.sub(lambda m: f"#{pos[m.group(1)]}", expr)


def guesses_for(spec, variant, template, active=None):
    """Seed expressions for one run from the guesses spec (multicity/config/guesses.json): "single" / "city" lists
    for variant A, "single_B" / "city_B" for B; template dicts get p1 / p2 expanded to the category count. None when
    spec is None."""
    if spec is None:
        return None
    active = ACTIVE if active is None else active
    tpl_single = template in ("single", "shared") or (template or "").startswith("single")
    key = ("single" if tpl_single else "city") + ("_B" if variant == "B" else "")
    out = []
    for g in spec.get(key, []):
        if isinstance(g, str):           # single / shared templates are TemplateExpressions with component f only
            out.append({"f": _positional(g, active)})
            continue
        C = n_categories(template)
        out.append({k: (_positional(v, active) if isinstance(v, str) else [float(v)] * C)
                    for k, v in g.items() if k in ("f", "g", "h", "p1", "p2")})      # provenance keys stay out of PySR
    return out


def make_model(data, variant, template, seed, niterations, procs, **overrides):
    from pysr import PySRRegressor
    kw = dict(
        expression_spec=template_spec(variant, template),
        loss_function_expression=LOSS,
        loss_scale="linear",
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["log", "nexp(x) = exp(-x)"],
        extra_sympy_mappings={"nexp": lambda x: sympy.exp(-x)},
        constraints={"^": (-1, 1)},
        nested_constraints={"log": {"log": 0, "nexp": 0}, "nexp": {"nexp": 0, "log": 0}},
        maxsize=30,
        maxdepth=10,
        niterations=niterations,
        populations=max(16, 2 * procs),
        procs=procs,
        parallelism="multithreading",
        batching=False,
        precision=32,                 # eta in Float32; the loss accumulates in Float64
        ncycles_per_iteration=100,
        random_state=seed,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    kw.update(overrides)
    return PySRRegressor(**kw)
