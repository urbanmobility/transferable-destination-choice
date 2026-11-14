"""Call point 0 — LLM-proposed seed skeletons for the PySR searches (plan 2 addendum, 11-04).

propose(): one local-LLM call per lens (distance decay, attraction, intervening opportunities and competition, urban
  form and population heterogeneity), each asking for N_PER_LENS skeletons eta = f + p1[city] g + p2[city] h written
  with variable names; the input carries only the variable table with pooled quantiles, the template, the operators
  and two reference forms (llm.check_input whitelist).
validate_expr(): sympy parse; only the searched variables (VARS minus s_health), numbers, log, nexp(x) = exp(-x) and
  + - * / ^ with a numeric or single-variable exponent.
screen(): code verdict on the pooled search rows — finite eta, constants refitted (single-category form), structural
  duplicates removed.
diverse(): greedy by training loss, skipping a skeleton whose eta correlates > rho with one already chosen.
allocate(): every run gets the hand seeds (config/guesses.json) plus a different, family-rotated subset; written as
  config/guesses/<tag>.json in the guesses.json format (run_queue.sh uses it when present).
select_file(): the accepted skeletons as a selection-format file for the library (provenance source_seed).

Round 4 (design decision of 11-04, after the challenger diagnostic of results/multicity/diag/): every round-3 seed kept
log distance plus a mass term, and the eta-correlation filter could not see it (the shared distance component makes
eta correlate across structurally different skeletons), so intervening opportunities (S_ij) were never searched.
backbone(): structural class decided by code from how distance and S_ij enter eta — power (log distance),
  kernel_other (distance only non-logarithmically), opportunity (S_ij, no distance), gravity_opportunity (both),
  no_distance.
diverse_by_class(): the correlation filter runs inside each class, up to per_class skeletons per class.
FAMILY_LENSES / allocate_family(): one LLM lens per non-power class (no gravity reference forms in the input); every
  round-4 run gets the seeds of exactly one class (seed number -> class: FAMILY_SEEDS), without the hand gravity seeds.
CHALLENGERS: the three diagnostic structures, written to the library as select_challengers_diag.json (and screened
  with the round-4 proposals, so they can also seed their class).

The prompts were issued in Chinese in the original runs; the text here is a translation.

Usage (compute server): ODSR_ROOT=... ODSR_LINE=M ODSR_CITIES=... LLM_*=... python -I seed_agent.py QUEUE_FILE [families]
"""
import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import llm  # noqa: E402
import sr_data  # noqa: E402
from library import structure_key, to_single  # noqa: E402
from llm_points import VAR_MEANING  # noqa: E402
from pysr_setup import _positional  # noqa: E402
from refit_select import PENALTY, eta, loss, parse_equation, refit  # noqa: E402

N_PER_LENS, K_KEEP, RHO_MAX = 8, 24, 0.98
SEARCH_VARS = [v for v in sr_data.VARS if v != "s_health"]
FUNCS = {"log", "nexp"}
LENSES = [
    {"name": "distance", "focus": "Functional form of distance decay: power law, exponential, power-exponential (Tanner), "
                                  "log-normal, Box-Cox, shifted power law (e.g. (d + c)^(-β)), and different decay at "
                                  "short and long distances; attraction may use a simple size term."},
    {"name": "attraction", "focus": "Destination attraction: employment and POI size (logarithmic, saturating, weighted "
                                    "combinations), population, shares and diversity (entropy) of the facility mix, "
                                    "employment density; distance with a simple decay."},
    {"name": "opportunities", "focus": "Intervening opportunities and competition: radiation / intervening-opportunity "
                                       "models (S_ij is the population inside the circle centred on the origin with "
                                       "radius equal to the distance to the destination), competing destinations and "
                                       "accessibility (acc_i), the role of origin and destination density."},
    {"name": "form", "focus": "Urban form and population heterogeneity: distance to the CBD and to employment subcentres "
                              "(origin, destination), monocentric / polycentric structure, interactions of the income "
                              "quantile q_i with distance or attraction (different income groups have different "
                              "distance sensitivity)."},
]
SCHEMA = {"type": "object", "required": ["proposals"], "properties": {"proposals": {"type": "array", "items": {
    "type": "object", "required": ["family", "rationale", "f", "g", "h"],
    "properties": {"family": {"type": "string"}, "rationale": {"type": "string"}, "f": {"type": "string"},
                   "g": {"type": "string"}, "h": {"type": "string"}, "p1": {"type": "number"}, "p2": {"type": "number"}}}}}}
SYSTEM = ("You are an expert in urban travel and destination-choice modelling, familiar with the literature on gravity "
          "models, radiation / intervening-opportunity models, competing-destination models and urban spatial structure. "
          "Your answer must be a single JSON object satisfying the given JSON Schema.")
OPERATORS = ("Binary: + − * / ^ (the exponent of ^ may only be a numeric constant or a single variable); unary: log "
             "(natural logarithm, the argument must be positive), nexp(x) = exp(−x). No other functions (such as sqrt, "
             "exp, abs) are allowed. Numeric constants may be written freely; the search refits them.")
TEMPLATE = ("Conditional-logit utility η_ij = f(x) + p1[city] · g(x) + p2[city] · h(x): f is the part shared by all "
            "cities; g and h are multiplied by the city-specific coefficients p1 and p2 (e.g. if the distance-decay "
            "exponent varies by city, put log distance in g). Only alternative destinations of the same origin are "
            "compared, so terms that contain only origin variables (*_i) and are not multiplied by destination "
            "variables have no effect. Give reasonable initial values for p1 and p2.")
FAMILY_SEEDS = {"40": "opportunity", "41": "gravity_opportunity", "42": "kernel_other"}
FAMILY_LENSES = [
    {"name": "opportunity", "prefix": "F1", "references": [],
     "focus": "Intervening opportunities instead of distance: destination choice is determined by destination size and "
              "the intervening opportunities S_ij (the population inside the circle centred on the origin with radius "
              "equal to the distance to the destination, i.e. a 'rank distance'); may include a radiation-model-style "
              "competition denominator (with origin population P_i), destination density and composition. "
              "No component may contain d or logd."},
    {"name": "gravity_opportunity", "prefix": "F2", "references": [],
     "focus": "Distance decay and intervening opportunities together: every skeleton must contain both distance (logd "
              "or a shifted log(d + c)) and S_ij; explore how the two divide the work (e.g. a city-varying coefficient "
              "on S_ij, different short- and long-range roles, competition between S_ij and destination size)."},
    {"name": "kernel_other", "prefix": "F3", "references": [],
     "focus": "Non-logarithmic distance kernels: distance enters only in non-log form (exponential nexp(d / c), linear "
              "d, fractional powers of d, two-piece kernels, etc.); logd and log(d + c) must not be used; attraction "
              "may use size terms, density or the facility mix."},
]
CHALLENGERS = [
    {"id": "D-grav_disp", "family": "shifted gravity", "f": "log(N_j + 1.0) - 2.2 * log(d + 1.5)", "g": "1.0", "h": "1.0"},
    {"id": "D-io_rank", "family": "intervening opportunities instead of distance",
     "f": "log(N_j + 1.0) - 1.0 * log(S_ij + 1.0)", "g": "1.0", "h": "1.0"},
    {"id": "D-grav_disp_io", "family": "shifted gravity + intervening opportunities",
     "f": "log(N_j + 1.0) - 2.2 * log(d + 1.5) - 0.3 * log(S_ij + 1.0)", "g": "1.0", "h": "1.0"},
]
REFERENCES = [{"name": "power-law gravity", "f": "log(N_j + 1.0)", "g": "logd", "h": "d", "p1": -1.7, "p2": 0.0},
              {"name": "the form of round-1 skeleton A", "f": "log(0.0145 * E_j + N_j - 2.0 * s_office + 16.6)", "g": "nexp(E_j)",
               "h": "log(d + 2.7)", "p1": -2.2, "p2": -2.5}]


def validate_expr(s):
    """(ok, reason, sympy expression) for one component written with variable names."""
    s = (s or "").strip()
    if not s:
        return False, "parse error: empty", None
    for m in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\s*(\()?", s):
        name, call = m.group(1), m.group(2)
        if call and name not in FUNCS:
            return False, f"function {name} not allowed", None
        if not call and name not in SEARCH_VARS:
            return False, f"symbol {name} not a searched variable", None
    loc = {v: sympy.Symbol(v) for v in SEARCH_VARS}
    loc.update(log=sympy.log, nexp=lambda z: sympy.exp(-z))
    try:
        e = sympy.sympify(s.replace("^", "**"), locals=loc)
    except Exception as ex:  # noqa: BLE001 — anything sympy cannot read is a rejection
        return False, f"parse error: {type(ex).__name__}", None
    if not isinstance(e, sympy.Basic):
        return False, "parse error: not an expression", None
    for p in e.atoms(sympy.Pow):
        if not (p.exp.is_number or isinstance(p.exp, sympy.Symbol)):
            return False, f"exponent {p.exp} is neither a number nor one variable", None
    return True, "ok", e


def backbone(p):
    """Structural class of a skeleton from the variables its components use (see the module docstring)."""
    exprs = [e for e in (validate_expr(p.get(k, ""))[2] for k in ("f", "g", "h")) if e is not None]
    names = {x.name for e in exprs for x in e.free_symbols}
    has_s, has_d = "S_ij" in names, bool(names & {"d", "logd"})
    if has_s:
        return "gravity_opportunity" if has_d else "opportunity"
    if not has_d:
        return "no_distance"
    d = sympy.Symbol("d")
    logdist = "logd" in names or any(d in a.free_symbols for e in exprs for a in e.atoms(sympy.log))
    return "power" if logdist else "kernel_other"


def challenger_props():
    return [{**c, "lens": "diagnostic", "rationale": "11-04 challenger diagnostic (results/multicity/diag/challengers.json)"}
            for c in CHALLENGERS]


def var_summary(data):
    """Pooled quantiles of each searched variable on the screening rows (aggregates only)."""
    out = {}
    for v in SEARCH_VARS:
        c = data.X[:, sr_data.VARS.index(v)].astype(float)
        q = np.quantile(c, [0.0, 0.1, 0.5, 0.9, 1.0])
        out[v] = {"min": round(float(q[0]), 4), "q10": round(float(q[1]), 4), "q50": round(float(q[2]), 4),
                  "q90": round(float(q[3]), 4), "max": round(float(q[4]), 4), "zero_share": round(float((c == 0).mean()), 4)}
    return out


def payload(lens, summary):
    return {"lens": lens["name"], "focus": lens["focus"], "n": N_PER_LENS, "template": TEMPLATE, "operators": OPERATORS,
            "variables": {v: {"meaning": VAR_MEANING[v], **summary.get(v, {})} for v in SEARCH_VARS},
            "references": lens.get("references", REFERENCES)}


def _messages(p):
    user = (f"Lens: {p['focus']}\n\nUnder this lens, propose {p['n']} mutually different utility skeletons that make "
            "sense in travel-behaviour terms (three components f, g, h, written with the variable names of the table "
            "below), each with a family (a short family name) and a rationale (a one-sentence justification). Note: a "
            "variable with zero_share > 0 needs a positive shift when taking log or a negative power (e.g. "
            "log(N_j + 1.0)); keep the complexity moderate (no more than about 10 operations per component); do not "
            "just make small changes to the reference forms.\n\n"
            + json.dumps({k: p[k] for k in ("template", "operators", "variables", "references")}, ensure_ascii=False))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def propose(summary, effort="medium", lenses=None):
    out = []
    for k, lens in enumerate(LENSES if lenses is None else lenses):
        p = payload(lens, summary)
        llm.check_input(p)
        try:
            r = llm.call(_messages(p), SCHEMA, effort=effort, max_tokens=8000, timeout=900, name="seed_proposals")
        except llm.LLMError as e:
            print("lens", lens["name"], "failed:", str(e)[:200], flush=True)
            continue
        for i, x in enumerate(r["proposals"]):
            out.append({"id": f"{lens.get('prefix', f'L{k + 1}')}-{i + 1:02d}", "lens": lens["name"], **x})
    return out


def _abs(expr):
    return _positional(expr, sr_data.VARS)                  # variable names -> absolute #k (VARS position + 1)


def _complexity(c):
    return int(sum(len(list(sympy.preorder_traversal(e))) for e in c.comps.values()))


def screen(props, data):
    rows, seen = [], {}
    for p in props:
        r = {k: p.get(k) for k in ("id", "lens", "family", "rationale", "f", "g", "h")}
        r.update(status=None, loss=None, consts=None, equation=None, eta=None, p=None, complexity=None, backbone=None)
        bad = [(part, validate_expr(p.get(part, ""))) for part in ("f", "g", "h")]
        bad = [(part, v[1]) for part, v in bad if not v[0]]
        if bad:
            r["status"] = f"rejected: {bad[0][0]}: {bad[0][1]}"
            rows.append(r)
            continue
        r["backbone"] = backbone(p)
        p1, p2 = float(p.get("p1", -1.0) or 0.0), float(p.get("p2", 0.0) or 0.0)
        eq = f"f = {_abs(p['f'])}; g = {_abs(p['g'])}; h = {_abs(p['h'])}; p1 = [{p1!r}]; p2 = [{p2!r}]"
        c = parse_equation(eq, None, "A", 1)
        key = structure_key(c)
        if key in seen:
            r["status"] = f"duplicate of {seen[key]}"
            rows.append(r)
            continue
        if loss(c, data) >= PENALTY:
            # perturbed starts are not needed here: a start whose eta is not finite on the rows is a rejection
            r["status"] = "rejected: eta not finite on the search rows"
            rows.append(r)
            continue
        cr = refit(to_single(c), data)
        lo = loss(cr, data)
        if not lo < PENALTY:
            r["status"] = "rejected: eta not finite after the refit"
            rows.append(r)
            continue
        seen[key] = r["id"]
        r.update(status="ok", loss=float(lo), consts=[float(v) for v in cr.consts], equation=eq,
                 eta=eta(cr, data), p=[float(cr.consts[cr.n_lit]), float(cr.consts[cr.n_lit + 1])],
                 complexity=_complexity(c))
        rows.append(r)
    return rows


def diverse(rows, k=K_KEEP, rho=RHO_MAX):
    ok = sorted([r for r in rows if r["status"] == "ok"], key=lambda r: r["loss"])
    chosen = []
    for r in ok:
        e = np.asarray(r["eta"], float)
        if any(np.std(e) > 0 and np.std(c["eta"]) > 0 and abs(np.corrcoef(e, c["eta"])[0, 1]) > rho for c in chosen):
            continue
        chosen.append(r)
        if len(chosen) >= k:
            break
    return chosen


def diverse_by_class(rows, per_class=8, rho=RHO_MAX):
    groups = {}
    for r in rows:
        if r["status"] == "ok":
            groups.setdefault(r["backbone"], []).append(r)
    return {b: diverse(g, k=per_class, rho=rho) for b, g in groups.items()}


def _rotation(sel):
    """Interleave families (first-seen order), each family in its given order."""
    fams = {}
    for s in sel:
        fams.setdefault(s["family"], []).append(s)
    out, i = [], 0
    while len(out) < len(sel):
        for f in fams.values():
            if i < len(f):
                out.append(f[i])
        i += 1
    return out


def allocate(sel, tags, hand, per_run=6):
    order = _rotation(sel)
    out = {}
    for j, tag in enumerate(tags):
        pick = [order[(j * per_run + i) % len(order)] for i in range(min(per_run, len(order)))] if order else []
        single = "_single-" in tag
        variant = "B" if "_B_" in tag else "A"
        key = ("single" if single else "city") + ("_B" if variant == "B" else "")
        seeds = list(hand.get(key, []))
        for s in pick:
            p1, p2 = s["p"]
            if single:
                seeds.append(_single(s))
            else:
                seeds.append({"f": s["f"], "g": s["g"], "h": s["h"], "p1": p1, "p2": p2, "seed_id": s["id"]})
        out[tag] = {key: seeds, "seed_ids": [s["id"] for s in pick]}
    return out


def _pysr_str(e):
    """sympy -> a PySR-parsable string: binary + * / ^ only, nexp for exp(-z); a sign only on numeric literals."""
    if e.is_Number:
        v = float(e)
        return f"({v!r})" if v < 0 else repr(v)
    if e.is_Symbol:
        return e.name
    if isinstance(e, sympy.Add):
        return "(" + " + ".join(_pysr_str(a) for a in e.args) + ")"
    if isinstance(e, sympy.Mul):
        return "(" + " * ".join(_pysr_str(a) for a in e.args) + ")"
    if isinstance(e, sympy.Pow):
        if e.exp == -1:
            return f"(1.0 / {_pysr_str(e.base)})"
        return f"({_pysr_str(e.base)} ^ {_pysr_str(e.exp)})"
    if isinstance(e, sympy.log):
        return f"log({_pysr_str(e.args[0])})"
    if isinstance(e, sympy.exp):
        return f"nexp({_pysr_str(-e.args[0])})"
    raise ValueError(f"cannot print {type(e).__name__} for PySR")


def _cx(e):
    """PySR's node count of the printed form (every variable, constant and operator = 1; calibrated 11-06)."""
    if e.is_Number or e.is_Symbol:
        return 1
    if isinstance(e, (sympy.Add, sympy.Mul)):
        return sum(_cx(a) for a in e.args) + len(e.args) - 1
    if isinstance(e, sympy.Pow):
        return 2 + _cx(e.base) if e.exp == -1 else 1 + _cx(e.base) + _cx(e.exp)
    if isinstance(e, sympy.log):
        return 1 + _cx(e.args[0])
    if isinstance(e, sympy.exp):
        return 1 + _cx(-e.args[0])
    raise ValueError(f"cannot count {type(e).__name__}")


def pysr_complexity(s):
    """PySR complexity of a seed string after pysr_safe; None if invalid."""
    p = pysr_safe(s)
    if p is None:
        return None
    return _cx(validate_expr(p)[2])


def pysr_safe(s):
    """A seed string re-printed so PySR's parser accepts it (it has no unary '-': round 5, 11-06); None if invalid."""
    ok, _, e = validate_expr(s)
    if not ok:
        return None
    try:
        return _pysr_str(e)
    except ValueError:
        return None


def _single(s):
    p1, p2 = s["p"]
    f, g, h = (pysr_safe(s[k]) or s[k] for k in ("f", "g", "h"))
    return f"({f}) + ({p1!r}) * ({g}) + ({p2!r}) * ({h})"


def allocate_family(by, tags):
    """Round 4: each single-city run gets the seeds of the one class its seed number names (FAMILY_SEEDS), nothing else."""
    out = {}
    for tag in tags:
        if "_single-" not in tag:
            raise ValueError(f"{tag}: the structural-class round has single-city runs only")
        fam = FAMILY_SEEDS[tag.rsplit("_s", 1)[1]]
        pick = by.get(fam, [])
        out[tag] = {"single": [_single(s) for s in pick], "seed_ids": [s["id"] for s in pick], "family": fam}
    return out


def select_file(rows, path, source="seed_agent"):
    cands = [{"equation": r["equation"], "template": None, "complexity": r["complexity"], "consts_refit": r["consts"],
              "source_seed": r["id"], "family": r["family"], "lens": r["lens"], "train_loss_screen": r["loss"]}
             for r in rows if r["status"] == "ok"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"variant": "A", "source": source, "candidates": cands}, ensure_ascii=False, indent=1))
    return path


def main(queue_file, n_orig=15):
    import pandas as pd
    from config import CITIES_CSV, RESULTS
    import ledger
    scripts = Path(__file__).resolve().parents[1]
    cities = pd.read_csv(CITIES_CSV, dtype=str)
    regions = list(cities[cities["role"] == "search"]["city"])
    data = sr_data.build(regions, "tract", "train", "search", n_orig, 0)
    summ = var_summary(data)
    props = propose(summ)
    rows = screen(props, data)
    sel = diverse(rows)
    tags = [f"{r[0]}_{r[1]}_{r[2].replace(':', '-')}_s{r[3]}" for r in
            (l.split() for l in Path(queue_file).read_text().splitlines() if l.strip() and not l.startswith("#"))]
    hand = json.loads((scripts / "multicity/config/guesses.json").read_text())
    alloc = allocate(sel, tags, hand)
    gdir = scripts / "multicity/config/guesses"
    gdir.mkdir(parents=True, exist_ok=True)
    for tag, spec in alloc.items():
        (gdir / f"{tag}.json").write_text(json.dumps({"note": f"seed_agent {tag}", **spec}, ensure_ascii=False, indent=1))
    sf = select_file(sel, sr_data.ROOT / "results/sr/M/select_seeds_agent.json")
    lib = RESULTS / "library"
    lib.mkdir(parents=True, exist_ok=True)
    keep = [{k: v for k, v in r.items() if k != "eta"} for r in rows]
    (lib / "seeds.json").write_text(json.dumps({"proposals": len(props), "screen": keep,
                                                "selected": [r["id"] for r in sel]}, ensure_ascii=False, indent=1))
    from collections import Counter
    st = Counter(r["status"].split(":")[0] if r["status"].startswith("rejected") else r["status"].split(" of ")[0] for r in rows)
    ledger.add(f"Plan 2 call point 0 (seed agent): {len(props)} LLM proposals over {len(LENSES)} lenses -> {dict(st)}; "
               f"{len(sel)} diverse seeds kept (rho <= {RHO_MAX}); families {sorted({r['family'] for r in sel})}; "
               f"seeds.json sha {ledger.sha(lib / 'seeds.json')}, select_seeds_agent.json sha {ledger.sha(sf)}; "
               f"per-run guesses in config/guesses/ for {len(alloc)} runs of {Path(queue_file).name}")
    print(json.dumps({"proposals": len(props), "status": dict(st), "selected": [(r["id"], r["family"], round(r["loss"], 4))
                                                                                for r in sel]}, ensure_ascii=False, indent=1))
    return sel


def main_families(queue_file, n_orig=15, per_class=8):
    """Round 4: proposals from FAMILY_LENSES plus the diagnostic challengers, screened on the pooled search rows,
    diversity-filtered within each class, one class per run; the library gets the selected agent seeds
    (select_seeds_agent_r4.json) and the challengers (select_challengers_diag.json)."""
    import pandas as pd
    from config import CITIES_CSV, RESULTS
    import ledger
    scripts = Path(__file__).resolve().parents[1]
    cities = pd.read_csv(CITIES_CSV, dtype=str)
    regions = list(cities[cities["role"] == "search"]["city"])
    data = sr_data.build(regions, "tract", "train", "search", n_orig, 0)
    props = propose(var_summary(data), lenses=FAMILY_LENSES)
    rows = screen(props + challenger_props(), data)
    by = diverse_by_class([r for r in rows if r["backbone"] in FAMILY_SEEDS.values()], per_class=per_class)
    tags = [f"{r[0]}_{r[1]}_{r[2].replace(':', '-')}_s{r[3]}" for r in
            (l.split() for l in Path(queue_file).read_text().splitlines() if l.strip() and not l.startswith("#"))]
    alloc = allocate_family(by, tags)
    gdir = scripts / "multicity/config/guesses"
    gdir.mkdir(parents=True, exist_ok=True)
    for tag, spec in alloc.items():
        (gdir / f"{tag}.json").write_text(json.dumps({"note": f"seed_agent round 4 {tag}", **spec}, ensure_ascii=False, indent=1))
    srm = sr_data.ROOT / "results/sr/M"
    sel = [r for b in by.values() for r in b if r["lens"] != "diagnostic"]
    sa = select_file(sel, srm / "select_seeds_agent_r4.json", source="seed_agent_r4")
    sd = select_file([r for r in rows if r["lens"] == "diagnostic"], srm / "select_challengers_diag.json", source="diagnostic")
    lib = RESULTS / "library"
    lib.mkdir(parents=True, exist_ok=True)
    (lib / "seeds_r4.json").write_text(json.dumps(
        {"proposals": len(props), "screen": [{k: v for k, v in r.items() if k != "eta"} for r in rows],
         "selected": {b: [r["id"] for r in v] for b, v in by.items()}}, ensure_ascii=False, indent=1))
    from collections import Counter
    st = Counter((r["status"].split(":")[0] if r["status"].startswith("rejected") else r["status"].split(" of ")[0],
                  r["backbone"]) for r in rows)
    ledger.add(f"Plan 2 round 4 seeds (structural classes, design decision of 11-04): {len(props)} LLM proposals over "
               f"{len(FAMILY_LENSES)} class lenses + {len(CHALLENGERS)} diagnostic challengers -> "
               f"{ {f'{a}/{b}': n for (a, b), n in st.items()} }; per class kept "
               f"{ {b: [r['id'] for r in v] for b, v in by.items()} } (rho <= {RHO_MAX} within class); "
               f"seeds_r4.json sha {ledger.sha(lib / 'seeds_r4.json')}, select_seeds_agent_r4.json sha {ledger.sha(sa)}, "
               f"select_challengers_diag.json sha {ledger.sha(sd)}; per-run guesses for {len(alloc)} runs of {Path(queue_file).name}")
    print(json.dumps({"proposals": len(props), "status": {f"{a}/{b}": n for (a, b), n in st.items()},
                      "by_class": {b: [(r["id"], r["family"], round(r["loss"], 4)) for r in v] for b, v in by.items()}},
                     ensure_ascii=False, indent=1))
    return by


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "families":
        main_families(sys.argv[1])
    else:
        main(sys.argv[1])
