"""Code-decidable cleanliness checks of candidate skeletons (plan 4 Task 3).

origin_only_terms(c): top-level additive terms of f / g / h whose free variables are all origin variables (P_i, q_i,
  pd_i, rsub_i, rcbd_i, acc_i). In a conditional logit such a term is constant over an origin's choice set and does
  nothing; interactions with a pair / destination variable (q_i * logd) are not flagged.
unit_mixing(c): an addition below the top level (a function argument or a sum inside a top-level term) whose terms
  carry variables of different dimension classes (DIM: N_j + d, N_j + a_j), or two different count variables added
  without a numeric weight (S_ij + N_j; 0.0145 * E_j + N_j is a composite attraction and passes).
disguised_linear(c, consts_by_city, xmax_by_city): log(A + k x) or (A + k x)^p with A > 10 * |k| * max(x) in a city,
  or exp(k x) with |k| * max(x) < 1/10, under that city's refitted constants — the term is linear in x over the
  city's range (c0122: 619 log(rcbd_j + 161508)). Any city flags the candidate.
canonicalize(c, xmax): clean.canonical (literals substituted, top-level constants removed), then origin-only terms
  removed and, with xmax, every disguised-linear log(A + k x) rewritten as log A + (k / A) x (the constant dropped).
flags(c, consts_by_city, xmax_by_city): the three lists and n = the number of non-empty ones (0..3).
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
from clean import _lits, canonical, render  # noqa: E402
from refit_select import COMPS, XS, parse_equation  # noqa: E402
from sr_data import CLASSES, VARS  # noqa: E402

ORIGIN_VARS = ("P_i", "q_i", "pd_i", "rsub_i", "rcbd_i", "acc_i")
DIM = {"count": {"P_i", "P_j", "E_j", "N_j", "S_ij"}, "area": {"a_j"},
       "distance": {"d", "logd", "rsub_i", "rsub_j", "rcbd_i", "rcbd_j"}, "share": {f"s_{c}" for c in CLASSES},
       "density": {"pd_i", "pd_j", "jd_j"}, "entropy": {"H_j"}, "quantile": {"q_i"}, "access": {"acc_i"}}
LINEAR_RATIO = 10.0

NAME = {XS[k]: v for k, v in enumerate(VARS)}
CLASS_OF = {v: k for k, vs in DIM.items() for v in vs}
ORIGIN_X = {XS[VARS.index(v)] for v in ORIGIN_VARS}


def _vars(e):
    return {s for s in e.free_symbols if s in NAME}


def _substituted(c, consts=None):
    """{comp: expression with the literal values substituted} (consts: the candidate's own by default)."""
    if consts is None:
        lits = _lits(c, exact=True)
    else:
        lits = {sympy.Symbol(f"c{i}"): sympy.Float(float(v)) for i, v in enumerate(np.asarray(consts, float)[:c.n_lit])}
    return {name: c.comps[name].subs(lits) for name in COMPS if name in c.comps}


def _is_origin_only(t):
    v = _vars(t)
    return bool(v) and v <= ORIGIN_X


def origin_only_terms(c) -> list[str]:
    return [str(t) for e in _substituted(c).values() for t in sympy.Add.make_args(e) if _is_origin_only(t)]


def _term_class(t):
    return tuple(sorted({CLASS_OF[NAME[s]] for s in _vars(t)}))


def _mixes(add):
    terms = [t for t in add.args if _vars(t)]
    classes = {_term_class(t) for t in terms}
    if len(classes) > 1:
        return True
    if classes == {("count",)}:
        unweighted = {frozenset(_vars(t)) for t in terms if t.as_coeff_Mul()[0] == 1}
        return len(unweighted) >= 2
    return False


def unit_mixing(c) -> list[str]:
    out = []
    for e in _substituted(c).values():
        for node in sympy.preorder_traversal(e):
            if node is e or not isinstance(node, sympy.Add):        # the top-level sum of f / g / h is the utility
                continue
            s = str(node)
            if _mixes(node) and s not in out:
                out.append(s)
    return out


def _linear_part(arg):
    """arg = A + sum_i k_i x_i with numbers A, k_i -> (A, {x_i: k_i}); None for any other form."""
    A, ks = 0.0, {}
    for t in sympy.Add.make_args(arg):
        if t.is_number:
            A += float(t)
            continue
        k, rest = t.as_coeff_Mul()
        if rest not in NAME:
            return None
        ks[rest] = ks.get(rest, 0.0) + float(k)
    return A, ks


def _bound(ks, xmax):
    """max over the city's rows of |sum k_i x_i| (bounded by sum |k_i| max x_i); None if a maximum is unknown."""
    if any(NAME[x] not in xmax for x in ks):
        return None
    return float(sum(abs(k) * abs(float(xmax[NAME[x]])) for x, k in ks.items()))


def _disguised(atom, xmax):
    arg = atom.base if isinstance(atom, sympy.Pow) else atom.args[0]
    if isinstance(atom, sympy.Pow) and not atom.exp.is_number:
        return False
    lp = _linear_part(arg)
    if lp is None or not lp[1]:
        return False
    A, ks = lp
    b = _bound(ks, xmax)
    if b is None or b <= 0:
        return False
    if isinstance(atom, sympy.exp):
        return b * LINEAR_RATIO < 1.0
    return A > LINEAR_RATIO * b


def _atoms(e):
    return sorted(e.atoms(sympy.log, sympy.exp, sympy.Pow), key=str)


def disguised_linear(c, consts_by_city: dict, xmax_by_city: dict) -> list[str]:
    out = []
    for city, consts in consts_by_city.items():
        if consts is None or city not in xmax_by_city or len(consts) < c.n_lit:
            continue
        for e in _substituted(c, consts).values():
            for a in _atoms(e):
                s = str(a)
                if _disguised(a, xmax_by_city[city]) and s not in out:
                    out.append(s)
    return out


def _linearised(atom):
    A, ks = _linear_part(atom.args[0])
    return sympy.log(sympy.Float(A)) + sum(sympy.Float(k / A) * x for x, k in ks.items())


def canonicalize(c, xmax: dict | None = None):
    k = canonical(c)
    comps = {}
    for name, e in _substituted(k).items():
        if xmax:
            e = e.xreplace({a: _linearised(a) for a in e.atoms(sympy.log) if _disguised(a, xmax)})
        terms = [t for t in sympy.Add.make_args(sympy.expand_mul(e, deep=False))
                 if not t.is_number and not _is_origin_only(t)]
        comps[name] = sympy.Add(*terms) if terms else sympy.Integer(0)
    tmp = dataclasses.replace(k, comps=comps, n_lit=0, consts=np.asarray(k.consts[k.n_lit:], float))
    return parse_equation(render(tmp), k.template, k.variant, k.complexity)


def flags(c, consts_by_city: dict, xmax_by_city: dict) -> dict:
    o, u, d = origin_only_terms(c), unit_mixing(c), disguised_linear(c, consts_by_city, xmax_by_city)
    return {"origin_only": o, "unit_mixing": u, "disguised_linear": d, "n": int(bool(o)) + int(bool(u)) + int(bool(d))}
