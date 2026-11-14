"""Plan 4 Task 3: code-decidable cleanliness checks: origin-only terms, unit mixing, disguised linear
terms, canonicalisation. Positions: d=#1 logd=#2 P_i=#3 E_j=#5 N_j=#6 a_j=#7 q_i=#17 pd_i=#18 rcbd_j=#24 S_ij=#26."""
from refit_select import parse_equation

from checks import DIM, ORIGIN_VARS, canonicalize, disguised_linear, flags, origin_only_terms, unit_mixing


def _cand(eq):
    tail = "; p1 = [1.0]; p2 = [1.0]" if "g = " in eq else "; p1 = []; p2 = []"
    return parse_equation(eq + tail, None, "A", 1)


def test_dimension_table_covers_every_variable():
    from sr_data import VARS
    assert sorted(v for vs in DIM.values() for v in vs) == sorted(VARS)
    assert set(ORIGIN_VARS) == {"P_i", "q_i", "pd_i", "rsub_i", "rcbd_i", "acc_i"}


def test_origin_only_flags_additive_origin_terms():
    c = _cand("f = log(#6 + 1.0) - 2.0 * #18 - 1.5 * log(#1 + 1.0)")
    assert origin_only_terms(c) == ["-2.0*x18"]


def test_origin_only_ignores_interactions():
    c = _cand("f = log(#6 + 1.0); g = #17 * #2; h = #18 + log(#1 + 1.0)")
    assert origin_only_terms(c) == ["x18"]                       # h's x18 yes, g's x17*x2 no


def test_unit_mixing_rules():
    assert unit_mixing(_cand("f = log(#26 + #6 + 3.7)")) == ["x26 + x6 + 3.7"]           # two counts, unweighted
    assert unit_mixing(_cand("f = log(0.0145 * #5 + #6 + 1.0)")) == []                  # weighted counts ok
    assert unit_mixing(_cand("f = log(#6 + #1)")) == ["x1 + x6"]                        # count + distance
    assert unit_mixing(_cand("f = log(#1 + 2.1) + log(#24 + 0.2)")) == []               # top-level sum of terms ok
    assert unit_mixing(_cand("f = 0.5 * (#6 + #7) * #2")) == ["x6 + x7"]                # inside a top-level term
    assert unit_mixing(_cand("f = log(#6 + 1.0); g = #2; h = log(#1 + #24 + 1.0)")) == []   # distance + distance ok


def test_disguised_linear_uses_per_city_refit_constants():
    c = _cand("f = 0.07 * log(#24 + 0.16) + log(#6 + 3.0)")
    flagged = disguised_linear(c, {"35620": [619.1, 161508.0, 3.7]}, {"35620": {"rcbd_j": 120.0, "N_j": 5000.0}})
    assert flagged == ["log(x24 + 161508.0)"]
    assert disguised_linear(c, {"35620": [0.07, 0.16, 3.0]}, {"35620": {"rcbd_j": 120.0, "N_j": 5000.0}}) == []


def test_disguised_linear_exponential_with_huge_scale():
    c = _cand("f = log(#6 + 3.0) + 2.0 * nexp(#1 * 0.5)")                  # nexp(d / 2): curved over d <= 150
    xm = {"A": {"d": 150.0, "N_j": 5000.0}}
    assert disguised_linear(c, {"A": [3.0, 2.0, 0.5]}, xm) == []
    assert disguised_linear(c, {"A": [3.0, 2.0, 1e-5]}, xm) == ["exp(-1.0e-5*x1)"]   # nexp(d / 1e5): linear


def test_disguised_linear_any_city_and_missing_city_constants():
    c = _cand("f = 0.07 * log(#24 + 0.16) + log(#6 + 3.0)")
    consts = {"A": [0.07, 0.16, 3.0], "B": [619.1, 161508.0, 3.7], "C": None}       # C not fittable
    xm = {c_: {"rcbd_j": 120.0, "N_j": 5000.0} for c_ in "ABC"}
    assert disguised_linear(c, consts, xm) == ["log(x24 + 161508.0)"]


def test_canonicalize_strips_origin_terms_and_linearises():
    c = _cand("f = 619.0 * log(#24 + 161508.0) + log(#6 + 3.0) - 2.0 * #18 + 1.7")
    k = canonicalize(c, {"rcbd_j": 120.0, "N_j": 5000.0})
    import sympy
    s = str(k.comps["f"])
    assert "x18" not in s and "x24" in s and len(k.comps["f"].atoms(sympy.log)) == 1      # only log(N_j + c) left
    lits = dict(zip([f"c{i}" for i in range(k.n_lit)], k.consts[:k.n_lit]))
    assert any(abs(v - 619.0 / 161508.0) < 1e-9 for v in lits.values())        # (c / A) * x
    assert all(abs(v - 161508.0) > 1 for v in lits.values()) and all(abs(v - 1.7) > 1e-9 for v in lits.values())
    k0 = canonicalize(c, None)                                                   # no xmax: no linearisation
    assert "x18" not in str(k0.comps["f"]) and any(abs(v - 161508.0) < 1e-6 for v in k0.consts[:k0.n_lit])


def test_canonicalize_g_and_h_origin_terms():
    c = _cand("f = log(#6 + 1.0); g = #17 * #2 + #3; h = #18 + log(#1 + 1.0)")
    k = canonicalize(c, None)
    assert "x3" not in str(k.comps["g"]) and "x17*x2" in str(k.comps["g"]) and "x18" not in str(k.comps["h"])


def test_c0209_is_clean_and_c0122_is_not():
    c0209 = _cand("f = -2.09 * log(#1 + 2.12) + 0.062 * log(#24 + 0.2) - 0.062 * log(#26 + 2.05) + log(#6 + 3.04)")
    c0122 = _cand("f = -0.88 * log(#1 + 0.64) + 0.6 * log(#24 + 20.3) + log((#6 + 17.0) ^ 0.957) "
                  "- 0.6 * log(#26 + #6 + 14909.0)")
    xmax = {"35620": {"rcbd_j": 120.0, "N_j": 5000.0, "S_ij": 2e7, "d": 150.0}}
    f0 = flags(c0209, {"35620": list(c0209.consts)}, xmax)
    assert f0["n"] == 0 and f0 == {"origin_only": [], "unit_mixing": [], "disguised_linear": [], "n": 0}
    f = flags(c0122, {"35620": [619.1, 161508.0, -6.8, 189.0, -0.8, 20460.0, 21.7, 1.1]}, xmax)
    assert f["unit_mixing"] and f["disguised_linear"] and f["n"] == 2
