"""Task 10a: city roles (search / confirm / rest / excluded), population tiers and the substitution rule."""
import pandas as pd


def test_assign_substitutes():
    from tiers import assign
    qc = pd.DataFrame({"city": ["A", "B", "C", "D", "E"], "eligible": [True, False, True, True, True],
                       "support_mismatch": ["", "", "E_j", "", ""], "population": [6e6, 6e6, 3e6, 3e6, 1e6]})
    out = assign(qc, search=["A", "B", "C"], confirm=["D"]).set_index("city")
    assert out.loc["A", "role"] == "search" and out.loc["B", "role"] == "excluded"
    assert out.loc["C", "role"] == "rest" and out.loc["D", "role"] == "confirm"
    assert out.loc["E", "role"] == "rest"                       # no same-tier substitute for C (tier 2: only D, a confirm city)
    assert out.loc["B", "tier"] == 1 and out.loc["C", "tier"] == 2 and out.loc["E", "tier"] == 3
    assert out["substitute_for"].isna().all()


def test_assign_takes_substitute_and_drops_bad_confirm():
    from tiers import assign
    qc = pd.DataFrame({"city": ["A", "B", "F", "D", "G"], "eligible": [True, False, True, False, True],
                       "support_mismatch": ["", "", "", "", ""], "population": [6e6, 6e6, 7e6, 3e6, 2.5e6]})
    out = assign(qc, search=["A", "B"], confirm=["D", "G"]).set_index("city")
    assert out.loc["F", "role"] == "search" and out.loc["F", "substitute_for"] == "B"   # largest eligible tier-1 city
    assert out.loc["D", "role"] == "excluded" and out.loc["G", "role"] == "confirm"     # bad confirm city dropped, not replaced
    assert out.loc["B", "role"] == "excluded"


def test_assign_raises_when_a_listed_city_is_missing_from_qc():
    """Final review (Important 6): a search / confirmation city absent from the QC table (its QC crashed) must be
    loud, not silently left without a substitute."""
    import pytest
    from tiers import assign
    qc = pd.DataFrame({"city": ["A"], "eligible": [True], "support_mismatch": [""], "population": [6e6]})
    with pytest.raises(ValueError, match="Z"):
        assign(qc, search=["A", "Z"], confirm=[])


def test_main_refuses_to_change_written_roles(tmp_path):
    """Final review (Important 4): once roles are written they are frozen; a re-run that would change them raises
    unless force=True."""
    import pytest
    from tiers import main
    qc = pd.DataFrame({"city": ["A", "B", "C", "D"], "eligible": [True] * 4, "support_mismatch": [""] * 4,
                       "population": [6e6, 6e6, 3e6, 3e6]})
    qc_csv, cities = tmp_path / "cities_qc.csv", tmp_path / "cities.csv"
    qc.to_csv(qc_csv, index=False)
    pd.DataFrame({"city": ["A", "B", "C", "D"], "name": list("abcd")}).to_csv(cities, index=False)
    kw = dict(search=["A", "B"], confirm=["D"], qc_csv=qc_csv, cities_csv=cities, ledger_fn=lambda s: None)
    main(**kw)
    roles = pd.read_csv(cities, dtype=str).set_index("city")["role"]
    assert roles["A"] == "search" and roles["D"] == "confirm" and roles["C"] == "rest"
    main(**kw)                                                   # identical roles: allowed
    qc.loc[qc["city"] == "A", "eligible"] = False
    qc.to_csv(qc_csv, index=False)
    with pytest.raises(RuntimeError, match="role"):
        main(**kw)
    assert pd.read_csv(cities, dtype=str).set_index("city").loc["A", "role"] == "search"   # unchanged
    main(**kw, force=True)
    assert pd.read_csv(cities, dtype=str).set_index("city").loc["A", "role"] == "excluded"
