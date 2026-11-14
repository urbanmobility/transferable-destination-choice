"""Plan 2 Task 1: multi-city parameterisation of the round-1 search scripts — `cat` is the position of a row's
city in the run's `regions` list (not the global CITY code), SR_DROP_VARS removes variables from the search while
the x-numbering stays, the city<N> template, and refit_select._cat on positions."""
import importlib
import shutil

import numpy as np
import pytest

from city_fixture import make_city


def give_subcentre(root):
    """The fixture city has 10-60 jobs per km2, below the subcentre threshold, so r_sub / r_cbd are NaN everywhere
    (PySR rejects NaN inputs). One dense tract (60,000 jobs) makes a CBD and finite form variables; one tract with
    no jobs gives E_j = 0 rows in the training block (log(E_j) candidates are not fittable there: Review Focus 2)."""
    import pandas as pd
    from city_fixture import TRACTS
    p = root / "derived/multicity/tract_attrs.parquet"
    a = pd.read_parquet(p)
    a.loc[a["tract"] == "06999000200", "jobs"] = 60000
    a.loc[a["tract"] == "06999000100", "jobs"] = 0
    a.to_parquet(p, index=False)
    # validation-period trips from every origin to every other tract, so the validation block (one destination
    # fold of a 6-tract city) has visits and a finite per-visit NLL
    rows, k = [], 0
    for o in (TRACTS[0], TRACTS[3], TRACTS[5]):
        for d in TRACTS:
            if d != o:
                rows.append(("2024-05-05", k % 20, o, d, False, 1 + k % 2))
                k += 1
    t = pd.DataFrame(rows, columns=["date", "g", "o", "d", "work", "n"])
    t["g"] = t["g"].astype("int32")
    for ss in ("06", "32"):
        t[t["o"].str[:2] == ss].to_parquet(root / f"derived/multicity/trips/o_state={ss}/2024-05_0.parquet", index=False)
    pd.DataFrame({"date": ["2024-05-05"] * 3, "o": [TRACTS[0], TRACTS[3], TRACTS[5]], "active_devices": [4, 4, 4]}
                 ).to_parquet(root / "derived/multicity/budget/2024-05.parquet", index=False)


@pytest.fixture
def two_cities(tmp_path, monkeypatch):
    """TEST1 and TEST2 (a copy of TEST1's tables) with ODSR_CITIES = X0,TEST2,X1,TEST1: the global codes are 1 and 3,
    so a correct `cat` (0 / 1 by position in `regions`) differs from the code."""
    monkeypatch.setenv("ODSR_ROOT", str(tmp_path))
    monkeypatch.setenv("ODSR_LINE", "M")
    monkeypatch.setenv("ODSR_CITIES", "X0,TEST2,X1,TEST1")
    monkeypatch.delenv("SR_DROP_VARS", raising=False)
    import sr_data, config, form_vars, units, pysr_setup, refit_select  # noqa: E401
    mods = (sr_data, config, form_vars, units, pysr_setup, refit_select)
    for m in mods:
        importlib.reload(m)
    csv = tmp_path / "cities.csv"
    make_city(tmp_path, csv)
    give_subcentre(tmp_path)
    units.build_city("TEST1", tmp_path, cities_csv=csv)
    base = tmp_path / "derived/model_inputs/M"
    shutil.copytree(base / "TEST1", base / "TEST2")
    yield tmp_path
    monkeypatch.undo()
    for m in mods:
        importlib.reload(m)


def test_cat_is_position_in_regions(two_cities):
    import sr_data
    d = sr_data.build(["TEST1", "TEST2"], "tract", "train", "search", None, 0)
    assert set(np.unique(d.cat)) == {0, 1}
    assert (d.cat[d.city == sr_data.CITY["TEST1"]] == 0).all() and (d.cat[d.city == sr_data.CITY["TEST2"]] == 1).all()
    assert sr_data.CITY["TEST1"] == 3 and sr_data.CITY["TEST2"] == 1


def test_drop_vars_keeps_numbering(monkeypatch):
    monkeypatch.setenv("SR_DROP_VARS", "s_health")
    import pysr_setup
    importlib.reload(pysr_setup)
    assert "s_health" not in pysr_setup.ACTIVE and pysr_setup.K == 25
    assert pysr_setup.XNAMES[9] == "x11" and "x10" not in pysr_setup.XNAMES and pysr_setup.GID_ROW == 28
    monkeypatch.delenv("SR_DROP_VARS")
    importlib.reload(pysr_setup)
    assert pysr_setup.K == 26 and pysr_setup.XNAMES[9] == "x10" and pysr_setup.GID_ROW == 29


def test_design_matrix_city8_cat_and_width(two_cities, monkeypatch):
    monkeypatch.setenv("SR_DROP_VARS", "s_health")
    import sr_data, pysr_setup  # noqa: E401
    importlib.reload(pysr_setup)
    d = sr_data.build(["TEST1", "TEST2"], "tract", "train", "search", None, 0)
    M = pysr_setup.design_matrix(d, "city2")
    assert M.shape[1] == 25 + 3 and set(np.unique(M[:, -2])) == {1.0, 2.0}
    assert np.allclose(M[:, 9], d.X[:, 10])                      # column 10 of the design is x11 (s_education)
    assert pysr_setup.n_categories("city8") == 8
    spec = pysr_setup.template_spec("A", "city8")
    assert spec.parameters == {"p1": 8, "p2": 8} and "x10" not in spec.variable_names and "x11" in spec.variable_names


def test_refit_cat_uses_position(two_cities):
    import sr_data, refit_select  # noqa: E401
    d = sr_data.build(["TEST2", "TEST1"], "tract", "train", "search", None, 0)   # reversed order
    c = refit_select.parse_equation("f = log(#5 + 1.0); g = nexp(#5); h = log(#1 + 2.0); p1 = [0.1, 0.2]; p2 = [-2.0, -2.5]",
                                    "city2", "A", 10)
    cat = refit_select._cat(c, d)
    assert set(np.unique(cat)) == {0, 1} and (cat[d.city == sr_data.CITY["TEST2"]] == 0).all()
    c8 = refit_select.parse_equation("f = #1; g = #2; h = #3; p1 = [1,2,3,4,5,6,7,8]; p2 = [1,2,3,4,5,6,7,8]", "city8", "A", 5)
    assert set(np.unique(refit_select._cat(c8, d))) == {0, 1}
    assert (refit_select._cat(refit_select.parse_equation("f = #1; p1 = []; p2 = []", None, "A", 1), d) == 0).all()


def test_absolute_equation_maps_argument_positions_to_variable_numbers():
    """PySR prints #k as the k-th ARGUMENT of a template sub-expression (checked 11-03: data depending only on the
    column named x4, with x3 left out, prints #3). With s_health (x10) dropped, #10..#25 are x11..x26, so every
    equation read from a multi-city run must be renumbered before parsing."""
    import sr_data
    import refit_select as rs
    active = [v for v in sr_data.VARS if v != "s_health"]
    eq = "f = log(#5 + 1.0) - #9 * #10; g = nexp(#15); h = #25 ^ 2.0; p1 = [0.1, 0.2]; p2 = [1.0, 2.0]"
    out = rs.absolute_equation(eq, active)
    assert out == "f = log(#5 + 1.0) - #9 * #11; g = nexp(#16); h = #26 ^ 2.0; p1 = [0.1, 0.2]; p2 = [1.0, 2.0]"
    assert rs.absolute_equation(eq, None) == eq and rs.absolute_equation(eq, list(sr_data.VARS)) == eq
    c = rs.parse_equation(out, "city2", "A", 10)
    assert rs.XS[10] in c.comps["f"].free_symbols and rs.XS[9] not in c.comps["f"].free_symbols   # s_education, not s_health
    assert rs.XS[15] in c.comps["g"].free_symbols and rs.XS[25] in c.comps["h"].free_symbols


def test_select_renumbers_equations_of_runs_with_dropped_variables(tmp_path, monkeypatch):
    """select() reads config.json's active_vars and stores absolute numbering in its output (equation_pysr keeps
    the run's own string)."""
    import json
    import pandas as pd
    import refit_select as rs
    rd = tmp_path / "run"
    (rd / "pysr").mkdir(parents=True)
    import sr_data
    active = [v for v in sr_data.VARS if v != "s_health"]
    (rd / "config.json").write_text(json.dumps({"template": "single", "active_vars": active}))
    pd.DataFrame({"complexity": [4], "loss": [1.0], "equation": ["f = log(#15 + 1.0)"]}).to_csv(rd / "hall_of_fame.csv", index=False)
    hof, _ = rs._hof(rd)
    eqs = rs.run_equations(rd, json.loads((rd / "config.json").read_text()), hof)
    assert eqs == [("f = log(#16 + 1.0)", "f = log(#15 + 1.0)")]


def test_guesses_positional_and_template_shaped():
    """Round 2 of the search queue (11-04) seeds every run; the seeds are written with variable names and must reach
    PySR as positional #k of the run's active variables, with p1 / p2 expanded to the template's category count."""
    import json
    from pathlib import Path
    import sr_data
    import pysr_setup
    spec = json.loads((Path(__file__).resolve().parents[1] / "config/guesses.json").read_text())
    active = [v for v in sr_data.VARS if v != "s_health"]
    s = pysr_setup.guesses_for(spec, "A", "single", active)
    assert s[0] == {"f": "log(#6 + 1.0) - 1.7 * #2"} and "#13" in s[3]["f"] and "#14" not in s[3]["f"]   # s_office x14 -> #13
    c = pysr_setup.guesses_for(spec, "A", "city8", active)
    assert c[0] == {"f": "log(#6 + 1.0)", "g": "#2", "h": "#1", "p1": [-1.7] * 8, "p2": [0.0] * 8}
    assert c[3]["g"] == "nexp(#5)" and c[3]["h"] == "log(#1 + 2.7)"
    b = pysr_setup.guesses_for(spec, "B", "city8", active)
    assert b[0]["f"] == "log(#6 + 1.0)" and len(b[0]["p1"]) == 8
    assert pysr_setup.guesses_for(spec, "B", "single", active) == [{"f": "log(#6 + 1.0)"}, {"f": "log(#5 + 1.0) + log(#6 + 1.0)"}]
    full = pysr_setup.guesses_for(spec, "A", "single", list(sr_data.VARS))
    assert "#14" in full[3]["f"]                                                            # nothing dropped: x14 = #14
    assert pysr_setup.guesses_for(None, "A", "single", active) is None


def test_hof_of_a_run_stopped_before_its_first_front_is_empty(tmp_path):
    """Round 2's pooled runs were stopped before PySR wrote any front; the selection must skip them, not crash."""
    import refit_select as rs
    rd = tmp_path / "run"
    (rd / "pysr").mkdir(parents=True)
    hof, finished = rs._hof(rd)
    assert len(hof) == 0 and finished is False and list(hof.columns) == ["complexity", "loss", "equation"]
