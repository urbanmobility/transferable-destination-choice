"""Exploration final-review fixes: the stages (fix, e1, e2, types and the two reports) run end to end on a small
synthetic world in a temporary directory; BH takes the normal-approximation p (I1); tract_area is a confounder, not a
form cell (I2); the type results are on disk before any plotting and survive a plotting failure (I3)."""
import json

import numpy as np
import pandas as pd
import pytest

import city_outcomes as co
import city_types as ct
import explore_form as ef
import ledger
import p7seal

BANDS = ["0_1km", "1_2km", "2_5km", "5_10km", "10_20km"]
CURVES = [f"{q}_{b}" for b in BANDS for q in ("ed", "eN")]
FORM = ["x1", "x2", "x3", "x4"]
OUTCOMES = CURVES + ["y_strong", "y_ta", "y_null"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """44 cities (discovery 24, law-test 16, confirmation 4) of two types: the behaviour curves and the family support
    follow the type; y_strong follows x1, y_ta follows tract_area; everything else is noise."""
    rng = np.random.default_rng(7)
    roles = {"search": [f"s{i}" for i in range(4)], "discovery": [f"d{i}" for i in range(20)],
             "lawtest": [f"l{i}" for i in range(16)], "confirm": [f"f{i}" for i in range(4)]}
    groups = {"discovery": roles["search"] + roles["discovery"], "lawtest": roles["lawtest"], "confirm": roles["confirm"]}
    cities = sum(groups.values(), [])
    n = len(cities)
    typ = np.arange(n) % 2
    form = pd.DataFrame({"city": cities, "log_population": rng.normal(13, 1, n), "log_jobs": rng.normal(12, 1, n),
                         "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    extra = pd.DataFrame({"city": cities, "x3": rng.normal(size=n), "x4": rng.normal(size=n),
                          "tract_area": rng.normal(size=n)})
    out = pd.DataFrame({"city": cities, "block": "test_dest", "log_population": form["log_population"],
                        "cls": np.where(typ == 0, "kernel", "opp")})
    for b in BANDS:
        out[f"ed_{b}"] = -1.0 - typ + rng.normal(scale=0.1, size=n)
        out[f"eN_{b}"] = 0.8 - 0.4 * typ + rng.normal(scale=0.05, size=n)
    out["y_strong"] = 2 * form["x1"] + rng.normal(scale=0.2, size=n)
    out["y_ta"] = 2 * extra["tract_area"] + rng.normal(scale=0.2, size=n)
    out["y_null"] = rng.normal(size=n)
    p7 = tmp_path / "p7"
    p7.mkdir()
    for g, cs in groups.items():
        fo, fc = ef.GROUP_FILES[g]
        out[out["city"].isin(cs)].to_parquet(p7 / fo)
        extra[extra["city"].isin(cs)].to_csv(p7 / fc, index=False)
    rel = [(o, c, 0.0, 1.0, 0.1, 0.3, 0.2, 0.1) for o in ("y_strong", "y_null", "ed_5_10km") for c in FORM]
    rel += [("y_ta", "tract_area", 0.0, 1.0, 0.1, 0.3, 0.2, 0.1), ("y_null", "log_population", 0.1, 0.2, np.nan, np.nan, np.nan, 1.0)]
    pd.DataFrame(rel, columns=["outcome", "covariate", "ci_lo", "ci_hi", "pci_lo", "pci_hi", "partial_slope",
                               "r_size"]).to_csv(p7 / "relations.csv", index=False)
    cfg = {"seed": 1, "groups": list(groups), "outcomes": OUTCOMES,
           "covariates": ["log_population", "log_jobs"] + FORM + ["tract_area"], "size": "log_population",
           "e1": {"exclude_covariates": ["log_population", "log_jobs", "tract_area"], "confounders": ["tract_area"],
                  "size_proxy_r": 0.8, "permutations": 50, "variants": ["main", "with_registered", "with_confounders"]},
           "e2": {"bootstrap": 50, "fdr_q": 0.1, "fdr_p": "normal", "family_exclude": ["log_population", "log_jobs"],
                  "confounders": ["tract_area"], "tract_area_control": True,
                  "axes": {"exclude": ["log_population", "log_jobs", "tract_area"], "var_target": 0.7, "max_axes": 5,
                           "tract_area_control": True}},
           "e3": {"families": {"K": ["k1"], "O": ["o1"]}, "floor": "global_min", "behaviour_bands": BANDS, "k": [2, 3],
                  "stability": {"frac": 0.8, "B": 20, "min_ari": 0.6},
                  "prediction_strength": {"train": "discovery", "test": "lawtest", "min": 0.8}},
           "e4": {"bootstrap": 50, "kw_fdr_q": 0.1, "logit_C": 1.0}}
    (tmp_path / "explore_form.json").write_text(json.dumps(cfg))
    share = {"k1": np.where(typ == 0, 0.99, 0.80), "o1": np.where(typ == 0, 0.80, 0.99)}
    cv = pd.DataFrame([(c, cid, 4.2 - (share[cid][i] + rng.normal(scale=0.005)) * 0.2, True)
                       for i, c in enumerate(cities) for cid in ("k1", "o1")], columns=["city", "cid", "val_nll", "finite"])
    refs = {c: {"grav_power": {"nll_per_visit": 4.2}, "M2_glm": {"nll_per_visit": 4.0}} for c in cities}
    lines = []
    monkeypatch.setattr(ledger, "add", lines.append)
    monkeypatch.setattr(ef, "CONFIG", tmp_path / "explore_form.json")
    monkeypatch.setattr(ef, "OUT", tmp_path / "explore")
    monkeypatch.setattr(ef, "FIXED", tmp_path / "explore" / "CONFIG_FIXED")
    monkeypatch.setattr(ef.config, "results_dir", lambda line=None: tmp_path)
    monkeypatch.setattr(ef.config, "cities_with_role", lambda r: roles[r])
    monkeypatch.setattr(ef.laws, "covariates", lambda cs, allow_confirm=False: form.set_index("city").loc[list(cs)].reset_index())
    monkeypatch.setattr(p7seal, "require_signed", lambda: [{"outcome": "y_strong", "covariate": "x2"}])
    monkeypatch.setattr(co, "_cv", lambda block: cv)
    monkeypatch.setattr(co, "_refs", lambda block, cs: refs)
    monkeypatch.setattr(ct, "city_xy", lambda cs: pd.DataFrame({"x": 0.0, "y": 0.0}, index=list(cs)))
    monkeypatch.setattr(ct, "plot_map", lambda xy, labels, path, title="": path.write_bytes(b"png"))
    return {"out": tmp_path / "explore", "ledger": lines, "n": n}


def test_stage_e1_runs_end_to_end(world):
    ef.main(["fix"])
    s = ef.main(["e1"])
    assert s["main"]["n_pairs"] == 11 and s["with_registered"]["n_pairs"] == 12 and s["with_confounders"]["n_pairs"] == 12
    assert 0 < s["main"]["perm_p"] <= 1 and s["main"]["perm_B"] == 50
    pairs = pd.read_csv(world["out"] / "e1_pairs.csv")
    assert len(pairs) == 35 and set(pairs["variant"]) == {"main", "with_registered", "with_confounders"}
    assert json.loads((world["out"] / "e1_summary.json").read_text())["main"]["agree"] == s["main"]["agree"]
    assert len(world["ledger"]) == 2 and "E-1" in world["ledger"][1]


def test_stage_e2_fdr_takes_the_normal_p_so_an_isolated_strong_cell_passes(world):
    """52 form cells and B = 50: the bootstrap p cannot go below 1/50 = 0.02 > the rank-1 BH threshold 0.1/52."""
    ef.main(["fix"])
    e = ef.main(["e2"]).set_index(["outcome", "covariate"])
    strong = e.loc[("y_strong", "x1")]
    assert strong["p"] == pytest.approx(1 / 50) and strong["p_norm"] < 1e-6 and bool(strong["bh"])
    from scipy.stats import norm
    assert strong["p_norm"] == pytest.approx(2 * norm.sf(abs(strong["std_slope"] / strong["se"])), rel=1e-9)
    assert int(e["bh"].sum()) < 10                                                     # the noise cells do not pass


def test_stage_e2_keeps_tract_area_out_of_the_form_family(world):
    ef.main(["fix"])
    e = ef.main(["e2"])
    assert e.groupby("role").size().to_dict() == {"confounder": 13, "form": 52, "size": 26}
    ta = e[e["covariate"] == "tract_area"].set_index("outcome")
    assert not ta["bh"].any() and ta.loc["y_ta", "eb"] == ta.loc["y_ta", "std_slope"] and ta.loc["y_ta", "p_norm"] < 1e-6
    saved = pd.read_csv(world["out"] / "e2_effects.csv")
    assert list(saved["role"]) == list(e["role"])
    assert "52 cells beyond size" in world["ledger"][-1]
    ef.main(["e1"])
    md = ef.main(["report"]).read_text()
    assert "52 cells beyond size" in md and "| tract_area |" not in md and "Tract area" in md and "normal approximation" in md
    assert "nan" not in md.lower() and "| y_strong | x1 |" in md


def _types_inputs():
    ef.main(["fix"])
    ef.main(["e2"])


def test_stage_types_and_report_run_end_to_end(world):
    _types_inputs()
    res = ct.STAGES["types"]()
    sk, bh = res["features"]["skeleton"], res["features"]["behaviour"]
    assert sk["n"] == world["n"] and sk["k"] in (2, 3) and bh["k"] in (2, 3) and "agreement_ari" in res
    assert sum(t["n"] for t in sk["describe"]["types"].values()) == world["n"]
    assert sk["map"] == "types_skeleton_map.png" and (world["out"] / sk["map"]).exists()
    assert json.loads((world["out"] / "types.json").read_text())["features"]["skeleton"]["k"] == sk["k"]
    members = pd.read_csv(world["out"] / "types_members.csv")
    assert len(members) == world["n"] and {"skeleton", "behaviour"} <= set(members.columns)
    assert "E-3" in world["ledger"][-1]
    md = ct.STAGES["report"]().read_text()
    assert "Exploratory" in md and f"**k = {sk['k']}**" in md and "nan" not in md.lower()


def test_stage_types_writes_results_before_plotting_and_survives_a_plot_error(world, monkeypatch):
    _types_inputs()
    seen = []

    def boom(xy, labels, path, title=""):
        seen.append(all((world["out"] / f).exists() for f in ("types.json", "types_members.csv", "types_skeleton_features.csv")))
        raise RuntimeError("no display")

    monkeypatch.setattr(ct, "plot_map", boom)
    res = ct.STAGES["types"]()
    assert seen and all(seen)                                                          # results on disk before any plot
    saved = json.loads((world["out"] / "types.json").read_text())
    assert saved["features"]["skeleton"]["k"] == res["features"]["skeleton"]["k"]
    assert "RuntimeError: no display" in saved["features"]["skeleton"]["map"]
    assert "E-3" in world["ledger"][-1]


def test_stage_axes_ta_adds_the_axis_level_tract_area_sensitivity_without_touching_e2(world):
    ef.main(["fix"])
    ef.main(["e1"])
    ef.main(["e2"])
    before = {f: (world["out"] / f).read_bytes() for f in ("e2_effects.csv", "e2_axis_effects.csv", "e2_axes.json")}
    ta = ef.main(["axes_ta"])
    assert all((world["out"] / f).read_bytes() == b for f, b in before.items())       # the E-2 files are not rewritten
    saved = pd.read_csv(world["out"] / "e2_axis_effects_ta.csv")
    assert len(saved) == len(ta) and set(saved["term"]) >= {"log_population", "axis1", "tract_area"}
    assert saved.groupby("outcome").size().nunique() == 1 and set(saved["outcome"]) == set(OUTCOMES)
    row = saved[(saved["outcome"] == "y_ta") & (saved["term"] == "tract_area")].iloc[0]
    assert row["ci_lo"] > 0.5                                                          # y_ta follows tract_area
    assert "axis-level tract-area sensitivity" in world["ledger"][-1]
    md = ef.main(["report"]).read_text()
    assert "With tract area as an extra control" in md and "nan" not in md.lower()


def test_stage_axes_ta_refuses_a_config_without_the_switch(world):
    cfg = json.loads(ef.CONFIG.read_text())
    del cfg["e2"]["axes"]["tract_area_control"]
    ef.CONFIG.write_text(json.dumps(cfg))
    ef.main(["fix"])
    ef.main(["e2"])
    with pytest.raises(ValueError, match="tract_area_control"):
        ef.main(["axes_ta"])
    md_inputs = ef.main(["e1"])
    assert md_inputs and "With tract area as an extra control" not in ef.main(["report"]).read_text()     # no sensitivity file, no line
