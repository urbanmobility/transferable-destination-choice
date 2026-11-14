"""Paper figures, Tasks 4-9: every figure builds from the exported tables, keeps to its size, carries its panel
letters, shows keywords only, and draws one point per city or model where it should."""
import importlib

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import data
import style

pytestmark = pytest.mark.skipif(not data.available(), reason="the exported figure tables are not on this machine")

SPEC = {     # name: (width mm, panels)
    "fig1": (183, "abcd"), "fig2": (183, "abcd"), "fig3": (183, "abcd"), "fig4": (183, "abcd"),
    "fig5": (183, "abcde"), "fig6": (183, "abcd"), "fig7": (120, "abc"),
}


@pytest.fixture(scope="module")
def tables():
    return data.load()


@pytest.fixture(scope="module")
def built(tables):
    out = {n: importlib.import_module(n).build(tables) for n in SPEC}
    yield out
    plt.close("all")


@pytest.mark.parametrize("name", sorted(SPEC))
def test_size_panels_and_text(name, built):
    fig, src = built[name]
    w, panels = SPEC[name]
    wi, hi = fig.get_size_inches()
    assert wi == pytest.approx(w / 25.4) and hi * 25.4 <= style.MAX_H
    letters = sorted(t.get_text() for t in fig.findobj(plt.matplotlib.text.Text) if t.get_gid() == "panel-label")
    assert letters == list(panels)
    assert style.check_text(fig) == []
    # 2026-11-12: no figure carries an "exploratory" tag (nor grey axes for it): the legends say what is exploratory
    assert not [t for t in fig.findobj(plt.matplotlib.text.Text) if "exploratory" in t.get_text().lower() or "post hoc" in t.get_text().lower()]
    if name in ("fig4", "fig7"):                  # the panels that used to be greyed as exploratory have ordinary axes
        grey = plt.matplotlib.colors.to_rgba(style.PALETTE["grey"])
        assert not [a for a in fig.axes if a.spines["bottom"].get_visible() and a.spines["bottom"].get_edgecolor() == grey]
    assert src and all(len(df) for df in src.values())


def test_colours_keep_their_meaning(built):
    """Orange appears only where confirmation cities or a threshold are drawn: no orange line in the figures without
    a threshold, and in Fig. 1 orange fills only the confirmation row and the confirmation markers."""
    import matplotlib.colors as mc
    orange = mc.to_hex(style.PALETTE["confirmation"])
    for name in ("fig1", "fig2", "fig3", "fig5"):
        fig, _ = built[name]
        for ax in fig.axes:
            for line in ax.lines:
                assert mc.to_hex(line.get_color()) != orange, name
    fig, _ = built["fig1"]                              # held-out data are hatched in black, never orange
    n_orange = sum(1 for ax in fig.axes for pch in ax.patches if mc.to_hex(pch.get_facecolor()) == orange
                   or mc.to_hex(pch.get_edgecolor()) == orange)
    assert n_orange == 0


def test_hollow_model_markers_use_the_validated_grey(built):
    """Population-mass models are drawn with PALETTE["prior_pop"] (colour-blind checked against the teal)."""
    import matplotlib.colors as mc
    fig, _ = built["fig5"]
    edges = {mc.to_hex(c) for ax in fig.axes for coll in ax.collections for c in coll.get_edgecolor()}
    assert mc.to_hex(style.PALETTE["prior_pop"]) in edges and mc.to_hex(style.PALETTE["grey"]) not in edges


def test_fig1_counts(built, tables):
    _, src = built["fig1"]
    assert len(src["a"]) == 67 and src["a"]["role"].value_counts().to_dict() == {"discovery": 27, "lawtest": 26, "search": 8, "confirm": 6}
    assert not {"N", "population", "trips", "count"} & set(src["b_c"].columns)          # classes only
    assert set(src["b_c"]["flow_class"]) <= {0, 1, 2, 3} and set(src["b_c"]["poi_class"]) <= {1, 2, 3, 4, 5}
    assert src["c_grid"].to_dict("records") == [{"block": "search", "period": "train", "use": "fit"},
                                                {"block": "validation", "period": "validation", "use": "select"},
                                                {"block": "test", "period": "test", "use": "report"}]
    dd = src["d"]                                        # the small plots of the workflow strip are real results
    assert (dd["stage"] == "Hypotheses").sum() == 35 and (dd["stage"] == "Held-out test").sum() == 26
    assert not set(dd.loc[dd["stage"] == "Hypotheses", "item"]) & set(dd.loc[dd["stage"] == "Held-out test", "item"])
    b = dd[dd["stage"] == "Benchmark"]                   # our formula first, then four earlier models, all below it
    assert b["item"].tolist()[0] == "c0355r1" and len(b) == 5 and (b["x"].iloc[1:] < b["x"].iloc[0]).all()


def test_fig1_names_three_city_groups_and_a_search_subset(built):
    fig, _ = built["fig1"]
    texts = {t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)}
    assert {"Discovery  35", "formula-search subset  8", "Held-out  26", "Confirmation  6"} <= texts
    assert not any("Discovery," in t for t in texts)                       # no second "Discovery" group
    assert {"inner\nloop", "outer loop", "LLM agent", "symbolic regression", "generates formulas", "scoring",
            "ranked automatically", "proposes", "hypotheses", "Freeze", "3 formulas"} <= texts
    assert {"discovery fit", "supported", "not", "opened after freeze", "6 unseen", "test period & block", "in all 67 cities",
            "ours", "fit share", "24 benchmark versions"} <= texts
    assert {"35 cities", "26 cities", "8 cities", "32 cities"} <= texts
    assert not {"code", "fixed rules", "pre-registered", "67 cities", "6 cities"} & texts   # wording judged unclear in review


def test_fig2_counts(built):
    _, src = built["fig2"]
    assert src["a_b"][src["a_b"]["model"] == "Flexible"]["city"].nunique() == 67
    assert src["d"].groupby("model")["city"].nunique().to_dict() == {
        "Scaled kernel": 67, "Tempered power law": 67, "Two-regime power law": 67}


def test_fig3_counts_and_the_shape_hypotheses(built, tables):
    """Panel c: the six registered size hypotheses about the shape of the decay, each with its quantity and model. The
    transfer-loss hypothesis (P7-02, not supported) is not here — it must stay in Fig. 4a (2026-11-12)."""
    fig, src = built["fig3"]
    assert len(src["a"]) == 67 and len(src["b"]) == 67
    c = src["c"]
    assert c["id"].tolist() == ["H3-01", "H4-03", "H4-01", "H4-02", "H1-01", "P7-01"] and c["passed"].all()
    assert c["model"].tolist()[-1] == "Two-regime power law"
    p7 = src4 = built["fig4"][1]["a"]
    row = p7[p7["id"] == "P7-02"]
    assert len(row) == 1 and not bool(row["supported"].item())            # the failed hypothesis is still shown, in Fig. 4a
    texts = [t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)]
    assert {"Population\nslope of", "estimated with", "Held-out test", "supported", "Two-regime power law", "log distance",
            "Distance band (km)", "Discovery cities", "Held-out cities", "Confirmation cities", "Discovery fit"} <= set(texts)
    assert not any("Transfer" in s or "Decay" in s for s in texts) and texts.count("supported") == 1
    assert len(src["d"]) == 18 and src["d"]["band_to"].max() == 60
    assert (c["r_lo"] <= c["ratio"]).all() and (c["ratio"] <= c["r_hi"]).all()
    assert c["ratio"].to_numpy() == pytest.approx((c["test_slope"] / c["disc_slope"]).to_numpy())


def test_fig4_counts(built, tables):
    _, src = built["fig4"]
    assert len(src["a"]) == 10 and int(src["a"]["supported"].sum()) == 1
    assert len(src["b"]) == 69 and src["b"]["agree"].mean() == pytest.approx(tables["e1_summary"]["agree"])
    assert src["c"].shape == (14, 13) and int(src["c_fdr"].drop(columns="covariate").to_numpy().sum()) > 0
    assert len(src["d"]) == 67 and "e" not in src                   # the city-types panel is not drawn (2026-11-12)


def test_fig4_shows_the_criterion_row_by_row(built):
    """2026-11-12: what "supported" means must be visible — the Holm-adjusted p of every row, which reproduces
    the recorded verdicts, and a legend that names both intervals."""
    fig, src = built["fig4"]
    a = src["a"]
    assert ((a["p_holm"] <= 0.05) == a["holm"].astype(bool)).all() and (a["p_holm"] >= a["p"]).all()
    assert a["p_holm"].round(3).tolist()[:3] == [0.002, 1.0, 0.094]
    txt = {t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)}
    assert {"<0.01", "held-out 95% interval", "discovery 95% interval", "not supported", "69 further relations",
            "Slope, discovery cities", "Slope, held-out cities", "Standardised\nslope", "significant,\nFDR 10%"} <= txt
    assert not {"Effect (s.d.)", "FDR 0.10", "Cities typed by", "Number of types"} & txt


def test_fig5_shows_both_versions_and_all_four_contrasts(built, tables):
    _, src = built["fig5"]
    a = src["a"]
    assert len(a) == 13 and int(a["population"].notna().sum()) == 11 and int(a["poi"].notna().sum()) == 12
    assert set(a.loc[a["this_study"], "family"]) == {"strexp3", "S0"} and a["source"].str.len().gt(0).all()
    o = src["a_ours"]                                               # our three formulas, the scaled kernel first and highest
    assert o["model"].tolist() == ["c0355r1", "c0262", "c0514c1"] and o["share"].is_monotonic_decreasing
    assert o["share"].min() > a["poi"].max()                        # each above every earlier model (medians; only the first is tested)
    assert (a.dropna(subset=["population", "poi"])["poi"] > a.dropna(subset=["population", "poi"])["population"]).all()
    assert list(src["b"].columns) == ["city", "C1", "C2", "C4", "C3", "set"] and len(src["b"]) == 32
    assert len(src["e"]) == 29
    con = tables["p9_contrasts"]["contrasts"]
    assert [con[k]["wins"] for k in ("C1", "C2", "C4", "C3")] == [26, 26, 23, 18]


def test_fig5_sets_the_block_contrasts_beside_the_all_destination_ones(built, tables):
    """Panel c: the pre-specified held-out-block contrasts (primary) next to the all-destination ones, which were added
    afterwards — the legend says so, the figure carries no tag."""
    fig, src = built["fig5"]
    c = src["c"]
    assert c[["contrast", "scoring"]].to_records(index=False).tolist() == [
        ("Fit", "held-out block"), ("Fit", "all destinations"), ("Zero-shot", "held-out block"), ("Zero-shot", "all destinations")]
    assert c["wins"].tolist() == [26, 29, 26, 29] and (c["n"] == 32).all()
    con = tables["p9_contrasts"]["contrasts"]
    assert c.loc[c["scoring"] == "held-out block", "median"].tolist() == pytest.approx([con["C1"]["median"], con["C2"]["median"]])
    assert "post hoc" not in {t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)}


def test_fig5_names_what_the_reader_compares(built, tables):
    """Panel a names the two reference lines and gives name, source and number of constants of every model, our three
    formulas included; the three markers are named as choices of the destination mass; the opponent in b and c is named
    as in panel a and the fourth contrast says it is against uniform constants; panel d shows the four parameter sources."""
    fig, src = built["fig5"]
    texts = [t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)]
    assert {"gravity", "flexible model", "Model", "Source", "Constants", "Scaled kernel", "Tempered power law", "Two-regime power law",
            "Stretched exp., relaxed *", "after Guo 2026", "Kwon 2023", "Simini 2012", "Tiered gravity", "* baseline, this study",
            "Destination mass", "population", "POI count", "observed trips", "local trips", "no local trips", "vs uniform"} <= set(texts)
    assert texts.count("ours") == 3 and not {"Own trip volume", "own trips", "free constants", "vs uniform constants"} & set(texts)
    assert texts.count("vs Stretched exp. *") == 5 and "vs best prior" not in texts and "constants" not in texts
    assert texts.count("flexible model") == 2 and "flexible" not in texts      # the upper reference has one name (a and e)
    d = src["d"]
    assert d["model"].tolist() == ["c0355r1", "strexp3_N", "grav_power_N", "kwon"]
    assert {"local", "uniform", "size", "form"} <= set(d.columns) and d[["local", "uniform", "size", "form"]].notna().all().all()


def test_fig7_counts(built):
    _, src = built["fig7"]
    a = src["a"]
    assert len(a) == 67 and len(src["c"]) == 67
    big = a[a["pop"] > 3.99e6]
    assert (len(big), int((big["gain"] > 0).sum()), int((a[a["pop"] <= 3.99e6]["gain"] > 0).sum())) == (14, 12, 26)
    assert src["b"]["n"].tolist() == [53, 53, 14, 14]
    assert (int(((big["set"] == "source") & (big["gain"] > 0)).sum()), int((big["set"] == "source").sum())) == (9, 9)
    assert (int(((big["set"] == "target") & (big["gain"] > 0)).sum()), int((big["set"] == "target").sum())) == (3, 5)


def test_fig4_labels_say_what_the_measures_are():
    import fig4
    assert fig4.COV["jobs_housing"] == "Jobs–housing separation"          # the index is 1 for full separation
    assert fig4.COV["secondary_centres"] == "Subcentre − CBD jobs"        # a difference of job shares, not a count


def test_fig1_labels_the_eight_largest_metros(built, tables):
    fig, _ = built["fig1"]
    import fig1
    top = tables["cities"].sort_values("population", ascending=False).head(8)["city"]
    want = {fig1.ABBR[c] for c in top}
    shown = {t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)} & set(fig1.ABBR.values())
    assert shown == want and "DFW" in want and "SEA" not in want


def test_lines_and_marks_are_named_in_the_figure(built):
    """Self-review of Figs 1 and 5 (2026-11-12): every fitted or reference line and every
    filled/hollow distinction is named where it is drawn, and the words found unclear are gone."""
    def texts(name):
        return {t.get_text() for t in built[name][0].findobj(plt.matplotlib.text.Text)}
    t2, t3, t4, t6 = texts("fig2"), texts("fig3"), texts("fig4"), texts("fig7")
    assert {"flexible model", "gravity at 0", "Flexible model, 67 cities", "median", "City population (million)"} <= t2
    assert not {"flexible", "Flexible model", "67 cities"} & t2 and not any("Constant" in s for s in t2)
    assert {"Discovery fit", "no effect", "same slope"} <= t3 and not {"none", "full"} & t3
    assert {"no effect", "same slope", "opposite sign", "Fit, all cities", "Discovery cities"} <= t4 and not {"none", "full"} & t4
    for name in ("fig2", "fig3", "fig4", "fig5", "fig7"):            # no city-set legend drops the word "cities"
        assert not {"Discovery", "Held-out", "Confirmation"} & texts(name), name
    assert any(s.startswith("same sign") for s in t4) and any(s.startswith("median ratio") for s in t4)
    assert {"running median", "size-predicted better", "discovery fit", "Gain over uniform"} <= t6
    assert not any("source" in s or "target" in s for s in t6)


def test_fig2_share_axis_starts_where_the_data_begin(built, tables):
    import fig2
    fig, src = built["fig2"]
    assert src["d"]["share"].min() >= fig2.SHARE_MIN           # nothing is cut off on the left
    assert int((src["d"]["share"] > fig2.SHARE_MAX).sum()) == 4  # the four off-scale shares are still named by value


def test_fig2_shows_the_whole_model_and_all_three_formulas_in_b(built):
    """Second design (2026-11-12): panel c gives the choice probability as well as the utilities; the
    elasticities are defined in the panels that plot them; panel b draws all three formulas; no symbol list in c."""
    import fig2
    fig, src = built["fig2"]
    texts = [t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)]
    assert {fig2.PROB, fig2.E_D, fig2.E_N, "Destination choice probability"} <= set(texts)
    assert all(texts.count(fig2.EQ[f[0]]) == 1 for f in fig2.FORMULAS) and "Elasticities" not in texts
    assert any(s.endswith("turning distance") for s in texts)                     # theta_1 is named where its formula stands
    assert not any("tract" in s or "POI count" in s for s in texts)               # the symbols are explained in the legend
    ax_b = next(a for a in fig.axes if a.get_ylabel().startswith("Size elasticity"))
    heavy = [ln for ln in ax_b.lines if ln.get_linewidth() >= 1.2]
    assert len(heavy) == 3                                                        # the three formulas, not one
    assert set(src["a_b"]["model"]) == {"Flexible", "Scaled kernel", "Tempered power law", "Two-regime power law"}


def test_fig6_shows_flows_as_classes_and_counts_what_it_says(built, tables):
    """Fig. 6: the 2,000 largest flows of Denver, observed and predicted with constants no Denver
    trip entered. Only rank tiers are drawn and written (the raw data are licensed); the number under every map is the
    number of that model's 2,000 largest flows that are also among the 2,000 largest observed."""
    import fig6
    fig, src = built["fig6"]
    a = src["a"]
    tiers = [c for c in a.columns if c.startswith("tier_")]
    assert len(tiers) == 8 and all(c == "tier_observed" or c.endswith("|uniform") for c in tiers)     # the uniform constants only
    assert set(a[tiers].to_numpy().ravel()) <= {0, 1, 2, 3, 4, 5} and (a[tiers] > 0).sum().eq(fig6.N_TOP).all()
    assert not {"flow", "trips", "count", "T"} & set(a.columns)
    sh = src["a_shared"].set_index("model")["in_observed_top"]
    for m in sh.index:
        assert sh[m] == int(((a["tier_observed"] > 0) & (a[f"tier_{m}|uniform"] > 0)).sum())
    ov = tables["od"]["overlap"].set_index(["source", "K"])["overlap"]
    assert all(sh[m] / fig6.N_TOP == pytest.approx(ov[(f"{m}|uniform", fig6.N_TOP)]) for m in sh.index)   # the same number as panel b
    assert sh.idxmax() == "c0355r1"
    txt = [t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)]
    assert "Top 2,000 flows" in txt and f"{sh['c0355r1']:,} in observed top 2,000" in txt and not any("also observed" in s for s in txt)
    assert {"Observed, Denver", "Scaled kernel, ours", "Gravity", "Radiation", "Stretched exponential", "as published", "with POI count",
            "zero-shot predictions"} <= set(txt)                     # the figure says these are transfers, not fits
    assert txt.count("as published") == 4 and txt.count("with POI count") == 4      # the rows of a, b's legend, the halves of c, the groups of d
    assert not any("population" in s.lower() for s in txt)          # population is the published setting: no suffix


def test_fig6_curves_and_cpc_cover_the_same_seven_models(built):
    _, src = built["fig6"]
    models = src["d"]["model"].tolist()
    assert models == ["c0355r1", "grav_power_P", "grav_power_N", "rad_P", "rad_N", "strexp1_P", "strexp1_N"]
    assert set(src["b"]["source"]) == {f"{m}|uniform" for m in models} and sorted(src["b"]["K"].unique()) == [100, 200, 500, 1000, 2000, 5000]
    c = src["c"]
    assert c["observed"].sum() == pytest.approx(1.0) and all(c[f"{m}|uniform"].sum() == pytest.approx(1.0) for m in models)
    assert src["d"].set_index("model")["cpc"].idxmax() == "c0355r1"
    assert src["d"]["version"].tolist() == ["ours"] + ["as published", "with POI count"] * 3


def test_fig6_every_line_has_a_legend_entry(built):
    """Panel b has seven lines and seven legend entries; panel c has its own; panel d lists six rows, the published
    versions above the POI-count versions."""
    fig, _ = built["fig6"]
    ax_b = next(a for a in fig.axes if a.get_ylabel() == "Share also predicted")
    legends = [c for c in ax_b.get_children() if isinstance(c, plt.matplotlib.legend.Legend)]
    assert len(ax_b.lines) == 7 and sum(len(lg.get_texts()) for lg in legends) == 7
    assert sorted(lg.get_title().get_text() for lg in legends) == ["", "as published", "with POI count"]
    ax_c = next(a for a in fig.axes if a.get_ylabel() == "Predicted / observed trips")
    assert [t.get_text() for t in ax_c.get_legend().get_texts()] == ["Scaled kernel, ours", "Gravity", "Radiation", "Stretched exponential"]
    ax_d = next(a for a in fig.axes if a.get_xlabel() == "CPC")
    assert [t.get_text() for t in ax_d.get_yticklabels()] == ["Gravity", "Radiation", "Stretched exponential"] * 2
    assert ax_d.get_legend() is None


def test_the_od_figure_is_drawn_the_same_way_for_every_confirmation_city(tables):
    """The supplement shows the Fig. 6 drawing for the other five confirmation cities. All six are
    exported with the same settings; each builds, keeps to the keyword rule and counts what it says."""
    import fig6
    cities = data.od_cities()
    conf = set(tables["cities"].loc[tables["cities"]["role"] == "confirm", "city"])
    assert cities[0] == data.DENVER and set(cities) == conf and len(cities) == 6
    for city in cities[1:]:
        od = data.load_od(city)
        fig, src = fig6.build(tables, od)
        assert style.check_text(fig) == []
        a, sh = src["a"], src["a_shared"].set_index("model")["in_observed_top"]
        assert (a.filter(like="tier_") > 0).sum().eq(fig6.N_TOP).all() and len(sh) == 7
        txt = {t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)}
        name = od["summary"]["name"].split(",")[0].split("-")[0]
        assert f"Observed, {name}" in txt and "Denver" not in " ".join(txt) and f"{sh['c0355r1']:,} in observed top 2,000" in txt
        plt.close(fig)


def test_the_six_city_table_says_what_the_ledger_says(tmp_path, monkeypatch):
    """Supplementary table (2026-11-12): the Fig. 6 numbers for all six confirmation cities, in place of the five
    figures that are not shown — ours first in every city at K = 2,000 and in CPC, and the cells where it is not first."""
    import supp_tables as st
    monkeypatch.setattr(st, "OUT", tmp_path)
    out = st.build()
    t = pd.read_csv(out[0], dtype={"city": str})
    assert len(t) == 6 * 7 and t["city"].nunique() == 6
    for col in ("in_observed_top", "cpc"):
        assert (t.loc[t.groupby("city")[col].idxmax(), "model"] == "c0355r1").all()
    den = t[(t["city"] == data.DENVER)].set_index("model")["in_observed_top"]
    assert den["c0355r1"] == 1164 and den["grav_power_N"] == 1102
    ex = st.exceptions(pd.read_csv(out[1], dtype={"city": str}))
    assert sorted(zip(ex["name"], ex["K"])) == [("Philadelphia", 100), ("Philadelphia", 200), ("Phoenix", 200), ("Phoenix", 500),
                                               ("Sacramento", 100)]
    assert (ex["best_other"] == "strexp1_N").all()
    md = out[2].read_text()
    assert "Philadelphia, K = 100" in md and "| Scaled kernel (ours) | 1,164 | 1,002 | 1,168 | 1,202 | 1,164 | 1,098 |" in md



def test_fig3_revision_labels(built):
    """Revision 2026-11-13: the turning distance is r (not theta_1, which the Scaled kernel uses), drawn with a
    log-log line; the variable-exponent row says that its elasticity is taken at 5 km."""
    fig, _ = built["fig3"]
    texts = {t.get_text() for t in fig.findobj(plt.matplotlib.text.Text)}
    assert "log-log slope" in texts and "Variable exponent (5 km)" in texts
    assert not any(r"\theta_1" in s for s in texts)
