"""Fig. 3 | City size shapes the law.

Claim: the larger the city, the steeper the mid-distance decay and the shorter the turning distance — also in cities
that took no part in finding it.
  a  flexible-model distance elasticity at 5-10 km against population (hero panel); line fitted on discovery cities
  b  turning distance of the two-regime power law against population; an inset sketches what turns (the decay
     goes from slope -1 to slope -2 around theta_1)
  c  the six registered size hypotheses about the shape of the decay (one law in five formulations, and the turning
     distance): held-out slope as a fraction of the discovery slope. Every row names its quantity and its model. The
     registered hypothesis on transfer loss (not supported) is in Fig. 4a, not here.
  d  decay profiles of small, medium and large cities, as steps over the six distance bands
Second design, 2026-11-12.
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

import stats
import style

POP_TICKS = [1e6, 2e6, 5e6, 1e7, 2e7]


def _pop_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(5e5, 2.4e7)
    ax.set_xticks(POP_TICKS, labels=["1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("Population (million)")


def _size_panel(ax, t, ycol, law, unit="", loglog=False):
    """Discovery-city OLS line with its band, all cities as points. loglog: the line is fitted to ln y (drawn back on
    the linear axis) and the text gives the log-log slopes, the held-out one with its city-bootstrap interval."""
    disc = t[t["set"] == "discovery"]
    grid = np.linspace(np.log(5.5e5), np.log(2.2e7), 60)
    yd = np.log(disc[ycol]) if loglog else disc[ycol]
    fit, lo, hi, slope = stats.ols_band(disc["log_population"], yd, grid)
    tf = np.exp if loglog else (lambda v: v)
    ax.fill_between(np.exp(grid), tf(lo), tf(hi), color=style.PALETTE["light"], lw=0, zorder=1)
    ax.plot(np.exp(grid), tf(fit), color=style.PALETTE["discovery"], lw=1.0, zorder=2)
    style.scatter_sets(ax, np.exp(t["log_population"]), t[ycol], t["set"], s=9)
    _pop_axis(ax)
    num = lambda v: f"{v:.2f}".replace("-", "\u2212")  # noqa: E731
    if loglog:
        import rev_tables
        h = t[t["set"] == "heldout"]
        b, l, u = rev_tables.ols_boot(np.log(h[ycol]), h["log_population"])
        law = {"disc_slope": slope, "test_slope": b, "ci_lo": l, "ci_hi": u}
    return (f"discovery  {num(law['disc_slope'])}\n"
            f"held-out  {num(law['test_slope'])} [{num(law['ci_lo'])}, {num(law['ci_hi'])}]")


def turning_inset(ax):
    """What turns: the two-regime power law, d^-1 (d + theta_1)^-1, falls with slope -1 before theta_1 and -2 after it."""
    P = style.PALETTE
    ins = ax.inset_axes([0.60, 0.43, 0.37, 0.26])
    th = 5.0
    x = np.geomspace(0.25, 100, 200)
    yk = np.log(1 / (x * (x + th)))
    ins.plot(np.log(x), yk, color=P["ours_3"], lw=1.4, zorder=3)
    y0 = np.log(1 / (th * 2 * th))
    ins.plot([np.log(th)] * 2, [yk.min() - 0.3, y0], color=P["ref"], lw=0.5, ls=(0, (3, 2)), zorder=2)
    ins.scatter([np.log(th)], [y0], s=10, facecolor="white", edgecolor=P["ours_3"], linewidth=0.9, zorder=4)
    ins.text(np.log(th) + 0.25, yk.min() + 0.25, r"$r$", fontsize=6.5, ha="left", va="bottom", color=P["heldout"])
    ins.text(np.log(0.32), np.log(1 / (0.6 * (0.6 + th))) - 3.3, r"slope $-1$", fontsize=5.5, color=P["grey"], ha="left", va="center")
    ins.text(np.log(95), np.log(1 / (22 * (22 + th))) + 1.5, r"slope $-2$", fontsize=5.5, color=P["grey"], ha="right", va="center")
    ins.set_title("Two-regime power law", fontsize=6, color=P["ours_3"], fontweight="bold", pad=2.5, loc="left")   # colour as in Fig. 2
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_xlabel("log distance", fontsize=5.5, color=P["grey"], labelpad=1.5)
    ins.set_ylabel("log odds", fontsize=5.5, color=P["grey"], labelpad=1.5)
    ins.set_xlim(np.log(0.25), np.log(100))
    ins.set_ylim(yk.min() - 0.3, yk.max() + 0.6)
    for sp in ins.spines.values():
        sp.set_linewidth(0.4)
        sp.set_edgecolor(P["grey"])
    return ins


EDGES = [0.5, 1, 2, 5, 10, 20, 60]            # the six distance bands of the flexible model, on the plotted range
SHAPE_LAWS = ["H3-01", "H4-03", "H4-01", "H4-02", "H1-01", "P7-01"]      # the registered size hypotheses about the shape of the decay
MODEL = ["Flexible model", "Scaled kernel", "Tempered power law", "Two-regime power law", "Variable exponent (5 km)",
         "Two-regime power law"]


def build(d: dict):
    style.apply()
    fig = style.figure(style.W2, 108)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1.12, 1], hspace=0.50, wspace=0.24,
                          **style.layout(fig, 13, 5, 6, 10))
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    low = gs[1, :].subgridspec(1, 2, width_ratios=[1.62, 1], wspace=0.24)       # c needs room for two label columns
    ax_c, ax_d = fig.add_subplot(low[0]), fig.add_subplot(low[1])
    cities, laws = d["cities"], d["size_laws"].set_index("id")
    flex = d["bands"][(d["bands"]["model"] == "Flexible") & (d["bands"]["band"] == 3)][["city", "ed"]]
    ta = cities.merge(flex, on="city")
    G = style.PALETTE["grey"]

    # ---- a
    txt = _size_panel(ax_a, ta, "ed", laws.loc["H3-01"])
    ax_a.set_ylabel(r"$e_d$ at 5–10 km")
    ax_a.text(0.97, 0.96, "slope", transform=ax_a.transAxes, ha="right", va="top", fontsize=6, color=G)
    ax_a.text(0.97, 0.89, txt, transform=ax_a.transAxes, ha="right", va="top", fontsize=6, linespacing=1.25)
    style.set_legend(ax_a, loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=1.6,
                     extra=[Line2D([], [], color=style.PALETTE["discovery"], lw=1.0, label="Discovery fit")])
    style.panel_label(ax_a, "a", dx=30)

    # ---- b: the turning distance, with a sketch of what turns
    tb = d["turning"]
    txt = _size_panel(ax_b, tb, "theta_turn", laws.loc["P7-01"], loglog=True)      # revision: log-log line (no zero crossing)
    ax_b.set_ylabel(r"Turning distance $r$ (km)")
    turning_inset(ax_b)
    ax_b.set_ylim(0, 21)
    ax_b.set_yticks([0, 5, 10, 15, 20])
    ax_b.text(0.97, 0.96, "log-log slope", transform=ax_b.transAxes, ha="right", va="top", fontsize=6, color=G)
    ax_b.text(0.97, 0.89, txt, transform=ax_b.transAxes, ha="right", va="top", fontsize=6, linespacing=1.25)
    style.panel_label(ax_b, "b", dx=24)

    # ---- c: replication forest of the six shape hypotheses. The third registered size hypothesis (transfer loss grows
    # with size, P7-02, not supported) is not about the shape of the decay and would read ambiguously here;
    # it is shown in Fig. 4a, row 2, and the legend of this figure must say so.
    L = d["size_laws"].set_index("id").loc[SHAPE_LAWS].reset_index()
    if not L["passed"].all():
        raise ValueError("panel c writes one verdict for all its rows; a row that is not supported needs a new layout")
    r = [stats.replication_ratio(a, b, lo, hi) for a, b, lo, hi in zip(L["disc_slope"], L["test_slope"], L["ci_lo"], L["ci_hi"])]
    L["ratio"], L["r_lo"], L["r_hi"] = zip(*r)
    L["model"] = MODEL
    y = np.arange(len(L))[::-1].astype(float)
    y[5:] -= 0.5                                                    # a gap below the five formulations of one law
    w_mm = fig.get_size_inches()[0] / style.MM
    pos = ax_c.get_position()
    ax_c.set_position([pos.x0 + 43 / w_mm, pos.y0, pos.width - (43 + 16) / w_mm, pos.height])   # two label columns, one verdict column

    def col(mm):                                  # x = mm from the left edge of the axes (negative = left of it), y = data
        return blended_transform_factory(ax_c.transAxes + ScaledTranslation(mm * style.MM, 0, fig.dpi_scale_trans), ax_c.transData)
    for xv in (0, 1):
        ax_c.axvline(xv, color=style.PALETTE["ref"], lw=0.5, ls=(0, (4, 3)) if xv else "-", zorder=1)
    for yi, row in zip(y, L.itertuples()):
        ax_c.plot([row.r_lo, row.r_hi], [yi, yi], color=style.PALETTE["heldout"], lw=0.9, zorder=3)
        ax_c.scatter([row.ratio], [yi], s=16, zorder=4, marker="o", linewidth=0.8, facecolor=style.PALETTE["heldout"],
                     edgecolor=style.PALETTE["heldout"])
    top = y.max() + 0.95
    kw = {"fontsize": 6.5, "va": "center", "ha": "left", "clip_on": False}
    hd = {"fontsize": 6, "color": G, "va": "bottom", "ha": "left", "clip_on": False}
    for yi, m in zip(y, MODEL):
        ax_c.text(0, yi, m, transform=col(-27), **kw)
    ax_c.text(0, (y[0] + y[4]) / 2, r"$e_d$ at" + "\n5–10 km", transform=col(-42), linespacing=1.05, **kw)
    ax_c.plot([0, 0], [y[4] - 0.3, y[0] + 0.3], transform=col(-28.6), color=G, lw=0.5, clip_on=False)
    ax_c.text(0, y[5], "Turning\n" + r"distance $r$", transform=col(-42), linespacing=1.05, **kw)
    ax_c.text(0, top, "Population\nslope of", transform=col(-42), linespacing=1.05, **hd)       # the x axis of panels a and b, by name
    ax_c.text(0, top, "estimated with", transform=col(-27), **hd)
    ax_c.text(1, top, "Held-out test", transform=col(1.5), **hd)
    ax_c.plot([1, 1], [y.min() - 0.3, y.max() + 0.3], transform=col(1.5), color=G, lw=0.5, clip_on=False)   # one verdict for all six
    ax_c.text(1, (y.min() + y.max()) / 2, "supported", transform=col(3.0), fontsize=6, va="center", ha="left",
              color=style.PALETTE["heldout"], clip_on=False)
    ax_c.text(1.0, top, "same slope", ha="center", va="bottom", fontsize=6, color=style.PALETTE["ref"])
    ax_c.text(0.0, top, "no effect", ha="center", va="bottom", fontsize=6, color=style.PALETTE["ref"])
    ax_c.set_yticks([])
    ax_c.spines["left"].set_visible(False)
    ax_c.set_xlim(-0.6, 1.75)
    ax_c.set_ylim(y.min() - 0.7, y.max() + 0.9)
    ax_c.set_xlabel("Held-out / discovery slope")
    style.panel_label(ax_c, "c", dx=30 + 43 / 25.4 * 72)

    # ---- d: profiles by size, drawn as steps — the flexible model has one elasticity per distance band
    f = d["bands"][d["bands"]["model"] == "Flexible"].merge(cities[["city", "log_population"]], on="city")
    lab = dict(zip(cities["city"], stats.size_terciles(cities["log_population"])))
    f["size"] = f["city"].map(lab)
    cmap = style.size_cmap()
    src_d = []
    for name, cpos in (("small", 0.12), ("medium", 0.55), ("large", 0.97)):
        g = f[f["size"] == name].groupby("band").agg(x=("d_ref", "median"), y=("ed", "median"),
                                                    q1=("ed", lambda v: v.quantile(0.25)), q3=("ed", lambda v: v.quantile(0.75)))
        c = cmap(cpos)
        ax_d.stairs(g["q3"].to_numpy(), EDGES, baseline=g["q1"].to_numpy(), fill=True, color=c, alpha=0.20, lw=0, zorder=1)
        ax_d.stairs(g["y"].to_numpy(), EDGES, baseline=None, color=c, lw=1.3, zorder=3)
        ax_d.plot([], [], color=c, lw=1.3, label=name)
        src_d.append(g.reset_index().assign(size=name, band_from=EDGES[:-1], band_to=EDGES[1:]))
    ax_d.set_xscale("log")
    ax_d.set_xlim(0.5, 60)
    ax_d.legend(loc="lower left", title="City population", title_fontsize=6, handlelength=1.8, labelspacing=0.3, alignment="left")
    ax_d.set_xticks([1, 2, 5, 10, 20, 50], labels=["1", "2", "5", "10", "20", "50"])
    ax_d.minorticks_off()
    ax_d.set_xlabel("Distance band (km)")
    ax_d.set_ylabel(r"Distance elasticity $e_d$")
    style.panel_label(ax_d, "d", dx=24)

    src = {"a": ta, "b": tb, "c": L, "d": pd.concat(src_d)}
    return fig, src


def main():
    import data
    fig, src = build(data.load())
    for p in style.save(fig, "fig3", src):
        print(p)


if __name__ == "__main__":
    main()
