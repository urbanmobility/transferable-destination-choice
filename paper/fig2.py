"""Fig. 2 | A simple family of laws.

Claim: distance decay is not a pure power law, the role of destination size grows with distance, and one
five-constant formula captures both while staying close to the flexible model.
  a  distance elasticity against distance: every city's flexible-model estimate (thin, coloured by city size) and
     the median curves of the three formulas (hero panel); the definition of e_d in the panel
  b  size elasticity against distance, all three formulas: only the scaled kernel rises with distance, the other two
     are flat by construction (their size term is ln(N_j + theta_2)); the definition of e_N in the panel
  c  the whole model: the destination choice probability and the three utilities (symbols are explained in the legend)
  d  fit share of the three formulas in the 67 cities (0 = gravity, 1 = flexible model); the axis starts at
     SHARE_MIN, where the data begin, and says so ("gravity at 0")
Design: the flexible model's legend entry shows that it is one line per city coloured by city size; b shows the
three formulas; c gives the choice probability, not only the utility; the elasticities are defined where they are
plotted; d keeps names and medians in a column beside the rows.
"""
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

import style

FORMULAS = [("Scaled kernel", "ours", "-", 1.5), ("Tempered power law", "ours_2", (0, (4, 1.5)), 1.2),
            ("Two-regime power law", "ours_3", (0, (1, 1)), 1.4)]
EQ = {"Scaled kernel": r"$\eta_{ij} = a_j^{\,\theta_4} - \theta_0\, d^{\,\theta_1} / (N_j + \theta_2)^{\theta_3}$",
      "Tempered power law": r"$\eta_{ij} = -\ln d - \theta_0\, d^{\,\theta_1} + \ln(N_j + \theta_2) - e^{-s_j} - e^{-a_j}$",
      "Two-regime power law": r"$\eta_{ij} = \theta_0 \ln a_j - \ln d - \ln(d + r) + \ln(N_j + \theta_2)$"}
PROB = r"$P(j\,|\,i) \;=\; \exp(\eta_{ij}) \;\,/\; \sum_{k}\, \exp(\eta_{ik})$"      # on one line, so that no subscript shrinks
E_D = r"$e_d = \partial\eta_{ij}\,/\,\partial \ln d$"            # the two elasticities, defined in the panels that plot them
E_N = r"$e_N = \partial\eta_{ij}\,/\,\partial \ln N_j$"
SHARE_MIN, SHARE_MAX = 0.72, 1.25


def _city_lines(ax, b, col, logpop, cmap, norm):
    f = b[b["model"] == "Flexible"]
    for c, g in f.groupby("city"):
        g = g.sort_values("band")
        ax.plot(g["d_ref"], g[col], color=cmap(norm(logpop[c])), lw=0.45, alpha=0.65, zorder=1, solid_capstyle="round")
    return f["city"].nunique()


def _median_curve(b, model, col):
    g = b[b["model"] == model].groupby("band").agg(x=("d_ref", "median"), y=(col, "median")).sort_index()
    return g["x"].to_numpy(), g["y"].to_numpy()


def _legend(ax, b, col, cmap, n, **kw):
    """The three formulas (median over the cities) and the flexible model. The flexible model's entry is three short
    lines in three colours of the city-size ramp, with the number of cities: one line per city, coloured by its size."""
    hs = []
    for name, key, ls, lw in FORMULAS:
        x, y = _median_curve(b, name, col)
        ax.plot(x, y, color=style.PALETTE[key], ls=ls, lw=lw, zorder=5)
        hs.append(Line2D([], [], color=style.PALETTE[key], ls=ls, lw=lw))
    flex = tuple(Line2D([], [], color=cmap(v), lw=0.9) for v in (0.08, 0.5, 0.95))
    ax.legend(hs + [flex], [f[0] for f in FORMULAS] + [f"Flexible model, {n} cities"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)}, handlelength=2.2, **kw)


def model_panel(ax, fig):
    """Panel c: the whole model, not only the utility — the probability of each destination and the three utilities.
    Units: mm from the top-left corner of the panel."""
    P = style.PALETTE
    ax.set_axis_off()
    bb = ax.get_position()
    W, H = bb.width * fig.get_size_inches()[0] / style.MM, bb.height * fig.get_size_inches()[1] / style.MM
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    head = {"fontsize": 6, "color": P["grey"], "va": "center", "ha": "left"}
    ax.text(0, H - 1.6, "Destination choice probability", **head)
    ax.text(0, H - 8.0, PROB, fontsize=8.5, va="center")
    ax.plot([0, W], [H - 12.6, H - 12.6], color=P["light"], lw=0.6)
    ax.text(0, H - 15.8, r"Discovered utilities $\eta_{ij}$", **head)       # found by the formula search, not assumed
    for i, (name, key, ls, lw) in enumerate(FORMULAS):
        y0 = H - 21.2 - i * 8.6
        ax.plot([0, 4.6], [y0, y0], color=P[key], ls=ls, lw=lw, solid_capstyle="butt")
        ax.text(6.4, y0, name, fontsize=6.5, fontweight="bold", va="center", color=P[key])
        ax.text(0, y0 - 4.0, EQ[name], fontsize=6.5, va="center")
        if name == "Two-regime power law":                           # what Fig. 3b plots
            ax.text(W, y0, r"$r$  turning distance", fontsize=5.5, color=P["grey"], va="center", ha="right")


def build(d: dict):
    style.apply()
    fig = style.figure(style.W2, 138)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1], height_ratios=[1, 1.0, 0.76], hspace=0.62, wspace=0.26,
                          **style.layout(fig, 13, 4, 6, 10))
    ax_a = fig.add_subplot(gs[0:2, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])
    b, cities = d["bands"], d["cities"]
    logpop = cities.set_index("city")["log_population"]
    cmap, norm = style.size_cmap(), Normalize(np.log(6e5), np.log(2e7))

    # ---- a: distance elasticity
    n = _city_lines(ax_a, b, "ed", logpop, cmap, norm)
    _legend(ax_a, b, "ed", cmap, n, loc="lower left", bbox_to_anchor=(0.0, 0.02), labelspacing=0.3)
    pl = float(np.median(b.loc[b["model"] == "Flexible", "ed"]))
    ax_a.axhline(pl, color=style.PALETTE["ref"], lw=0.5, ls=(0, (4, 3)), zorder=2)
    ax_a.text(0.55, pl + 0.04, "power law", color=style.PALETTE["ref"], fontsize=6, va="bottom")
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.5, 60)
    ax_a.set_ylim(-3.6, 0.2)
    ax_a.set_xticks([1, 2, 5, 10, 20, 50], labels=["1", "2", "5", "10", "20", "50"])
    ax_a.minorticks_off()
    ax_a.set_xlabel("Distance (km)")
    ax_a.set_ylabel(r"Distance elasticity $e_d$")
    ax_a.text(0.03, 0.965, E_D, transform=ax_a.transAxes, fontsize=7.5, va="center", ha="left")
    cax = ax_a.inset_axes([0.60, 0.92, 0.34, 0.025])
    cb = fig.colorbar(ScalarMappable(norm, cmap), cax=cax, orientation="horizontal")
    cb.set_ticks(np.log([1e6, 3e6, 1e7]), labels=["1", "3", "10"])
    cb.outline.set_linewidth(0.3)
    cax.tick_params(length=1.5, width=0.3, labelsize=5.5, pad=1)
    cax.set_title("City population (million)", fontsize=5.5, pad=2)
    style.panel_label(ax_a, "a", dx=30)

    # ---- b: size elasticity, all three formulas
    _city_lines(ax_b, b, "eN", logpop, cmap, norm)
    _legend(ax_b, b, "eN", cmap, n, loc="upper left", bbox_to_anchor=(0.0, 1.03), labelspacing=0.25)
    ax_b.text(0.985, 0.945, E_N, transform=ax_b.transAxes, fontsize=7.5, va="center", ha="right")
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.5, 60)
    ax_b.set_ylim(-0.3, 1.75)
    ax_b.set_xticks([1, 2, 5, 10, 20, 50], labels=["1", "2", "5", "10", "20", "50"])
    ax_b.minorticks_off()
    ax_b.set_xlabel("Distance (km)")
    ax_b.set_ylabel(r"Size elasticity $e_N$")
    style.panel_label(ax_b, "b", dx=26)

    # ---- c: the model. The box runs from below b's axis title to the foot of panel a.
    pa, pb, pc = ax_a.get_position(), ax_b.get_position(), ax_c.get_position()
    h_mm = fig.get_size_inches()[1] / style.MM
    ax_c.set_position([pc.x0, pa.y0 - 10.0 / h_mm, pc.width, (pb.y0 - 12.5 / h_mm) - (pa.y0 - 10.0 / h_mm)])
    model_panel(ax_c, fig)
    style.panel_label(ax_c, "c", dx=14, dy=-4)

    # ---- d: fit share. The axis starts where the data begin; names and medians sit in a column left of the rows.
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory
    w_mm = fig.get_size_inches()[0] / style.MM
    pos = ax_d.get_position()
    ax_d.set_position([pos.x0 + 44 / w_mm, pos.y0, pos.width - 44 / w_mm, pos.height])

    def col(mm):                                  # x = mm left of the axes, y = data
        return blended_transform_factory(ax_d.transAxes + ScaledTranslation(-mm * style.MM, 0, fig.dpi_scale_trans), ax_d.transData)
    sh = d["shares"].merge(cities[["city", "set"]], on="city")
    if sh["share"].min() < SHARE_MIN:
        raise ValueError(f"a fit share lies below the axis start {SHARE_MIN}; lower it rather than clip")
    rng = np.random.default_rng(1)
    for i, (name, key, _, _) in enumerate(FORMULAS):
        g = sh[sh["model"] == name]
        y0 = len(FORMULAS) - 1 - i
        inside = g[g["share"] <= SHARE_MAX]
        vp = ax_d.violinplot(inside["share"], positions=[y0 + 0.10], orientation="horizontal", widths=0.66,
                             showextrema=False, side="high")
        for body in vp["bodies"]:
            body.set(facecolor=style.PALETTE[key], edgecolor="none", alpha=0.32)
        style.scatter_sets(ax_d, inside["share"].to_numpy(), y0 - 0.13 + rng.uniform(-0.10, 0.10, len(inside)),
                           inside["set"].to_numpy(), s=6)
        med = float(g["share"].median())
        ax_d.plot([med, med], [y0 - 0.30, y0 + 0.40], color=style.PALETTE[key], lw=1.2, zorder=6)
        ax_d.text(0, y0 + 0.05, name, transform=col(44), color=style.PALETTE[key], fontsize=6.5, fontweight="bold", va="center", clip_on=False)
        ax_d.text(0, y0 + 0.05, f"{med:.2f}", transform=col(5), color=style.PALETTE[key], fontsize=6.5, va="center", ha="right", clip_on=False)
        off = g[g["share"] > SHARE_MAX].sort_values("share")
        if len(off):                                         # every off-scale city is named by its value
            ax_d.plot(SHARE_MAX + 0.008, y0 - 0.13, marker=">", ms=2.6, color=style.PALETTE["grey"], clip_on=False, zorder=7)
            ax_d.text(SHARE_MAX - 0.004, y0 - 0.36, "off-scale " + ", ".join(f"{v:.2f}" for v in off["share"]), fontsize=5.5,
                      color=style.PALETTE["grey"], ha="right", va="center")
    top = len(FORMULAS) - 0.42
    ax_d.text(0, top, "median", transform=col(5), color=style.PALETTE["grey"], fontsize=6, va="bottom", ha="right", clip_on=False)
    ax_d.axvline(1.0, color=style.PALETTE["ref"], lw=0.5, ls=(0, (4, 3)), zorder=1)
    ax_d.text(1.0 + 0.004, top, "flexible model", color=style.PALETTE["ref"], fontsize=6, va="bottom")
    ax_d.plot(SHARE_MIN + 0.004, top + 0.13, marker="<", ms=2.6, color=style.PALETTE["ref"], zorder=7)      # the lower reference is off the axis
    ax_d.text(SHARE_MIN + 0.011, top, "gravity at 0", color=style.PALETTE["ref"], fontsize=6, va="bottom")
    ax_d.set_yticks([])
    ax_d.spines["left"].set_visible(False)
    ax_d.set_xlim(SHARE_MIN, SHARE_MAX + 0.02)
    ax_d.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ax_d.set_ylim(-0.55, len(FORMULAS) - 0.15)
    ax_d.set_xlabel("Fit share")
    style.set_legend(ax_d, loc="lower right", bbox_to_anchor=(1.0, 0.985), ncols=3, columnspacing=1.2)
    style.panel_label(ax_d, "d", dx=30 + 44 / 25.4 * 72)

    src = {"a_b": b[b["model"].isin(["Flexible"] + [f[0] for f in FORMULAS])], "d": sh}
    return fig, src


def main():
    import data
    fig, src = build(data.load())
    for p in style.save(fig, "fig2", src):
        print(p)


if __name__ == "__main__":
    main()
