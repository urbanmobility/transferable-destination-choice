"""Fig. 7 | Where size-adjusted constants matter (exploratory).

Claim: below about four million people one uniform set of constants is enough; larger metros lose more under uniform
constants, and constants predicted from city size recover most of that loss.
  a  gain of size-predicted over uniform constants against population, 67 cities, test block (hero panel)
  b  transfer loss under uniform and under size-predicted constants, below and above four million
  c  the two size-dependent constants of the scaled kernel against population
No panel carries an "exploratory" tag or grey axes: the legend says that the whole
figure is exploratory. Renumbered from Fig. 6 on 2026-11-12, when the OD-flow figure became Fig. 6. Self-review 2026-11-12: a — the blue line is named, the win counts carry the city-set markers of the legend instead
of "source" and "target", the axis says what the gain is over; c — the constants and the fitted line are named.
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import stats
import style

P = style.PALETTE
THRESH = float(np.exp(15.2))                      # about four million
SET = {"source": "discovery", "search": "discovery", "lawtest": "heldout", "confirm": "confirmation"}


def _pop_axis(ax, label=True):
    ax.set_xscale("log")
    ax.set_xlim(5e5, 2.4e7)
    ax.set_xticks([1e6, 2e6, 5e6, 1e7, 2e7], labels=["1", "2", "5", "10", "20"] if label else [""] * 5)
    ax.minorticks_off()
    if label:
        ax.set_xlabel("Population (million)")


def build(d: dict):
    style.apply()
    fig = style.figure(style.W15, 88)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], height_ratios=[1.2, 1, 1], hspace=0.52, wspace=0.42,
                          **style.layout(fig, 13, 4, 7, 10))
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b, ax_c1, ax_c2 = (fig.add_subplot(gs[i, 1]) for i in range(3))
    c3 = d["c3_cities"].copy()
    c3["marker"] = c3["role"].map(SET)
    c3["pop"] = np.exp(c3["logpop"])

    # ---- a: gain against population
    ax_a.axhline(0, color=P["ref"], lw=0.5, zorder=1)
    ax_a.axvline(THRESH, color=P["threshold"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    xs, ys = stats.running_median(c3["logpop"], c3["gain"], k=13)
    ax_a.plot(np.exp(xs), ys, color=P["ours"], lw=1.4, zorder=2)
    style.scatter_sets(ax_a, c3["pop"], c3["gain"], c3["marker"], s=10)
    big = c3[c3["pop"] > THRESH]
    small = c3[c3["pop"] <= THRESH]
    def wins(g, k):                                           # source cities and target cities, counted apart
        return f"{int((g.loc[g['set'] == k, 'gain'] > 0).sum())}/{int((g['set'] == k).sum())}"

    def mark(x, kind):                                        # a city-set marker in axes coordinates, as in the legend
        ax_a.scatter([x], [yw], s=9 * (0.8 if kind == "confirmation" else 1.0), transform=ax_a.transAxes, clip_on=False, zorder=6,
                     **style.MARKERS[kind])
    xt = (np.log(THRESH) - np.log(5e5)) / (np.log(2.4e7) - np.log(5e5))                      # the threshold in axes coordinates
    yw, num = 0.925, {"transform": ax_a.transAxes, "fontsize": 6.5, "color": P["ref"], "va": "center"}
    ax_a.text(xt, 0.985, "size-predicted better", transform=ax_a.transAxes, fontsize=5.5, color=P["grey"], va="center", ha="center",
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    ax_a.text(xt - 0.025, yw, wins(small, "target"), ha="right", **num)                     # below the threshold, read right to left
    mark(xt - 0.162, "confirmation")
    mark(xt - 0.189, "heldout")
    ax_a.text(xt - 0.232, yw, wins(small, "source"), ha="right", **num)
    mark(xt - 0.368, "discovery")
    mark(xt + 0.035, "discovery")                                                            # above the threshold
    ax_a.text(xt + 0.060, yw, wins(big, "source"), ha="left", **num)
    mark(xt + 0.160, "heldout")
    mark(xt + 0.188, "confirmation")
    ax_a.text(xt + 0.213, yw, wins(big, "target"), ha="left", **num)
    ax_a.text(1.45e7, 0.043, "running median", fontsize=5.5, color=P["ours"], ha="center", va="center")
    ax_a.text(THRESH * 1.05, -0.118, "4 million", fontsize=6, color=P["threshold"], va="bottom")   # drawn after seeing the data: said in the legend
    _pop_axis(ax_a)
    ax_a.set_ylim(-0.125, 0.158)                                 # room for the counts above the highest point
    ax_a.set_ylabel("Gain over uniform")
    style.set_legend(ax_a, loc="lower left", bbox_to_anchor=(0.0, 0.09))
    style.panel_label(ax_a, "a", dx=30)

    # ---- b: loss under uniform and under size-predicted constants
    src_b = []
    for i, (lab, g) in enumerate((("< 4 million", small), ("> 4 million", big))):
        for j, (col, face) in enumerate((("loss_uniform", "white"), ("loss_size", P["ours"]))):
            v = g[col].dropna()
            x = i + (j - 0.5) * 0.36
            q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
            ax_b.plot([x, x], [q1, q3], color=P["ours"], lw=0.8, zorder=2)
            ax_b.scatter([x], [med], s=16, facecolor=face, edgecolor=P["ours"], linewidth=0.9, zorder=3)
            src_b.append({"group": lab, "constants": col, "n": len(v), "q1": q1, "median": med, "q3": q3})
        m = [g[c].median() for c in ("loss_uniform", "loss_size")]
        ax_b.plot([i - 0.18, i + 0.18], m, color=P["ours"], lw=0.7, zorder=1)
    ax_b.axhline(0, color=P["ref"], lw=0.5, zorder=0)
    ax_b.set_xticks([0, 1], labels=[f"< 4 million\nn = {len(small)}", f"> 4 million\nn = {len(big)}"])
    ax_b.set_xlim(-0.55, 1.55)
    ax_b.set_ylim(-0.02, 0.13)
    ax_b.set_ylabel("Transfer loss")
    ax_b.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=P["ours"], mew=0.9, label="Uniform"),
                         Line2D([], [], ls="none", marker="o", ms=3.6, mfc=P["ours"], mec=P["ours"], label="Size-predicted")],
                loc="upper left", bbox_to_anchor=(0.0, 1.04), handletextpad=0.2, labelspacing=0.25)
    style.panel_label(ax_b, "b", dx=26)

    # ---- c: the two size-dependent constants
    k = d["consts"].copy()
    k["pop"] = np.exp(k["log_population"])
    grid = np.linspace(np.log(5.5e5), np.log(2.2e7), 40)
    for ax, col, lab in ((ax_c1, "theta1", r"$\theta_1$"), (ax_c2, "theta3", r"$\theta_3$")):
        from scipy.stats import theilslopes
        src = k[k["set"] == "discovery"]                    # the rule behind panels a and b: Theil-Sen on the source cities
        sl, ic, _, _ = theilslopes(src[col], src["log_population"])
        ax.plot(np.exp(grid), ic + sl * grid, color=P["discovery"], lw=0.9, zorder=2)
        style.scatter_sets(ax, k["pop"], k[col], k["set"], s=5)
        ax.set_ylabel("Constant " + lab)
        q = k[col].quantile([0.02, 0.98])
        pad = (q.iloc[1] - q.iloc[0]) * 0.25
        ax.set_ylim(q.iloc[0] - pad, q.iloc[1] + pad)
    ax_c1.text(0.98, 0.93, "discovery fit", transform=ax_c1.transAxes, fontsize=5.5, color=P["discovery"], ha="right", va="top")
    _pop_axis(ax_c1, label=False)
    _pop_axis(ax_c2)
    style.panel_label(ax_c1, "c", dx=26)

    src = {"a": c3, "b": pd.DataFrame(src_b), "c": k}
    return fig, src


def main():
    import data
    fig, src = build(data.load())
    for p in style.save(fig, "fig7", src):
        print(p)


if __name__ == "__main__":
    main()
