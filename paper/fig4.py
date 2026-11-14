"""Fig. 4 | Urban form beyond size.

Claim: the pre-registered form hypotheses did not hold out of sample — their effects shrank to about half; the weak
signal that remains sits in the spatial distribution of jobs.
  a  the ten registered hypotheses: held-out slope as a fraction of the discovery slope, Holm-adjusted p (hero panel)
  b  discovery against held-out standardized slopes of the 69 further beyond-size relations   (exploratory)
  c  standardized slopes of the elasticities on 14 form measures, 67 cities                   (exploratory)
  d  first form axis against long-distance decay net of size                                  (exploratory)
Design. a: the legend names both intervals and the criterion, and a column gives the
Holm-adjusted p of every row; b: says what the points are; c: plain words for the colour scale and the dots. The
city-types panel is not drawn (a negative result, too abstract as a picture): it is one sentence in the text and a
table in the supplement. Four panels on a two-by-two grid — a over c, b over d.
No panel carries an "exploratory" tag or grey axes: the legend says that b, c and d are exploratory.
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

import stats
import style

P = style.PALETTE
COV = {"log_population": "Population", "road_density": "Road density", "compactness": "Compactness",
       "subcentre_job_share": "Subcentre job share", "circuity": "Circuity", "pop_density_gini": "Density inequality",
       "cbd_job_share": "CBD job share", "jobs_housing": "Jobs–housing separation", "n_subcentres": "Subcentres",
       "no_car_share": "Car-free households", "orientation_entropy": "Street-grid entropy", "poi_mix": "POI mix",
       "secondary_centres": "Subcentre − CBD jobs", "transit_per_1k": "Transit supply", "water_share": "Water share"}
OUT = {"theta_turn_c0514c1": "turning distance", "transfer_loss_c0355r1": "transfer loss",
       "ed_5_10km_c0262": "decay 5–10 km, tempered", "ed_5_10km_c0355r1": "decay 5–10 km, kernel", "ed_10_20km": "decay 10–20 km",
       "ed_1_2km": "decay 1–2 km", "ed_20km": "decay >20 km", "slope_eN_logd_c0355r1": "size-effect slope"}
BANDS = ["0_1km", "1_2km", "2_5km", "5_10km", "10_20km", "20km"]
BAND_LAB = ["0–1", "1–2", "2–5", "5–10", "10–20", ">20"]


def holm_adjusted(p) -> np.ndarray:
    """Holm's step-down adjusted p-values: a hypothesis is supported at level alpha exactly when its adjusted p <= alpha."""
    p = np.asarray(p, float)
    m, order = len(p), np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    return out


def build(d: dict):
    style.apply()
    fig = style.figure(style.W2, 128)
    grid = fig.add_gridspec(2, 2, width_ratios=[2.39, 1], hspace=0.42, wspace=0.205, **style.layout(fig, 6, 5, 7, 10))   # right column: two 46 mm squares
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(grid[i, j]) for i in (0, 1) for j in (0, 1))
    w_mm = fig.get_size_inches()[0] / style.MM

    # ---- a: the ten registered hypotheses
    pos = ax_a.get_position()
    ax_a.set_position([pos.x0 + 60 / w_mm, pos.y0, pos.width - 60 / w_mm, pos.height])
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    def col(mm):                                  # x = mm left of the axes, y = data
        return blended_transform_factory(ax_a.transAxes + ScaledTranslation(-mm * style.MM, 0, fig.dpi_scale_trans), ax_a.transData)
    T = d["p7"].copy()
    r = [stats.replication_ratio(a, b, lo, hi) for a, b, lo, hi in zip(T["disc_slope"], T["test_slope"], T["ci_lo"], T["ci_hi"])]
    T["ratio"], T["r_lo"], T["r_hi"] = zip(*r)
    y = np.arange(len(T))[::-1].astype(float)
    y[:2] += 0.6                                                    # the two size hypotheses, set apart
    xmax, xlim = 2.05, 3.55
    T["p_holm"] = holm_adjusted(T["p"])
    if ((T["p_holm"] <= 0.05) != T["holm"].astype(bool)).any():
        raise ValueError("the adjusted p-values do not reproduce the recorded Holm verdicts")
    ax_a.axvline(0, color=P["ref"], lw=0.5, zorder=1)
    ax_a.axvline(1, color=P["ref"], lw=0.5, ls=(0, (4, 3)), zorder=1)
    for n_row, (yi, row) in enumerate(zip(y, T.itertuples()), start=1):
        ok = bool(row.supported)
        dl, dh = sorted((row.disc_lo / row.disc_slope, row.disc_hi / row.disc_slope))     # the discovery estimate's own interval
        ax_a.plot([dl, dh], [yi, yi], color=P["light"], lw=3.2, zorder=2, solid_capstyle="butt")
        ax_a.plot([max(row.r_lo, -0.75), min(row.r_hi, xmax)], [yi, yi], color=P["heldout"], lw=0.9, zorder=3)
        if row.r_hi > xmax:
            ax_a.plot(xmax, yi, marker=">", ms=2.6, color=P["heldout"], zorder=3, clip_on=False)
        ax_a.scatter([row.ratio], [yi], s=17, zorder=4, linewidth=0.8, facecolor=P["heldout"] if ok else "white", edgecolor=P["heldout"])
        ax_a.text(0, yi, COV[row.covariate], transform=col(50), fontsize=6.5, va="center", ha="left", clip_on=False)
        ax_a.text(0, yi, OUT[row.outcome], transform=col(25), fontsize=6.5, va="center", ha="left", color=P["grey"], clip_on=False)
        ax_a.text(0, yi, str(n_row), transform=col(56), fontsize=5.5, va="center", ha="right", color=P["grey"], clip_on=False)
    for yi, row in zip(y, T.itertuples()):                         # the criterion, row by row: Holm-adjusted p of the held-out slope
        ok = bool(row.supported)
        txt = "<0.01" if row.p_holm < 0.01 else f"{row.p_holm:.2f}"
        ax_a.text(2.50, yi, txt, fontsize=6, ha="center", va="center", color=P["heldout"] if ok else P["grey"],
                  fontweight="bold" if ok else "normal")
    ax_a.text(2.50, y.max() + 0.75, r"adjusted $p$", fontsize=6, ha="center", va="bottom", color=P["ref"])
    for xv, lab in ((-0.26, "no effect"), (1, "same slope")):          # the first one a little left of its line, clear of the second
        ax_a.text(xv, y.max() + 0.75, lab, ha="center", va="bottom", fontsize=6, color=P["ref"])
    ax_a.text(0, y.max() + 0.75, "City measure", transform=col(50), fontsize=6, color=P["ref"], va="bottom")
    ax_a.text(0, y.max() + 0.75, "Outcome", transform=col(25), fontsize=6, color=P["ref"], va="bottom")
    bx, tx, ha = 2.98, 3.06, "left"
    ax_a.plot([bx, bx], [y[1] - 0.3, y[0] + 0.3], color=P["grey"], lw=0.5)
    ax_a.text(tx, (y[0] + y[1]) / 2, "size", fontsize=6.5, color=P["grey"], va="center", ha=ha, fontstyle="italic")
    ax_a.plot([bx, bx], [y[9] - 0.3, y[2] + 0.3], color=P["grey"], lw=0.5)
    ax_a.text(tx, (y[2] + y[9]) / 2, "form", fontsize=6.5, color=P["grey"], va="center", ha=ha, fontstyle="italic")
    ax_a.set_yticks([])
    ax_a.spines["left"].set_visible(False)
    ax_a.set_xlim(-0.8, xlim)
    ax_a.set_xticks([0, 1, 2])
    ax_a.spines["bottom"].set_bounds(-0.8, xmax)
    ax_a.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax_a.set_xlabel("Held-out / discovery slope")
    k = P["heldout"]
    ax_a.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.6, mfc=k, mec=k, label=r"supported, adjusted $p$ < 0.05"),
                         Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=k, mew=0.8, label="not supported"),
                         Line2D([], [], color=k, lw=0.9, label="held-out 95% interval"),
                         Line2D([], [], color=P["light"], lw=3.2, label="discovery 95% interval")],
                loc="upper right", bbox_to_anchor=(0.74, -0.165), ncols=2, handletextpad=0.7, columnspacing=2.0, handlelength=1.4, labelspacing=0.35)
    style.panel_label(ax_a, "a", dx=178)

    # ---- b: 69 beyond-size relations, discovery against held-out
    e1, summ = d["e1_pairs"], d["e1_summary"]
    lim = 0.95
    for sx, sy in ((0, 0), (-lim, -lim)):
        ax_b.add_patch(Rectangle((sx, sy), lim, lim, facecolor=P["light"], edgecolor="none", zorder=0))
    ax_b.plot([-lim, lim], [-lim, lim], color=P["ref"], lw=0.5, ls=(0, (4, 3)), zorder=1)
    ax_b.plot([-lim, lim], [-lim * summ["median_ratio"], lim * summ["median_ratio"]], color=P["ref"], lw=1.0, zorder=2)
    ag = e1["agree"].astype(bool).to_numpy()
    ax_b.scatter(e1.loc[ag, "d_std"], e1.loc[ag, "l_std"], s=9, facecolor=P["heldout"], edgecolor="none", zorder=3)
    ax_b.scatter(e1.loc[~ag, "d_std"], e1.loc[~ag, "l_std"], s=9, facecolor="white", edgecolor=P["heldout"], linewidth=0.6, zorder=3)
    kb = P["heldout"]
    ax_b.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.0, mfc=kb, mec=kb, label=f"same sign  {summ['agree']:.0%}"),
                         Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=kb, mew=0.6, label="opposite sign"),
                         Line2D([], [], color=P["ref"], lw=1.0, label=f"median ratio  {summ['median_ratio']:.2f}")],
                loc="upper left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.4, labelspacing=0.3, handlelength=1.4,
)
    ax_b.text(0, 1.01, f"{len(e1)} further relations", transform=ax_b.transAxes, fontsize=6, va="bottom", ha="left")
    ax_b.text(0.70, 0.80, "1:1", fontsize=6, color=P["ref"], rotation=45, ha="center", va="center")
    ax_b.set_xlim(-lim, lim)
    ax_b.set_ylim(-lim, lim)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xticks([-0.5, 0, 0.5], labels=["−0.5", "0", "0.5"])
    ax_b.set_yticks([-0.5, 0, 0.5], labels=["−0.5", "0", "0.5"])
    ax_b.set_xlabel("Slope, discovery cities")
    ax_b.set_ylabel("Slope, held-out cities")
    style.panel_label(ax_b, "b", dx=26)

    # ---- c: effects heat map
    pos = ax_c.get_position()
    c_w = 55 / w_mm                               # the heat map starts under the "Outcome" column of panel a
    ax_c.set_position([pos.x0 + 35 / w_mm, pos.y0 + 0.022, c_w, pos.height - 0.075])
    c_right = (pos.x1 - pos.x0 - 35 / w_mm) / c_w                  # right edge of the left column, in axes units
    e2 = d["e2_effects"]
    cols = [f"ed_{b}" for b in BANDS] + [f"eN_{b}" for b in BANDS]
    M = e2[e2["outcome"].isin(cols)].pivot(index="covariate", columns="outcome", values="std_slope")[cols]
    S = e2[e2["outcome"].isin(cols)].pivot(index="covariate", columns="outcome", values="bh")[cols].astype(bool)
    order = M.abs().mean(axis=1).sort_values(ascending=False).index
    M, S = M.loc[order], S.loc[order]
    xs = np.r_[np.arange(6), np.arange(6) + 6.5]
    vmax = 0.6
    for j, xj in enumerate(xs):
        im = ax_c.pcolormesh([xj - 0.5, xj + 0.5], np.arange(len(M) + 1) - 0.5, M.iloc[:, [j]].to_numpy(), cmap=style.DIVERGING,
                             vmin=-vmax, vmax=vmax, edgecolors="white", linewidth=0.4)
    yy, xx = np.nonzero(S.to_numpy())
    ax_c.scatter(xs[xx], yy, s=2.2, color="black", zorder=3, linewidths=0)
    ax_c.set_xticks(xs, labels=BAND_LAB * 2, rotation=90)
    ax_c.set_yticks(range(len(M)), labels=[COV[c] for c in M.index])
    ax_c.invert_yaxis()
    ax_c.tick_params(length=0, labelsize=5)
    ax_c.tick_params(axis="y", labelsize=6)
    for sp in ax_c.spines.values():
        sp.set_visible(False)
    ax_c.set_xlabel("Distance band (km)")
    ax_c.text(2.5, -0.8, "Distance\n" + r"elasticity $e_d$", ha="center", va="bottom", fontsize=6, linespacing=1.0)
    ax_c.text(9.0, -0.8, "Size\n" + r"elasticity $e_N$", ha="center", va="bottom", fontsize=6, linespacing=1.0)
    cax = ax_c.inset_axes([1.035, 0.32, 0.018, 0.42])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-0.5, 0, 0.5], labels=["−0.5", "0", "0.5"])
    cb.outline.set_visible(False)
    cax.tick_params(length=1.5, width=0.3, labelsize=5.5, pad=1)
    cax.set_title("Standardised\nslope", fontsize=5.5, pad=3, loc="left", linespacing=1.0)
    ax_c.scatter([], [], s=2.2, color="black", label="significant,\nFDR 10%")
    ax_c.legend(loc="lower left", bbox_to_anchor=(1.0, 0.02), handletextpad=0.1, fontsize=5.5)
    style.panel_label(ax_c, "c", dx=107, dy=22)                     # level with d

    # ---- d: first form axis against long-distance decay net of size
    a1 = d["axis1"]
    grid = np.linspace(a1["axis1"].min(), a1["axis1"].max(), 50)
    fit, lo, hi, slope = stats.ols_band(a1["axis1"], a1["ed_20km_resid"], grid)
    ax_d.fill_between(grid, lo, hi, color=P["light"], lw=0, zorder=1)
    ax_d.plot(grid, fit, color=P["ref"], lw=1.0, zorder=2)
    style.scatter_sets(ax_d, a1["axis1"], a1["ed_20km_resid"], a1["set"], s=7)
    ax_d.axhline(0, color=P["ref"], lw=0.4, zorder=0)
    ax_d.set_xlabel("Job-dispersion axis")
    ax_d.set_ylabel(r"$e_d$ >20 km, size-adjusted")
    ax_d.set_ylim(-1.12, 0.92)                                     # room for the legend below the points
    ax_b.apply_aspect()                                            # the same box as panel b, straight below it
    pb, pd_ = ax_b.get_position(), ax_d.get_position()
    ax_d.set_position(Bbox.from_extents(pb.x0, pd_.y0, pb.x1, pd_.y1))
    ax_d.set_yticks([-0.8, -0.4, 0, 0.4, 0.8], labels=["−0.8", "−0.4", "0", "0.4", "0.8"])
    style.set_legend(ax_d, loc="lower right", bbox_to_anchor=(1.0, -0.01), fontsize=5.5, handlelength=1.4,
                     extra=[Line2D([], [], color=P["ref"], lw=1.0, label="Fit, all cities")])
    style.panel_label(ax_d, "d", dx=26)

    return fig, {"a": T, "b": e1, "c": M.reset_index(), "c_fdr": S.reset_index(), "d": a1}


def main():
    import data
    fig, src = build(data.load())
    for p in style.save(fig, "fig4", src):
        print(p)


if __name__ == "__main__":
    main()
