"""Fig. 5 | Benchmark against earlier models and transfer without data.

Claim: replacing population by the POI count as the destination mass is the largest single step; beyond it the
Reach model still fits and transfers better than the strongest earlier model.
  a  median fit share of every earlier model family in the 32 target cities: population version -> POI version
     (hero panel); our three formulas on top, the Reach model also as a vertical line; a small table on the left
     gives the name, the source and the number of free constants of every model
  b  the four pre-specified contrasts, city by city (the one that was not supported included); the fourth is the
     Reach model against itself and is set apart
  c  the fit and the zero-shot contrast on the pre-specified held-out block beside the same contrasts with every
     destination in the choice set (added afterwards, 2026-11-12 — the legend says so)
  d  the cost of transfer over the four parameter sources: local fit, uniform, size-predicted, size + form
  e  CPC against fit share for all 29 model versions

Layout: two columns that share their edges, placed in mm. Third design (2026-11-12): a — rows for our other
two formulas, the source of every earlier model beside its name, the three markers named as three choices of the
destination mass; b — the size-predicted row reads "vs uniform"; c — no tag on the second scoring; d — taller,
points set further apart, squares for the tiered gravity model, a legend in the order of the lines.
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import style

NAMES = {"strexp3": "Stretched exp., relaxed", "S0": "Shifted gravity", "strexp1": "Stretched exponential",
         "ops": "OPS", "grav_power": "Power gravity", "cab_pow": "Cabanas-Tirapu I", "cab_exp": "Cabanas-Tirapu II",
         "ext_rad": "Extended radiation", "grav_exp": "Exponential gravity", "io": "Intervening opportunities",
         "pwo": "PWO", "rad": "Radiation", "kwon": "Tiered gravity"}
CONTRASTS = [("C1", "Fit"), ("C2", "Zero-shot"), ("C4", "Zero-shot vs fit"), ("C3", "Size-predicted")]
XMIN = -0.62
SELF_LABEL = "vs uniform"                  # the fourth contrast: the Reach model, size-predicted against uniform constants
OPP_LABEL = "vs Stretched exp. *"          # the strongest earlier model, named as in panel a (and in its colour)
# Where each earlier model comes from, written beside its name. Both gravity models are the
# production-constrained member of Wilson's family, free mass exponent, power or exponential deterrence (source check
# 2026-11-12: label Wilson 1971; Zipf 1946 is cited in the text as the origin of the gravity hypothesis).
SOURCE = {"strexp3": "after Guo 2026", "S0": "this study", "strexp1": "Guo 2026", "ops": "Liu & Yan 2019", "grav_power": "Wilson 1971",
          "cab_pow": "2025", "cab_exp": "2025", "ext_rad": "Yang 2014", "grav_exp": "Wilson 1971", "io": "Schneider 1959",
          "pwo": "Yan 2014", "rad": "Simini 2012", "kwon": "Kwon 2023"}
OTHER_OURS = [("c0262", "Tempered power law", "ours_2"), ("c0514c1", "Two-regime power law", "ours_3")]
LEVELS = [("local", "Local fit"), ("uniform", "Uniform"), ("size", "Size-predicted"), ("form", "Size + form")]
P = style.PALETTE


def _num(v, nd=2):
    return f"{v:.{nd}f}".replace("-", "−")


def both_scorings(ax, con: dict, alldest: dict):
    """The fit and the zero-shot contrast on the pre-specified held-out block and with every destination in the choice
    set (supplementary, post hoc): median paired share difference, its interval, and the win count."""
    rows = [("Fit", con["C1"], alldest["paired"]["local|target"]["all_destinations"]),
            ("Zero-shot", con["C2"], alldest["paired"]["uniform|target"]["all_destinations"])]
    xmax = 0.118
    ax.axvline(0, color=P["ref"], lw=0.5, zorder=1)
    out = []
    for i, (lab, blk, alld) in enumerate(rows):
        y0 = 1.0 - i
        for r, dy, col, mk, name in ((blk, 0.20, P["ours"], "o", "held-out block"), (alld, -0.20, P["ours_2"], "s", "all destinations")):
            yy = y0 + dy
            ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=col, lw=1.3, zorder=3, solid_capstyle="butt")
            ax.scatter([r["median"]], [yy], s=17, marker=mk, facecolor=col, edgecolor="white", linewidth=0.3, zorder=4)
            ax.text(xmax + 0.004, yy, f"{r['wins']}/{r['n']}", fontsize=6.5, va="center", ha="left", color=col, fontweight="bold")
            ax.text(r["ci_hi"] + 0.004, yy, _num(r["median"], 3), fontsize=5.5, va="center", ha="left", color=col)
            out.append({"contrast": lab, "scoring": name, "median": r["median"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                        "wins": r["wins"], "n": r["n"]})
        ax.text(-0.012, y0 + 0.08, lab, transform=ax.get_yaxis_transform(), fontsize=6.5, ha="right", va="center")
        ax.text(-0.012, y0 - 0.22, OPP_LABEL, transform=ax.get_yaxis_transform(), fontsize=5.5, color=P["prior_poi"], ha="right", va="center")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.004, xmax)
    ax.set_ylim(-0.55, 1.62)
    ax.set_xticks([0, 0.05, 0.10], labels=["0", "0.05", "0.10"])
    ax.set_xlabel("Share difference")
    ax.text(xmax + 0.004, 1.52, "wins", fontsize=6, color=P["grey"], va="bottom")
    return pd.DataFrame(out)


def header(ax, items, y_mm: float = 3.4, step_mm: float = 3.1):
    """A legend line (or lines) just above the axes, left-aligned with it: items = [[(marker, face, edge, label, note)], ...],
    one inner list per line from the top; note is an optional grey italic word after the label."""
    from matplotlib.transforms import ScaledTranslation
    fig = ax.figure
    for k, line in enumerate(items):
        up = (y_mm + (len(items) - 1 - k) * step_mm) * style.MM
        x = 1.2
        for marker, face, edge, label, note in line:
            tr = ax.transAxes + ScaledTranslation(x * style.MM, up, fig.dpi_scale_trans)
            ax.plot([0], [1], marker=marker, ms=3.7, mfc=face, mec=edge, mew=0.3 if edge == "white" else 0.7, ls="none", transform=tr,
                    clip_on=False, zorder=6)
            t = ax.text(0, 1, label, transform=ax.transAxes + ScaledTranslation((x + 1.8) * style.MM, up, fig.dpi_scale_trans),
                        fontsize=6, va="center", ha="left")
            w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / style.MM
            x += 1.8 + w + 1.2
            if note:
                n = ax.text(0, 1, note, transform=ax.transAxes + ScaledTranslation(x * style.MM, up, fig.dpi_scale_trans), fontsize=5.5,
                            fontstyle="italic", color=P["grey"], va="center", ha="left")
                x += n.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / style.MM + 1.2
            x += 2.2


def build(d: dict):
    style.apply()
    alldest = d["alldest_summary"]
    # Two columns that share their edges: left (a above, the transfer panel below) and right (the contrasts, the two
    # scorings, CPC). Positions in mm from the left and from the top.
    Wmm, Hmm = style.W2, 158
    fig = style.figure(Wmm, Hmm)

    def place(x0, x1, ytop, ybot):
        return fig.add_axes([x0 / Wmm, 1 - ybot / Hmm, (x1 - x0) / Wmm, (ybot - ytop) / Hmm])
    LX, RX, RW = (54, 99), (127, 168), 178                     # left axes, right axes, right edge of the wins column
    bottom = (122, 149)
    ax_a = place(*LX, 8, 104)
    ax_b = place(*RX, 11, 58)
    ax_c = place(*RX, 79, 96)
    ax_d = place(14, 100, *bottom)
    ax_e = place(115, RW, *bottom)
    dx_r = 70                                                  # panel letters: one x per column (points left of the axes)
    dx_bottom = (33.5, 36.5)                                   # ... and the same two x positions for the bottom row
    s = d["p9_summary"]
    loc = s[s["level"] == "local"].set_index("model")
    ours = float(loc.loc["c0355r1", "share_med"])

    # ---- a: population -> POI count
    fam = loc[loc["kind"] == "prior"].copy()
    best = fam.groupby("family")["share_med"].max().sort_values(ascending=False)
    order = [f for f in best.index if f != "kwon"] + ["kwon"]
    y = {f: len(order) - 1 - i for i, f in enumerate(order)}
    top = len(order) + len(OTHER_OURS)                          # the row of the Reach model
    # the lines stop below their names, so no line runs through a label
    ax_a.plot([0, 0], [-1.0, top + 0.75], color=P["ref"], lw=0.5, ls=(0, (4, 3)), zorder=1)
    ax_a.plot([1, 1], [-1.0, top + 0.75], color=P["flexible"], lw=0.5, ls=(0, (4, 3)), zorder=1)
    ax_a.plot([ours, ours], [-1.0, top], color=P["ours"], lw=1.4, zorder=2, solid_capstyle="butt")
    src_a, kfree = [], {}
    for f in order:
        g = fam[fam["family"] == f]
        vp = g.loc[g["mass"] == "P", "share_med"]
        vn = g.loc[g["mass"] == "N", "share_med"]
        vt = g.loc[g["mass"] == "T", "share_med"]
        yy = y[f]
        xp = float(vp.iloc[0]) if len(vp) else None
        xn = float(vn.iloc[0]) if len(vn) else None
        if xp is not None and xn is not None and xn > XMIN:
            x0 = max(xp, XMIN)
            ax_a.add_patch(FancyArrowPatch((x0 + 0.02, yy), (xn - 0.025, yy), arrowstyle="-|>", mutation_scale=5,
                                           lw=0.6, color=P["prior_pop"], zorder=2, shrinkA=0, shrinkB=0))
        both_off = xp is not None and xn is not None and xp < XMIN and xn < XMIN
        if xp is not None:
            if xp < XMIN:
                yo = yy + (0.2 if both_off else 0)
                ax_a.plot(XMIN, yo, marker="<", ms=3.2, mfc="white", mec=P["prior_pop"], mew=1.0, zorder=4)
                ax_a.text(XMIN + 0.035, yo + (0 if both_off else 0.3), _num(xp, 1), fontsize=5.5, color=P["grey"], va="center")
            else:
                ax_a.scatter([xp], [yy], s=17, facecolor="white", edgecolor=P["prior_pop"], linewidth=1.0, zorder=4)
        if xn is not None:
            if xn < XMIN:
                ax_a.plot(XMIN, yy - 0.2, marker="<", ms=3.2, mfc=P["prior_poi"], mec=P["prior_poi"], zorder=5)
                ax_a.text(XMIN + 0.035, yy - 0.2, _num(xn, 1), fontsize=5.5, color=P["prior_poi"], va="center")
            else:
                ax_a.scatter([xn], [yy], s=19, facecolor=P["prior_poi"], edgecolor="white", linewidth=0.3, zorder=5)
        if len(vt):
            ax_a.scatter([float(vt.iloc[0])], [yy], s=17, marker="s", facecolor="white", edgecolor=P["prior_pop"], linewidth=1.0, zorder=4)
        k = int(g["n_free"].max())
        kfree[f] = k
        src_a.append({"family": f, "label": NAMES[f], "population": xp, "poi": xn,
                      "observed_trips": float(vt.iloc[0]) if len(vt) else None, "n_free": k, "source": SOURCE[f],
                      "this_study": f in ("strexp3", "S0")})
    ax_a.scatter([ours], [top], s=24, facecolor=P["ours"], edgecolor="white", linewidth=0.3, zorder=6)
    ax_a.text(ours - 0.035, top, f"{ours:.2f}", color=P["ours"], fontsize=6.5, fontweight="bold", ha="right", va="center")
    k_ours = int(loc.loc["c0355r1", "n_free"])
    own = {"strexp3", "S0"}                                  # strengthened or built in this study, not from the literature
    # The two reference lines are named above the plot, in the row of the column headers. The names sit in a small
    # table: model, source, number of free constants. The star marks a comparison model built in this study (not taken
    # from the literature) — the Reach model is our result, not a baseline, so it carries no star.
    ax_a.text(0, top + 0.95, "gravity", color=P["ref"], fontsize=6, ha="center", va="bottom")
    ax_a.text(1.045, top + 0.95, "flexible model", color=P["ref"], fontsize=6, ha="right", va="bottom")
    src_ours = [{"model": "c0355r1", "label": "Reach model", "share": ours, "n_free": k_ours}]
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    def col(mm):                                  # x = mm left of the axes, y = data
        return blended_transform_factory(ax_a.transAxes + ScaledTranslation(-mm * style.MM, 0, fig.dpi_scale_trans), ax_a.transData)
    NAME, SRC, NFREE = col(51), col(23.5), col(6.5)

    def row(yy, name, source, k, colour="black", weight="normal", source_colour=None):
        ax_a.text(0, yy, name, transform=NAME, fontsize=6.5, va="center", ha="left", color=colour, fontweight=weight, clip_on=False)
        ax_a.text(0, yy, source, transform=SRC, fontsize=5.5, va="center", ha="left", color=source_colour or P["grey"], clip_on=False)
        ax_a.text(0, yy, str(k), transform=NFREE, fontsize=6, va="center", ha="center", color=colour if weight == "bold" else P["grey"],
                  fontweight=weight, clip_on=False)
    row(top, "Reach model", "ours", k_ours, P["ours"], "bold", P["ours"])
    for i, (m, name, ck) in enumerate(OTHER_OURS, start=1):         # our other two formulas: reported, not part of the contrasts
        v = float(loc.loc[m, "share_med"])
        ax_a.scatter([v], [top - i], s=19, facecolor=P[ck], edgecolor="white", linewidth=0.3, zorder=6)
        ax_a.text(v - 0.035, top - i, f"{v:.2f}", color=P["ours_2"], fontsize=6, ha="right", va="center")
        row(top - i, name, "ours", int(loc.loc[m, "n_free"]), P["ours_2"], "normal", P["ours_2"])
        src_ours.append({"model": m, "label": name, "share": v, "n_free": int(loc.loc[m, "n_free"])})
    for f in order:
        row(y[f], NAMES[f] + (" *" if f in own else ""), SOURCE[f], kfree[f])
    for tr, lab in ((NAME, "Model"), (SRC, "Source")):
        ax_a.text(0, top + 0.95, lab, transform=tr, fontsize=6, color=P["ref"], va="bottom", ha="left", clip_on=False)
    ax_a.text(0, top + 0.95, "Constants", transform=NFREE, fontsize=6, color=P["ref"], va="bottom", ha="center", clip_on=False)
    ax_a.text(0, -0.78, "* strengthened benchmark", transform=NAME, fontsize=5.5, color=P["grey"], va="center", ha="left", clip_on=False)
    ax_a.set_yticks([])
    ax_a.tick_params(axis="y", length=0)
    ax_a.spines["left"].set_visible(False)
    ax_a.set_xlim(XMIN - 0.03, 1.10)
    ax_a.set_ylim(-1.0, top + 1.1)
    ax_a.set_xticks([-0.5, 0, 0.5, 1.0], labels=["−0.5", "0", "0.5", "1"])
    ax_a.set_xlabel("Fit share")
    h = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=P["prior_pop"], mew=1.0, label="population"),
         Line2D([], [], ls="none", marker="o", ms=3.8, mfc=P["prior_poi"], mec="white", mew=0.3, label="POI count"),
         Line2D([], [], ls="none", marker="s", ms=3.4, mfc="white", mec=P["prior_pop"], mew=1.0, label="observed trips")]
    ax_a.legend(handles=h, loc="upper left", bbox_to_anchor=(0.05, 2.62), bbox_transform=ax_a.transData, handletextpad=0.3,
                labelspacing=0.3, title="Destination mass", title_fontsize=6, alignment="left")   # the three markers: three choices of the mass
    style.panel_label(ax_a, "a", dx=147, dy=10)                 # clear of the column headers

    # ---- b: the four pre-specified contrasts
    con, diffs = d["p9_contrasts"]["contrasts"], d["p9_diffs"]
    grp = d["p9_target"].drop_duplicates("city").set_index("city")["group"].map({"lawtest": "heldout", "confirm": "confirmation"})
    rng = np.random.default_rng(3)
    lim = (-0.16, 0.42)
    ax_b.axvline(0, color=P["ref"], lw=0.5, zorder=1)
    for i, (cid, lab) in enumerate(CONTRASTS):
        y0 = len(CONTRASTS) - 1 - i
        v = diffs[cid].to_numpy(float)
        sets = grp.reindex(diffs["city"]).to_numpy()
        if v.min() < lim[0] or v.max() > lim[1]:
            raise ValueError(f"{cid}: a city difference lies outside the axis {lim}; widen it rather than clip")
        style.scatter_sets(ax_b, v, y0 - 0.18 + rng.uniform(-0.13, 0.13, len(v)), sets, s=5.5, order=("heldout", "confirmation"))
        r = con[cid]
        ok = r["verdict"] == "better"
        ax_b.plot([r["ci_lo"], r["ci_hi"]], [y0 + 0.22, y0 + 0.22], color=P["ours"], lw=1.3, zorder=5, solid_capstyle="butt")
        ax_b.scatter([r["median"]], [y0 + 0.22], s=18, zorder=6, facecolor=P["ours"] if ok else "white", edgecolor=P["ours"], linewidth=0.9)
        ax_b.text(lim[1] + 0.005, y0 + 0.02, f"{r['wins']}/{r['n']}", fontsize=6.5, va="center", ha="left",
                  color=P["ours"] if ok else P["grey"], fontweight="bold" if ok else "normal")
        ax_b.text(lim[1] - 0.004, y0 + 0.24, f"{_num(r['median'], 3)} [{_num(r['ci_lo'], 3)}, {_num(r['ci_hi'], 3)}]", fontsize=5.5,
                  va="center", ha="right", color=P["ours"] if ok else P["grey"])
        versus = (OPP_LABEL, P["prior_poi"])
        if cid == "C3":                           # the Reach model against itself, not against the stretched exp.
            versus = (SELF_LABEL, P["ours"])      # (otherwise it reads as the latter) — in our colour and set apart by a rule
            ax_b.plot([-0.47, 1.17], [y0 + 0.5, y0 + 0.5], transform=ax_b.get_yaxis_transform(), color=P["light"], lw=0.6, clip_on=False, zorder=0)
        ax_b.text(-0.012, y0 - 0.27, versus[0], transform=ax_b.get_yaxis_transform(), fontsize=5.5, color=versus[1], ha="right", va="center")
    ax_b.set_yticks([len(CONTRASTS) - 1 - i + 0.08 for i in range(len(CONTRASTS))], labels=[c[1] for c in CONTRASTS])
    ax_b.tick_params(axis="y", length=0)
    ax_b.spines["left"].set_visible(False)
    ax_b.set_xlim(*lim)
    ax_b.set_ylim(-0.6, len(CONTRASTS) - 0.4)
    ax_b.set_xlabel("Share difference")
    ax_b.text(lim[1] + 0.005, len(CONTRASTS) - 0.42, "wins", fontsize=6, color=P["grey"], va="bottom")
    M = style.MARKERS
    header(ax_b, [[("o", M["heldout"]["facecolor"], "white", style.SET_LABEL["heldout"], None),
                   ("D", M["confirmation"]["facecolor"], "white", style.SET_LABEL["confirmation"], None)]])
    style.panel_label(ax_b, "b", dx=dx_r, dy=7)

    # ---- c: the two scorings side by side (that the second was added afterwards is said in the legend, not here)
    src_c = both_scorings(ax_c, con, alldest)
    header(ax_c, [[("o", P["ours"], "white", "held-out block", None)],
                  [("s", P["ours_2"], "white", "all destinations", None)]])
    style.panel_label(ax_c, "c", dx=dx_r, dy=16)

    # ---- d: the cost of transfer
    rows = [("c0355r1", "Reach model, ours", P["ours"], "-", 1.5),
            ("strexp3_N", NAMES["strexp3"] + " *", P["prior_poi"], "-", 1.1),
            ("grav_power_N", NAMES["grav_power"], P["prior_poi"], (0, (3, 1.5)), 1.0), ("kwon", NAMES["kwon"], P["grey"], "-", 1.0)]
    lv = s[s["level"].isin([k for k, _ in LEVELS])].set_index(["model", "level"])
    src_d, ends = [], []
    for off, (m, lab, col, ls, lw) in zip((-0.09, -0.03, 0.03, 0.09), rows):
        ys = [float(lv.loc[(m, k), "share_med"]) for k, _ in LEVELS]
        for j, (k, _) in enumerate(LEVELS):
            ax_d.plot([j + off, j + off], [lv.loc[(m, k), "share_q1"], lv.loc[(m, k), "share_q3"]], color=col, lw=0.6, alpha=0.6, zorder=2)
        mk, mfc = ("s", "white") if m == "kwon" else ("o", col)    # the hollow square of panel a
        ax_d.plot(np.arange(len(LEVELS)) + off, ys, color=col, ls=ls, lw=lw, marker=mk, ms=2.6, zorder=3, mfc=mfc, mec=col)
        ends.append((ys[0], lab, col, ls, lw, mk, mfc))
        src_d.append({"model": m, "label": lab, **{k: v for (k, _), v in zip(LEVELS, ys)}})
    # A legend in the order the lines have over most of the panel, not labels at the line ends: two of the lines end at
    # the same value (they could not be told apart).
    for n_lab, (yv, lab, col, ls, lw, mk, mfc) in enumerate(sorted(ends, key=lambda e: -e[0])):
        x0, ypos = len(LEVELS) - 1 + 0.30, 0.955 - 0.068 * n_lab
        ax_d.plot([x0 + 0.03, x0 + 0.17, x0 + 0.31], [ypos] * 3, color=col, ls=ls, lw=lw, marker=mk, ms=2.6, mfc=mfc, mec=col,
                  markevery=[1], solid_capstyle="butt", clip_on=False)
        ax_d.text(x0 + 0.40, ypos, lab, color=col, fontsize=6, va="center")
    ax_d.set_ylabel("Fit share")
    ax_d.set_xticks(range(len(LEVELS)), labels=[lab for _, lab in LEVELS])
    ax_d.set_xlim(-0.35, 5.0)
    ax_d.set_ylim(0.55, 1.07)
    ax_d.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_d.spines["bottom"].set_bounds(-0.15, 3.15)
    ax_d.spines["left"].set_bounds(0.55, 1.0)
    ax_d.plot([0.72, 3.2], [1.035, 1.035], color=P["grey"], lw=0.5, clip_on=False)          # the three sources that use no local trips
    ax_d.text(1.96, 1.045, "transferred", fontsize=5.5, color=P["grey"], ha="center", va="bottom")
    ax_d.plot([-0.2, 0.28], [1.035, 1.035], color=P["grey"], lw=0.5, clip_on=False)
    ax_d.text(0.04, 1.045, "fitted locally", fontsize=5.5, color=P["grey"], ha="center", va="bottom")
    style.panel_label(ax_d, "d", dx=dx_bottom[0], dy=5)

    # ---- e: CPC against share
    m2 = float(d["p9_refs"]["m2_cpc"].median())
    ax_e.axhline(m2, color=P["flexible"], lw=0.5, ls=(0, (4, 3)), zorder=1)
    ax_e.text(-0.58, m2 + 0.004, "flexible model", fontsize=6, color=P["ref"], va="bottom")
    q = loc.reset_index()
    off = q["share_med"] < XMIN
    pop = (q["kind"] == "prior") & (q["mass"] == "P")
    poi = (q["kind"] == "prior") & (q["mass"] == "N")
    kw = q["model"] == "kwon"
    oth = (q["kind"] == "ours") & (q["model"] != "c0355r1")
    me = q["model"] == "c0355r1"

    def draw(ax, zoom=False):
        k = 1.25 if zoom else 1.0
        ax.scatter(q.loc[pop & ~off, "share_med"], q.loc[pop & ~off, "cpc_med"], s=13 * k, facecolor="white", edgecolor=P["prior_pop"], linewidth=1.0, zorder=3)
        ax.scatter(q.loc[kw, "share_med"], q.loc[kw, "cpc_med"], s=13 * k, marker="s", facecolor="white", edgecolor=P["prior_pop"], linewidth=1.0, zorder=3)
        ax.scatter(q.loc[poi & ~off, "share_med"], q.loc[poi & ~off, "cpc_med"], s=15 * k, facecolor=P["prior_poi"], edgecolor="white", linewidth=0.3, zorder=4)
        ax.scatter(q.loc[oth, "share_med"], q.loc[oth, "cpc_med"], s=13 * k, facecolor=P["ours_3"], edgecolor="white", linewidth=0.3, zorder=5)
        ax.scatter(q.loc[me, "share_med"], q.loc[me, "cpc_med"], s=24 * k, facecolor=P["ours"], edgecolor="white", linewidth=0.3, zorder=6)
    draw(ax_e)
    for r in q[off].itertuples():                                # off the share axis: marked, never drawn as ordinary points
        face = P["prior_poi"] if r.mass == "N" else "white"
        ax_e.plot(XMIN, r.cpc_med, marker="<", ms=3.0, mfc=face, mec=P["prior_poi"] if r.mass == "N" else P["prior_pop"], mew=0.9, zorder=4)
    zx, zy = (0.875, 0.985), (0.6305, 0.6415)
    axz = ax_e.inset_axes([0.57, 0.13, 0.37, 0.42])               # right of the population-mass points, so its ticks touch nothing
    draw(axz, zoom=True)
    axz.set_xlim(*zx)
    axz.set_ylim(*zy)
    axz.set_xticks([0.9, 0.95])
    axz.set_yticks([0.632, 0.640])
    axz.tick_params(labelsize=5, length=1.5, pad=1)
    for sp in axz.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.4)
        sp.set_edgecolor(P["grey"])
    ax_e.indicate_inset_zoom(axz, edgecolor=P["grey"], linewidth=0.4, alpha=1.0)
    mx, my = float(q.loc[me, "share_med"].iloc[0]), float(q.loc[me, "cpc_med"].iloc[0])
    axz.annotate("Reach model", (mx, my), xytext=(0.70, 0.10), textcoords="axes fraction", fontsize=5.5, color=P["ours"], ha="center",
                 arrowprops={"arrowstyle": "-", "lw": 0.4, "color": P["ours"]})
    axz.text(0.04, 0.93, "our other\nformulas", transform=axz.transAxes, fontsize=5.5, color=P["ours_2"], va="top", linespacing=1.0)
    ax_e.text(XMIN + 0.04, 0.392, "off-scale", fontsize=5.5, color=P["grey"], va="center")
    ax_e.set_xlim(XMIN - 0.03, 1.08)
    ax_e.set_ylim(0.38, 0.67)
    ax_e.set_xticks([-0.5, 0, 0.5, 1.0], labels=["−0.5", "0", "0.5", "1"])
    ax_e.set_xlabel("Fit share")
    ax_e.set_ylabel("CPC")
    style.panel_label(ax_e, "e", dx=dx_bottom[1], dy=5)

    src = {"a": pd.DataFrame(src_a), "a_ours": pd.DataFrame(src_ours), "b": diffs[["city"] + [c[0] for c in CONTRASTS]].assign(set=grp.reindex(diffs["city"]).to_numpy()),
           "c": src_c, "d": pd.DataFrame(src_d), "e": q[["model", "family", "mass", "kind", "n_free", "share_med", "cpc_med"]]}
    return fig, src


def main():
    import data
    fig, src = build(data.load())
    for p in style.save(fig, "fig5", src):
        print(p)


if __name__ == "__main__":
    main()
