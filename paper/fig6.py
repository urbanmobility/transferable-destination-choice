"""Fig. 6 | Predicted flows in a city that was never used.

Claim: with constants that no Denver trip entered, the Reach model reproduces the largest flows of the city; the
earlier models — as published or with the POI count as the mass — put them in the wrong places.
  a  the 2,000 largest flows: observed above ours; power gravity, radiation and the stretched exponential of Guo et
     al., as published above the same model with the POI count as the mass (hero panel)
  b  share of the K largest observed flows that are among the model's K largest
  c  predicted over observed trips by trip distance, the published versions and the POI-count versions side by side
     (in one panel the two versions of a model would lie on each other and could not be told apart)
  d  CPC of the same models
Every prediction is a transfer, not a fit: the constants are plan 9's uniform constants (medians over the source
cities), so no Denver trip enters them; the figure says "zero-shot predictions". In b and d the published version has
a hollow marker and the POI-count version a filled one, as in Fig. 5a.
One city (Denver: the example of Fig. 1b, a confirmation city), test period x all destinations, the uniform constants
of plan 9, flows = T_i * P(j|i). An illustration, not a test: nothing here is a pre-specified contrast. The choices
(city, constants, models) were fixed before the export; the drawing was tried in several ways and the
one without a distance cut was chosen. Only classes and aggregates were exported (the raw data are licensed).
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import style

P = style.PALETTE
LEVEL = "uniform"
OURS = "c0355r1"
FAMILIES = [("grav_power", "Gravity", "o", "-"), ("rad", "Radiation", "s", (0, (4, 1.5))), ("strexp1", "Stretched exponential", "^", (0, (1, 1.2)))]
SHORT = {}                                    # full names everywhere: an abbreviation would end in a full stop
VERSION = {"P": ("as published", P["grey"]), "N": ("with POI count", P["prior_poi"])}     # a grey as dark as the green: equal weight
TIER_WIDTH = (1.3, 1.0, 0.7, 0.45, 0.3)           # rank tiers of a flow: top 100, 200, 500, 1,000, 2,000
TIER_ALPHA = (0.9, 0.85, 0.75, 0.6, 0.5)
N_TOP = 2000


def key(m: str) -> str:
    return f"{m}|{LEVEL}"


def window(t, pad=0.04):
    """The box of the end points of the observed flows drawn (0.3rd to 99.7th percentile)."""
    xs, ys = np.r_[t["ox"], t["dx"]], np.r_[t["oy"], t["dy"]]
    x0, x1, y0, y1 = np.percentile(xs, 0.3), np.percentile(xs, 99.7), np.percentile(ys, 0.3), np.percentile(ys, 99.7)
    return x0 - pad * (x1 - x0), x1 + pad * (x1 - x0), y0 - pad * (y1 - y0), y1 + pad * (y1 - y0)


def fit_window(win, aspect: float):
    """Widen the box in one direction until it has the panel's width / height: every map then fills its cell, whatever
    the shape of the city (a tall city would otherwise shrink its panels and move their titles)."""
    x0, x1, y0, y1 = win
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        pad = (aspect * h - w) / 2
        return x0 - pad, x1 + pad, y0, y1
    pad = (w / aspect - h) / 2
    return x0, x1, y0 - pad, y1 + pad


def flow_map(ax, g, t, win, src, colour):
    """The N_TOP largest flows of one source, the largest drawn last; width and opacity by rank tier."""
    g.plot(ax=ax, facecolor="#F1F1F1", edgecolor="white", linewidth=0.15, zorder=0)
    for tier in range(len(TIER_WIDTH), 0, -1):
        s = t[t[f"tier_{src}"] == tier]
        seg = np.stack([np.c_[s["ox"], s["oy"]], np.c_[s["dx"], s["dy"]]], axis=1)
        ax.add_collection(LineCollection(seg, colors=colour, linewidths=TIER_WIDTH[tier - 1], alpha=TIER_ALPHA[tier - 1],
                                         capstyle="round", zorder=2 + (6 - tier)))
    ax.set_xlim(win[0], win[1])
    ax.set_ylim(win[2], win[3])
    ax.set_aspect("equal")
    ax.set_axis_off()


def build(d: dict, od: dict = None):
    """od: the OD-flow tables of another confirmation city (data.load_od) — the same figure for the supplement."""
    style.apply()
    od = od or d["od"]
    place_name = od["summary"]["name"].split(",")[0].split("-")[0]          # "Denver-Aurora-Centennial, CO" -> "Denver"
    g = gpd.read_file(od["tracts"])
    g = g.to_crs(g.estimate_utm_crs())                                      # the city's UTM zone: north is up
    cen = g.assign(GEOID=g["GEOID"].astype(str)).set_index("GEOID").geometry.centroid
    t = od["pairs"].copy()
    for end in ("o", "d"):
        t[f"{end}x"], t[f"{end}y"] = t[end].map(cen.x), t[end].map(cen.y)
    if t[["ox", "dx"]].isna().any().any():
        raise ValueError("a flow ends in a tract without geometry")
    obs = t["tier_observed"] > 0
    if int(obs.sum()) != N_TOP:
        raise ValueError(f"{int(obs.sum())} observed flows, not {N_TOP}")
    win = window(t[obs])

    H = 144
    fig = style.figure(style.W2, H)
    # the maps and the row of curves have their own margins (mm): the maps need none on the left, the curves need the axis
    top = fig.add_gridspec(2, 5, wspace=0.03, hspace=0.27, width_ratios=[1, 0.09, 1, 1, 1], **style.layout(fig, 5, 2, 8, H - 80))
    def place(x0, x1, ytop=92, ybot=126):         # the lower row, in mm from the left and from the top: d needs a wide label column
        return fig.add_axes([x0 / style.W2, 1 - ybot / H, (x1 - x0) / style.W2, (ybot - ytop) / H])
    fig.text(1 - 2 / style.W2, 1 - 1.5 / H, "zero-shot predictions", fontsize=6, fontstyle="italic", color=P["grey"], ha="right", va="top")

    # ---- a: the 2,000 largest flows, observed and predicted
    def under(ax, text):                          # under the map, so that no line runs through the numbers
        ax.text(0.0, -0.03, text, transform=ax.transAxes, fontsize=6, va="top")
    ax_obs = fig.add_subplot(top[0, 0])
    box = ax_obs.get_position()
    win = fit_window(win, (box.width * style.W2) / (box.height * H))
    flow_map(ax_obs, g, t, win, "observed", P["heldout"])
    ax_obs.set_title(f"Observed, {place_name}", fontsize=7, pad=3, color=P["heldout"])
    under(ax_obs, f"Top {N_TOP:,} flows")
    shared = {}
    ax = fig.add_subplot(top[1, 0])
    flow_map(ax, g, t, win, key(OURS), P["ours"])
    ax.text(0.5, 1.02, "Reach model, ours", transform=ax.transAxes, fontsize=7, color=P["ours"], ha="center", va="bottom")
    shared[OURS] = int((obs & (t[f"tier_{key(OURS)}"] > 0)).sum())
    under(ax, f"{shared[OURS]:,} in observed top {N_TOP:,}")
    for c, (f, name, _, _) in enumerate(FAMILIES, start=2):
        for r, v in enumerate("PN"):
            m = f"{f}_{v}"
            lab, colour = VERSION[v]
            ax = fig.add_subplot(top[r, c])
            flow_map(ax, g, t, win, key(m), colour)
            shared[m] = int((obs & (t[f"tier_{key(m)}"] > 0)).sum())
            under(ax, f"{shared[m]:,} in observed top {N_TOP:,}")
            if r == 0:
                ax.set_title(name, fontsize=7, pad=3, color="black")
            if c == 2:
                ax.text(-0.035, 0.5, lab, transform=ax.transAxes, rotation=90, fontsize=6.5, color=colour, ha="right", va="center")
    style.panel_label(ax_obs, "a", dx=10, dy=4)

    models = [(OURS, "Reach model, ours", P["ours"], "o", "-", 1.5)] + \
             [(f"{f}_{v}", name, VERSION[v][1], mk, ls, 1.0) for f, name, mk, ls in FAMILIES for v in "PN"]

    def face(m, colour):                          # hollow = as published, filled = with the POI count (as in Fig. 5a)
        return "white" if m.endswith("_P") else colour

    # ---- b: how many of the largest observed flows a model also ranks among its largest
    ax_b = place(14, 64)
    ov = od["overlap"]
    for m, name, colour, mk, ls, lw in models[::-1]:
        s = ov[ov["source"] == key(m)].sort_values("K")
        ax_b.plot(s["K"], s["overlap"], color=colour, lw=lw, ls=ls, marker=mk, ms=2.8, mfc=face(m, colour),
                  mec=colour if m.endswith("_P") else "white", mew=0.7 if m.endswith("_P") else 0.3, zorder=5 if m == OURS else 3)
    ax_b.set_xscale("log")
    ks = sorted(ov["K"].unique())
    ax_b.set_xticks(ks, labels=[f"{k:,}" for k in ks])
    ax_b.minorticks_off()
    if ov.loc[ov["source"].isin([key(m) for m, *_ in models]), "overlap"].max() > 0.65:
        raise ValueError("a recovery share reaches into the legend band")
    ax_b.set_ylim(0.0, 0.95)                      # from 0, so that no line of any city is cut; the band above 0.65 holds the
    ax_b.spines["left"].set_bounds(0.0, 0.65)     # legend: one entry for each of the seven lines
    ax_b.set_yticks([0, 0.2, 0.4, 0.6], labels=["0", "0.2", "0.4", "0.6"])
    ax_b.set_xlabel(r"$K$ largest observed flows")
    ax_b.set_ylabel("Share also predicted", y=0.33)
    for v, loc, x0 in (("P", "upper left", -0.03), ("N", "upper right", 1.08)):
        lab, colour = VERSION[v]
        hollow = v == "P"
        leg = ax_b.legend(handles=[Line2D([], [], color=colour, lw=1.0, ls=ls, marker=mk, ms=2.8, mfc="white" if hollow else colour,
                                          mec=colour if hollow else "white", mew=0.7 if hollow else 0.3, label=SHORT.get(name, name))
                                   for _, name, mk, ls in FAMILIES],
                          loc=loc, bbox_to_anchor=(x0, 1.04), fontsize=5.5, handlelength=2.0, handletextpad=0.4, labelspacing=0.2, title=lab,
                          title_fontsize=5.5,
                          alignment="left")
        leg.get_title().set_color(colour)
        ax_b.add_artist(leg)
    ax_b.legend(handles=[Line2D([], [], color=P["ours"], lw=1.5, marker="o", ms=2.8, mec="white", mew=0.3, label="Reach model, ours")],
                loc="lower right", bbox_to_anchor=(1.03, -0.02), fontsize=5.5, handlelength=2.6)
    style.panel_label(ax_b, "b", dx=26)

    # ---- c: trips by distance, predicted over observed — one small panel per version: the two versions of a model
    # nearly coincide here (the mass changes where trips go, hardly how far), so in one panel they hid each other
    ds = od["dist"]
    x = np.sqrt(ds["dist_from"] * ds["dist_to"])
    keep = ds["observed"] >= 0.002                 # the distance classes that hold at least 0.2% of the observed trips
    ax_c = None
    for k, v in enumerate("PN"):
        ax = place(79 + 26 * k, 103 + 26 * k)
        ax_c = ax_c or ax
        ax.axhline(1, color=P["ref"], lw=0.5, zorder=1)
        for m, name, colour, mk, ls, lw in models[::-1]:
            if m == OURS or m.endswith(f"_{v}"):
                ax.plot(x[keep], (ds[key(m)] / ds["observed"])[keep], color=colour, lw=lw, ls=ls, zorder=5 if m == OURS else 3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.03, 14)
        ax.set_xticks([1, 3, 10, 30], labels=["1", "3", "10", "30"])
        ax.set_yticks([0.1, 0.3, 1, 3, 10], labels=["0.1", "0.3", "1", "3", "10"] if k == 0 else [""] * 5)
        ax.minorticks_off()
        ax.set_title(VERSION[v][0], fontsize=6.5, color=VERSION[v][1], pad=3)
    ax_c.set_xlabel("Trip distance (km)", x=1.042)
    ax_c.set_ylabel("Predicted / observed trips")
    dark = P["heldout"]                            # the line style says which model; the colour is the version, named above each half
    ax_c.legend(handles=[Line2D([], [], color=P["ours"], lw=1.5, label="Reach model, ours")] +
                        [Line2D([], [], color=dark, lw=1.0, ls=ls, label=SHORT.get(name, name)) for _, name, _, ls in FAMILIES],
                loc="upper center", bbox_to_anchor=(1.042, -0.27), ncols=2, fontsize=5.5, handlelength=2.6, labelspacing=0.25, columnspacing=1.6)
    style.panel_label(ax_c, "c", dx=26)

    # ---- d: CPC — six rows, the published versions above the POI-count versions; ours as a line
    ax_d = place(153, 180)
    cp = {m: float(od["summary"]["models"][key(m)]["cpc"]) for m, *_ in models}
    n_f = len(FAMILIES)
    y_of = {(v, i): (n_f + 1.1 if v == "P" else 0.0) + (n_f - 1 - i) for v in "PN" for i in range(n_f)}
    top_y = max(y_of.values())
    ax_d.plot([cp[OURS]] * 2, [-0.6, top_y + 1.25], color=P["ours"], lw=1.4, solid_capstyle="butt", zorder=2)
    ax_d.text(cp[OURS], top_y + 1.35, "ours", color=P["ours"], fontsize=6, ha="center", va="bottom")
    ticks, labels = [], []
    for v in "PN":
        lab, colour = VERSION[v]
        for i, (f, name, mk, _) in enumerate(FAMILIES):
            yy = y_of[(v, i)]
            ax_d.plot([0.24, cp[f"{f}_{v}"]], [yy, yy], color=P["light"], lw=0.6, zorder=1)
            ax_d.scatter([cp[f"{f}_{v}"]], [yy], s=16, marker=mk, facecolor="white" if v == "P" else colour, edgecolor=colour if v == "P" else "white",
                         linewidth=0.9 if v == "P" else 0.3, zorder=4)
            ticks.append(yy)
            labels.append(SHORT.get(name, name))
        ax_d.text(-0.04, y_of[(v, 0)] + 0.72, lab, transform=ax_d.get_yaxis_transform(), fontsize=6, color=colour, ha="right", va="center")
    ax_d.set_yticks(ticks, labels=labels)
    ax_d.tick_params(axis="y", length=0, labelsize=6)
    ax_d.spines["left"].set_visible(False)
    ax_d.set_ylim(-0.6, top_y + 1.9)
    ax_d.set_xlim(0.24, 0.60)
    ax_d.set_xticks([0.3, 0.4, 0.5, 0.6])
    ax_d.set_xlabel("CPC")
    style.panel_label(ax_d, "d", dx=70)

    tiers = ["tier_observed"] + [f"tier_{key(m)}" for m, *_ in models]
    src = {"a": t.loc[(t[tiers] > 0).any(axis=1), ["o", "d", "dist_km"] + tiers].reset_index(drop=True),
           "a_shared": pd.DataFrame([{"model": m, "in_observed_top": n, "of": N_TOP} for m, n in shared.items()]),
           "b": ov[ov["source"].isin([key(m) for m, *_ in models])].reset_index(drop=True),
           "c": ds[["dist_from", "dist_to", "observed"] + [key(m) for m, *_ in models]],
           "d": pd.DataFrame([{"model": m, "label": name, "version": "ours" if m == OURS else VERSION[m[-1]][0], "cpc": cp[m]}
                              for m, name, *_ in models])}
    return fig, src


def main():
    import data
    fig, src = build(data.load())
    for p in style.save(fig, "fig6", src):
        print(p)


if __name__ == "__main__":
    main()
