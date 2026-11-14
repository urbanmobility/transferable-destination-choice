"""Fig. 1 | Data, design and workflow.

Claim: 67 metropolitan areas are split into non-overlapping groups, and a period and a block of destinations are
held out as well, so every test runs on data that took no part in finding what is tested.
  a  the 67 metros: circle area proportional to population; three city groups — Discovery (35), Held-out (26),
     Confirmation (6); the 8 discovery cities used for the formula search carry a dot (a subset, not a fourth group)
  b  one metro, centre: the choice set (tracts shaded by POI-count quintile) and the main destinations of one home
  c  the holdout scheme as one object: the destination blocks on the map and a periods x blocks grid that says which
     cell is used to fit, to select and to report
  d  the workflow, drawn; every stage names what its picture shows:
     Formula search  the inner loop — symbolic regression generates formulas, they are scored (fit to trips,
                     ranked automatically), the LLM agent proposes and rewrites;
     Hypotheses      a pattern across the discovery cities -> the LLM agent proposes -> a list of hypotheses;
     Held-out test   the same relation on the held-out cities against the discovery fit; supported or not;
     Freeze          3 formulas;
     Confirmation    opened after the freeze: 6 unseen cities + the test period & block in all 67 cities;
     Benchmark       fit share of our formula above four earlier models.
     The small plots are real results: 5-10 km decay against city size (flexible model) on the discovery and on the
     held-out cities, and the median fit shares of plan 9.
Held-out data — the test block and the test period — are hatched in black; orange is kept for the confirmation
cities. The example city carries classes only (quintiles, blocks, three flow ranks): no tract-level counts.
"""
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

import stats
import style

P = style.PALETTE
ROLE_STYLE = {"search": {"facecolor": "white", "edgecolor": P["discovery"], "linewidth": 0.7},
              "discovery": {"facecolor": "white", "edgecolor": P["discovery"], "linewidth": 0.7},
              "lawtest": {"facecolor": P["heldout"], "edgecolor": "white", "linewidth": 0.3},
              "confirm": {"facecolor": P["confirmation"], "edgecolor": "white", "linewidth": 0.3}}
ABBR = {"35620": "NYC", "31080": "LA", "16980": "CHI", "BAY9": "SF", "19100": "DFW", "26420": "HOU", "47900": "DC",
        "33100": "MIA", "37980": "PHL", "12060": "ATL", "14460": "BOS", "38060": "PHX"}
OFFSET = {"NYC": (10, 2), "LA": (-10, 0), "CHI": (-9, 3), "SF": (-8, 0), "DFW": (-10, 0), "HOU": (0, -9), "DC": (9, -4),
          "MIA": (9, 0), "PHL": (10, -6), "ATL": (9, 0)}
POI_RAMP = ["#F5F5F3", "#DEDEDB", "#BFBFBB", "#93938F", "#5E5E5A"]          # neutral: teal is kept for model colours
SEARCH_FILL, VAL_FILL = "#DDDDDA", "#8F8F8A"
WINDOW_B_KM, WINDOW_C_KM = 18.75, 40.0
EXCLUDE = {"02", "15", "72", "78", "60", "66", "69"}
WB = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0}
BENCH = ("strexp1", "grav_power", "ext_rad", "io")                          # the families of the small benchmark plot


def _area(pop):
    """Marker area proportional to population."""
    return 80.0 * np.asarray(pop, float) / 2e7


def load_geo():
    import geopandas as gpd
    st = gpd.read_file(style.EXT / "cb_2020_us_state_20m.shp")
    st = st[~st["STATEFP"].isin(EXCLUDE)].to_crs("EPSG:5070")
    ex = gpd.read_file(style.DATA / "example_city.geojson").to_crs("EPSG:5070")
    return st, ex


def _frame(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.4)
        sp.set_edgecolor(P["grey"])


# --------------------------------------------------------------------------------------------- glyphs (mm units)
def group_marker(ax, x, y, kind, s=14):
    """A city-group marker as on the map."""
    if kind in ("discovery", "search"):
        ax.scatter([x], [y], s=s, facecolor="white", edgecolor=P["discovery"], linewidth=0.7, zorder=5)
        if kind == "search":
            ax.scatter([x], [y], s=s * 0.12, facecolor=P["discovery"], edgecolor="none", zorder=6)
    elif kind == "heldout":
        ax.scatter([x], [y], s=s, facecolor=P["heldout"], edgecolor="white", linewidth=0.3, zorder=5)
    else:
        ax.scatter([x], [y], s=s, facecolor=P["confirmation"], edgecolor="white", linewidth=0.3, zorder=5)


def tree_glyph(ax, x, y, h=5.6):
    """A small expression tree: what symbolic regression searches over."""
    c = P["ours"]
    n = {"r": (x, y + h / 2), "a": (x - h * 0.42, y), "b": (x + h * 0.42, y), "c": (x - h * 0.68, y - h / 2),
         "d": (x - h * 0.16, y - h / 2), "e": (x + h * 0.42, y - h / 2)}
    for u, v in (("r", "a"), ("r", "b"), ("a", "c"), ("a", "d"), ("b", "e")):
        ax.plot([n[u][0], n[v][0]], [n[u][1], n[v][1]], color=c, lw=0.6, zorder=3)
    for k, (px, py) in n.items():
        ax.add_patch(Circle((px, py), h * 0.11, facecolor="white" if k in ("c", "d", "e") else c, edgecolor=c, linewidth=0.6, zorder=4))


def agent_glyph(ax, x, y, s=5.2):
    """The LLM agent: a rounded tile with a four-point spark."""
    ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s, boxstyle=f"round,pad=0,rounding_size={s * 0.28}", facecolor="#EFEFEC",
                                edgecolor=P["ref"], linewidth=0.6, zorder=3))
    r, q = s * 0.34, s * 0.09
    pts = [(x, y + r), (x + q, y + q), (x + r, y), (x + q, y - q), (x, y - r), (x - q, y - q), (x - r, y), (x - q, y + q)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=P["ref"], edgecolor="none", zorder=4))


def rank_glyph(ax, x, y, s=5.2):
    """Scoring: a ranked list, the best candidate on top."""
    ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s, boxstyle=f"round,pad=0,rounding_size={s * 0.28}", facecolor="white",
                                edgecolor=P["heldout"], linewidth=0.6, zorder=3))
    for k, (frac, col) in enumerate(((0.62, P["ours"]), (0.46, P["prior_pop"]), (0.30, P["prior_pop"]))):
        yy = y + s * 0.24 - k * s * 0.24
        ax.plot([x - s * 0.31, x - s * 0.31 + s * frac], [yy, yy], color=col, lw=1.3, solid_capstyle="butt", zorder=4)


def doc_glyph(ax, x, y, w=5.0, h=6.6):
    """A list of hypotheses: a sheet with three bulleted lines."""
    c = w * 0.28
    pts = [(x - w / 2, y - h / 2), (x + w / 2, y - h / 2), (x + w / 2, y + h / 2 - c), (x + w / 2 - c, y + h / 2), (x - w / 2, y + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=P["heldout"], linewidth=0.6, zorder=3))
    for k in range(3):
        yy = y + h * 0.16 - k * h * 0.21
        ax.add_patch(Circle((x - w * 0.27, yy), w * 0.055, facecolor=P["heldout"], edgecolor="none", zorder=4))
        ax.plot([x - w * 0.12, x + w * (0.14 if k == 0 else 0.30)], [yy, yy], color=P["grey"], lw=0.6, zorder=4)


def tick_glyph(ax, x, y, s=2.4):
    ax.plot([x - s * 0.45, x - s * 0.10, x + s * 0.50], [y, y - s * 0.36, y + s * 0.42], color=P["heldout"], lw=1.0, zorder=4,
            solid_capstyle="round", solid_joinstyle="round")


def cross_glyph(ax, x, y, s=2.0):
    for a, b in (((-1, -1), (1, 1)), ((-1, 1), (1, -1))):
        ax.plot([x + a[0] * s * 0.4, x + b[0] * s * 0.4], [y + a[1] * s * 0.4, y + b[1] * s * 0.4], color=P["grey"], lw=1.0, zorder=4,
                solid_capstyle="round")


def mini_axes(ax, x0, y0, w, h, xlab, ylab, key):
    ax.plot([x0, x0, x0 + w], [y0 + h, y0, y0], color=P["ref"], lw=0.5, zorder=3)
    ax.text(x0 + w / 2, y0 - 2.0, xlab, **key)
    ax.text(x0 - 1.5, y0 + h / 2, ylab, rotation=90, **key)


def lock_glyph(ax, x, y, s=4.2):
    ax.add_patch(Arc((x, y + s * 0.18), s * 0.5, s * 0.62, theta1=0, theta2=180, color=P["heldout"], lw=0.8, zorder=4))
    ax.add_patch(FancyBboxPatch((x - s * 0.38, y - s * 0.42), s * 0.76, s * 0.6, boxstyle=f"round,pad=0,rounding_size={s * 0.1}",
                                facecolor=P["heldout"], edgecolor="none", zorder=5))


def _step(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=5, lw=0.7, color=P["ref"], zorder=2))


# --------------------------------------------------------------------------------------------- panel d
def workflow(ax, fig, d) -> list:
    """Panel d: the workflow drawn as pictures (units: mm). Every stage names what its picture shows; the small plots
    are real results. Returns the rows of the source table."""
    cities = d["cities"]
    n = cities["role"].value_counts()
    ax.set_axis_off()
    bb = ax.get_position(original=True)
    W, H = bb.width * fig.get_size_inches()[0] / style.MM, bb.height * fig.get_size_inches()[1] / style.MM
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ytit, ycity, cy = H - 3.2, 3.2, H * 0.50 + 0.6
    head = {"fontsize": 6.5, "fontweight": "bold", "ha": "center", "va": "center"}
    key = {"fontsize": 5, "color": P["grey"], "ha": "center", "va": "center"}
    row = {"fontsize": 5.5, "va": "center"}
    src = []

    # ---- 1 formula search: the inner loop; every node says what it does
    cx, R = 25.0, 9.5
    ax.text(cx, ytit, "Formula search", **head)
    pos = {k: (cx + R * np.cos(np.radians(a)), cy + R * np.sin(np.radians(a))) for k, a in (("sr", 90), ("agent", 210), ("score", 330))}
    for a0, a1 in ((66, -4), (304, 236), (184, 114)):                       # clockwise arcs between the three nodes
        t = np.radians(np.linspace(a0, a1, 30))
        ax.plot(cx + R * np.cos(t), cy + R * np.sin(t), color=P["ref"], lw=0.7, zorder=2)
        ax.add_patch(FancyArrowPatch((cx + R * np.cos(t[-2]), cy + R * np.sin(t[-2])), (cx + R * np.cos(t[-1]), cy + R * np.sin(t[-1])),
                                     arrowstyle="-|>", mutation_scale=5, lw=0, color=P["ref"], zorder=2, shrinkA=0, shrinkB=0))
    tree_glyph(ax, *pos["sr"])
    agent_glyph(ax, *pos["agent"])
    rank_glyph(ax, *pos["score"])
    ax.text(pos["sr"][0] - 5.2, pos["sr"][1] + 2.9, "symbolic regression", fontsize=5.5, color=P["ours"], ha="right", va="center")
    ax.text(pos["sr"][0] - 5.2, pos["sr"][1] + 0.7, "generates formulas", fontsize=5, color=P["grey"], ha="right", va="center")
    ax.text(pos["agent"][0] - 3.6, pos["agent"][1] + 1.6, "LLM agent", fontsize=5.5, color=P["ref"], ha="right", va="center")
    ax.text(pos["agent"][0] - 3.6, pos["agent"][1] - 0.6, "proposes,", fontsize=5, color=P["grey"], ha="right", va="center")
    ax.text(pos["agent"][0] - 3.6, pos["agent"][1] - 2.6, "rewrites", fontsize=5, color=P["grey"], ha="right", va="center")
    ax.text(pos["score"][0] + 3.6, pos["score"][1] + 1.6, "scoring", fontsize=5.5, color=P["heldout"], ha="left", va="center")
    ax.text(pos["score"][0] + 3.6, pos["score"][1] - 0.6, "fit to trips,", fontsize=5, color=P["grey"], ha="left", va="center")
    ax.text(pos["score"][0] + 3.6, pos["score"][1] - 2.6, "ranked automatically", fontsize=5, color=P["grey"], ha="left", va="center")
    ax.text(cx, cy - 0.4, "inner\nloop", fontsize=5.5, color=P["grey"], ha="center", va="center", linespacing=1.0, fontstyle="italic")
    group_marker(ax, cx - 4.2, ycity, "search")
    ax.text(cx - 2.6, ycity, f"{n['search']} cities", **row)

    # the relation both small plots show: mid-distance decay against city size (flexible model; real values)
    flex = d["bands"][(d["bands"]["model"] == "Flexible") & (d["bands"]["band"] == 3)][["city", "ed"]]
    t = cities.merge(flex, on="city")
    x_lo, x_hi = t["log_population"].min(), t["log_population"].max()
    y_lo, y_hi = t["ed"].min(), t["ed"].max()
    disc, held = t[t["set"] == "discovery"], t[t["set"] == "heldout"]
    w, hgt = 14.5, 12.0

    def uv(g):
        return (g["log_population"] - x_lo) / (x_hi - x_lo), (g["ed"] - y_lo) / (y_hi - y_lo)

    def line(g):
        u, v = uv(g)
        fit, _, _, _ = stats.ols_band(u, v, np.array([0.0, 1.0]))
        return fit

    def to_mm(x0, y0, u, v):
        return x0 + (0.05 + 0.9 * np.asarray(u)) * w, y0 + (0.06 + 0.88 * np.asarray(v)) * hgt

    # ---- 2 hypotheses: pattern across the discovery cities -> the agent proposes -> a registered list
    hx0, hy0 = 57.0, cy - 6.3
    ax.text(hx0 + 14.5, ytit, "Hypotheses", **head)
    mini_axes(ax, hx0, hy0, w, hgt, "city size", "decay", key)
    ax.scatter(*to_mm(hx0, hy0, *uv(disc)), s=4, facecolor="white", edgecolor=P["discovery"], linewidth=0.5, zorder=4)
    fd = line(disc)
    ax.plot(*to_mm(hx0, hy0, [0, 1], fd), color=P["discovery"], lw=0.8, zorder=3)
    ax_, dx_ = hx0 + w + 5.0, hx0 + w + 12.6                                  # the agent, then the list
    _step(ax, hx0 + w + 1.4, dx_ - 3.6, cy)
    agent_glyph(ax, ax_, cy + 4.6, s=3.8)
    ax.text(ax_, cy + 8.2, "LLM agent", fontsize=5, color=P["ref"], ha="center", va="center")
    ax.text(ax_, cy - 2.2, "proposes", **key)
    doc_glyph(ax, dx_, cy)
    ax.text(dx_, cy - 5.6, "hypotheses", **key)
    group_marker(ax, hx0 + 8.5, ycity, "discovery")
    ax.text(hx0 + 10.1, ycity, f"{len(disc)} cities", **row)
    src += [{"stage": "Hypotheses", "item": c, "x": x, "y": y} for c, x, y in zip(disc["city"], disc["log_population"], disc["ed"])]

    # ---- 3 held-out test: the same relation on the held-out cities, against the discovery fit
    tx0 = 98.5
    ax.text(tx0 + 11.0, ytit, "Held-out test", **head)
    mini_axes(ax, tx0, hy0, w, hgt, "city size", "decay", key)
    ax.plot(*to_mm(tx0, hy0, [0, 1], fd), color=P["discovery"], lw=0.8, ls=(0, (3, 1.6)), zorder=3)
    ax.scatter(*to_mm(tx0, hy0, *uv(held)), s=4.5, facecolor=P["heldout"], edgecolor="white", linewidth=0.25, zorder=4)
    ax.plot(*to_mm(tx0, hy0, [0, 1], line(held)), color=P["heldout"], lw=0.9, zorder=5)
    kx_, ky_ = tx0 + w * 0.50, hy0 + hgt + 1.4                                # which line is which
    ax.plot([kx_, kx_ + 2.4], [ky_, ky_], color=P["discovery"], lw=0.8, ls=(0, (3, 1.6)), zorder=3)
    ax.text(kx_ + 3.0, ky_, "discovery fit", fontsize=5, color=P["grey"], ha="left", va="center")
    vx = tx0 + w + 3.0                                                        # the two possible verdicts
    tick_glyph(ax, vx, cy + 2.4)
    ax.text(vx + 1.9, cy + 2.4, "supported", fontsize=5, color=P["heldout"], ha="left", va="center")
    cross_glyph(ax, vx, cy - 2.0)
    ax.text(vx + 1.9, cy - 2.0, "not", fontsize=5, color=P["grey"], ha="left", va="center")
    group_marker(ax, tx0 + 6.0, ycity, "heldout")
    ax.text(tx0 + 7.6, ycity, f"{n['lawtest']} cities", **row)
    src += [{"stage": "Held-out test", "item": c, "x": x, "y": y} for c, x, y in zip(held["city"], held["log_population"], held["ed"])]

    # the outer loop, from the test back to the search; the forward steps
    ax.add_patch(FancyArrowPatch((tx0 + 5.0, cy + 10.4), (cx + 9.0, cy + 9.4), connectionstyle="arc3,rad=0.10", arrowstyle="-|>",
                                 mutation_scale=5, lw=0.7, color=P["ref"], zorder=2, shrinkA=0, shrinkB=0))
    ax.text(49.0, cy + 14.2, "outer loop", fontsize=5.5, color=P["grey"], ha="center", va="center", fontstyle="italic")
    _step(ax, 48.0, 52.5, cy)
    _step(ax, 90.6, 94.4, cy)

    # ---- 4 freeze: what is frozen
    fx = 129.5
    ax.plot([fx, fx], [ycity + 2.6, cy - 6.6], color=P["heldout"], lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.plot([fx, fx], [cy + 3.6, ytit - 2.4], color=P["heldout"], lw=0.6, ls=(0, (3, 2)), zorder=2)
    lock_glyph(ax, fx, cy)
    ax.text(fx, ytit, "Freeze", **head)
    ax.text(fx, cy - 4.6, "3 formulas", **key)

    # ---- 5 confirmation: the two things opened after the freeze, each named
    kx = 146.0
    ax.text(kx, ytit, "Confirmation", **head)
    ax.text(kx, cy + 8.2, "opened after freeze", fontsize=5, color=P["grey"], ha="center", va="center", fontstyle="italic")
    ux = kx - 7.4                                                             # the unseen cities, as on the map
    for i in range(int(n["confirm"])):
        ax.scatter([ux - 1.6 + (i % 2) * 3.2], [cy + 3.3 - (i // 2) * 3.0], s=13, facecolor=P["confirmation"], edgecolor="white",
                   linewidth=0.3, zorder=4)
    ax.text(ux, cy - 5.6, f"{n['confirm']} unseen", **key)
    ax.text(ux, cy - 7.5, "cities", **key)
    ax.text(kx - 1.0, cy + 0.3, "+", fontsize=7, color=P["ref"], ha="center", va="center")
    gx, cell = kx + 2.9, 2.5                                                  # the periods x blocks grid of panel c, its test cell hatched
    for i in range(3):
        for j in range(3):
            on = (i, j) == (2, 2)
            ax.add_patch(Rectangle((gx + j * cell + 0.15, cy + 3.75 - (i + 1) * cell + 0.15), cell - 0.3, cell - 0.3, zorder=3,
                                   facecolor="white" if on else "#ECECEA", edgecolor=P["heldout"] if on else "none", linewidth=0.5,
                                   hatch="//////" if on else None))
    ax.text(gx + 1.5 * cell, cy - 5.6, "test period & block", **key)
    ax.text(gx + 1.5 * cell, cy - 7.5, "in all 67 cities", **key)       # both captions carry their own count: no city row here
    _step(ax, fx + 2.6, fx + 6.4, cy)
    _step(ax, 158.0, 161.5, cy)

    # ---- 6 benchmark: fit share, our formula above four earlier models (real values)
    bx0, sc = 164.5, 9.2                                                      # x of share 0, mm per unit share
    ax.text(bx0 + 5.0, ytit, "Benchmark", **head)
    s = d["p9_summary"]
    loc = s[s["level"] == "local"].set_index("model")["share_med"]
    bars = [("c0355r1", P["ours"])] + [(f"{fam}_N", P["prior_poi"]) for fam in BENCH]
    for i, (m, col) in enumerate(bars):
        yy = cy + 4.9 - i * 2.5
        ax.add_patch(Rectangle((bx0, yy - 0.8), sc * float(loc[m]), 1.6, facecolor=col, edgecolor="none", zorder=3))
        src.append({"stage": "Benchmark", "item": m, "x": float(loc[m]), "y": None})
    ax.text(bx0 + sc * float(loc["c0355r1"]) + 0.7, cy + 4.9, "ours", fontsize=5.5, color=P["ours"], ha="left", va="center", fontweight="bold")
    ax.plot([bx0, bx0], [cy - 6.3, cy + 6.3], color=P["ref"], lw=0.5, zorder=4)
    ax.add_patch(FancyArrowPatch((bx0, cy - 6.3), (bx0 + sc * 1.05, cy - 6.3), arrowstyle="-|>", mutation_scale=4, lw=0.5, color=P["ref"],
                                 zorder=4, shrinkA=0, shrinkB=0))
    ax.text(bx0 + sc * 0.5, cy - 8.3, "fit share", **key)
    n_prior = int(((s["level"] == "local") & (s["kind"] == "prior")).sum())
    ax.text(bx0 + sc * 0.5, cy - 10.6, f"{n_prior} benchmark versions", fontsize=5, color=P["prior_poi"], ha="center", va="center")
    group_marker(ax, bx0 - 2.6, ycity, "discovery")
    ax.add_patch(FancyArrowPatch((bx0 - 1.3, ycity), (bx0 + 1.7, ycity), arrowstyle="-|>", mutation_scale=4, lw=0.5, color=P["ref"], zorder=2))
    group_marker(ax, bx0 + 3.0, ycity, "heldout")
    group_marker(ax, bx0 + 5.2, ycity, "confirmation")
    ax.text(bx0 + 6.8, ycity, f"{n['lawtest'] + n['confirm']} cities", **row)
    style.panel_label(ax, "d", dx=-2, dy=-12)
    return src


# --------------------------------------------------------------------------------------------- the figure
def build(d: dict, geo=None):
    style.apply()
    st, ex = geo or load_geo()
    fig = style.figure(style.W2, 122)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.75, 1], hspace=0.06, **style.layout(fig, 3, 3, 4, 2))
    top = outer[0].subgridspec(2, 2, width_ratios=[1.9, 1], wspace=0.03, hspace=0.10)
    ax_a = fig.add_subplot(top[:, 0])
    ax_b = fig.add_subplot(top[0, 1])
    cgs = top[1, 1].subgridspec(1, 2, width_ratios=[0.78, 1.3], wspace=0.05)
    ax_c, ax_g = fig.add_subplot(cgs[0]), fig.add_subplot(cgs[1])
    ax_d = fig.add_subplot(outer[1])
    cities = d["cities"]

    # ---- a: the map — three groups; the search subset is a dot inside the discovery ring
    st.plot(ax=ax_a, facecolor="#F3F3F1", edgecolor="white", linewidth=0.5, zorder=1)
    for role in ("confirm", "lawtest", "discovery", "search"):
        g = cities[cities["role"] == role].sort_values("population", ascending=False)
        ax_a.scatter(g["x"], g["y"], s=_area(g["population"]), zorder=4 if role == "confirm" else 3, **ROLE_STYLE[role])
        if role == "search":
            ax_a.scatter(g["x"], g["y"], s=_area(g["population"]) * 0.10, zorder=3, facecolor=P["discovery"], edgecolor="none")
    for r in cities.sort_values("population", ascending=False).head(8).itertuples():
        dx, dy = OFFSET.get(ABBR[r.city], (9, 0))
        ax_a.annotate(ABBR[r.city], (r.x, r.y), xytext=(dx, dy), textcoords="offset points", fontsize=5.5, color=P["ref"],
                      ha="left" if dx > 0 else "right" if dx < 0 else "center", va="center")
    ax_a.set_axis_off()
    ax_a.set_aspect("equal")
    ax_a.set_xlim(-2.45e6, 2.35e6)
    ax_a.set_ylim(1.4e5, 3.25e6)
    n = cities["role"].value_counts()
    g_, k_ = P["discovery"], P["heldout"]
    h = [Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=g_, mew=0.7, label=f"Discovery  {n['search'] + n['discovery']}"),
         Line2D([], [], ls="none", marker=r"$\odot$", ms=4.6, mfc=g_, mec=g_, mew=0.2, label=f"formula-search subset  {n['search']}"),
         Line2D([], [], ls="none", marker="o", ms=4, mfc=k_, mec=k_, label=f"Held-out  {n['lawtest']}"),
         Line2D([], [], ls="none", marker="o", ms=4, mfc=P["confirmation"], mec=P["confirmation"], label=f"Confirmation  {n['confirm']}")]
    leg = ax_a.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.3, labelspacing=0.35)
    ax_a.add_artist(leg)
    hs = [Line2D([], [], ls="none", marker="o", ms=np.sqrt(_area(p)), mfc="none", mec=P["grey"], mew=0.5, label=lab)
          for p, lab in ((1e6, "1"), (5e6, "5"), (2e7, "20"))]
    ax_a.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.50, 0.0), title="Population (million)", title_fontsize=6,
                ncols=3, handletextpad=0.4, columnspacing=1.2, labelspacing=0.3)
    style.panel_label(ax_a, "a", dx=-2, dy=-8)

    # ---- b: one city — the choice set and the main destinations of one home tract
    home = ex[ex["home"]].geometry.iloc[0].centroid

    def box(km, ax):
        w = km * 1000
        bb = ax.get_position()
        fw, fh = fig.get_size_inches()
        ratio = (bb.height * fh) / (bb.width * fw)
        return (home.x - w, home.x + w, home.y - w * ratio, home.y + w * ratio)
    bb = box(WINDOW_B_KM, ax_b)
    view = ex.cx[bb[0]:bb[1], bb[2]:bb[3]]
    view.plot(ax=ax_b, column="poi_class", cmap=ListedColormap(POI_RAMP), vmin=1, vmax=5, edgecolor="white", linewidth=0.15, zorder=1)
    far = None
    for r in ex[ex["flow_class"] > 0].sort_values("flow_class").itertuples():
        c = r.geometry.centroid
        a = FancyArrowPatch((home.x, home.y), (c.x, c.y), connectionstyle="arc3,rad=0.22", arrowstyle="-",
                            lw={1: 0.5, 2: 0.85, 3: 1.3}[r.flow_class], color=P["heldout"], zorder=3)
        a.set_path_effects([pe.withStroke(linewidth={1: 1.3, 2: 1.7, 3: 2.2}[r.flow_class], foreground="white")])
        ax_b.add_patch(a)
        ax_b.scatter([c.x], [c.y], s=4.5, facecolor=P["heldout"], edgecolor="white", linewidth=0.3, zorder=4)
        if far is None or (c.x - home.x) ** 2 + (c.y - home.y) ** 2 > (far.x - home.x) ** 2 + (far.y - home.y) ** 2:
            far = c
    ax_b.scatter([home.x], [home.y], marker="*", s=46, facecolor="white", edgecolor=P["heldout"], linewidth=0.6, zorder=5)
    ax_b.annotate("home", (home.x, home.y), xytext=(-5, -8), textcoords="offset points", fontsize=6, ha="right", bbox=WB, zorder=6)
    ax_b.annotate("destination", (far.x, far.y), xytext=(6, 5), textcoords="offset points", fontsize=6, ha="left", bbox=WB, zorder=6)
    ax_b.add_patch(Rectangle((0.015, 0.03), 0.30, 0.24, transform=ax_b.transAxes, facecolor="white", edgecolor="none", alpha=0.88, zorder=6))
    cax = ax_b.inset_axes([0.035, 0.095, 0.26, 0.045], zorder=7)
    for i, col in enumerate(POI_RAMP):
        cax.add_patch(Rectangle((i, 0), 1, 1, facecolor=col, edgecolor="none"))
    cax.set_xlim(0, 5)
    cax.set_ylim(0, 1)
    cax.set_axis_off()
    ax_b.text(0.035, 0.16, "POI count", transform=ax_b.transAxes, fontsize=5.5, va="bottom", zorder=8)
    ax_b.text(0.035, 0.085, "low", transform=ax_b.transAxes, fontsize=5, va="top", color=P["grey"], zorder=8)
    ax_b.text(0.295, 0.085, "high", transform=ax_b.transAxes, fontsize=5, va="top", ha="right", color=P["grey"], zorder=8)
    ax_b.set_xlim(bb[0], bb[1])
    ax_b.set_ylim(bb[2], bb[3])
    _frame(ax_b)
    style.panel_label(ax_b, "b", dx=-3, dy=-9).set_bbox(WB)

    # ---- c: the holdout scheme — the destination blocks on the map (three fills, as in the grid's rows) ...
    bc = box(WINDOW_C_KM, ax_c)
    wide = ex.cx[bc[0]:bc[1], bc[2]:bc[3]]
    for fold in (2, 3, 4):
        g = wide[wide["block"] == fold]
        if len(g):
            g.plot(ax=ax_c, facecolor=SEARCH_FILL, edgecolor=SEARCH_FILL, linewidth=0.3, zorder=1)
    g = wide[wide["block"] == 1]
    if len(g):
        g.plot(ax=ax_c, facecolor=VAL_FILL, edgecolor=VAL_FILL, linewidth=0.3, zorder=1)
    te = wide[wide["block"] == 0]
    if len(te):
        te.plot(ax=ax_c, facecolor="white", edgecolor="white", linewidth=0.3, zorder=2)
        te.plot(ax=ax_c, facecolor="none", edgecolor=P["heldout"], linewidth=0, hatch="//////", zorder=3)
    ax_c.scatter([home.x], [home.y], marker="*", s=30, facecolor="white", edgecolor=P["heldout"], linewidth=0.5, zorder=5)
    xr = bc[1] - 0.07 * (bc[1] - bc[0])
    ax_c.plot([xr - 20000, xr], [bc[2] + 5000] * 2, color=P["heldout"], lw=1.0, zorder=5)
    ax_c.text(xr - 10000, bc[2] + 8000, "20 km", fontsize=5.5, ha="center", va="bottom", bbox=WB, zorder=5)
    ax_c.set_xlim(bc[0], bc[1])
    ax_c.set_ylim(bc[2], bc[3])
    _frame(ax_c)
    style.panel_label(ax_c, "c", dx=-3, dy=-9).set_bbox(WB)

    # ---- ... and the periods x blocks grid: which cell is used for what
    ax_g.set_axis_off()
    ax_g.set_xlim(0, 10)
    ax_g.set_ylim(0, 10)
    cols = [("train", "Mar–Apr"), ("validation", "1–15\nMay"), ("test", "16–31\nMay")]
    rows = [("search", {"facecolor": SEARCH_FILL, "edgecolor": "none"}),
            ("validation", {"facecolor": VAL_FILL, "edgecolor": "none"}),
            ("test", {"facecolor": "white", "edgecolor": P["heldout"], "linewidth": 0.5, "hatch": "//////"})]
    use = {(0, 0): "fit", (1, 1): "select", (2, 2): "report"}
    x0, cw, y0, rh = 3.7, 2.05, 6.3, 1.7
    ax_g.text(x0 + 1.5 * cw, 9.9, "Period", fontsize=6, ha="center", va="top", color=P["grey"])
    ax_g.text(0.15, 8.1, "Block", fontsize=6, ha="left", va="center", color=P["grey"])
    for j, (nm, dates) in enumerate(cols):
        ax_g.text(x0 + (j + 0.5) * cw, 8.1, nm, fontsize=5.2, ha="center", va="center")
        ax_g.text(x0 + (j + 0.5) * cw, 1.15, dates, fontsize=5, ha="center", va="center", color=P["grey"], linespacing=1.0)
    for i, (nm, kw) in enumerate(rows):
        yc = y0 - i * rh
        ax_g.add_patch(Rectangle((0.15, yc - 0.40), 0.75, 0.80, **kw))
        ax_g.text(1.08, yc, nm, fontsize=5.2, va="center")
        for j in range(3):
            on = (i, j) in use
            ax_g.add_patch(Rectangle((x0 + j * cw + 0.07, yc - rh / 2 + 0.09), cw - 0.14, rh - 0.18, zorder=1,
                                     facecolor=P["heldout"] if on else "#F3F3F1", edgecolor="none"))
            if on:
                ax_g.text(x0 + (j + 0.5) * cw, yc, use[(i, j)], fontsize=5.5, ha="center", va="center", color="white", fontweight="bold")

    # ---- d: the workflow, drawn
    src_d = workflow(ax_d, fig, d)

    src = {"a": cities[["city", "name", "role", "set", "population", "x", "y"]],
           "b_c": pd.DataFrame(ex.drop(columns="geometry")),
           "c_grid": pd.DataFrame([{"block": rows[i][0], "period": cols[j][0], "use": v} for (i, j), v in use.items()]),
           "d": pd.DataFrame(src_d)}
    return fig, src


def main():
    import data
    fig, src = build(data.load())
    for p in style.save(fig, "fig1", src):
        print(p)


if __name__ == "__main__":
    main()
