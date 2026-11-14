"""Shared style of the paper figures: sizes in mm, one palette whose
colours keep their meaning across all figures, keyword-only text, editable vector exports with per-panel source data.

Colour-blind check (all pairs, protan / deutan / tritan and normal vision, 2026-11-11): the model colours
ours / prior_poi / prior_pop and the city-set colours discovery / heldout / confirmation pass; a
grey teal for prior_poi does not (it merges with the grey under protanopia). The greys are deliberately neutral, so
every grey category also differs by shape: hollow against filled markers."""
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import ScaledTranslation

MM = 1 / 25.4
W2, W15, W1, MAX_H = 183, 120, 89, 170          # widths and the height limit, mm
ROOT = Path(os.environ.get("ODSR_PAPER", Path(__file__).resolve().parents[2] / "paper"))
DATA, FIG_DIR, SRC_DIR, EXT = ROOT / "figdata", ROOT / "figures", ROOT / "source_data", ROOT / "external"

_ORANGE = "#D96F00"
PALETTE = {
    "ours": "#1F4E79",            # the main formula: deep blue
    "ours_2": "#4E86B8",          # our other formulas: lighter steps of the same blue
    "ours_3": "#8FB4D6",
    "prior_poi": "#00998A",       # earlier models with the POI count as the mass: teal, filled
    "prior_pop": "#A8A8A8",       # earlier models as published (population mass): light grey, hollow
    "flexible": "#000000",        # the flexible model, the upper reference: black dashed
    "discovery": "#8F8F8F",       # cities used to find things: grey, hollow
    "heldout": "#2A2A2A",         # held-out test cities: near black, filled
    "confirmation": _ORANGE,      # confirmation cities: the only accent colour
    "threshold": _ORANGE,         # ... also used for a threshold the reader must see
    "grey": "#7A7A7A",            # secondary text
    "light": "#E9E9E9",           # bands and backgrounds
    "ref": "#555555",             # reference lines
}
MARKERS = {
    "discovery": {"marker": "o", "facecolor": "none", "edgecolor": PALETTE["discovery"], "linewidth": 0.6},
    "heldout": {"marker": "o", "facecolor": PALETTE["heldout"], "edgecolor": "none", "linewidth": 0},
    "confirmation": {"marker": "D", "facecolor": PALETTE["confirmation"], "edgecolor": "white", "linewidth": 0.3},
}
SET_LABEL = {"discovery": "Discovery cities", "heldout": "Held-out cities", "confirmation": "Confirmation cities"}
DIVERGING = "RdBu_r"


def size_cmap():
    """City size: one sequential ramp, pale sand to deep indigo."""
    return LinearSegmentedColormap.from_list("city_size", ["#DDCB94", "#A9BFA0", "#6F9FB0", "#3B6A9C", "#1B2A5C"])


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "legend.frameon": False, "legend.handlelength": 1.2, "legend.borderaxespad": 0.2,
        "axes.spines.right": False, "axes.spines.top": False, "axes.grid": False, "axes.linewidth": 0.5,
        "axes.labelpad": 2.0, "axes.titlepad": 3.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "lines.linewidth": 1.0, "lines.markersize": 3.0, "patch.linewidth": 0.5, "hatch.linewidth": 0.4,
        "figure.dpi": 150, "savefig.dpi": 600, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def figure(width_mm: float, height_mm: float):
    if height_mm > MAX_H:
        raise ValueError(f"figure height {height_mm} mm exceeds {MAX_H} mm")
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


def panel_label(ax, letter: str, dx: float = 14, dy: float = 3):
    """Small bold lowercase letter just outside the top-left corner (dx, dy in points)."""
    if not (len(letter) == 1 and letter.islower()):
        raise ValueError(f"panel label {letter!r}: one lowercase letter")
    tr = ax.transAxes + ScaledTranslation(-dx / 72, dy / 72, ax.figure.dpi_scale_trans)
    return ax.text(0, 1, letter, transform=tr, fontsize=8, fontweight="bold", va="bottom", ha="left", gid="panel-label")


_MATH = re.compile(r"\$[^$]*\$")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-']*")


def check_text(fig, max_words: int = 3) -> list:
    """Texts that break the keyword rule: more than max_words words (math and numbers do not count) or a sentence
    (ends with a full stop). Sorted, duplicates removed."""
    bad = set()
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text().strip()
        if not s or not t.get_visible():
            continue
        plain = _MATH.sub(" ", s)
        if len(_WORD.findall(plain)) > max_words or re.search(r"[A-Za-z]\.$", plain.strip()):
            bad.add(s)
    return sorted(bad)


def save(fig, stem: str, source: dict, fig_dir=None, src_dir=None) -> list:
    """PDF and SVG (editable text), 600-dpi TIFF, a PNG preview, and one CSV per panel. The figure keeps its exact
    size (nothing is cropped). Refuses a figure whose text breaks the keyword rule."""
    bad = check_text(fig)
    if bad:
        raise ValueError(f"{stem}: text beyond keywords: {bad}")
    fig_dir, src_dir = Path(fig_dir or FIG_DIR), Path(src_dir or SRC_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, kw in ((".pdf", {}), (".svg", {}), (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
                    (".png", {"dpi": 300})):
        p = fig_dir / f"{stem}{ext}"
        fig.savefig(p, **kw)
        if ext == ".tiff":                                   # journals want flattened RGB, not RGBA
            from PIL import Image
            Image.open(p).convert("RGB").save(p, compression="tiff_lzw", dpi=(600, 600))
        out.append(p)
    for panel, df in source.items():
        df.to_csv(src_dir / f"{stem}_{panel}.csv", index=False)
    return out


def scatter_sets(ax, x, y, sets, s: float = 9, order=("discovery", "heldout", "confirmation"), **kw):
    """Points drawn with the city-set markers (discovery hollow grey, held-out filled, confirmation orange diamond)."""
    import numpy as np
    x, y, sets = np.asarray(x, float), np.asarray(y, float), np.asarray(sets)
    out = {}
    for z, k in enumerate(order):
        m = sets == k
        if m.any():
            out[k] = ax.scatter(x[m], y[m], s=s * (0.8 if k == "confirmation" else 1.0), zorder=3 + z, **MARKERS[k], **kw)
    return out


def set_legend(ax, sets=("discovery", "heldout", "confirmation"), extra=(), **kw):
    """Legend of the city-set markers; extra = further handles (a fitted line, say) listed after them."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], linestyle="none", marker=MARKERS[k]["marker"], markersize=3.2 if k != "confirmation" else 2.9,
                markerfacecolor=MARKERS[k]["facecolor"], markeredgecolor=MARKERS[k]["edgecolor"] if MARKERS[k]["edgecolor"] != "none" else MARKERS[k]["facecolor"],
                markeredgewidth=max(MARKERS[k]["linewidth"], 0.3), label=SET_LABEL[k]) for k in sets]
    return ax.legend(handles=h + list(extra), **{"handletextpad": 0.3, "labelspacing": 0.25, **kw})


def layout(fig, left: float, right: float, top: float, bottom: float) -> dict:
    """Margins in mm -> the fractions a GridSpec takes."""
    w, h = fig.get_size_inches() / MM
    return {"left": left / w, "right": 1 - right / w, "top": 1 - top / h, "bottom": bottom / h}
