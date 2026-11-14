"""Additional file 1, Fig. S1 (revision 2026-11-13): the distance-size regularity under four measurements.

Size elasticity of the banded flexible model against distance, at one reference point shared by every city and band
(rev_banded.py): a POI count; b employment as the banded mass; c employment with log POI count held fixed;
d POI count, non-work visits. Thin lines: cities (bands with >= 1,000 training visits); thick: cross-city median.
Panel a also shows the Scaled kernel's median at the same points. Text: the number of cities whose elasticity rises
with ln d (OLS slope > 0).
"""
import numpy as np
import pandas as pd

import style

RV = style.DATA / "rev"
PANELS = [("fixed", "POI count", "$e_N$"), ("emp", "Employment", "$e_E$"), ("empN", "Employment, POI fixed", "$e_E$"),
          ("nonwork", "Non-work visits", "$e_N$")]


def load(variant: str) -> pd.DataFrame:
    t = pd.read_parquet(RV / f"banded_{variant}.parquet")
    t = t[(t["ref"] == "fixed") & (t["visits"] >= 1000)].copy()
    t["city"] = t["city"].astype(str)
    return t


def rising(g: pd.DataFrame) -> tuple:
    s = [np.polyfit(np.log(c["d_ref"]), c["eN"], 1)[0] for _, c in g.groupby("city") if len(c) >= 3]
    return int(np.sum(np.array(s) > 0)), len(s)


def build():
    style.apply()
    fig = style.figure(183, 58)
    axes = [fig.add_axes([0.07 + k * 0.235, 0.22, 0.19, 0.62]) for k in range(4)]
    P = style.PALETTE
    src = {}
    for k, (ax, (v, title, ylab)) in enumerate(zip(axes, PANELS)):
        t = load(v)
        f = t[t["model"] == "Flexible"]
        for _, g in f.groupby("city"):
            g = g.sort_values("d_ref")
            ax.plot(g["d_ref"], g["eN"], color=P["discovery"], lw=0.35, alpha=0.45, zorder=1)
        med = f.groupby("d_ref")["eN"].median()
        ax.plot(med.index, med.values, color=P["flexible"], lw=1.6, zorder=3, label="Flexible, median")
        ax.plot([], [], color=P["discovery"], lw=0.6, label="Flexible, cities")
        if v == "fixed":
            sk = t[t["model"] == "Scaled kernel"].groupby("d_ref")["eN"].median()
            ax.plot(sk.index, sk.values, color=P["ours"], lw=1.6, zorder=4, label="Scaled kernel, median")
            ax.legend(loc="upper left", fontsize=5.5, handlelength=1.6, labelspacing=0.3, frameon=False)
        n_up, n = rising(f)
        ax.text(0.97, 0.04, f"rising {n_up}/{n}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=P["ref"])
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10, 20], labels=["0.5", "1", "2", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_xlim(0.4, 36)
        ax.axhline(0, color=P["light"], lw=0.6, zorder=0)
        ax.set_title(title, fontsize=7, loc="left", pad=3)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel(ylab)
        style.panel_label(ax, "abcd"[k], dx=22)
        src[v] = f[["city", "band", "d_ref", "eN"]].assign(model="Flexible")
        if v == "fixed":
            src[v] = pd.concat([src[v], t[t["model"] == "Scaled kernel"][["city", "band", "d_ref", "eN"]].assign(model="Scaled kernel")])
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(min(lo, -0.2), max(hi, 1.45))
    return fig, src


if __name__ == "__main__":
    fig, src = build()
    print(style.save(fig, "figS1", src)[0])
