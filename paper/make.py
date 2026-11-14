"""Build the paper figures: make.py [fig1 ... fig7 | tables | supp] (the seven main figures by default).
"supp" redraws Fig. 6 for the other confirmation cities — kept for checking, not part of the paper."""
import importlib
import sys

import data
import style

FIGS = ("fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7")


def supplement(d):
    """The OD-flow figure (Fig. 6) for the other confirmation cities: figures/supp/figS_od_<city>.* — not in the paper
    nor in the supplement; their numbers are in tableS_od_cities."""
    import fig6
    import matplotlib.pyplot as plt
    for city in data.od_cities()[1:]:
        od = data.load_od(city)
        fig, src = fig6.build(d, od)
        out = style.save(fig, f"figS_od_{city}", src, fig_dir=style.FIG_DIR / "supp", src_dir=style.SRC_DIR / "supp")
        plt.close(fig)
        print(f"figS_od_{city}", od["summary"]["name"], "->", out[0].parent)


def main(argv):
    names = argv or FIGS
    d = data.load()
    for n in names:
        if n == "supp":
            supplement(d)
            continue
        if n == "tables":
            import supp_tables
            for path in supp_tables.build():
                print("table ->", path)
            continue
        fig, src = importlib.import_module(n).build(d)
        out = style.save(fig, n, src)
        print(n, "->", out[0].parent, f"({len(src)} source tables)")


if __name__ == "__main__":
    main(sys.argv[1:])
