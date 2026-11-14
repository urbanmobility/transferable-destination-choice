"""Supplementary tables of the paper: make.py tables -> data/paper/tables/.

tableS_od_cities: the numbers of the Fig. 6 analysis for all six confirmation cities. The maps of the
other five cities are not shown (decided after viewing them); the table keeps on record that Denver, fixed before
anything was computed, is not a favourable exception. For every city and each of the seven models of Fig. 6, with the
uniform constants (no local trips): how many of the 2,000 largest observed flows are among the model's 2,000 largest,
and the CPC. A second table gives the recovery share for every K, because at small K the ranking is not uniform.
"""
import pandas as pd

import data
import style

OUT = style.ROOT / "tables"
MODELS = [("c0355r1", "Scaled kernel (ours)"), ("grav_power_P", "Gravity"), ("grav_power_N", "Gravity, POI count"),
          ("rad_P", "Radiation"), ("rad_N", "Radiation, POI count"), ("strexp1_P", "Stretched exponential"),
          ("strexp1_N", "Stretched exponential, POI count")]
N_TOP = 2000


def short(name: str) -> str:
    return name.split(",")[0].split("-")[0]              # "Denver-Aurora-Centennial, CO" -> "Denver"


def od_cities() -> dict:
    """{"cities": one row per city and model, "by_k": the recovery share for every K}."""
    rows, by_k = [], []
    for city in data.od_cities():
        od = data.load_od(city)
        ov = od["overlap"].set_index(["source", "K"])["overlap"]
        for m, label in MODELS:
            k = f"{m}|uniform"
            rows.append({"city": city, "name": short(od["summary"]["name"]), "model": m, "label": label,
                         "in_observed_top": int(round(ov[(k, N_TOP)] * N_TOP)), "cpc": float(od["summary"]["models"][k]["cpc"])})
            by_k += [{"city": city, "name": short(od["summary"]["name"]), "model": m, "label": label, "K": int(K), "share": float(v)}
                     for (src, K), v in ov.items() if src == k]
    return {"cities": pd.DataFrame(rows), "by_k": pd.DataFrame(by_k)}


def exceptions(by_k: pd.DataFrame) -> pd.DataFrame:
    """The (city, K) cells in which another model recovers more of the largest flows than the scaled kernel."""
    p = by_k.pivot_table(index=["name", "K"], columns="model", values="share")
    best = p.drop(columns="c0355r1").idxmax(axis=1)
    worse = p["c0355r1"] < p.drop(columns="c0355r1").max(axis=1)
    return pd.DataFrame({"best_other": best[worse], "ours": p.loc[worse, "c0355r1"], "other": p.drop(columns="c0355r1").max(axis=1)[worse]}).reset_index()


def markdown(t: pd.DataFrame, ex: pd.DataFrame) -> str:
    cities = list(dict.fromkeys(t["name"]))
    L = ["Table S (OD flows in the six confirmation cities). Uniform constants, no local trips; test period, all destinations.", "",
         f"Of the {N_TOP:,} largest observed flows, the number that are also among the model's {N_TOP:,} largest", "",
         "| Model | " + " | ".join(cities) + " |", "|---|" + "---|" * len(cities)]
    for _, label in MODELS:
        g = t[t["label"] == label].set_index("name")
        L.append(f"| {label} | " + " | ".join(f"{int(g.loc[c, 'in_observed_top']):,}" for c in cities) + " |")
    L += ["", "CPC", "", "| Model | " + " | ".join(cities) + " |", "|---|" + "---|" * len(cities)]
    for _, label in MODELS:
        g = t[t["label"] == label].set_index("name")
        L.append(f"| {label} | " + " | ".join(f"{g.loc[c, 'cpc']:.3f}" for c in cities) + " |")
    L += ["", "Cells in which another model recovers more of the K largest observed flows than the scaled kernel:", ""]
    lab = dict(MODELS)
    L += [f"- {r.name}, K = {r.K:,}: {lab[r.best_other]} {r.other:.3f}, scaled kernel {r.ours:.3f}" for r in ex.itertuples()] or ["- none"]
    return "\n".join(L) + "\n"


def build() -> list:
    t = od_cities()
    ex = exceptions(t["by_k"])
    OUT.mkdir(parents=True, exist_ok=True)
    out = [OUT / "tableS_od_cities.csv", OUT / "tableS_od_cities_byK.csv", OUT / "tableS_od_cities.md"]
    t["cities"].to_csv(out[0], index=False)
    t["by_k"].to_csv(out[1], index=False)
    out[2].write_text(markdown(t["cities"], ex))
    return out


if __name__ == "__main__":
    for p in build():
        print(p)
