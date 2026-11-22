"""Fig. 4b (2026-11-20): the ten registered hypotheses of Fig. 4a on the axes of the 69 further relations.

Writes data/paper/figdata/rev/e1_tested.csv: the standardised discovery and held-out slopes of the ten hypotheses,
on the same definition as figdata/e1_pairs.csv (y and x z-scored within each city set; form characteristics
controlled for log population).
  form rows (P7-03 to P7-10)  the "with_registered" variant of the exploration output
                              (figdata/rev/explore_e1_pairs.csv, a copy of results/multicity/explore/e1_pairs.csv)
                              minus its "main" variant;
  size rows (P7-01, P7-02)    computed here with the exploration's own function from the city-level outcomes in
                              figdata/rev. The characteristic is log population itself, so nothing is controlled for.
Checks: the "main" variant equals figdata/e1_pairs.csv; the form rows whose characteristic is in
figdata/rev/covariates*.csv are recomputed here and must agree to 1e-9.
Run it before amend_theta2.py: like the exploration itself, it uses the city fits of record, and the amendment
replaces the Reach model's 5-10 km elasticity of five cities in figdata/rev/city_outcomes*.parquet.
Usage: python e1_tested.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[2] / "paper"
FD = PAPER / "figdata"
RV = FD / "rev"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multicity"))
SIZE = "log_population"


def _frames(cities: pd.DataFrame, outcomes: str, covariates: str, city_set: str) -> tuple:
    keep = set(cities.loc[cities["set"] == city_set, "city"])
    o = pd.read_parquet(RV / outcomes)
    o = o.assign(city=o["city"].astype(str))
    if "block" in o:
        o = o[o["block"] == "test_dest"]
    o = o[o["city"].isin(keep)].drop(columns=["block", SIZE], errors="ignore").reset_index(drop=True)
    c = pd.read_csv(RV / covariates, dtype={"city": str})
    c = c[c["city"].isin(keep)].merge(cities[["city", SIZE]], on="city").reset_index(drop=True)
    if not (len(o) == len(keep) == len(c)):
        raise ValueError(f"{city_set}: {len(o)} outcome rows and {len(c)} covariate rows for {len(keep)} cities")
    return o, c


def build() -> pd.DataFrame:
    import explore_form as ef
    cities = pd.read_csv(FD / "cities.csv", dtype={"city": str})
    p7 = pd.read_csv(FD / "p7.csv")
    allv = pd.read_csv(RV / "explore_e1_pairs.csv")
    main = allv[allv["variant"] == "main"].reset_index(drop=True)
    local = pd.read_csv(FD / "e1_pairs.csv")
    if not (len(main) == len(local) and np.allclose(main["d_std"], local["d_std"]) and np.allclose(main["l_std"], local["l_std"])):
        raise ValueError("explore_e1_pairs.csv (main) differs from figdata/e1_pairs.csv")
    seen = set(zip(main["outcome"], main["covariate"]))
    reg = allv[allv["variant"] == "with_registered"]
    form = reg[[(o, c) not in seen for o, c in zip(reg["outcome"], reg["covariate"])]][["outcome", "covariate", "d_std", "l_std"]]
    want = p7[p7["group"] == "form"]
    if set(zip(form["outcome"], form["covariate"])) != set(zip(want["outcome"], want["covariate"])):
        raise ValueError("the registered form pairs of the exploration output are not rows 3-10 of Fig. 4a")
    od, cd = _frames(cities, "city_outcomes.parquet", "covariates.csv", "discovery")
    ol, cl = _frames(cities, "city_outcomes_lawtest.parquet", "covariates_lawtest.csv", "heldout")
    chk = form[form["covariate"].isin(cd.columns)]
    pairs = list(zip(chk["outcome"], chk["covariate"]))
    gap = max(np.abs(ef.std_partial_slopes(od, cd, pairs, (SIZE,)) - chk["d_std"]).max(),
              np.abs(ef.std_partial_slopes(ol, cl, pairs, (SIZE,)) - chk["l_std"]).max())
    if not (len(pairs) >= 3 and gap < 1e-9):
        raise ValueError(f"the local recomputation of {len(pairs)} form rows differs by {gap}")
    sz = p7[p7["group"] == "size"]
    pairs = list(zip(sz["outcome"], sz["covariate"]))
    size = pd.DataFrame({"outcome": sz["outcome"].to_numpy(), "covariate": sz["covariate"].to_numpy(),
                         "d_std": ef.std_partial_slopes(od, cd, pairs, (SIZE,)), "l_std": ef.std_partial_slopes(ol, cl, pairs, (SIZE,))})
    t = p7[["id", "group", "outcome", "covariate"]].assign(row=np.arange(1, len(p7) + 1))
    t = t.merge(pd.concat([size, form], ignore_index=True), on=["outcome", "covariate"], how="left", validate="one_to_one")
    if t[["d_std", "l_std"]].isna().any().any():
        raise ValueError("a registered hypothesis has no standardised slope")
    t["agree"] = np.sign(t["d_std"]) == np.sign(t["l_std"])
    return t[["id", "row", "group", "outcome", "covariate", "d_std", "l_std", "agree"]]


def main():
    t = build()
    t.to_csv(RV / "e1_tested.csv", index=False)
    print(t.round(3).to_string(index=False))
    print("wrote", RV / "e1_tested.csv")


if __name__ == "__main__":
    main()
