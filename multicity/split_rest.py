"""Plan 4 Task 1: split the 53 rest cities into discovery / lawtest, once.

Population terciles among the rest cities (population from form_metrics, else from the QC table) -> inside each tercile
sort by pop_density_gini ascending (missing gini last, by city id; exact ties broken by a seeded RNG) -> alternate:
1st discovery, 2nd lawtest, ... Cities without any population form a trailing group ordered by city id.
The frozen `tier` column of cities.csv is not touched; the tercile is returned as `split_tier` and not written back.
Usage: ODSR_ROOT=... python -I split_rest.py   (refuses to run if a discovery / lawtest role already exists)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CITIES_CSV, RESULTS, SPLIT_SEED  # noqa: E402
import ledger  # noqa: E402

ROLES = ("discovery", "lawtest")


def split_rest(cities: pd.DataFrame, form: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    out = cities.copy()
    out["city"] = out["city"].astype(str)
    out["split_tier"] = np.nan
    out["split_rank"] = np.nan
    rest = out.loc[out["role"] == "rest", ["city"]].copy()
    f = form.copy()
    f["city"] = f["city"].astype(str)
    rest = rest.merge(f[["city"] + [c for c in ("population", "pop_density_gini") if c in f]], on="city", how="left")
    for c in ("population", "pop_density_gini"):
        if c not in rest:
            rest[c] = np.nan
    rest = rest.sort_values("city").reset_index(drop=True)
    has = rest["population"].notna()
    rest["split_tier"] = 4.0                                     # no population at all: trailing group
    if has.any():
        r = rest.loc[has, "population"].rank(method="first") - 1
        rest.loc[has, "split_tier"] = np.floor(3 * r / has.sum()).astype(int) + 1
    rng = np.random.default_rng(seed)
    rest["_tie"] = rng.permutation(len(rest))                    # drawn in city-id order: reproducible
    rest["_nogini"] = rest["pop_density_gini"].isna()
    rest.loc[rest["_nogini"], "_tie"] = 0                        # missing gini: by city id only
    rest = rest.sort_values(["split_tier", "_nogini", "pop_density_gini", "_tie", "city"], na_position="last")
    rest["split_rank"] = rest.groupby("split_tier").cumcount()
    rest["new_role"] = [ROLES[k % 2] for k in rest["split_rank"]]
    m = rest.set_index("city")
    idx = out["role"] == "rest"
    out.loc[idx, "split_tier"] = out.loc[idx, "city"].map(m["split_tier"]).to_numpy()
    out.loc[idx, "split_rank"] = out.loc[idx, "city"].map(m["split_rank"]).to_numpy()
    out.loc[idx, "role"] = out.loc[idx, "city"].map(m["new_role"]).to_numpy()
    return out


def main() -> None:
    cities = pd.read_csv(CITIES_CSV, dtype=str)
    if cities["role"].isin(ROLES).any():
        sys.exit("split_rest: discovery / lawtest roles already exist; the split is never redone")
    form = pd.read_csv(RESULTS / "form_metrics.csv", dtype={"city": str})
    qc = pd.read_csv(RESULTS / "qc/cities_qc.csv", dtype={"city": str})[["city", "population"]]
    rest_ids = set(cities.loc[cities["role"] == "rest", "city"])
    form = form[form["city"].isin(rest_ids)]                    # only the rest cities' rows are used
    qc = qc[qc["city"].isin(rest_ids)]
    pop = qc.merge(form[["city", "population", "pop_density_gini"]], on="city", how="left", suffixes=("_qc", ""))
    pop["population"] = pop["population"].fillna(pop["population_qc"])
    out = split_rest(cities, pop[["city", "population", "pop_density_gini"]])
    disc = out.loc[out["role"] == "discovery", "city"].tolist()
    law = out.loc[out["role"] == "lawtest", "city"].tolist()
    out.drop(columns=["split_tier", "split_rank"]).to_csv(CITIES_CSV, index=False)
    n_gini = int(pop["pop_density_gini"].notna().sum())
    ledger.add(f"Task 1 (plan 4): rest split (population terciles from form_metrics/QC, gini ascending, alternate; seed "
               f"{SPLIT_SEED}; {n_gini}/{len(pop)} with gini) discovery {len(disc)} [{' '.join(disc)}] / lawtest "
               f"{len(law)} [{' '.join(law)}]; cities.csv sha {ledger.sha(CITIES_CSV)}; form_metrics.csv sha "
               f"{ledger.sha(RESULTS / 'form_metrics.csv')}")
    print(f"discovery {len(disc)} lawtest {len(law)}")


if __name__ == "__main__":
    main()
