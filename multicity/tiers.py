"""City roles (design §5.2): search cities (SR search + library), sealed confirmation cities, the rest (library fit
only), excluded (ineligible). Tiers by ACS population: >= 5M -> 1, 2-5M -> 2, else 3. A search city that fails QC
(ineligible or support_mismatch) is replaced by the largest eligible same-tier city that is neither a confirmation
city nor already a search city; a confirmation city that fails QC is dropped, not replaced.

Writes role, tier, substitute_for into config/cities.csv and a ledger entry; the list is not changed afterwards.
Usage: ODSR_ROOT=... python -I tiers.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CITIES_CSV, CONFIRM, RESULTS, SEARCH  # noqa: E402
import ledger  # noqa: E402

TIER_BOUNDS = (5_000_000, 2_000_000)


def tier(population):
    return 1 if population >= TIER_BOUNDS[0] else 2 if population >= TIER_BOUNDS[1] else 3


def assign(cities_qc, search, confirm):
    df = cities_qc.copy().set_index("city")
    missing = [c for c in list(search) + list(confirm) if c not in df.index]
    if missing:   # a listed city without a QC row means its QC crashed; it must not silently lose its role
        raise ValueError(f"search / confirmation cities missing from the QC table: {missing}")
    df["support_mismatch"] = df["support_mismatch"].fillna("").astype(str)
    df["tier"] = df["population"].apply(tier)
    df["role"] = np.where(df["eligible"], "rest", "excluded")
    df["substitute_for"] = None
    for c in confirm:
        if c in df.index and df.loc[c, "eligible"]:
            df.loc[c, "role"] = "confirm"
    for c in search:
        ok = c in df.index and bool(df.loc[c, "eligible"]) and df.loc[c, "support_mismatch"] == ""
        if ok:
            df.loc[c, "role"] = "search"
            continue
        t = df.loc[c, "tier"]
        cand = df[(df["role"] == "rest") & (df["tier"] == t) & (df["support_mismatch"] == "")].sort_values("population", ascending=False)
        if len(cand):
            sub = cand.index[0]
            df.loc[sub, "role"] = "search"
            df.loc[sub, "substitute_for"] = c
    return df.reset_index()[["city", "role", "tier", "substitute_for", "population", "eligible"]]


def main(search=SEARCH, confirm=CONFIRM, qc_csv=None, cities_csv=None, force=False, ledger_fn=ledger.add):
    """Roles already written are frozen: a run whose assignment differs raises unless force=True (then the override
    goes to the ledger). An identical re-run is allowed."""
    qc_csv = Path(qc_csv or RESULTS / "qc/cities_qc.csv")
    cities_csv = Path(cities_csv or CITIES_CSV)
    qc = pd.read_csv(qc_csv, dtype={"city": str})
    out = assign(qc, search, confirm)
    cities = pd.read_csv(cities_csv, dtype=str)
    if "role" in cities.columns:
        old, new = cities.set_index("city")["role"], out.set_index("city")["role"]
        changed = [(c, old[c], new[c]) for c in new.index if c in old.index and old[c] not in (new[c], "no_qc")]
        if changed and not force:
            raise RuntimeError(f"roles already written in {cities_csv} would change (the list is frozen; pass --force "
                               f"and record the decision in the ledger): {changed[:10]}")
        if changed:
            ledger_fn(f"Task 10: roles overridden with --force: {changed[:20]}")
    for col in ("role", "tier", "substitute_for"):
        cities = cities.drop(columns=[col], errors="ignore")
    cities = cities.merge(out[["city", "role", "tier", "substitute_for"]], on="city", how="left")
    cities["role"] = cities["role"].fillna("no_qc")
    cities.to_csv(cities_csv, index=False)
    roles = out.groupby("role")["city"].apply(list).to_dict()
    ledger_fn(f"Task 10: city roles written to {cities_csv.name} (sha {ledger.sha(cities_csv)}): "
              f"search {roles.get('search', [])}, confirm {roles.get('confirm', [])}, rest {len(roles.get('rest', []))}, "
              f"excluded {len(roles.get('excluded', []))}; substitutes "
              f"{out.dropna(subset=['substitute_for'])[['city', 'substitute_for']].values.tolist()}")
    print(roles.get("search"), roles.get("confirm"), len(roles.get("rest", [])), len(roles.get("excluded", [])))
    return out


if __name__ == "__main__":
    main(force="--force" in sys.argv[1:])
