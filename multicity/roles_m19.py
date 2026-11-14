"""City roles of the 2019 line M19 (plan 7; design §8 item 1): among the 129 metros with built units,
the 67 cities of 2024 (search / discovery / lawtest / confirm) are `old` (descriptive cross-year replication),
the others that pass the 2019 QC are `new` (the independent replication sample), the rest `excluded`.

Usage: ODSR_LINE=M19 ... python -I roles_m19.py init     (config/cities_M19.csv, every units city "pending")
       ODSR_LINE=M19 ... python -I roles_m19.py assign   (roles from results/multicity19/qc/cities_qc.csv)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

OLD_ROLES = ("search", "discovery", "lawtest", "confirm")
CONFIG = Path(__file__).resolve().parent / "config"


def init_table(cities: pd.DataFrame, units_list: list) -> pd.DataFrame:
    """The 2024 city table restricted to the units cities, role 'pending' and no tier (same columns)."""
    out = cities[cities["city"].isin(set(units_list))].copy()
    out["role"] = "pending"
    for c in ("tier", "substitute_for"):
        if c in out:
            out[c] = None
    return out.reset_index(drop=True)


def assign(qc: pd.DataFrame, roles_2024: pd.DataFrame) -> pd.DataFrame:
    """city, role: old = a 2024 study city (whatever its 2019 QC), new = eligible in 2019 and not old, else excluded."""
    old = set(roles_2024.loc[roles_2024["role"].isin(OLD_ROLES), "city"])
    elig = qc["eligible"].astype(bool)
    rows = [{"city": c, "role": "old" if c in old else ("new" if bool(elig.get(c, False)) else "excluded")}
            for c in sorted(set(qc.index) | (old & set(roles_2024["city"])))]
    return pd.DataFrame(rows)


def main(argv):
    import ledger
    if config.LINE != "M19":
        sys.exit("roles_m19.py runs on the M19 line (ODSR_LINE=M19)")
    c24 = pd.read_csv(CONFIG / "cities.csv", dtype=str)
    if argv == ["init"]:
        units = [x.strip() for x in (CONFIG / "cities_units.txt").read_text().split() if x.strip()]
        t = init_table(c24, units)
        t.to_csv(config.CITIES_CSV, index=False)
        ledger.add(f"Plan 7 M19: {config.CITIES_CSV.name} initialised with {len(t)} units cities (role pending)")
        return
    if argv == ["assign"]:
        qc = pd.read_csv(config.RESULTS / "qc/cities_qc.csv", dtype={"city": str}).set_index("city")
        r = assign(qc, c24[["city", "role"]]).set_index("city")["role"]
        t = pd.read_csv(config.CITIES_CSV, dtype=str)
        t["role"] = t["city"].map(r).fillna("excluded")
        t.to_csv(config.CITIES_CSV, index=False)
        n = t["role"].value_counts().to_dict()
        ledger.add(f"Plan 7 M19 roles: old {n.get('old', 0)}, new {n.get('new', 0)}, excluded {n.get('excluded', 0)} "
                   f"({config.CITIES_CSV.name} sha {ledger.sha(config.CITIES_CSV)})")
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
