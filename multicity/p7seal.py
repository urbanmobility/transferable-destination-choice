"""Plan 7 sign-off guard (plan 7 design, sign-off rule).

Before the registered hypotheses are signed off (results/multicity/laws/registered_p7.json), nothing may compute an
outcome on the sealed data lines (M19 = 2019 non-work, M_nw = 2024 non-work) or a new outcome / covariate of a
law-test city. The paths live under the 2024 line's results whatever ODSR_LINE says, so every line sees the same
signature. `sign` is run only after the explicit sign-off.

Usage (after sign-off): python -I p7seal.py sign
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import sr_data  # noqa: E402

LAWS_M = sr_data.ROOT / "results/multicity/laws"
REG = LAWS_M / "registered_p7.json"
SIGNED = LAWS_M / "P7_SIGNED"
SEALED_LINES = ("M19", "M_nw")
CITIES_M = Path(__file__).resolve().parent / "config" / "cities.csv"     # the 2024 roles, whatever the line


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def is_signed() -> bool:
    return SIGNED.exists() and REG.exists() and SIGNED.read_text().strip() == _sha(REG)


def require_signed() -> list:
    """The registered hypotheses; RuntimeError unless the sign-off signature matches the registration file."""
    if not is_signed():
        raise RuntimeError(f"plan 7: {REG} is not signed off (or changed after signing)")
    return json.loads(REG.read_text())


def current_line() -> str:
    return sr_data.LINE


def guard_line(line: str) -> None:
    if line in SEALED_LINES and not is_signed():
        raise RuntimeError(f"plan 7: data line {line} is sealed until {REG.name} is signed off")


def _lawtest() -> set:
    import pandas as pd
    c = pd.read_csv(CITIES_M, dtype=str)
    return set(c.loc[c["role"] == "lawtest", "city"])


def guard_cities(cities) -> None:
    """Refuses before signing: any call on a sealed line, and any law-test city (2024 roles) on any line."""
    guard_line(current_line())
    law = _lawtest()
    hit = sorted(set(map(str, cities)) & law)
    if hit and not is_signed():
        raise RuntimeError(f"plan 7: law-test cities {hit[:5]} are sealed until {REG.name} is signed off")


def sign(log) -> str:
    """Write the sign-off signature (sha256 of the registration file). Only after the explicit sign-off."""
    if not REG.exists():
        raise FileNotFoundError(REG)
    if SIGNED.exists():
        raise FileExistsError(SIGNED)
    sha = _sha(REG)
    SIGNED.write_text(sha + "\n")
    log(f"plan 7 registration signed off: {REG.name} sha256 {sha}")
    return sha


if __name__ == "__main__":
    if sys.argv[1:] == ["sign"]:
        import ledger
        print(sign(ledger.add))
    else:
        sys.exit("usage: p7seal.py sign")
