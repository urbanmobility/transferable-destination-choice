"""Raw-shard integrity (global check): every parquet shard starts and ends with the PAR1 magic bytes and
every day of every month has at least one shard (monthly tables: every month). The check exists because 6 of 47,135
downloaded shards were 38-byte HTTP 503/500 error bodies saved as .parquet.

Output: results/multicity/qc/shards.json ; exit 1 when a shard is bad or a day is missing.
Usage: python -I shards.py
"""
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MONTHS, RESULTS, VERASET  # noqa: E402

TABLES = ("visits", "work-visits", "home-location")
MONTHLY = ("home-location",)
MAGIC = b"PAR1"


def _ok(path):
    try:
        if os.path.getsize(path) < 12:
            return False
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(-4, 2)
            return head == MAGIC and f.read(4) == MAGIC
    except OSError:
        return False


def check(src, tables, months):
    src = Path(src)
    out = {"n_files": 0, "bad": [], "per_day": {}, "days_missing": {}}
    for t in tables:
        per_day, missing = {}, []
        for m in months:
            files = sorted(glob.glob(str(src / t / f"{m}*.parquet")))
            out["n_files"] += len(files)
            out["bad"] += [f for f in files if not _ok(f)]
            if t in MONTHLY:
                keys = [m]
                n = {m: len(files)}
            else:
                first = pd.Timestamp(f"{m}-01")
                keys = [d.strftime("%Y-%m-%d") for d in pd.date_range(first, first + pd.offsets.MonthEnd(0))]
                n = {}
                for f in files:
                    d = os.path.basename(f)[:10]
                    n[d] = n.get(d, 0) + 1
            per_day.update({k: v for k, v in n.items() if v})
            missing += [k for k in keys if not n.get(k)]
        out["per_day"][t] = per_day
        out["days_missing"][t] = missing
    return out


def main():
    r = check(VERASET, TABLES, MONTHS)
    (RESULTS / "qc").mkdir(parents=True, exist_ok=True)
    (RESULTS / "qc/shards.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({"n_files": r["n_files"], "bad": len(r["bad"]),
                      "days_missing": {t: len(v) for t, v in r["days_missing"].items()}}))
    sys.exit(1 if r["bad"] or any(r["days_missing"].values()) else 0)


if __name__ == "__main__":
    main()
