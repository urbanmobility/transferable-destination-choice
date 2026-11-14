"""Append-only ledger of the multi-city pipeline: one timestamped line per entry in multicity/LEDGER.md."""
import hashlib
import time
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "LEDGER.md"


def add(text):
    with open(LEDGER, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M %Z')} {text}\n")


def sha(path):
    """First 12 hex digits of the sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]
