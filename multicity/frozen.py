"""The freeze record and its guard (plan 6 Task 1; design decision of 2026-11-10).

freeze(): writes results/multicity/frozen/library.json (the frozen library: cid, role, equation, template, variant,
  complexity; the sha256 of the protocol (plan 6), config/cities.csv and scoring.py; the time) and
  results/multicity/FROZEN (= the sha256 of library.json); one ledger line. Refuses a second freeze.
require(): the library entries once FROZEN exists and matches library.json and every pinned input (protocol,
  cities.csv, scoring.py, the evaluation set, the pool, the law files) still has its frozen sha256; RuntimeError
  otherwise. Every path that reads a confirmation city or the test period calls it first.
clean_library(): the cids with role main / second / optional (the clean library of the coverage claims).
unseal_env(): {"ODSR_UNSEAL": "1"} after require() (refs.run_city of a confirmation city).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

_RES_M = config.line_constants("M", config.ROOT)["RESULTS"]     # plan 7: the freeze lives on the M line, every line sees it
FROZEN = _RES_M / "FROZEN"
FROZEN_DIR = _RES_M / "frozen"
ROLES = ("main", "second", "optional", "comparison", "reference")
CLEAN_ROLES = ("main", "second", "optional")


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def is_frozen() -> bool:
    return FROZEN.exists()


def freeze(library_cfg: Path, protocol: Path, pool: dict, log, pinned: dict | None = None) -> str:
    """pinned: {name: path} of further inputs whose sha256 require() re-checks (besides the protocol, cities.csv and
    scoring.py); a missing pinned file refuses the freeze."""
    if FROZEN.exists():
        raise FileExistsError(f"{FROZEN} exists: the library is frozen once")
    pins = {"protocol": Path(protocol), "scoring": Path(__file__).resolve().parent / "scoring.py"}
    if Path(config.CITIES_CSV).exists():
        pins["cities_csv"] = Path(config.CITIES_CSV)
    for k, v in (pinned or {}).items():
        if not Path(v).exists():
            raise ValueError(f"pinned input {k} missing: {v}")
        pins[k] = Path(v)
    entries = json.loads(Path(library_cfg).read_text())["entries"]
    for e in entries:
        if e["cid"] not in pool:
            raise ValueError(f"{e['cid']} is not in the pool")
        if e["role"] not in ROLES:
            raise ValueError(f"{e['cid']}: unknown role {e['role']!r}")
    if not any(e["role"] == "main" for e in entries):
        raise ValueError("the library needs a main skeleton")
    rec = {"entries": [{"cid": e["cid"], "role": e["role"],
                        **{k: pool[e["cid"]].get(k) for k in ("equation", "template", "variant", "complexity")}}
                       for e in entries],
           "protocol": {"path": str(protocol), "sha": _sha(protocol)},
           "cities_csv_sha": _sha(config.CITIES_CSV) if Path(config.CITIES_CSV).exists() else None,
           "scoring_sha": _sha(Path(__file__).resolve().parent / "scoring.py"),
           "pinned": {k: {"path": str(v), "sha": _sha(v)} for k, v in pins.items()},
           "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    p = FROZEN_DIR / "library.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    sha = _sha(p)
    FROZEN.write_text(sha + "\n")
    pins_txt = ", ".join(k + " " + v["sha"][:12] for k, v in rec["pinned"].items())
    log(f"FREEZE (plan 6): library {[(e['cid'], e['role']) for e in rec['entries']]}; library.json sha256 {sha}; "
        f"protocol {Path(protocol).name} sha256 {rec['protocol']['sha'][:12]}; cities.csv {str(rec['cities_csv_sha'])[:12]}; "
        f"scoring.py {rec['scoring_sha'][:12]}; pinned {pins_txt}. From now on the confirmation cities and the test period may be read; no "
        "rule, library or parameterisation changes.")
    return sha


def _local(path: str) -> Path:
    """A pinned path as recorded on the machine that froze the library, mapped onto this data root when the recorded
    absolute path does not exist here (the record keeps the original paths; its own hash must not change)."""
    p = Path(path)
    if p.exists():
        return p
    for anchor in ("/scripts/", "/results/", "/derived/"):
        i = path.find(anchor)
        if i >= 0:
            return Path(config.ROOT) / path[i + 1:]
    return p


def require() -> list:
    if not FROZEN.exists():
        raise RuntimeError("not frozen: the confirmation cities and the test period stay sealed (plan 6)")
    p = FROZEN_DIR / "library.json"
    if not p.exists() or _sha(p) != FROZEN.read_text().strip():
        raise RuntimeError(f"freeze record mismatch: {p} does not match {FROZEN}")
    rec = json.loads(p.read_text())
    for k, v in rec.get("pinned", {}).items():
        p = _local(v["path"])
        if not p.exists() or _sha(p) != v["sha"]:
            raise RuntimeError(f"frozen input {k} changed after the freeze: {p}")
    return rec["entries"]


def clean_library() -> list:
    return [e["cid"] for e in require() if e["role"] in CLEAN_ROLES]


def unseal_env() -> dict:
    require()
    return {"ODSR_UNSEAL": "1"}
