"""Plan 8 (holdout test of the registered hypotheses; design decision A1, 2026-11-10, before any holdout result):
the primary holdout test of the 10 signed plan-7 hypotheses - the 26 law-test cities, 2024, all-purpose trips.

Pass rule (registration, primary part): the city-bootstrap (B = 5000) 95% CI of the slope (partial slope with the
registered control) excludes 0 in the registered direction (explore_p7.run_test), and Holm over the registered family
(m = 10, apart from the 15 earlier laws); pass = supported. There is no "confirmed" level (A1). The result file is
written once; the registration must still carry its sign-off signature.

Usage (run_stage.sh, M line): p8_test.py test | report
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import explore_p7 as ex  # noqa: E402

B = 5000
RULING = ("A1 2026-11-10: primary layer only (26 law-test cities, 2024, all-purpose trips); pass = supported; "
          "no confirmed level, the registration's independent replication does not apply")


def _res() -> Path:
    return config.results_dir("M")


def results_path() -> Path:
    return _res() / "laws" / "results_p7.json"


def primary_frames() -> tuple:
    """(outcomes, covariates, cities) of the 26 law-test cities, merged as in the exploration."""
    cities = [str(c) for c in config.cities_with_role("lawtest")]
    p7 = _res() / "p7"
    out, cov = ex.frames(pd.read_parquet(p7 / "city_outcomes_lawtest.parquet"), cities, p7 / "covariates_lawtest.csv")
    return out, cov, cities


def _needs(h) -> list:
    return [h["covariate"]] + ([h["control"]] if h.get("control") else [])


def check_complete(hyps, out, cov, cities) -> None:
    """Every registered outcome column exists, every city has an outcome row and every registered covariate /
    control value (outcome values may be NaN)."""
    cities = [str(c) for c in cities]
    miss = sorted(set(cities) - set(out["city"].astype(str)))
    if miss:
        raise ValueError(f"no outcome row for {len(miss)} cities, e.g. {miss[:5]}")
    c = cov.assign(city=cov["city"].astype(str)).set_index("city")
    for h in hyps:
        if h["outcome"] not in out.columns:
            raise ValueError(f"{h['id']}: outcome {h['outcome']} absent from the outcome table")
        for k in _needs(h):
            if k not in c.columns:
                raise ValueError(f"{h['id']}: covariate {k} absent")
            v = c[k].reindex(cities)
            bad = [x for x, val in v.items() if pd.isna(val)]
            if bad:
                raise ValueError(f"{h['id']}: {k} missing for {len(bad)} cities, e.g. {bad[:5]}")


def run_tier(hyps, out, cov, B: int = B) -> list:
    """run_test per hypothesis (registration order), then Holm over these p values (m = number of hypotheses);
    pass = holm and pass_raw = supported."""
    import laws
    rows = []
    for h in hyps:
        r = dict(ex.run_test(h, out, cov, B=B))
        r.update(id=h["id"], outcome=h["outcome"], covariate=h["covariate"], control=h.get("control"),
                 direction=h["direction"])
        rows.append(r)
    ps = [float(r["p"]) if r.get("p") is not None and np.isfinite(float(r["p"])) else 1.0 for r in rows]
    for r, k in zip(rows, laws.holm(ps)):
        r.update(m=len(rows), holm=bool(k), **{"pass": bool(k and r["pass_raw"])})
        r["supported"] = r["pass"]
    return rows


def stage_test() -> dict:
    import ledger
    import p7seal
    if config.line_of(p7seal.current_line()) != "M":                 # unknown lines (e.g. round 1's L) are M
        raise ValueError(f"the primary test runs on the M line (laws.covariates reads its form_metrics), not "
                         f"{p7seal.current_line()}")
    hyps = p7seal.require_signed()
    path = results_path()
    if path.exists():
        raise FileExistsError(f"{path} exists: the primary test is written once (no rerun without an explicit decision)")
    out, cov, cities = primary_frames()
    check_complete(hyps, out, cov, cities)
    rows = run_tier(hyps, out, cov)
    res = {"cities": list(cities), "registration_sha": p7seal._sha(p7seal.REG), "design_decisions": [RULING], "B": B,
           "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    ns = [int(r["n"]) for r in rows]
    ledger.add(f"Plan 8 primary test (A1): {len(cities)} law-test cities, B {B}, Holm m {len(rows)}: supported "
               f"{sum(r['supported'] for r in rows)}/{len(rows)} "
               f"({', '.join(r['id'] + (' yes' if r['supported'] else ' no') for r in rows)}); n {min(ns)}-{max(ns)}; "
               f"registration sha {res['registration_sha'][:12]}; {path.name} sha {ledger.sha(path)}")
    return res


def _f(v, nd=3) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{v:.{nd}g}" if np.isfinite(v) else "—"


def _yn(v) -> str:
    return "—" if v is None else ("yes" if v else "no")


def render(results: dict, hyps: list) -> str:
    rows = {r["id"]: r for r in results["rows"]}
    k = sum(1 for r in results["rows"] if r.get("supported"))
    L = ["# Holdout test of the plan 7 registered hypotheses (plan 8)", "",
         f"Test cities: {len(results['cities'])} law-test cities (2024, all trips); registration file sha {results['registration_sha'][:12]}; "
         f"city bootstrap B = {results.get('B', B)}; Holm correction within the {len(results['rows'])} registered hypotheses.", "",
         "Decision A1 (2026-11-10, before any holdout result was computed): primary test only, pass = supported; "
         "the registration's independent replication (cities newly eligible in 2019) and the \"confirmed\" level do not apply.", "",
         f"**Supported {k}/{len(results['rows'])}**", "",
         "| ID | Statement | Outcome ~ covariate | Control | Direction | Discovery evidence: n, slope [95% interval] | Test cities: n, slope [95% interval] | p | Holm | Supported |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for h in hyps:
        r, e = rows.get(h["id"], {}), h.get("evidence") or {}
        ci = e.get("ci") or [None, None]
        L.append(f"| {h['id']} | {h.get('statement', '')} | {h['outcome']} ~ {h['covariate']} | {h.get('control') or '—'} | "
                 f"{h['direction']} | {e.get('n', '—')}, {_f(e.get('stat'))} [{_f(ci[0])}, {_f(ci[1])}] | "
                 f"{r.get('n', '—')}, {_f(r.get('stat'))} [{_f(r.get('ci_lo'))}, {_f(r.get('ci_hi'))}] | "
                 f"{_f(r.get('p'), 2)} | {_yn(r.get('holm'))} | {_yn(r.get('supported'))} |")
    L += ["", "Slope magnitudes are reported only and do not enter the verdict; cities with a missing outcome are left out of that hypothesis's test (n as above)."]
    return "\n".join(L) + "\n"


def stage_report() -> Path:
    import p7seal
    hyps = p7seal.require_signed()
    md = render(json.loads(results_path().read_text()), hyps)
    p = _res() / "p8" / "report_p8.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md)
    return p


STAGES = {"test": stage_test, "report": stage_report}

if __name__ == "__main__":
    if sys.argv[1:2] not in ([k] for k in STAGES) or len(sys.argv) != 2:
        sys.exit(__doc__)
    STAGES[sys.argv[1]]()
