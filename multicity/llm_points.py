"""LLM call points 1-3 (plan 2 Task 5): qc_summary (city-level QC anomalies), simplify (an algebraic
simplification proposal, verified by clean.verify_simplification before it replaces anything), readability (a 1-5
score, the meaning of each term, the variables used — checked against the variable table). Inputs are city-level
summaries and formulas only (llm.check_input); confirmation cities never appear.
The prompts and variable descriptions were issued in Chinese in the original runs; the text here is an English
translation.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr"))
import llm  # noqa: E402
from sr_data import VARS  # noqa: E402

VAR_MEANING = {
    "d": "distance from origin to destination (km, floor 0.05)",
    "logd": "natural logarithm of distance",
    "P_i": "resident population of the origin tract",
    "P_j": "resident population of the destination tract",
    "E_j": "number of jobs in the destination tract (LODES)",
    "N_j": "total number of POIs in the destination tract (sum of the 8 classes)",
    "a_j": "land area of the destination tract (km², floor 0.01)",
    "s_food": "share of food POIs at the destination (Dirichlet-smoothed)",
    "s_retail": "share of retail POIs at the destination (Dirichlet-smoothed)",
    "s_health": "share of health POIs; always 0 with the Veraset data, removed",
    "s_education": "share of education POIs at the destination (Dirichlet-smoothed)",
    "s_leisure": "share of leisure and entertainment POIs at the destination (Dirichlet-smoothed)",
    "s_service": "share of service POIs at the destination (Dirichlet-smoothed)",
    "s_office": "share of office / business POIs at the destination (Dirichlet-smoothed)",
    "s_transit": "share of transport-hub POIs at the destination (Dirichlet-smoothed)",
    "H_j": "entropy of the destination's POI class composition (classes present)",
    "q_i": "income quantile of the origin tract (0–1)",
    "pd_i": "population density of the origin tract",
    "pd_j": "population density of the destination tract",
    "jd_j": "job density of the destination tract",
    "rsub_i": "distance from the origin to the nearest employment subcentre",
    "rsub_j": "distance from the destination to the nearest employment subcentre",
    "rcbd_i": "distance from the origin to the CBD",
    "rcbd_j": "distance from the destination to the CBD",
    "acc_i": "job accessibility of the origin (distance-decay-weighted number of jobs)",
    "S_ij": "intervening opportunities: population within the circle centred on the origin with radius equal to the "
            "distance to the destination (the S of the radiation model)",
}
XVAR = {f"x{i + 1}": v for i, v in enumerate(VARS)}
VALID_TOKENS = set(XVAR) | set(VARS)

SCHEMAS = {
    "qc_summary": {"type": "object", "additionalProperties": False, "required": ["flags"],
                   "properties": {"flags": {"type": "array", "items": {
                       "type": "object", "additionalProperties": False, "required": ["city", "flag", "note"],
                       "properties": {"city": {"type": "string"}, "flag": {"type": "string"}, "note": {"type": "string"}}}}}},
    "simplify": {"type": "object", "additionalProperties": False, "required": ["sympy", "reason"],
                 "properties": {"sympy": {"type": "string"}, "reason": {"type": "string"}}},
    "readability": {"type": "object", "additionalProperties": False, "required": ["score", "terms", "uses_variables"],
                    "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5},
                                   "terms": {"type": "array", "items": {
                                       "type": "object", "additionalProperties": False, "required": ["term", "meaning"],
                                       "properties": {"term": {"type": "string"}, "meaning": {"type": "string"}}}},
                                   "uses_variables": {"type": "array", "items": {"type": "string"}}}},
}

SYSTEM = ("You are a reviewer in urban travel demand modelling (conditional logit of destination choice). The answer "
          "must be a single JSON object satisfying the given JSON Schema, "
          "with no other text. Variables use the x1..x26 notation, with meanings in the table the user gives; "
          "constants are numeric.")


def _vars_text(payload):
    v = payload.get("variables") or {}
    return "\n".join(f"- {k}: {m}" for k, m in v.items()) or "(no variable table)"


def qc_summary(payload):
    """payload: {"cities": [city-level QC rows]} -> {"flags": [{"city", "flag", "note"}]}."""
    user = ("Below is the quality-check summary of several cities (city-level indicators: eligible = passes, "
            "n_origins = number of origins, train_trips = number of training trips, r2 = panel representativeness, "
            "income_ratio = income-group coverage ratio, coverage = share of out-of-city destinations, "
            "support_mismatch = variables with zero values only in the validation/test blocks, nonfinite = non-finite "
            "variables, flags = flags set by the code). Point out anomalies that deserve the researchers' attention, "
            "each with city, a short flag name and a one-sentence note in Chinese; if there are none, flags is an "
            "empty array. "
            "Do not repeat the flags already set by the code; only add cases that need human judgement.\n\n"
            + json.dumps(payload.get("cities", []), ensure_ascii=False))
    return llm.call([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    SCHEMAS["qc_summary"], effort="low", max_tokens=4096, name="qc_summary")


def simplify(payload, verify=None):
    """payload: {"cid", "equation" (sympy string of eta in x1..x26), "variables", optional "fits"}; verify(cid,
    sympy) -> clean.verify_simplification result decides acceptance; without a verifier accepted is None."""
    user = ("Given the sympy expression of a conditional-logit utility skeleton η(x) (variable meanings in the table "
            "below, constants are numeric), propose an algebraically equivalent or very close, more concise "
            "form: merge constants, drop additive constants with no identifiable contribution to η (a uniform shift "
            "of utility does not change choice probabilities), collect like terms, simplify nesting. "
            "Use only the variable symbols in the table and numeric constants; do not change the functional "
            "structure (positions of logs, powers, exponentials) and do not introduce new variables. "
            "The sympy field is a valid sympy expression string (** for powers, log is the natural log); reason "
            "says in one sentence what was done.\n\n"
            f"η = {payload.get('equation')}\n\nVariables:\n{_vars_text(payload)}")
    if payload.get("fits"):
        user += "\n\nFit summary of each search city (city level): " + json.dumps(payload["fits"], ensure_ascii=False)
    r = llm.call([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                 SCHEMAS["simplify"], effort="low", name="simplify")
    out = {"cid": payload.get("cid"), "sympy": r["sympy"], "reason": r["reason"], "accepted": None}
    if verify is not None:
        v = verify(payload.get("cid"), r["sympy"])
        out["accepted"] = bool(v.get("accepted"))
        out["verify"] = {k: val for k, val in v.items() if k != "candidate"}
    return out


def readability(payload):
    """payload: {"cid", "equation", "variables"} -> score 1-5, terms with meanings, uses_variables (checked)."""
    user = ("Given a conditional-logit utility skeleton η(x) and the variable meanings: (1) explain in Chinese the "
            "travel-behaviour meaning of each top-level term of η (e.g. 'log attraction of job size', "
            "'power-law distance decay'); (2) give a readability score of 1–5 (5 = every term has a clear meaning "
            "that can be explained to urban planners, 1 = most terms cannot be explained); "
            "(3) list in uses_variables the variable symbols (x names) used by η.\n\n"
            f"η = {payload.get('equation')}\n\nVariables:\n{_vars_text(payload)}")
    r = llm.call([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                 SCHEMAS["readability"], effort="low", name="readability")
    bad = [v for v in r["uses_variables"] if v not in VALID_TOKENS]
    r["invalid"] = bool(bad)
    r["invalid_reason"] = f"unknown variables {bad}" if bad else ""
    return r
