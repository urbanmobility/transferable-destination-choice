"""MiniMax call points of the agent loop (plan 4 Tasks 9, 10, 12, 14).

Every point is one question, one JSON answer (llm.call: schema-validated, logged); its input is assembled by code and
passes llm.check_input; its output is either verified by code before it takes effect or only displayed.
diversity_seeds: lens_for_class(cls) — the lens text of one structural class for seed_agent.propose (8 seeds each).
The prompts were issued in Chinese in the original runs; the text here is an English translation.
"""
ORIGIN_RULE = ("Origin variables (P_i, q_i, pd_i, rsub_i, rcbd_i, acc_i) may appear only in interaction terms with "
               "destination or distance variables (e.g. q_i * logd); as a standalone additive term they have no effect "
               "in a conditional logit and the code removes them. "
               "These seeds are for single-city search, with a complexity limit of 30 nodes (each variable, constant "
               "and operator counts 1; f, g and h counted together): "
               "each whole η must not exceed 20 nodes; in a single city g and h only multiply by one more constant, "
               "so write 1.0 unless needed.")
_LENSES = {
    "power": ("P0", "Log distance (power law or shifted power law log(d + c)) as the backbone: explore the composition "
                    "of attractiveness (saturation and weighting of size, composition, density) "
                    "and how the distance elasticity varies with origin or destination characteristics. "),
    "kernel_other": ("P1", "Non-logarithmic distance kernel: distance enters only in non-log form (exponential "
                           "nexp(d / c), linear d, fractional powers of d, two-piece kernels, etc.); "
                           "logd or log(d + c) must not be used; attractiveness may use size terms, density or "
                           "facility composition. "),
    "opportunity": ("P2", "Intervening opportunities instead of distance: destination choice is determined by "
                          "destination size and the intervening opportunities S_ij (the population within the circle "
                          "centred on the origin with radius equal to the distance to the destination, i.e. a 'rank "
                          "distance'), and may include radiation-model-style competition terms, destination density "
                          "and composition. No component may contain d or logd. "),
    "gravity_opportunity": ("P3", "Distance decay and intervening opportunities together: every expression must "
                                  "contain both distance (logd or log(d + c)) and S_ij; explore how the two divide "
                                  "the work (short versus long range, competition between S_ij and destination "
                                  "size). "),
    "no_distance": ("P4", "No distance and no intervening opportunities: destination choice is determined only by "
                          "destination size, composition, density, distance to the centre and to subcentres "
                          "(rcbd_j, rsub_j) and their interactions with origin characteristics; d, logd and S_ij "
                          "must not be used. "),
}


def lens_for_class(cls: str) -> dict:
    prefix, focus = _LENSES[cls]
    return {"name": cls, "focus": focus + ORIGIN_RULE, "prefix": prefix, "references": []}


# ---- review (Task 10): problems, a cleaned version, insight; the cleaned version is verified by code ----
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402

REVIEW_SCHEMA = {"type": "object", "required": ["problems", "cleaned_sympy", "changes", "expected_fit_cost", "insight"],
                 "properties": {"problems": {"type": "array", "items": {
                     "type": "object", "required": ["issue", "evidence"],
                     "properties": {"issue": {"type": "string"}, "evidence": {"type": "string"}}}},
                     "cleaned_sympy": {"type": "string"}, "changes": {"type": "string"},
                     "expected_fit_cost": {"type": "string"}, "insight": {"type": "string"}}}
SYSTEM_RESEARCH = ("You are a research assistant in urban travel demand modelling and symbolic regression, working on a "
                   "multi-city project to discover OD travel laws. The goal is a better and more credible "
                   "utility skeleton formula: it should have interpretable, insightful parts (not every term has to be "
                   "interpretable), must not obtain its fit through numerical tricks, and must fit clearly better than "
                   "the classical gravity and radiation models. Answer only from the material the user gives; the answer "
                   "is one JSON object. Variables use the x1..x26 notation (meanings in the material).")
REVIEW_TASK = ("Task: for this candidate (1) list the problems in the expression (numerical degeneracy, "
               "non-identifiability, fit obtained through numerical tricks, adding quantities of different dimensions, "
               "origin-only terms, uninterpretable parts, etc.), giving the evidence for each; (2) give a cleaned "
               "version: one sympy expression η (only x symbols and numeric constants, no template parameters; the code "
               "will later refit the constants from multiple starts on the 8 search cities, so reasonable initial values "
               "suffice; give an empty string if no cleaning is needed), and say what was removed or changed and how "
               "much fit is expected to be lost; "
               "(3) point out the parts of the expression that have explanatory power and insight. The code verifies "
               "the cleaned version: it is accepted only if the mean loss increase over the 8 cities is at most 3% of "
               "each city's gravity-to-M2 gap "
               "and no city becomes unfittable (e.g. a negative argument of log).")


def review(payload: dict) -> dict:
    user = "Material:\n" + json.dumps(payload, ensure_ascii=False, indent=1) + "\n\n" + REVIEW_TASK
    return llm.call([{"role": "system", "content": SYSTEM_RESEARCH}, {"role": "user", "content": user}],
                    REVIEW_SCHEMA, max_tokens=32000, name="review")


# ---- hypotheses (Task 12): law statements from the city-level summary; registered only through laws.register ----
HYP_FIELDS = ("statement", "skeleton", "parameter", "covariate", "direction", "test", "criterion")
HYPOTHESES_SCHEMA = {"type": "object", "required": ["hypotheses"], "properties": {"hypotheses": {"type": "array", "items": {
    "type": "object", "required": list(HYP_FIELDS),
    "properties": {**{k: {"type": "string"} for k in HYP_FIELDS}, "rationale": {"type": "string"}}}}}}
HYP_TASK = ("Task: read the cross-city summary above (relations of the parameters, derived quantities and term "
            "presence on the discovery cities with city characteristics) and propose at most 5 law hypotheses, "
            "each of the form 'parameter X varies with city characteristic Y' or 'term Z exists only in a certain type "
            "of city'. Fields: statement (stated in Chinese), skeleton (skeleton cid), "
            "parameter (a parameter name from identifiable, e.g. f:c1 or elasticity_5km; for term_exist_prop, a term "
            "from term_existence), "
            "covariate (a name from covariates), direction (+ or -), test (one of methods), criterion (the decision "
            "rule, written concretely following the description in methods), rationale (the basis, one or two "
            "sentences). These hypotheses will be registered and then each tested once on a separate set of test "
            "cities never seen before, with Holm correction; "
            "the same 'skeleton x parameter x city characteristic' can be registered only once in the whole run "
            "(registered lists the hypotheses already registered and their test-city results: pass is true / false, "
            "null means not yet tested), and non-identifiable parameters are rejected. Prefer hypotheses with large "
            "effects that make mechanistic sense; "
            "do not propose hypotheses just to fill the quota.")


def hypotheses(payload: dict) -> dict:
    user = "Material:\n" + json.dumps(payload, ensure_ascii=False, indent=1) + "\n\n" + HYP_TASK
    return llm.call([{"role": "system", "content": SYSTEM_RESEARCH}, {"role": "user", "content": user}],
                    HYPOTHESES_SCHEMA, max_tokens=32000, name="hypotheses")


# ---- stuck_report (Task 14): the diagnosis when the loop is stuck; displayed to the researchers, never acted on ----
STUCK_SCHEMA = {"type": "object", "required": ["tried", "evidence", "why_framework", "proposal"],
                "properties": {k: {"type": "string"} for k in ("tried", "evidence", "why_framework", "proposal")}}
STUCK_TASK = ("Task: the loop stopped on a predefined 'stuck' condition (the reason field). From the material, write a "
              "diagnosis for the human researchers: tried (what was tried), "
              "evidence (evidence, citing numbers from the material), why_framework (why this is considered a problem "
              "of the framework rather than something a few more search rounds would solve), proposal "
              "(the proposed change to the procedure, down to which step to change; only the human researchers can "
              "change the evaluation criteria, the sealed data and the budget, you can only make suggestions).")


def stuck_report(payload: dict) -> dict:
    user = "Material:\n" + json.dumps(payload, ensure_ascii=False, indent=1) + "\n\n" + STUCK_TASK
    return llm.call([{"role": "system", "content": SYSTEM_RESEARCH}, {"role": "user", "content": user}],
                    STUCK_SCHEMA, max_tokens=32000, name="stuck_report")
