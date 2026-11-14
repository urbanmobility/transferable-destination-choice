"""Scoring rules of the agent loop (plan 4 Task 6; agent-in-the-loop design §5.7 — frozen).

A refs_city dict is one city's reference models ({model: {"nll_per_visit": ...}}): refs/<city>.json["models"] for
validation-period scores (outer loop), refs/<city>.json["inner"] for fold-4 scores (inner loop; refs.inner_refs).
delta_city: tolerance of every "close" judgement = MARGIN x (NLL_grav_power - NLL_M2_glm) (design decision of 11-05;
  independent of the share zero point).
share: (NLL_zero - NLL) / (NLL_zero - NLL_M2), zero = S0 (corrected gravity, reported) or grav_power (classic).
passes_floor: NLL below the best of classic power / exponential gravity and extended radiation (missing ignored).
near_best: (cid, city) in the city's 1-SE set or at most max(se_vs_best, delta_city) above the city's best fittable.
term_exists: removing the term raises the NLL by >= delta_city.
rank: representatives that pass the floor in >= 1 city, ordered lexicographically: near-best (at GATE_MARGIN) in >=
  min_cities search cities; code cleanliness (n_flags = checks.flags n + 1 if degenerate; 0 first); complexity; more
  near-best cities; cid. The first `top` with n_near >= min_cities are gate1 (the loop's top 8).

Multipliers (design decision of 2026-11-10, "V3"; plans 2-5 used one multiplier 0.03 everywhere and a chain merge):
  MARGIN        0.03  conclusions: discovery / confirmation near-best (fit_library), coverage, term_exists, review
  GATE_MARGIN   0.05  the search-city gate (rank): a screen, so missing a good form costs more than a spare one
  MERGE_MARGIN  0.03  the equivalence merge (merge.loss_close): a wrong merge loses a form for good
  MERGE_LINKAGE "leader" (no chain: each candidate joins the first representative it is directly equivalent to);
                "chain" = union-find, the registered rule
registered_rule(m=0.03): the plans 2-5 rule (every multiplier m, chain) inside the block, for reproduction.
"""
import contextlib

MARGIN = 0.03
GATE_MARGIN = 0.05
MERGE_MARGIN = 0.03
MERGE_LINKAGE = "leader"
FLOOR_MODELS = ("grav_power", "grav_exp", "ext_rad")
MIN_CITIES, TOP = 4, 8


def _nll(refs_city, model):
    v = (refs_city or {}).get(model) or {}
    x = v.get("nll_per_visit")
    return float(x) if x is not None else None


def delta_city(refs_city: dict, margin: float | None = None) -> float:
    g, m2 = _nll(refs_city, "grav_power"), _nll(refs_city, "M2_glm")
    if g is None or m2 is None:
        return 0.0
    return (MARGIN if margin is None else margin) * max(g - m2, 0.0)


@contextlib.contextmanager
def registered_rule(m: float = 0.03):
    """The rule of plans 2-5 inside the block: one multiplier m for conclusions, gate and merge; chain merge."""
    global MARGIN, GATE_MARGIN, MERGE_MARGIN, MERGE_LINKAGE
    old = (MARGIN, GATE_MARGIN, MERGE_MARGIN, MERGE_LINKAGE)
    MARGIN = GATE_MARGIN = MERGE_MARGIN = float(m)
    MERGE_LINKAGE = "chain"
    try:
        yield
    finally:
        MARGIN, GATE_MARGIN, MERGE_MARGIN, MERGE_LINKAGE = old


def share(nll, refs_city: dict, zero: str = "S0"):
    z, m2 = _nll(refs_city, zero), _nll(refs_city, "M2_glm")
    if nll is None or z is None or m2 is None or not z > m2:
        return None
    return (z - float(nll)) / (z - m2)


def passes_floor(nll, refs_city: dict) -> bool:
    fl = [v for v in (_nll(refs_city, m) for m in FLOOR_MODELS) if v is not None]
    return nll is not None and bool(fl) and float(nll) < min(fl)


def term_exists(delta_nll, refs_city: dict) -> bool:
    return delta_nll is not None and float(delta_nll) >= delta_city(refs_city)


def _ok(fits):
    return fits[fits["finite"].astype(bool) & fits["val_nll"].notna()]


def near_best(fits, refs: dict, margin: float | None = None) -> set:
    """margin None: MARGIN (conclusions)."""
    w = fits["within_1se"].astype(bool)
    out = set(zip(fits.loc[w, "cid"], fits.loc[w, "city"]))
    ok = _ok(fits)
    best = ok.groupby("city")["val_nll"].min()
    for r in ok.itertuples():
        se = float(r.se_vs_best) if getattr(r, "se_vs_best", None) is not None and r.se_vs_best == r.se_vs_best else 0.0
        if r.val_nll - best[r.city] <= max(se, delta_city(refs.get(r.city), margin)):
            out.add((r.cid, r.city))
    return out


def n_flags(flags_cid: dict | None) -> int:
    f = flags_cid or {}
    return int(f.get("n", 0)) + int(bool(f.get("degenerate")))


def rank(reps: list, fits, refs: dict, flags: dict, min_cities: int = MIN_CITIES, top: int = TOP,
         margin: float | None = None) -> list:
    """margin None: GATE_MARGIN."""
    near = near_best(fits, refs, GATE_MARGIN if margin is None else margin)
    ok = _ok(fits)
    floor = {(r.cid, r.city) for r in ok.itertuples() if passes_floor(r.val_nll, refs.get(r.city))}
    out = []
    for r in reps:
        members = [r["cid"]] + list(r.get("merged_from", []))
        fl = sorted({city for cid, city in floor if cid in members})
        if not fl:
            continue
        n1 = fits[fits["cid"].isin(members) & fits["within_1se"].astype(bool)]["city"].nunique()
        nc = sorted({city for cid, city in near if cid in members})
        out.append({**r, "n_1se": int(n1), "n_near": len(nc), "near_cities": nc, "floor_cities": fl,
                    "n_flags": n_flags(flags.get(r["cid"]))})
    out.sort(key=lambda r: (r["n_near"] < min_cities, r["n_flags"], r["complexity"], -r["n_near"], r["cid"]))
    k = 0
    for r in out:
        r["gate1"] = bool(r["n_near"] >= min_cities and k < top)
        k += int(r["gate1"])
    return out
