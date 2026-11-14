"""Plan 9: summaries over cities, paired share differences with a city bootstrap, the opponent rule and the verdicts
(design §6.4, §6.5). Pure functions.

A city where a model gives no finite NLL is a failure: its share is -inf in the quantiles, and a loss in a paired
comparison (a tie when both sides fail). Nothing is pooled across cities.
"""
import numpy as np
import pandas as pd

VERDICTS = ("better", "worse", "no distinguishable difference")
RULES = {     # opponent -> (level, originals only?, exclude models that need the city's own trips?)
    "fit": ("local", False, False),
    "zero_shot": ("uniform", False, True),
    "fit_original": ("local", True, False),
    "zero_shot_original": ("uniform", True, True),
}


def _quantile(v, q: float) -> float:
    """Linear-interpolation quantile that survives infinities (the nearer order statistic when one end is infinite)."""
    v = np.sort(np.asarray(v, float))
    if not len(v):
        return np.nan
    pos = q * (len(v) - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    if lo == hi:
        return float(v[lo])
    if not (np.isfinite(v[lo]) and np.isfinite(v[hi])):
        return float(v[lo] if pos - lo < 0.5 else v[hi])
    return float(v[lo] + (pos - lo) * (v[hi] - v[lo]))


def summary(t: pd.DataFrame) -> pd.DataFrame:
    """One row per model x level: n, n_failed and median / quartiles of share (failures = -inf), nll and cpc (the
    cities that did not fail)."""
    rows = []
    for (m, lv), g in t.groupby(["model", "level"], sort=False):
        ok = ~g["failed"].astype(bool)
        r = {"model": m, "level": lv, "n": int(len(g)), "n_failed": int((~ok).sum())}
        for col, v in (("share", np.where(ok, g["share"], -np.inf)), ("nll", g.loc[ok, "nll"]), ("cpc", g.loc[ok, "cpc"])):
            for tag, q in (("med", 0.5), ("q1", 0.25), ("q3", 0.75)):
                r[f"{col}_{tag}"] = _quantile(v, q)
        rows.append(r)
    return pd.DataFrame(rows)


def _side(t, key) -> pd.DataFrame:
    g = t[(t["model"] == key[0]) & (t["level"] == key[1])]
    if not len(g):
        raise ValueError(f"no rows for {key}")
    if not g["city"].is_unique:
        raise ValueError(f"several rows per city for {key}")
    return g.set_index("city")


def paired(t: pd.DataFrame, a: tuple, b: tuple) -> pd.Series:
    """Share of a = (model, level) minus share of b, by city: -inf where a fails, +inf where b fails, 0 where both."""
    A, B = _side(t, a), _side(t, b)
    if set(A.index) != set(B.index):
        raise ValueError(f"{a} and {b} cover different cities")
    B = B.loc[A.index]
    fa, fb = A["failed"].to_numpy(bool), B["failed"].to_numpy(bool)
    for side, g, f in ((a, A, fa), (b, B, fb)):
        if np.isnan(g["share"].to_numpy(float)[~f]).any():
            raise ValueError(f"{side}: a city that did not fail has no share")
    with np.errstate(invalid="ignore"):
        d = A["share"].to_numpy(float) - B["share"].to_numpy(float)
    d = np.where(fa & fb, 0.0, np.where(fa, -np.inf, np.where(fb, np.inf, d)))
    return pd.Series(d, index=A.index, name="diff")


def _medians(S: np.ndarray) -> np.ndarray:
    """Row medians that survive infinities: an infinite middle value wins; opposite infinities give 0."""
    S = np.sort(S, axis=1)
    n = S.shape[1]
    lo, hi = S[:, (n - 1) // 2], S[:, n // 2]
    with np.errstate(invalid="ignore"):
        mid = (lo + hi) / 2
    both = np.isinf(lo) & np.isinf(hi) & (lo != hi)
    return np.where(np.isfinite(lo) & np.isfinite(hi), mid, np.where(both, 0.0, np.where(np.isinf(lo), lo, hi)))


def contrast(diff: pd.Series, B: int, seed: int) -> dict:
    """Median paired difference, its city-bootstrap 95% interval, the two-sided bootstrap p ((k + 1) / (B + 1) form)
    and the win counts."""
    d = np.asarray(diff, float)
    n = len(d)
    bs = _medians(d[np.random.default_rng(seed).integers(0, n, (B, n))])
    p = 2 * (1 + min((bs <= 0).sum(), (bs >= 0).sum())) / (B + 1)          # never exactly 0
    return {"n": int(n), "median": float(_medians(d[None, :])[0]), "ci_lo": _quantile(bs, 0.025),
            "ci_hi": _quantile(bs, 0.975), "p": float(min(1.0, p)),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()), "B": int(B)}


def select_opponents(s: pd.DataFrame, meta: dict) -> dict:
    """The four opponents of the primary contrasts from the source pre-run summary: the earlier model with the
    highest median share at the rule's level. Ties: fewer free parameters, then the name."""
    out = {}
    for key, (level, originals, no_trips) in RULES.items():
        g = s[(s["level"] == level) & s["model"].isin(list(meta))]
        cand = [(-float(r.share_med), meta[r.model]["n_free"], r.model) for r in g.itertuples()
                if (meta[r.model]["original"] or not originals) and not (no_trips and meta[r.model]["needs_trips"])
                and np.isfinite(r.share_med)]
        if not cand:
            raise ValueError(f"no eligible earlier model for opponent {key!r}")
        v, _, m = min(cand)
        out[key] = {"model": m, "share_med": -v}
    return out


def judge(results: dict, primary: list, alpha: float = 0.05) -> dict:
    """Holm over the primary contrasts' p values; "better" / "worse" when the interval excludes 0 on that side and the
    contrast passes Holm, otherwise "no distinguishable difference". Other (descriptive) contrasts get no verdict."""
    import laws
    out = {k: dict(r, ci_excludes_zero=bool(r["ci_lo"] > 0 or r["ci_hi"] < 0), holm=None, verdict=None)
           for k, r in results.items()}
    for k, h in zip(primary, laws.holm([float(results[k]["p"]) for k in primary], alpha)):
        r = out[k]
        r["holm"] = bool(h)
        r["verdict"] = "better" if h and r["ci_lo"] > 0 else "worse" if h and r["ci_hi"] < 0 else "no distinguishable difference"
    return out
