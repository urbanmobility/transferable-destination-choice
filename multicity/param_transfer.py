"""Plan 9: the constants of a model for a city WITHOUT its own trip data (design §6.3), from the other cities' fits.

  uniform   element-wise median of the donors' constants;
  size      each constant on a Theil-Sen line against the size covariate, read at the city;
  form      each constant by ridge regression on size + form covariates (standardised on the donors; the constant
            clipped to the donors' quantiles first, because a few city fits land far out).
"local" (the city's own refit) is the fourth level and is not computed here.
"""
import numpy as np
import pandas as pd

LEVELS = ("local", "uniform", "size", "form")
ALPHAS = np.logspace(-1, 3, 13)


def predict(K: dict, cov: pd.DataFrame, city: str, size: str, form: list, min_donors: int = 10, alphas=ALPHAS,
            clip=(0.05, 0.95)) -> dict:
    """K = {city: constants} of one model, cov indexed by city -> {level: constants for `city`}. Donors are the
    cities of K other than `city` that have covariates; {} with fewer than min_donors donors or no covariates for
    `city`. A constant that does not vary across the donors (a fixed numeral) passes through unchanged."""
    from scipy.stats import theilslopes
    from sklearn.linear_model import RidgeCV
    donors = [c for c in K if c != city and c in cov.index]
    if len(donors) < min_donors or city not in cov.index:
        return {}
    Y = np.array([K[c] for c in donors], float)
    fixed = np.ptp(Y, axis=0) == 0
    out = {"uniform": np.median(Y, axis=0)}
    x, x0 = cov.loc[donors, size].to_numpy(float), float(cov.loc[city, size])
    cols = [size] + list(form)
    X = cov.loc[donors, cols].astype(float)
    mu, sd = X.mean(), X.std(ddof=1).replace(0, 1.0).fillna(1.0)
    Z = ((X - mu) / sd).fillna(0.0).to_numpy()
    z0 = ((cov.loc[[city], cols].astype(float) - mu) / sd).fillna(0.0).to_numpy()
    s, f = Y[0].copy(), Y[0].copy()
    for k in np.flatnonzero(~fixed):
        sl, ic, _, _ = theilslopes(Y[:, k], x)
        s[k] = ic + sl * x0
        lo = np.quantile(Y[:, k], clip[0], method="higher")      # order statistics, not interpolated: with ~35
        hi = np.quantile(Y[:, k], clip[1], method="lower")       # donors an interpolated bound follows one outlier
        f[k] = float(RidgeCV(alphas=alphas).fit(Z, np.clip(Y[:, k], lo, hi)).predict(z0)[0])
    out["size"], out["form"] = s, f
    return out
