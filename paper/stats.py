"""Small statistics shown in the paper figures."""
import numpy as np


def replication_ratio(disc: float, test: float, lo: float, hi: float, min_abs: float = 1e-9) -> tuple:
    """Held-out estimate and its interval divided by the discovery estimate: 1 = the same size, 0 = nothing left.
    A negative denominator swaps the interval ends; a denominator near 0 gives nan."""
    if not (np.isfinite(disc) and abs(disc) > min_abs):
        return (float("nan"),) * 3
    a, b = lo / disc, hi / disc
    return float(test / disc), float(min(a, b)), float(max(a, b))


def ols_band(x, y, grid, level: float = 0.95) -> tuple:
    """OLS line y ~ x evaluated on grid with its confidence band -> (fit, lo, hi, slope). Missing values dropped."""
    from scipy import stats as sps
    x, y, grid = np.asarray(x, float), np.asarray(y, float), np.asarray(grid, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    b, a = np.polyfit(x, y, 1)
    fit = a + b * grid
    s2 = float(((y - (a + b * x)) ** 2).sum() / max(n - 2, 1))
    se = np.sqrt(s2 * (1 / n + (grid - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()))
    t = sps.t.ppf(0.5 + level / 2, max(n - 2, 1))
    return fit, fit - t * se, fit + t * se, float(b)


def running_median(x, y, k: int = 9) -> tuple:
    """Median of y in a window of k neighbours along sorted x -> (sorted x, running median)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    h = k // 2
    return x, np.array([np.median(y[max(i - h, 0):i + h + 1]) for i in range(len(x))])


def size_terciles(logpop) -> np.ndarray:
    """small / medium / large by rank of city size (equal counts)."""
    r = np.argsort(np.argsort(np.asarray(logpop, float), kind="stable"), kind="stable")
    return np.array(["small", "medium", "large"])[np.minimum(3 * r // len(r), 2)]
