"""The Scaled kernel: a compact, transferable utility for intra-urban destination choice.

    eta_ij = a_j ** theta4 - theta0 * d_ij ** theta1 / (N_j + theta2) ** theta3
    P(j | i) = exp(eta_ij) / sum_k exp(eta_ik)

d_ij: straight-line distance between the origin and destination tracts in km (floored at 0.05 km);
N_j: number of points of interest (places) in destination tract j; a_j: land area of tract j in km^2.

DEFAULT holds the uniform parameter vector of the paper (median of the 35 discovery-city fits, Additional file 1,
Table S5). With it the formula retained a median 94% of the likelihood gap between population-based gravity and a
22-parameter flexible model in 32 cities whose visits played no part in estimating it. Only numpy is needed.
"""
from __future__ import annotations

import numpy as np

DEFAULT = {
    "theta0": 8.844391445272066,
    "theta1": 0.21189336782799995,
    "theta2": 14.982145306944018,
    "theta3": 0.12747186922393527,
    "theta4": 0.12887300017697215,
}
D_MIN_KM = 0.05


def utility(d_km, n_poi, area_km2, params: dict | None = None) -> np.ndarray:
    """eta for every (origin, destination) pair; arrays broadcast (e.g. d as an origins x destinations matrix and
    n_poi, area_km2 as destination vectors)."""
    p = {**DEFAULT, **(params or {})}
    d = np.maximum(np.asarray(d_km, float), D_MIN_KM)
    n = np.asarray(n_poi, float) + p["theta2"]
    if np.any(n <= 0):
        raise ValueError("N_j + theta2 must be positive for every destination")
    return np.asarray(area_km2, float) ** p["theta4"] - p["theta0"] * d ** p["theta1"] / n ** p["theta3"]


def choice_probabilities(d_km, n_poi, area_km2, params: dict | None = None, exclude_self: bool = True) -> np.ndarray:
    """P(j | i) for an origins x destinations distance matrix. exclude_self: origin i is not a destination of itself
    (the diagonal of a square matrix gets probability 0), as in the paper."""
    eta = np.array(utility(d_km, n_poi, area_km2, params), float)
    if eta.ndim != 2:
        raise ValueError("d_km must be an origins x destinations matrix")
    if exclude_self and eta.shape[0] == eta.shape[1]:
        np.fill_diagonal(eta, -np.inf)
    eta -= eta.max(axis=1, keepdims=True)
    w = np.exp(eta)
    return w / w.sum(axis=1, keepdims=True)


def flows(origin_totals, d_km, n_poi, area_km2, params: dict | None = None, exclude_self: bool = True) -> np.ndarray:
    """Production-constrained flows T_i * P(j | i)."""
    P = choice_probabilities(d_km, n_poi, area_km2, params, exclude_self)
    return np.asarray(origin_totals, float)[:, None] * P


def elasticities(d_km, n_poi, params: dict | None = None) -> tuple:
    """(e_d, e_N): d eta / d ln d and d eta / d ln N_j (Additional file 1, Note S1). e_N rises with distance exactly
    when theta3 > 0, since d e_N / d ln d = theta1 * e_N."""
    p = {**DEFAULT, **(params or {})}
    d = np.maximum(np.asarray(d_km, float), D_MIN_KM)
    n = np.asarray(n_poi, float)
    k = p["theta0"] * d ** p["theta1"]
    e_d = -p["theta1"] * k / (n + p["theta2"]) ** p["theta3"]
    e_n = p["theta3"] * k * n / (n + p["theta2"]) ** (p["theta3"] + 1)
    return e_d, e_n


def distance_scale(n_poi, params: dict | None = None) -> np.ndarray:
    """Scale l_j (km) of the stretched-exponential kernel: the distance term equals (d / l_j) ** theta1, with
    l_j = theta0 ** (-1 / theta1) * (N_j + theta2) ** (theta3 / theta1), so the scale grows with the POI count."""
    p = {**DEFAULT, **(params or {})}
    return p["theta0"] ** (-1 / p["theta1"]) * (np.asarray(n_poi, float) + p["theta2"]) ** (p["theta3"] / p["theta1"])
