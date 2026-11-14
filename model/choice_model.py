"""Fixed-budget conditional choice model (plan §4.2) and the gate ladder (§4.3, §7.4).

    p_ij = exp(eta_ij) / sum_{l in J_i} exp(eta_il),   eta_ij = s_{k(i)}(d_ij) + lambda_{k(i)} g(x_j)

s_k: distance response, a B-spline in log d with fixed knots (pairwise-distance quantiles of the
choice set, no outcome data), shared (k = 1) or per SES group; g: linear in destination features
(GLM) or in a penalised spline expansion of them (GAM); lambda_k = 1 except in M4lambda
(lambda_1 = 1). Loss: conditional negative log-likelihood -sum T log p (+ ridge on the spline
coefficients for GAM). Prediction for a split uses that split's observed budgets O_i.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize


def bspline_basis(x, knots_inner, degree=3, lo=None, hi=None):
    """B-spline basis (no intercept column) on x with the given interior knots."""
    lo = x.min() if lo is None else lo
    hi = x.max() if hi is None else hi
    t = np.r_[[lo] * (degree + 1), knots_inner, [hi] * (degree + 1)]
    n = len(t) - degree - 1
    xc = np.clip(x, lo, hi)
    B = np.empty((len(x), n))
    for k in range(n):
        c = np.zeros(n)
        c[k] = 1
        B[:, k] = BSpline(t, c, degree, extrapolate=False)(xc)
    B = np.nan_to_num(B)
    return B[:, 1:]  # drop one column: the origin constant absorbs a level shift


@dataclass
class Design:
    """Pair-level distance basis and destination-level features for one city / scale."""
    B: np.ndarray            # (N, N, nb) distance basis
    X: np.ndarray            # (N, p) destination features (standardised)
    groups: np.ndarray       # (N,) origin group 0..G-1 (0 for shared models)
    n_groups: int
    origin_mask: np.ndarray  # (N,) origins used in the loss
    dest_mask: np.ndarray    # (N,) units in the choice set
    penalty: np.ndarray = field(default=None)  # (p,) ridge weights on X coefficients (GAM)
    Z: np.ndarray = field(default=None)        # (N, N, q) optional pair features (M3s interaction, M3.5 deficit)


class ChoiceModel:
    def __init__(self, design, group_distance=False, group_scale=False, ridge=0.0, shape_weights=None):
        """shape_weights: group weights pi_k; if given, adds the rank-1 shape term a_k h(x_j) of M4 with
        h = X c and a = E alpha, E an orthonormal basis of {a : sum_k pi_k a_k = 0}."""
        self.d = design
        self.gd = group_distance
        self.gs = group_scale
        self.ridge = ridge
        G = design.n_groups
        nb, p = design.B.shape[2], design.X.shape[1]
        self.n_s = nb * (G if group_distance else 1)
        self.n_b = p
        self.n_l = (G - 1) if group_scale else 0
        self.shape = shape_weights is not None
        self.n_c = p if self.shape else 0
        self.n_a = (G - 1) if self.shape else 0
        if self.shape:
            pi = np.asarray(shape_weights, float)
            self.E = np.linalg.svd(pi[None, :])[2][1:].T          # (G, G-1), columns orthogonal to pi
        avail = design.dest_mask[None, :] & ~np.eye(len(design.dest_mask), dtype=bool)
        self.avail = avail[design.origin_mask]
        self.Bo = design.B[design.origin_mask]                # (No, N, nb)
        self.go = design.groups[design.origin_mask]
        self.Zo = None if design.Z is None else design.Z[design.origin_mask]
        self.n_z = 0 if design.Z is None else design.Z.shape[2]
        self.scale = 1.0  # objective multiplier; fit() sets 1 / training visits (per-visit loss)

    def unpack(self, th):
        s = th[:self.n_s]
        b = th[self.n_s:self.n_s + self.n_b]
        lam = np.ones(self.d.n_groups)
        if self.n_l:
            lam[1:] = np.exp(th[self.n_s + self.n_b:self.n_s + self.n_b + self.n_l])
        return s, b, lam

    def extras(self, th):
        """(c, a, theta_pair): shape direction coefficients, group loadings, pair-feature coefficients."""
        k = self.n_s + self.n_b + self.n_l
        c = th[k:k + self.n_c]
        a = self.E @ th[k + self.n_c:k + self.n_c + self.n_a] if self.shape else None
        z = th[k + self.n_c + self.n_a:k + self.n_c + self.n_a + self.n_z]
        return c, a, z

    def n_params(self):
        return self.n_s + self.n_b + self.n_l + self.n_c + self.n_a + self.n_z

    def eta(self, th):
        s, b, lam = self.unpack(th)
        nb = self.d.B.shape[2]
        if self.gd:
            S = s.reshape(self.d.n_groups, nb)[self.go]       # (No, nb)
            dist = np.einsum("ijk,ik->ij", self.Bo, S)
        else:
            dist = self.Bo @ s
        g = self.d.X @ b                                      # (N,)
        e = dist + lam[self.go][:, None] * g[None, :]
        c, a, z = self.extras(th)
        if self.shape:
            e = e + a[self.go][:, None] * (self.d.X @ c)[None, :]
        if self.n_z:
            e = e + self.Zo @ z
        return e, g

    def logp(self, th):
        e, g = self.eta(th)
        e = np.where(self.avail, e, -np.inf)
        m = e.max(1, keepdims=True)
        lse = m + np.log(np.exp(e - m).sum(1, keepdims=True))
        return e - lse, g

    def loss_grad(self, th, T):
        lp, g = self.logp(th)
        P = np.exp(lp)
        To = T
        O = To.sum(1, keepdims=True)
        pos = To > 0
        nll = -np.sum(To[pos] * lp[pos])
        R = np.where(self.avail, To - O * P, 0.0)             # d(-nll)/d eta
        s, b, lam = self.unpack(th)
        nb = self.d.B.shape[2]
        if self.gd:
            gs = np.zeros((self.d.n_groups, nb))
            for k in range(self.d.n_groups):
                sel = self.go == k
                gs[k] = np.einsum("ij,ijk->k", R[sel], self.Bo[sel])
            gs = gs.ravel()
        else:
            gs = np.einsum("ij,ijk->k", R, self.Bo)
        Rl = R * lam[self.go][:, None]
        gb = self.d.X.T @ Rl.sum(0)
        grad = [-gs, -gb]
        if self.n_l:
            gl = np.array([(R[self.go == k] @ g).sum() * lam[k] for k in range(1, self.d.n_groups)])
            grad.append(-gl)
        c, a, z = self.extras(th)
        if self.shape:
            h = self.d.X @ c
            grad.append(-(self.d.X.T @ (R * a[self.go][:, None]).sum(0)))
            gk = np.array([(R[self.go == k] @ h).sum() for k in range(self.d.n_groups)])
            grad.append(-(self.E.T @ gk))
        if self.n_z:
            grad.append(-np.einsum("ij,ijq->q", R, self.Zo))
        grad = np.concatenate(grad)
        nll, grad = nll * self.scale, grad * self.scale
        if self.ridge and self.d.penalty is not None:   # ridge is on the per-visit scale
            pen = self.ridge * self.d.penalty
            nll += 0.5 * np.sum(pen * b ** 2)
            grad[self.n_s:self.n_s + self.n_b] += pen * b
            if self.shape:
                k = self.n_s + self.n_b + self.n_l
                nll += 0.5 * np.sum(pen * c ** 2)
                grad[k:k + self.n_c] += pen * c
        return nll, grad

    def fit(self, T, th0=None, maxiter=3000, fixed=None):
        """fixed: optional {parameter index: value}; those parameters are held at the value."""
        T = np.where(self.avail, T[self.d.origin_mask], 0.0)  # visits outside the choice set are dropped
        self.scale = 1.0 / T.sum()
        n = self.n_params()
        th0 = np.zeros(n) if th0 is None else np.array(th0, dtype=float)
        bounds = None
        if fixed:
            bounds = [(None, None)] * n
            for k, v in fixed.items():
                th0[k] = v
                bounds[k] = (v, v)
        res = minimize(self.loss_grad, th0, args=(T,), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "maxcor": 30, "gtol": 1e-8, "ftol": 1e-13})
        self.theta, self.result = res.x, res
        return self

    def evaluate(self, T):
        """Held-out scores with the split's own budgets O_i (plan §4.2, §7.2)."""
        T = np.where(self.avail, T[self.d.origin_mask].astype(np.float64), 0.0)
        lp, _ = self.logp(self.theta)
        O = T.sum(1, keepdims=True)
        mu = O * np.exp(lp)
        pos = T > 0
        nll = -np.sum(T[pos] * lp[pos])
        dev = 2 * np.sum(T[pos] * np.log(T[pos] / mu[pos]))   # sum(mu) = sum(T) per origin
        cpc = 2 * np.minimum(T, mu).sum() / (T.sum() + mu.sum())
        return {"visits": float(T.sum()), "nll_per_visit": nll / T.sum(), "deviance": dev, "cpc": cpc}

    def noise_deviance(self, T, S=10, seed=0):
        """Expected deviance of multinomial samples drawn from this model with T's origin budgets."""
        T = np.where(self.avail, T[self.d.origin_mask], 0.0)
        P = np.exp(self.logp(self.theta)[0])
        O = np.rint(T.sum(1)).astype(np.int64)
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(S):
            dev = 0.0
            for i in np.flatnonzero(O):
                a = self.avail[i]
                p = P[i, a] / P[i, a].sum()
                x = rng.multinomial(O[i], p)
                k = x > 0
                dev += 2 * np.sum(x[k] * np.log(x[k] / (O[i] * p[k])))
            out.append(dev)
        return float(np.mean(out)), float(np.std(out))
