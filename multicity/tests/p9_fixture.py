"""Synthetic cities for the plan-9 tests: a loaded-city dict like sr_data.load (units, distances, trips, masks) whose
trips follow log(N_j + 1) - beta log d, and the long-format rows / matrix arrays built from it."""
import numpy as np
import pandas as pd

from sr_data import SRData, VARS

FOLD_OF = {"search": (2, 3, 4), "val": (1,), "fold:0": (0,)}


def toy_city(seed: int, n_units: int = 30, beta: float = 1.5, trips: int = 400, zero_mass: bool = False,
             eta_fn=None) -> dict:
    """A city of n_units units in a 10 km square. zero_mass: unit 0 has no POI, unit 1 no population (both stay in
    the choice set). eta_fn(d) -> N x N utilities replaces the gravity that generates the trips."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10, (n_units, 2))
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    pop = rng.integers(500, 5000, n_units).astype(float)
    poi = rng.integers(1, 60, n_units).astype(float)
    if zero_mass:
        poi[0], pop[1] = 0.0, 0.0
    u = pd.DataFrame({"unit": np.arange(n_units), "x": xy[:, 0], "y": xy[:, 1], "population": pop, "N": poi})
    d = {"u": u, "D": D, "N": n_units, "dest": np.ones(n_units, bool), "orig": np.ones(n_units, bool),
         "groups": np.arange(n_units) % 3, "fold": np.arange(n_units) % 5}
    e = eta_fn(d) if eta_fn is not None else np.log(poi + 1.0)[None, :] - beta * np.log(np.maximum(D, 0.05))
    e = np.where(np.eye(n_units, dtype=bool), -np.inf, e)
    p = np.exp(e - e.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    d["T"] = {sp: np.array([rng.multinomial(trips, p[i]) for i in range(n_units)], float)
              for sp in ("train", "val", "test")}
    return d


def toy_rows(d: dict, origins, split: str, block: str) -> SRData:
    """Long-format rows of the origins x the block's destinations (self excluded), columns d, logd, P_i, P_j, N_j."""
    dm = d["dest"] & np.isin(d["fold"], FOLD_OF[block])
    pop, poi = d["u"]["population"].to_numpy(float), d["u"]["N"].to_numpy(float)
    X, y, off, oi, dj = [], [], [0], [], []
    for i in origins:
        js = np.flatnonzero(dm & (np.arange(d["N"]) != i))
        x = np.zeros((len(js), len(VARS)), np.float32)
        x[:, 0] = np.maximum(d["D"][i, js], 0.05)
        x[:, 1] = np.log(x[:, 0])
        x[:, 2], x[:, 3], x[:, 5] = pop[i], pop[js], poi[js]
        X.append(x)
        y.append(d["T"][split][i, js])
        off.append(off[-1] + len(js))
        oi += [i] * len(js)
        dj += list(js)
    n = off[-1]
    z = np.zeros(n, int)
    return SRData(X=np.concatenate(X), names=list(VARS), y=np.concatenate(y), w=np.ones(n), offsets=np.array(off),
                  city=z, ses=z, offset_B=np.zeros(n), origin_unit=np.array(oi), dest_unit=np.array(dj))


def toy_inputs(seed: int, n_units: int = 30, n_orig: int = 10, beta: float = 1.5, eval_split: str = "val",
               eval_block: str = "val", **kw) -> dict:
    """{"d", "tr", "ev", "A"}: training rows on n_orig sampled origins x search block, scoring rows on every origin x
    the evaluation block, and the matrix arrays of the same blocks."""
    import prior_models as pm
    from form_vars import intervening
    d = toy_city(seed, n_units, beta, **kw)
    A = pm.city_arrays(d, d["fold"], eval_split, eval_block, intervening(d["D"], d["u"]["population"].to_numpy(float)),
                       n_orig=n_orig, seed=0)
    return {"d": d, "A": A, "tr": toy_rows(d, A["io"]["tr"], "train", "search"),
            "ev": toy_rows(d, A["io"]["ev"], eval_split, eval_block)}
