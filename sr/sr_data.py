"""Shared data access for the SR round: region inputs and destination folds (Task 1), long-format
SR tables (Task 2).
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))

# ODSR_ROOT: data root (default: the parent of the scripts directory); ODSR_LINE: model_inputs line (L = round 1, M = multi-city);
# ODSR_CITIES: comma-separated city names in category order (multi-city); unset = the round-1 defaults.
ROOT = Path(os.environ.get("ODSR_ROOT", Path(__file__).resolve().parents[2]))
LINE = os.environ.get("ODSR_LINE", "L")
SPLITS = ("train", "val", "test")
K_FOLDS, N_CLUST, FOLD_SEED = 5, 20, 20261031


def load(region, scale):
    """Units, distances, counts and the masks of run_gate_l.py."""
    base = ROOT / f"derived/model_inputs/{LINE}/{region}/{scale}"
    u = pd.read_parquet(base / "units.parquet")
    D = np.load(base / "D.npy").astype(np.float64)
    T = {sp: np.load(base / f"T_{sp}.npz")["n_dev_5"].astype(np.float64) for sp in SPLITS}
    dest = ~((u["population"] == 0) & (u["N"] == 0)).to_numpy()
    orig = (u["origin_included"].fillna(False).astype(bool) & (u["ses"] > 0) & dest).to_numpy()
    groups = np.clip(u["ses"].to_numpy() - 1, 0, None)
    return {"base": base, "u": u, "D": D, "T": T, "dest": dest, "orig": orig, "groups": groups, "N": len(u)}


def folds(region, scale=None):
    """Outer destination fold 0..4 per unit (-1 outside the choice set), as in ceiling_cv2_l.py."""
    d = load(region, scale) if isinstance(region, str) else region
    xy = d["u"][["x", "y"]].to_numpy(float)
    k = min(N_CLUST, int(d["dest"].sum()))          # tiny (test) cities: one cluster per unit
    _, lab = kmeans2(xy[d["dest"]], k, seed=FOLD_SEED, minit="++")
    fold_of_cluster = np.random.default_rng(FOLD_SEED).permutation(np.arange(k) % K_FOLDS)
    f = np.full(d["N"], -1)
    f[np.flatnonzero(d["dest"])] = fold_of_cluster[lab]
    return f


_cities = os.environ.get("ODSR_CITIES")
CITY = ({c: k for k, c in enumerate(_cities.split(","))} if _cities
        else {"LA": 0, "SFBAY": 1, "SDCA": 2, "SRAA": 3, "RSBO": 4})
CLASSES = ("food", "retail", "health", "education", "leisure", "service", "office", "transit")
VARS = (["d", "logd", "P_i", "P_j", "E_j", "N_j", "a_j"] + [f"s_{c}" for c in CLASSES]
        + ["H_j", "q_i", "pd_i", "pd_j", "jd_j", "rsub_i", "rsub_j", "rcbd_i", "rcbd_j", "acc_i", "S_ij"])
BLOCKS = {"test": (0,), "val": (1,), "search": (2, 3, 4), "search23": (2, 3)}   # search23: agent loop (fold 4 ranks)


@dataclass
class SRData:
    X: np.ndarray            # float32 (rows, K), columns VARS
    names: list
    y: np.ndarray            # counts of the split
    w: np.ndarray            # 1 for available rows (only available rows are stored)
    offsets: np.ndarray      # row range of each origin, rows contiguous by origin
    city: np.ndarray         # CITY code per row
    ses: np.ndarray          # SES group 0..2 of the origin per row
    offset_B: np.ndarray     # variant-B distance spline offset (0 if not requested)
    origin_unit: np.ndarray  # unit index of the origin within its region
    dest_unit: np.ndarray    # unit index of the destination within its region
    logw: np.ndarray = None  # log(1 / inclusion prob.) per row (sampled alternatives); 0 = full choice set
    cat: np.ndarray = None   # position of the row's city in the `regions` of build() (template category, 0-based)

    def __post_init__(self):
        if self.logw is None:
            self.logw = np.zeros(len(self.y))
        if self.cat is None:     # round-1 callers: the two regions LA, SFBAY are also codes 0, 1
            self.cat = self.city.copy()


def dest_mask(d, fold, block):
    if block == "all":
        return d["dest"].copy()
    if block.startswith("fold:"):
        return d["dest"] & (fold == int(block[5:]))
    if block.startswith("notfold:"):
        return d["dest"] & (fold != int(block[8:]))
    return d["dest"] & np.isin(fold, BLOCKS[block])


def income_rank(u, orig_like):
    """Population-weighted within-city percentile of median household income (0..1)."""
    inc, pop = u["income"].to_numpy(float), u["population"].to_numpy(float)
    ok = np.isfinite(inc) & (pop > 0)
    order = np.argsort(inc[ok])
    cw = np.cumsum(pop[ok][order]) / pop[ok].sum()
    q = np.full(len(u), 0.5)
    vals = np.empty(ok.sum())
    vals[order] = cw - 0.5 * pop[ok][order] / pop[ok].sum()
    q[ok] = vals
    return q


def unit_table(d):
    """Unit-level variables (raw, positive scales) used to assemble the pair rows."""
    from run_gate_l import features   # noqa: F401  (smoothing identical to the ladder)
    u = d["u"]
    form = pd.read_parquet(d["base"] / "form.parquet").set_index("unit").reindex(u["unit"])
    n = u[[f"n_{c}" for c in CLASSES]].to_numpy(float)
    pbar = n.sum(0) / n.sum()
    sh = (n + pbar) / (n.sum(1, keepdims=True) + 1.0)
    # entropy over the classes present in the data: a class with no POI anywhere (Veraset carries no health places)
    # has share 0 and 0 * log 0 counts as 0, not NaN; unchanged when every class is present
    log_sh = np.log(np.where(sh > 0, sh, 1.0))
    t = {"P": u["population"].to_numpy(float), "E": u["jobs"].to_numpy(float), "N": u["N"].to_numpy(float),
         "a": u["area_km2"].clip(lower=0.01).to_numpy(float), "H": -(sh * log_sh).sum(1),
         "q": income_rank(u, d["orig"]), "pd": form["pop_dens"].to_numpy(float), "jd": form["job_dens"].to_numpy(float),
         "rsub": form["r_sub"].to_numpy(float), "rcbd": form["r_cbd"].to_numpy(float),
         "acc": form["acc_jobs"].to_numpy(float)}
    for k, c in enumerate(CLASSES):
        t[f"s_{c}"] = sh[:, k]
    return t


def sample_origins(d, n_orig, seed):
    io = np.flatnonzero(d["orig"])
    if n_orig is None or n_orig >= len(io):
        return io
    rng = np.random.default_rng(seed)
    g = d["groups"][io]
    take = []
    for k in range(3):
        ik = io[g == k]
        m = int(round(n_orig * len(ik) / len(io)))
        take.append(rng.choice(ik, size=min(m, len(ik)), replace=False))
    return np.sort(np.concatenate(take))


def distance_offset(region, scale, block):
    """Group-specific distance spline s_k(log d) fitted on train x block (variant B offset), (N, N)."""
    from choice_model import ChoiceModel, Design, bspline_basis
    from run_gate_l import DIST_DF
    d = load(region, scale)
    fold = folds(d)
    N, D, dest = d["N"], d["D"], d["dest"]
    logd = np.log(np.maximum(D, 0.05))
    off = ~np.eye(N, dtype=bool) & dest[None, :] & dest[:, None]
    kn = np.quantile(logd[off], np.linspace(0, 1, DIST_DF - 1)[1:-1])
    Bd = bspline_basis(logd.ravel(), kn, lo=logd[off].min(), hi=logd[off].max()).reshape(N, N, -1)
    des = Design(B=Bd, X=np.zeros((N, 1)), groups=d["groups"], n_groups=3, origin_mask=d["orig"],
                 dest_mask=dest_mask(d, fold, block))
    m = ChoiceModel(des, group_distance=True).fit(d["T"]["train"])
    full = np.zeros((N, N))
    full[d["orig"]] = m.eta(m.theta)[0]
    return full


def build(regions, scale, split, dest_block, n_orig, seed, offset_B=False, offset_block=None, neg_rate=None,
          neg_seed=0):
    """neg_rate: sampled alternatives for the search (McFadden): every destination with a positive count in
    this split is kept, zero-count destinations are kept with probability neg_rate and carry
    logw = log(1 / neg_rate), which the losses add to eta (unbiased estimate of each denominator).
    offset_block: destination block the variant-B spline is fitted on (default: the search block for
    search / val / test, else dest_block itself; pass the training block when scoring a held-out fold)."""
    parts = []
    regions = list(regions)
    for region in regions:
        d = load(region, scale)
        fold = folds(d)
        dm = dest_mask(d, fold, dest_block)
        t = unit_table(d)
        S = np.load(d["base"] / "S.npy").astype(np.float64)
        ob = offset_block or ("search" if dest_block in ("search", "val", "test") else dest_block)
        offB = distance_offset(region, scale, ob) if offset_B else None
        rng = np.random.default_rng(neg_seed + 1000 * CITY[region])
        for i in sample_origins(d, n_orig, seed):
            js = np.flatnonzero(dm & (np.arange(d["N"]) != i))
            lw = np.zeros(len(js))
            if neg_rate is not None:
                pos = d["T"][split][i, js] > 0
                keep = pos | (rng.random(len(js)) < neg_rate)
                js, lw = js[keep], np.where(pos[keep], 0.0, -np.log(neg_rate))
            if len(js) == 0:
                continue
            cols = {"d": np.maximum(d["D"][i, js], 0.05), "P_i": np.full(len(js), t["P"][i]), "P_j": t["P"][js],
                    "E_j": t["E"][js], "N_j": t["N"][js], "a_j": t["a"][js], "H_j": t["H"][js],
                    "q_i": np.full(len(js), t["q"][i]), "pd_i": np.full(len(js), t["pd"][i]), "pd_j": t["pd"][js],
                    "jd_j": t["jd"][js], "rsub_i": np.full(len(js), t["rsub"][i]), "rsub_j": t["rsub"][js],
                    "rcbd_i": np.full(len(js), t["rcbd"][i]), "rcbd_j": t["rcbd"][js],
                    "acc_i": np.full(len(js), t["acc"][i]), "S_ij": S[i, js]}
            cols["logd"] = np.log(cols["d"])
            for c in CLASSES:
                cols[f"s_{c}"] = t[f"s_{c}"][js]
            parts.append({"X": np.column_stack([cols[v] for v in VARS]).astype(np.float32),
                          "y": d["T"][split][i, js], "city": CITY[region], "cat": regions.index(region),
                          "ses": d["groups"][i],
                          "offB": offB[i, js] if offset_B else np.zeros(len(js)), "i": i, "js": js, "lw": lw})
    lens = np.array([len(p["y"]) for p in parts])
    return SRData(X=np.concatenate([p["X"] for p in parts]), names=list(VARS),
                  y=np.concatenate([p["y"] for p in parts]), w=np.ones(lens.sum()),
                  offsets=np.r_[0, np.cumsum(lens)],
                  city=np.repeat([p["city"] for p in parts], lens), ses=np.repeat([p["ses"] for p in parts], lens),
                  offset_B=np.concatenate([p["offB"] for p in parts]),
                  origin_unit=np.repeat([p["i"] for p in parts], lens),
                  dest_unit=np.concatenate([p["js"] for p in parts]),
                  logw=np.concatenate([p["lw"] for p in parts]),
                  cat=np.repeat([p["cat"] for p in parts], lens))
