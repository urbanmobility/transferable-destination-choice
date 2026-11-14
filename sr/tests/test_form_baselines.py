import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines_l import ext_radiation_eta, radiation_eta  # noqa: E402
from form_vars import intervening, subcenters  # noqa: E402


def test_intervening_simple():
    xy = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    pop = np.array([1, 2, 3, 4.0])
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    S = intervening(D, pop)
    assert S[0, 2] == 2
    assert S[0, 3] == 2 + 3
    assert S[1, 2] == 0          # unit 0 is at d=1, tying d(1,2)=1: excluded
    assert S[1, 0] == 0
    assert S[3, 0] == 3 + 2


def test_intervening_ties_excluded():
    xy = np.array([[0, 0], [1, 0], [-1, 0]], float)
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    S = intervening(D, np.array([5, 7, 11.0]))
    assert S[0, 1] == 0          # unit 2 is at the same distance as unit 1: not counted


def _grid(n=5):
    ij = [(r, c) for r in range(n) for c in range(n)]
    adj = {}
    for k, (r, c) in enumerate(ij):
        adj[k] = [kk for kk, (rr, cc) in enumerate(ij) if abs(rr - r) + abs(cc - c) == 1]
    return ij, adj


def test_subcenters_grid():
    ij, adj = _grid()
    jobs = np.array([4000.0 if 1 <= r <= 3 and 1 <= c <= 3 else 0.0 for r, c in ij])
    u = pd.DataFrame({"jobs": jobs, "area_km2": np.ones(len(ij))})
    sc = subcenters(u, adj)
    assert (sc >= 0).sum() == 9 and len(set(sc[sc >= 0])) == 1


def test_subcenters_total_threshold():
    ij, adj = _grid()
    jobs = np.zeros(len(ij))
    jobs[0] = 4000.0             # dense but only 4,000 jobs in total: not a subcentre
    jobs[24] = 12000.0
    u = pd.DataFrame({"jobs": jobs, "area_km2": np.ones(len(ij))})
    sc = subcenters(u, adj)
    assert sc[0] == -1 and sc[24] >= 0


def test_radiation_matches_formula():
    eta = radiation_eta(np.array([10.0]), np.array([5.0]), np.array([[20.0]]))
    assert np.isclose(eta[0, 0], np.log(5) - np.log(30) - np.log(35))


def test_ext_radiation_alpha1_close_to_radiation_for_large_masses():
    mi, mj = np.array([1e4, 3e4]), np.array([5e3, 8e3, 2e4])
    S = np.array([[1e4, 4e4, 9e4], [2e4, 0.0, 6e4]])
    diff = ext_radiation_eta(mi, mj, S, 1.0) - radiation_eta(mi, mj, S)
    assert np.allclose(diff - diff[:, :1], 0.0, atol=1e-3)   # per-origin constant only
