import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr_data import ROOT, VARS, build, folds, load  # noqa: E402


def test_folds_match_cv2():
    f = folds("SFBAY", "zcta")
    ref = json.loads((ROOT / "results/gate2/L/explore/Ccv2_SFBAY_zcta.json").read_text())["fold_sizes"]
    d = load("SFBAY", "zcta")
    assert np.bincount(f[d["dest"]], minlength=5).tolist() == ref
    assert (f[~d["dest"]] == -1).all()


def test_rows_exclude_unavailable():
    s = build(["SFBAY"], "zcta", "train", "search", 30, 0)
    d = load("SFBAY", "zcta")
    f = folds(d)
    oi, dj = s.origin_unit, s.dest_unit
    assert (oi != dj).all()
    assert d["dest"][dj].all()
    assert np.isin(f[dj], [2, 3, 4]).all()            # search block = folds other than 0 (test) and 1 (val)
    assert (s.w == 1).all()


def test_offsets_contiguous():
    s = build(["SFBAY"], "zcta", "train", "all", 25, 1)
    assert s.offsets[0] == 0 and s.offsets[-1] == len(s.y)
    assert (np.diff(s.offsets) > 0).all()
    for a, b in zip(s.offsets[:-1], s.offsets[1:]):
        assert len(set(s.origin_unit[a:b])) == 1


def test_counts_total():
    s = build(["SFBAY"], "zcta", "train", "all", None, 0)
    d = load("SFBAY", "zcta")
    av = d["dest"][None, :] & ~np.eye(d["N"], dtype=bool)
    assert np.isclose(s.y.sum(), d["T"]["train"][d["orig"]][av[d["orig"]]].sum())


def test_two_cities_and_variables():
    s = build(["LA", "SFBAY"], "zcta", "train", "search", 20, 0)
    assert set(np.unique(s.city)) == {0, 1}
    assert s.names == VARS and s.X.shape == (len(s.y), len(VARS))
    assert np.isfinite(s.X).all()
    assert (s.X[:, VARS.index("d")] > 0).all()


def test_offset_B_finite_and_decreasing_in_distance():
    s = build(["SFBAY"], "zcta", "train", "search", 20, 0, offset_B=True)
    assert np.isfinite(s.offset_B).all()
    d = s.X[:, VARS.index("d")]
    near, far = s.offset_B[d < np.quantile(d, 0.1)].mean(), s.offset_B[d > np.quantile(d, 0.9)].mean()
    assert near > far
