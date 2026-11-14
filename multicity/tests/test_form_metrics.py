"""Task 10b: urban-form metrics (population-density Gini, orientation entropy, subcentre shares, water share)."""
import json

import numpy as np
import pytest
import shapely

from test_units import city_env  # noqa: F401


def _lines(bearings_deg):
    return [shapely.LineString([(0, 0), (np.sin(np.radians(b)), np.cos(np.radians(b)))]) for b in bearings_deg]


def test_gini():
    from form_metrics import gini
    assert gini(np.array([1, 1, 1, 1.0]), np.array([1, 1, 1, 1.0])) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 100.0]), np.array([1, 1, 1, 1.0])) == pytest.approx(0.75)
    assert gini(np.array([0, 100.0]), np.array([3, 1.0])) == pytest.approx(0.75)        # weights matter


def test_orientation_entropy_and_order():
    from form_metrics import orientation_entropy, orientation_order
    grid = _lines([0, 90, 180, 270])
    assert orientation_entropy(grid) == pytest.approx(np.log(4))
    assert orientation_order(grid) == pytest.approx(1.0)
    spread = _lines(range(0, 360, 10))
    assert orientation_entropy(spread) == pytest.approx(np.log(36))
    assert orientation_order(spread) == pytest.approx(0.0)
    assert orientation_entropy([]) != orientation_entropy([])                           # nan for no roads


def test_compute_city(city_env):  # noqa: F811
    root, csv, units = city_env
    units.build_city("TEST1", root, cities_csv=csv)
    import geopandas as gpd
    base = root / "derived/model_inputs/M/TEST1/tract"
    gpd.GeoDataFrame(geometry=_lines([0, 90]), crs="EPSG:5070").to_file(base / "roads_builtup.gpkg", driver="GPKG")
    from form_metrics import compute
    m = compute("TEST1", root, cities_csv=csv)
    assert set(m) >= {"pop_density_gini", "cbd_job_share", "subcentre_job_share", "n_subcentres", "orientation_entropy",
                      "orientation_order", "water_share", "population", "jobs"}
    assert 0 <= m["pop_density_gini"] < 1 and m["n_subcentres"] == 0 and m["cbd_job_share"] == 0
    assert m["orientation_entropy"] == pytest.approx(np.log(4)) and m["population"] == 2100 and m["jobs"] == 210
    assert np.isnan(m["water_share"])                                                    # no AREAWATER files in the fixture
    assert json.loads((root / "results/multicity/form_metrics/TEST1.json").read_text())["population"] == 2100


def test_orientation_entropy_handles_clipped_geometry_collections():
    """Clipping roads to the built-up area yields GeometryCollections of lines and points (45/67 cities failed on 11-03)."""
    from form_metrics import orientation_entropy
    a, b = _lines([0, 90])
    mixed = shapely.GeometryCollection([a, shapely.Point(5, 5), shapely.MultiLineString([b])])
    assert orientation_entropy([mixed, shapely.Point(1, 1)]) == pytest.approx(orientation_entropy([a, b]))
