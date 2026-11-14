"""Task 9: per-city OSM features from state-level geojsonseq extracts."""
import json

import pytest
import shapely

from city_fixture import TRACTS
from test_units import city_env  # noqa: F401


def _feat(geom, props):
    return json.dumps({"type": "Feature", "properties": props, "geometry": shapely.geometry.mapping(geom)})


def _write_extracts(root):
    d = root / "derived/osm/us2024"
    d.mkdir(parents=True)
    # tract 06999000100: lon -118.0..-117.991, lat 34.0..34.009 ; tract 06999000200: lon -117.99..-117.981
    transit = [_feat(shapely.Point(-117.996, 34.004), {"@type": "n", "@id": 1, "highway": "bus_stop"}),
               _feat(shapely.Point(-117.993, 34.006), {"@type": "n", "@id": 2, "railway": "station"}),
               _feat(shapely.Point(-117.985, 34.004), {"@type": "n", "@id": 3, "amenity": "bus_station"}),
               _feat(shapely.Point(-117.994, 34.002), {"@type": "n", "@id": 4, "amenity": "bench"}),      # excluded
               _feat(shapely.Point(-117.996, 34.004), {"@type": "n", "@id": 1, "highway": "bus_stop"})]   # duplicate id
    park_box = shapely.box(-117.99, 34.0, -117.9855, 34.0045)
    # osmium export writes a closed way twice (LineString first, then the assembled area) with the same @type/@id
    parks = [_feat(park_box.exterior, {"@type": "way", "@id": 10, "leisure": "park"}),
             _feat(park_box, {"@type": "way", "@id": 10, "leisure": "park"})]
    roads = [_feat(shapely.LineString([(-117.997, 34.0), (-117.997, 34.009)]), {"@type": "w", "@id": 20, "highway": "primary"}),
             _feat(shapely.LineString([(-117.994, 34.0), (-117.994, 34.009)]), {"@type": "w", "@id": 21, "highway": "secondary"})]
    for name, feats in (("transit", transit), ("parks", parks), ("roads", roads)):
        (d / f"06_{name}.geojsonseq").write_text("\n".join(feats) + "\n")
    for name in ("transit", "parks", "roads"):
        (d / f"32_{name}.geojsonseq").write_text("")


def test_tract_features(city_env):  # noqa: F811
    root, csv, units = city_env
    _write_extracts(root)
    from osm_city import tract_features
    f = tract_features("TEST1", root, cities_csv=csv).set_index("unit")
    assert f.loc[TRACTS[0], "n_transit_osm"] == 2 and f.loc[TRACTS[1], "n_transit_osm"] == 1 and f.loc[TRACTS[3], "n_transit_osm"] == 0
    assert f.loc[TRACTS[0], "road_km"] == pytest.approx(2.0, rel=0.05) and f.loc[TRACTS[1], "road_km"] == 0
    assert f.loc[TRACTS[1], "park_share"] == pytest.approx(0.25, rel=0.05) and f.loc[TRACTS[0], "park_share"] == 0
    base = root / "derived/model_inputs/M/TEST1/tract"
    assert (base / "osm_features.parquet").exists() and (base / "roads_builtup.gpkg").exists()


def test_geofabrik_names():
    from config import CONUS_STATES
    from osm_city import GEOFABRIK
    assert set(GEOFABRIK) == set(CONUS_STATES.values())
    assert GEOFABRIK["CA"] == ["california/norcal", "california/socal"] and GEOFABRIK["NY"] == ["new-york"]
    assert GEOFABRIK["DC"] == ["district-of-columbia"] and GEOFABRIK["WV"] == ["west-virginia"]


def test_extract_state_skips_complete_extracts_and_never_truncates(tmp_path):
    """Review Focus: on the compute server the extracts are synced in and the pbf files are absent — extract_state must leave a
    complete extract untouched (skip) and must not truncate anything when it cannot run."""
    from osm_city import extract_state
    d = tmp_path / "derived/osm/us2024"
    d.mkdir(parents=True)
    for name in ("transit", "parks", "roads"):
        (d / f"32_{name}.geojsonseq").write_text('{"type":"Feature"}\n')
    assert extract_state("32", tmp_path) == "skip"
    for name in ("transit", "parks", "roads"):
        assert (d / f"32_{name}.geojsonseq").read_text() == '{"type":"Feature"}\n'
    (d / "32_roads.geojsonseq").unlink()                      # incomplete extract, no pbf -> error, nothing truncated
    with pytest.raises(FileNotFoundError):
        extract_state("32", tmp_path)
    assert (d / "32_transit.geojsonseq").read_text() == '{"type":"Feature"}\n'
