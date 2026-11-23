# Data

All paths are relative to a data root `$ODSR_ROOT`; this repository is cloned into `$ODSR_ROOT/scripts`.

| Input | Source | Where it goes |
|---|---|---|
| Visit records (`visits/`, `work-visits/`, `home-location/` parquet shards, March–May 2024) | Veraset, through the Dewey platform (licensed; not redistributed) | `$ODSR_VERASET` (default `$ODSR_ROOT/raw/veraset`) |
| ACS 2015–2019 5-year estimates, 2010 tracts: population, households, median income, households without a vehicle | US Census Bureau API | `external/acs/` |
| LODES7 workplace area characteristics 2019 (2018 where 2019 is missing) | LEHD | `external/lodes/lodes7/` (`attrs.py download`) |
| TIGER 2010 tracts and water areas | US Census Bureau | `external/tiger/raw/` (`attrs.py download`) |
| CBSA delineation file, July 2023 (`list1_2023.xlsx`) | OMB / Census | `external/cbsa/` |
| OpenStreetMap state extracts | Geofabrik | `external/osm/` (`osm_city.py download`); extraction needs `osmium-tool` |

Spatial units are 2010 census tracts, the vintage of the block-group codes in the visit records. Every derived table is
written under `derived/` and every result under `results/` of the data root; no input is modified.

What is counted: one home-based visit per device, calendar day and destination tract when the device stays at least five
minutes at a POI outside its home tract; the home tract is the block group reported in most study months. A tract's POI
count is the number of distinct places (placekeys) with at least one recorded visit in the study months
(`multicity/poi.py`). Visits to the device's inferred workplace are flagged and kept as a separate count
(`n_work_5`, `n_nonwork_5` in `T_<split>.npz`).

Two city-level covariates of Additional file 1, Table S8 (`results/city_level/city_controls.csv`): the mean commute
time, the sum of the ACS 2015–2019 5-year county estimates B08013_001E (aggregate travel time to work, minutes)
divided by the sum of B08303_001E (workers who did not work at home) over the counties of the city
(`multicity/config/cities.csv`); and the POI density, the median over home tracts of the number of POIs in the tracts
whose internal points lie within 5 km. The county table was retrieved from the US Census Bureau API
(`acs/acs5`, 2019, `for=county:*`) into `external/acs/acs5_2019_county_commute.csv`. Neither covariate is written by a
pipeline stage. The commute times are reproduced exactly from the county table; recomputing the POI density from
`units.parquet` and `D.npy` by the definition above gives the published values in 23 of the 67 cities and differs by
at most 2.1% in the others; with the recomputed values the population slopes of Table S8 change by less than 0.5%.

Only city-level aggregates are published in `results/`.
