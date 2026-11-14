"""L1a: OSM tag -> 8 coarse facility classes, exact key=value rules, first match wins.

Classes: food (food and drink), retail (everyday retail), health (health care), education, leisure (leisure,
culture, sports), service (public and personal services), office (office and business), transit (transport stops).
"exclude" marks tags that are
infrastructure or furniture (parking, benches, toilets), private or sub-facility features
(swimming pools, pitches, playgrounds), lodging, or duplicate transit features (platforms and
stop positions of a stop already mapped as bus_stop / station). Parks are not counted here:
leisure=park / dog_park and landuse=recreation_ground enter as area share (PARK_TAGS).
The rule list was drafted from the CA 2019-01-01 tag frequencies and the keyword lists of the
neuroGravity extractor (osm_feature_extractor.py). The 100-item check per city (LA, SFBAY;
2026-10-29) moved pharmacies to retail and dropped tourism=attraction; frozen after that check.
"""

ANY = None  # matches any value of the key

FOOD = ("restaurant", "fast_food", "cafe", "bar", "pub", "food_court", "ice_cream", "biergarten")
HEALTH_AMENITY = ("hospital", "clinic", "doctors", "dentist", "nursing_home")
EDUCATION_AMENITY = ("school", "kindergarten", "college", "university", "childcare", "language_school",
                     "driving_school", "music_school", "prep_school")
LEISURE_AMENITY = ("theatre", "cinema", "arts_centre", "library", "nightclub", "casino", "community_centre",
                   "events_venue", "planetarium", "social_centre")
SERVICE_AMENITY = ("bank", "post_office", "townhall", "police", "courthouse", "place_of_worship",
                   "social_facility", "car_wash", "veterinary", "bureau_de_change", "money_transfer",
                   "public_building", "car_rental")
RETAIL_AMENITY = ("marketplace", "fuel", "pharmacy")  # US chain drugstores are daily retail
TRANSIT_AMENITY = ("bus_station", "ferry_terminal")
SERVICE_SHOP = ("hairdresser", "beauty", "laundry", "dry_cleaning", "massage", "tailor", "tattoo",
                "travel_agency", "funeral_directors", "car_repair", "copyshop", "storage_rental",
                "pawnbroker", "nail_salon", "cosmetics_service", "tyres")
LEISURE_LEISURE = ("sports_centre", "fitness_centre", "stadium", "golf_course", "bowling_alley", "ice_rink",
                   "water_park", "dance", "amusement_arcade", "miniature_golf", "horse_riding", "sauna",
                   "disc_golf_course", "marina", "escape_game", "trampoline_park")
# tourism=attraction excluded: mostly sub-sites of theme parks / zoos (one park -> many objects)
LEISURE_TOURISM = ("museum", "gallery", "zoo", "theme_park", "aquarium")

RULES = [  # (key, values or ANY, class)
    ("amenity", FOOD, "food"),
    ("healthcare", ("pharmacy",), "retail"),
    ("amenity", HEALTH_AMENITY, "health"),
    ("healthcare", ANY, "health"),
    ("amenity", EDUCATION_AMENITY, "education"),
    ("amenity", LEISURE_AMENITY, "leisure"),
    ("amenity", SERVICE_AMENITY, "service"),
    ("amenity", RETAIL_AMENITY, "retail"),
    ("amenity", TRANSIT_AMENITY, "transit"),
    ("amenity", ANY, "exclude"),
    ("shop", ("vacant",), "exclude"),
    ("shop", ("optician",), "health"),
    ("shop", SERVICE_SHOP, "service"),
    ("shop", ANY, "retail"),
    ("office", ANY, "office"),
    ("leisure", LEISURE_LEISURE, "leisure"),
    ("tourism", LEISURE_TOURISM, "leisure"),
    ("highway", ("bus_stop",), "transit"),
    ("railway", ("station", "halt", "tram_stop", "subway_entrance"), "transit"),
    ("public_transport", ("station",), "transit"),
    ("aeroway", ("aerodrome",), "transit_airport"),  # only with an IATA code, see classify()
]
CLASSES = ("food", "retail", "health", "education", "leisure", "service", "office", "transit")
PARK_TAGS = (("leisure", "park"), ("leisure", "dog_park"), ("landuse", "recreation_ground"))


def classify(tags):
    """Class of an OSM object from its tag dict; None if no rule applies."""
    for key, values, cls in RULES:
        v = tags.get(key)
        if v is None or (values is not ANY and v not in values):
            continue
        if cls == "transit_airport":
            return "transit" if tags.get("iata") else "exclude"
        return cls
    return None


def is_park(tags):
    return any(tags.get(k) == v for k, v in PARK_TAGS)
