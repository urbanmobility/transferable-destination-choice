"""NAICS code -> the eight facility classes of round 1 (design §5.1.3; rules fixed in plan 1 Task 5).

Longest prefix wins. Codes outside the table are "other": counted in N (all POIs) but in none of the eight shares.
"""
RULES = [                      # (prefix, class) — longer prefixes listed first within a family
    ("512131", "leisure"), ("512132", "leisure"),      # cinemas
    ("722", "food"),
    ("721", "leisure"),                                # accommodation
    ("4881", "transit"), ("4811", "transit"), ("485", "transit"), ("482", "transit"),
    ("522", "service"), ("532", "service"), ("491", "service"), ("492", "service"),
    ("44", "retail"), ("45", "retail"),
    ("62", "health"),
    ("61", "education"),
    ("71", "leisure"),
    ("81", "service"),
    ("51", "office"), ("52", "office"), ("53", "office"), ("54", "office"), ("55", "office"), ("56", "office"),
    ("92", "office"),
]
CLASSES = ("food", "retail", "health", "education", "leisure", "service", "office", "transit")


def classify(naics):
    if naics is None:
        return "other"
    s = str(naics).strip()
    if not s or not s.isdigit():
        return "other"
    for prefix, cls in sorted(RULES, key=lambda r: -len(r[0])):
        if s.startswith(prefix):
            return cls
    return "other"


def sql_case(col):
    """DuckDB CASE expression equivalent to classify(col), generated from the same rule table."""
    whens = " ".join(f"WHEN starts_with({col}, '{p}') THEN '{c}'" for p, c in sorted(RULES, key=lambda r: -len(r[0])))
    return (f"CASE WHEN {col} IS NULL OR NOT regexp_matches(CAST({col} AS VARCHAR), '^[0-9]+$') THEN 'other' "
            f"{whens.replace(f'starts_with({col}', f'starts_with(CAST({col} AS VARCHAR)')} ELSE 'other' END")
