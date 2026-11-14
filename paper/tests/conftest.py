"""Import path for the paper-figure modules; a non-interactive backend."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
P = Path(__file__).resolve().parents[1]
if str(P) not in sys.path:
    sys.path.insert(0, str(P))
