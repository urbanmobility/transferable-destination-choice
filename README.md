# Transferable destination choice

Code and city-level results for

> J. Yang†, Z. Huang†, S. Huang, P. Sun, S. Xu, Y. Jin, M. C. González, Y. Xu.
> *A transferable formula captures distance-dependent destination attraction across US metropolitan areas.*
> Submitted to EPJ Data Science (2026). † Equal contribution.

We model where residents go when they leave home as a conditional-logit choice among the census tracts of their
metropolitan area, and search for a compact utility by symbolic regression on 192 million home-based visits to points
of interest (POIs) in 67 US metropolitan areas (March–May 2024). The search, fitted to the raw visit counts, found the
**Reach model**

$$\eta_{ij} = a_j^{\theta_4} - \frac{\theta_0\, d_{ij}^{\theta_1}}{(N_j+\theta_2)^{\theta_3}},\qquad P(j\mid i)=\frac{e^{\eta_{ij}}}{\sum_k e^{\eta_{ik}}},$$

with $d_{ij}$ the straight-line distance in km, $N_j$ the destination's POI count and $a_j$ its area in km². Its distance
scale grows with the destination's POI count, so a destination's size matters more the farther away it is — a
regularity that a flexible model with no built-in link between distance and size also shows. One parameter vector
estimated in 35 discovery cities retained a median 94% of the likelihood gap between population-based gravity and the
flexible model in 32 cities whose visits played no part in estimating it.

## Use the formula

`reach_model/` is a small, dependency-light module (numpy only) with the default parameters of the paper:

```python
import numpy as np
from reach_model import choice_probabilities, flows, DEFAULT

d = ...          # origins x destinations distance matrix, km
n_poi = ...      # POI count of every destination tract
area = ...       # land area of every destination tract, km^2
P = choice_probabilities(d, n_poi, area)            # rows sum to 1; an origin is not its own destination
T = flows(origin_totals, d, n_poi, area)            # production-constrained flows T_i * P(j|i)
```

`DEFAULT` is the uniform parameter vector of Additional file 1, Table S5. Run its tests with
`pytest reach_model`.

## Repository layout

| Path | Content |
|---|---|
| `reach_model/` | the formula, its elasticities and default parameters |
| `multicity/` | the full multi-city pipeline: data build, symbolic-regression search and candidate library, scoring, registered hypotheses, freeze and confirmation, benchmarks and parameter transfer, exploratory and revision analyses (`multicity/tests/` for unit tests) |
| `sr/` | data rows, PySR set-up and conditional-logit loss, refitting, reference models |
| `model/` | the conditional-logit engine and the flexible reference models |
| `osm/` | OpenStreetMap feature rules |
| `paper/` | figures and supplementary tables of the paper (`paper/make.py`) |
| `results/` | city-level results (see below) |
| `docs/` | data access and reproduction notes |

## Data

The visit records are licensed from Veraset through the Dewey platform and cannot be redistributed; access is governed
by the provider. All other inputs are public: 2015–2019 ACS 5-year estimates (2010 tracts), LODES7 workplace area
characteristics 2019, TIGER 2010 tracts and water areas, OMB CBSA delineations (July 2023) and OpenStreetMap extracts.
See [`docs/DATA.md`](docs/DATA.md) for the expected directory layout and [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the
order in which the stages were run.

## Results

`results/` contains only city-level aggregates, no device or tract-pair data:

- `city_level/` — per city: characteristics, fit (NLL) of every model, Reach-model constants with standard errors,
  elasticities by distance band, and the model ablations (also Additional file 2 of the paper);
- `benchmarks/` — the 29 model versions under four parameter sources in the 32 target cities, the selected opponents
  and the registered contrasts;
- `registry/` — the registered hypotheses and their held-out results (`registry_en.csv` gives all 25 in English, as in
  Additional file 1, Table S3);
- `freeze/` — the frozen formula library and the candidate pool it was drawn from;
- `figure_source_data/` — the city-level data behind the figures.

## Language

Code, comments and reports are in English. Texts that became part of the registered hypotheses (statement and criterion templates in `multicity/laws.py`, `city_outcomes.py`, `lawtest_extra.py`, `v3_library.py`, `explore_p7.py` and `multicity/config/p7_covariates.json`) keep their original Chinese wording so that the code reproduces the registered records in `results/registry/`; English versions are in `results/registry/registry_en.csv` and in comments next to them. The language-model prompts were issued in Chinese in the original runs; the prompts in the code are translations.

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Results in `results/`: CC BY 4.0 ([`results/LICENSE`](results/LICENSE)).

## Citation

See [`CITATION.cff`](CITATION.cff).
