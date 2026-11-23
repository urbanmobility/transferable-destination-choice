# Transferable destination choice

Code and city-level results for

> J. Yang†, Z. Huang†, S. Huang, P. Sun, S. Xu, Y. Jin, M. C. González, Y. Xu.
> *A transferable formula captures distance-dependent destination attraction across US metropolitan areas.*
> Submitted to EPJ Data Science (2026). † Equal contribution.

We model where residents go when they leave home as a conditional-logit choice among the census tracts of their
metropolitan area, and search for a compact utility by symbolic regression on 192.1 million home-based visits to points
of interest (POIs) from 41,688 home tracts in 67 US metropolitan areas (March–May 2024). The search, fitted to the raw
visit counts, found the **Reach model**

$$\eta_{ij} = a_j^{\theta_4} - \frac{\theta_0\, d_{ij}^{\theta_1}}{(N_j+\theta_2)^{\theta_3}},\qquad P(j\mid i)=\frac{e^{\eta_{ij}}}{\sum_k e^{\eta_{ik}}},\qquad \theta_2\ge 1,$$

with $d_{ij}$ the straight-line distance in km, $N_j$ the destination's POI count and $a_j$ its area in km². It combines
a destination-area term with a distance scale that expands with the POI count, so a destination's size matters more the
farther away it is — a regularity that a flexible model with no built-in link between distance and size also shows.
Fitted in each city, the Reach model closes a median 96% of the log-likelihood gap between population-based gravity
and a 22-parameter flexible benchmark. One parameter vector estimated in 35 discovery cities retained a median 94% of
that gap in 32 target cities without being fitted to their visits, and the Reach model outperformed the strongest
benchmark in 26 of the 32 cities, both fitted locally and transferred with this parameter vector.

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

`DEFAULT` is the uniform parameter vector of Additional file 1, Table S5. The shift is constrained to $\theta_2\ge 1$,
which keeps $N_j+\theta_2$ positive for tracts without POIs; `DEFAULT` satisfies the constraint, and the functions do
not enforce it. The default parameters need no fitting in the city where they are applied, but they need a measure of
destination activity: with the POI counts of this study, which are themselves obtained from visit records, they retain
a median share of 0.942 of the gravity-to-flexible gap in the 32 target cities; a version of the Reach model with
LODES workplace employment in place of the POI count, transferred in the same way, retains 0.84 (Additional file 1,
Table S7). Run the module's tests with `pytest reach_model`.

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

`results/` contains only city-level aggregates, no device-level, tract-level or tract-pair data:

- `city_level/` — per city: characteristics (`cities.csv`), NLL of every model (`fit_by_city.csv`), Reach-model
  constants with standard errors (`reach_model_constants.csv`), elasticities by distance band (`elasticities.csv`,
  with the variants of Additional file 1, Table S6: common reference point, employment, non-work and weekend visits,
  the flexible model with POI count and area only, and thirds of origins by distance to the CBD) and the urban-form
  relationships of Fig. 4a,b (`form_relations.csv`). These five files are the sheets of Additional file 2 of the
  paper. `ablation_by_city.csv` and `ablation_contrasts.csv` give the NLL and fit share of every model of the model
  comparisons per city, parameter source and input line, and their paired differences (Additional file 1, Table S7
  draws on them);
- `benchmarks/` — the 29 model versions (24 benchmark versions, the three formulas and two further expressions from
  the search) under four parameter sources in the 32 target cities, the selected opponents, the registered contrasts,
  and `seeds_bound.csv`: the Reach model and the strongest benchmark under five other assignments of destination
  blocks and five other samples of origins (Additional file 1, Table S11);
- `registry/` — the registered hypotheses and their held-out results (`registry_en.csv` gives all 25 in English, as in
  Additional file 1, Table S3);
- `freeze/` — the fixed formula library and the candidate pool it was drawn from;
- `figure_source_data/` — the city-level data behind the figures (the tract-level layers of Fig. 1b,c and the tract-pair
  flows of Fig. 6a are not included).

Fits of the Reach model. The Reach model's city-level constants, fits and elasticities in `city_level/`, and its fits
and those of its variants in the model comparisons (`sk`, `sk_E`, `sk_lna`, `sk_plus`, `sk_plus1`) and in the seed
checks, are constrained to $\theta_2\ge 1$. In the city fits the constraint is active in five of the 67 cities (10740,
15980, 46060, 19430, 40900), where $\theta_2=1$ has no standard error. `sk_frozen`, `tp_frozen` and `tr_frozen` are
the city fits of the three formulas as recorded when the library was fixed, and keep those values. The separable twins
(`sep_N`, `sep_E`) are not the Reach model and keep a free shift. A model whose utility is undefined on the scoring
block of a city has no NLL there.

## Language

Code, comments and reports are in English. Texts that became part of the registered hypotheses (statement and criterion templates in `multicity/laws.py`, `city_outcomes.py`, `lawtest_extra.py`, `v3_library.py`, `explore_p7.py` and `multicity/config/p7_covariates.json`) keep their original Chinese wording so that the code reproduces the registered records in `results/registry/`; English versions are in `results/registry/registry_en.csv` and in comments next to them. The language-model prompts were issued in Chinese in the original runs; the prompts in the code are translations.

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Results in `results/`: CC BY 4.0 ([`results/LICENSE`](results/LICENSE)).

## Citation

See [`CITATION.cff`](CITATION.cff).
