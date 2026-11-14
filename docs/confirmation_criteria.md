# Confirmation criteria

Fixed together with the formula library before the six confirmation cities and the test period were opened
(`multicity/confirm.py`, stage `freeze`).

## Constraints

- Before the freeze (while `results/multicity/FROZEN` does not exist) no code reads the test period, scores on
  destination fold 0 or fits a confirmation city; `frozen.require()` raises before the freeze.
- After the freeze no rule, library member or parameterisation changes and no new search is run; problems are recorded
  and reported.
- Selection rules (V3): margin for conclusions `scoring.MARGIN = 0.03` (primary), 0.05 as a supplement.
- Evaluation set: the 26 candidates in `prefreeze_mv/discovery/cities_val.parquet`; "near-best" is relative to the best
  of these 26 in the same city and scoring block.
- Constants are fitted on the training period × search blocks (folds 2–4), exactly as before the freeze. Test scoring:
  **double holdout** (primary): test period × fold 0, unseen destinations and period; **time-only holdout**: test period
  × fold 1.
- Cities: all 67 eligible cities (8 search, 27 discovery, 26 held-out, 6 confirmation).
- No language-model calls.

## Confirmation scheme

**Frozen library** (`multicity/config/frozen_library.json`):

| cid | role |
|---|---|
| c0355r1 | main (Reach model) |
| c0262 | second (Tempered power law) |
| c0514c1 | optional (Two-regime power law; counted in the clean library) |
| c0413c1r1 | comparison (variable-exponent control; fitted and reported, not counted in clean coverage) |
| c0079c1 | reference (shifted log-gravity control; reference only) |

The clean library = roles main, second and optional.

**P1 Library coverage** (descriptive): number of confirmation cities (k/6) in which a clean-library member is near-best
and better than the floor models; primary scoring double holdout, margin 0.03; also 0.05 and the validation scoring;
Clopper–Pearson 95% intervals beside the discovery and held-out coverage.

**P2 Test-period performance** (criterion): share_S0 of each library member on the double holdout. Median over the six
confirmation cities and the 10th–90th percentile range over the 35 discovery cities. **Met** when the confirmation
median lies within the discovery range.

**P3 Transfer vs local refit** (descriptive): for each member and confirmation city, share_S0 on the double holdout with
(i) local constants and (ii) the element-wise median of the 35 discovery cities' constants; cities where (ii) gives a
non-finite utility are counted separately.

**P4 Supported laws**:
- **H3-02** (criterion): the flexible model's slope of e_N on ln d is positive. **Met** when it holds in at least 5 of
  the 6 cities.
- **H1-01, H3-01, H4-01, H4-02, H4-03, H2-03** (one criterion each): fit OLS of the pre-freeze quantity on the
  hypothesis's city characteristic over the discovery and held-out cities; give each confirmation city's 95% prediction
  interval. **Met** when at least 5 of 6 confirmation cities fall inside. The slope within the confirmation cities is
  reported descriptively (n = 6).

**P5 Elasticity profiles** (descriptive): the comparison of e_d and e_N profiles with the flexible model, repeated in
the confirmation cities.

Outcome: all 12 criteria (5 under P2, 7 under P4) were met (Additional file 1, Table S3b).
