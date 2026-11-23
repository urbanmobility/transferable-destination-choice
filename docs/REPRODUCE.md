# Reproduction

Environment: `conda env create -f environment.yml` (PySR also installs Julia on first use; `sr/install_env.sh`).
Set `ODSR_ROOT` (data root; the repository lives in `$ODSR_ROOT/scripts`) and, for the search loop, the language-model
endpoint (`LLM_API`, `LLM_BASE_URL`, `LLM_MODEL`, and a key in `LLM_KEY_FILE` or `OPENAI_API_KEY`; there is no
default endpoint or key location; see `multicity/llm.py`). Long stages are run through `multicity/run_stage.sh SCRIPT STAGE [ARGS]`, which writes a log to
`results/multicity/logs/`. Each module's docstring lists its stages and outputs.

The stages, in the order they were run:

1. **City table and data build.** `cities.py`; `attrs.py download | build`; `osm_city.py download | extract | features`;
   `pipeline.sh shards homes trips poi attrs units qc osm units2 tiers form refs`.
2. **City groups.** `config/cities.csv` holds the roles (8 search, 6 confirmation); `split_rest.py` divides the other
   53 cities into 27 discovery and 26 held-out cities.
3. **Search.** PySR runs from `run_queue.sh config/queue_search*.txt` with `snap_hof.sh` (first rounds;
   `orchestrate.py` then selects, simplifies and fits the candidate library), followed by the iterative loop
   `loop.py run` (search rounds, candidate review, outer cycles that record and test cross-city hypotheses).
4. **Consolidation before the freeze.** `run_prefreeze.sh` (`prefreeze.py`), `banded.py`, `sensitivity.py`,
   `merge_variants.py`, `v3_library.py`, `lawtest_extra.py`.
5. **Freeze and confirmation.** `confirm.py freeze | refs | fit | rescore | banded | analyze | report`. After the freeze
   every stage that reads the test period or the confirmation cities calls `frozen.require()`, which refuses to run
   before the library is frozen or after a frozen input has changed.
6. **Second pre-specified set.** `form_p7.py`, `city_outcomes.py`, `explore_p7.py` (selection), `p7seal.py` (sign-off guard),
   `p8_test.py test | report`.
7. **Exploratory urban form.** `explore_form.py fix | e1 | e2 | axes_ta | report`, `city_types.py types | report`.
8. **Benchmarks and parameter transfer.** `p9.py fix | source | prerun | select | target | contrasts | report`,
   `p9_explore.py`, `alldest_check.py`, `od_maps.py`.
9. **Supplementary analyses.** `rev_models.py fit | transfer | report <line>` for the lines `all`, `nonwork`,
   `poitrain`, `all2`; `rev_banded.py run fixed | emp | empN | nonwork`; `rev_flows.py run`;
   `rev_weekend.py build | run`; `rev_matched.py run poiarea`; `rev_strata.py run`; `rev_seeds.py run`.
10. **The constraint $\theta_2\ge 1$.** The shift of the Reach model is constrained to $\theta_2\ge 1$. The stages above
    fit it without the bound; the following stages refit it with the bound wherever a fit falls below it:
    `rev_theta2.py run` (the city fits), `rev_theta2_full.py run` (their scores and elasticities),
    `rev_theta2_se.py run` (their standard errors), `rev_seeds.py run bound` (the seed checks, `seeds_bound.parquet`)
    and `rev_models_bound.py run <line>` for the lines `all`, `all2`, `nonwork`, `poitrain` (the model comparisons;
    output in `rev/bound/`). The frozen fits (`sk_frozen`, `tp_frozen`, `tr_frozen`) keep their recorded values, and
    the separable twins keep a free shift. With the bounded fits in place: `rev_seeds.py run allorig` (the formulas
    fitted on every origin, `seeds_allorig.parquet`) and `rev_round5.py decomp | knots | uncond | rstar`.
11. **Figures and tables.** `paper_figdata.py export` and a copy of `results/multicity/{p7,confirm,p9,refs_test,rev}`
    in `paper/figdata/rev/` (with `results/multicity/explore/e1_pairs.csv` as `explore_e1_pairs.csv`), taking for each
    line the files of `rev/bound/` (`models_<line>_*.parquet`, `summary_models_<line>.json`, `report_models_<line>.md`)
    in place of those of `rev/`; then, in
    this order: `paper/e1_tested.py`; `paper/rev_tables.py`; `paper/amend_theta2.py` and `paper/amend_theta2_se.py`,
    which write the bounded city fits into the exported tables (each runs once; keep a copy of the files they replace);
    `paper/rev_tables.py` again; `paper/make.py`, `paper/make.py tables`, `paper/figS1.py` and `paper/supp_tex.py`.
    `supp_tex.py` also reads three files that no stage above writes into `paper/`: `paper/tables/tableS_od_cities.csv`
    (from `make.py tables`), `results/multicity/qc/cities_qc.csv`, expected as `manuscript/sources/cities_qc.csv` next
    to the data root, and `paper/figdata/rev/city_controls_r5.csv`, which is `results/city_level/city_controls.csv` of
    this repository (definitions in [`DATA.md`](DATA.md)).

Unit tests: `pytest multicity/tests`, `pytest sr/tests`, `pytest paper/tests`, `pytest reach_model` (run each
directory separately). Tests that need the full data root are skipped or fail without it.

## Stage labels used in the code

Comments and log messages refer to the stages by number:

| Label | Stage |
|---|---|
| plan 1 | data pipeline (cities, visits, tracts, attributes, quality checks, reference models) |
| plan 2 | first search rounds and the candidate library |
| plan 4 | iterative search loop with candidate review and pre-specified cross-city hypotheses |
| plan 5 | consolidation before the freeze (rewrites, banded flexible model, selection rules) |
| plan 6 | freeze and confirmation ([`confirmation_criteria.md`](confirmation_criteria.md)) |
| plan 7 / plan 8 | selection, registration and held-out test of the second hypothesis set |
| plan 9 | benchmarks against earlier models and parameter transfer |
| revision | supplementary analyses (`rev_*.py`) |
