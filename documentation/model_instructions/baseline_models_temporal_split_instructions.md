# Instructions: Temporal split for the four existing baselines

**Status: DRAFT — not yet approved, do not execute.** Adds a third generalization test for
Chapman-Richards, average-by-age, linear regression, and random forest, alongside the two already
built this session:

| Split | Question it answers | Status |
|---|---|---|
| `plot_level_split` | Can the model interpolate between plots it's seen the general neighbourhood of? | Built, all 4 baselines |
| `spatial_block_split` | Does the model generalise to genuinely unseen forest compartments? | Built, all 4 baselines (`baseline_results.ipynb` section 8) |
| `temporal_split` | Does the model predict a real future survey it never trained on? | **This document** |

Same reasoning as the spatial-generalization work: today's model-comparison table is only honest
about *one* kind of generalization at a time. A model that looks great under `plot_level_split`
might be quietly relying on interpolation, spatial proximity, or just memorising "what 2021 looked
like" rather than learning something that holds up in a genuinely future survey. Running all three
splits on the same four models is what lets a claim like "RF generalises worse" or "the PINN is
more robust" mean something specific, rather than being true only for whichever split happened to
be tested.

## 1. What already exists (reuse, don't rebuild)

- **`temporal_split()`** — `models/common/splits.py`. Already implemented and tested
  (`models/common/test_splits.py::test_temporal_split`). Unlike `plot_level_split` and
  `spatial_block_split`, the same plot is *expected* to appear in both train and test here — that
  is not a leak, it's the whole point (the question is about *time*, not about unseen plots or
  unseen places). The function already prints a warning if any survey year present in the data
  isn't assigned to train/val/test, and already handles an unspecified `val_years=None`.
- **Year assignment, already decided and tested** — currently only defined locally inside
  `test_splits.py` as `TEMPORAL_YEARS`:
  ```python
  TEMPORAL_YEARS = {
      "4survey": {"train_years": [2008, 2012], "val_years": [2021], "test_years": [2023]},
      "6survey": {"train_years": [2002, 2006, 2008, 2012], "val_years": [2021], "test_years": [2023]},
  }
  ```
  This holds out the most recent survey (2023) as test and the second-most-recent (2021) as
  validation — confirmed this session as the year assignment to use, rather than folding 2021 into
  training. **Implementation note**: this constant currently lives in a test file. Move it to
  `models/common/splits.py` (or a small new shared constants location) so both this baseline work
  and the later DNN/PINN work read the exact same year assignment, rather than two copies that
  could silently drift apart.
- **Every plot has full year coverage** — confirmed directly against the actual data: 100% of
  plots in both cohorts appear in every survey year of their cohort (71,766/71,766 for 4survey,
  13,897/13,897 for 6survey — the cleaning pipeline already balances each cohort to exactly N
  surveys per plot before export). This means `temporal_split`'s train/val/test row counts will be
  exactly proportional to how many survey years fall in each bucket — there's no risk of a
  lopsided split from partial plot coverage.
- **`filter_data()`, `output_dir()`, and the `--split-type` pattern** — already built for
  `spatial_block_split` in `models/baselines/run_baselines.py` and `evaluate_baselines.py` this
  session. Extend the same pattern rather than writing new scripts: `output_dir()` gets a third
  branch (`split_type == "temporal"` → `outputs/temporal/...`), `build_split_for_cohort()` gets a
  third branch calling `temporal_split(df, year_col="LiDAR_year", **TEMPORAL_YEARS[cohort])`, and
  `--split-type` gains `"temporal"` as a third `choices` value in both scripts' `argparse` setup.
  `filter_data()` itself needs no change — it already runs before the split is computed, same as
  for the other two split types.

## 2. What's genuinely new here

- Wiring `temporal_split()` into `run_baselines.py`/`evaluate_baselines.py`'s `--split-type`
  branch (the function itself needs no changes, just the orchestration).
- Moving `TEMPORAL_YEARS` out of the test file into shared, importable code.
- A new comparison section in `baseline_results.ipynb` (or a clearly-scoped addition to the
  existing section 8) putting all three split types' results for all four baselines side by side.

## 3. Notebook section — what to add

Mirror section 8's structure (comparison table with `%change` from `plot_level_split`, a rank
table, a bar chart, and a plain-language interpretation cell) but extend it to three columns
(`plot_level`, `spatial_block`, `temporal`) rather than two. One thing to call out explicitly in
the interpretation text, since it's easy to misread: **a plot appearing in both a temporal split's
train and test rows is expected and correct**, not a leak — this is the opposite convention from
`spatial_block_split`'s buffering, and worth a one-line reminder in the notebook so it doesn't get
flagged as a bug later.

Given you mentioned wanting a map in future notebooks: a natural one for **this specific split**
would be a compartment-level map (reusing `models/common/geo.py::load_plot_coordinates()`, already
used for the spatial buffer) colored by each compartment's *temporal* degradation (e.g. RF's RMSE
change from `plot_level_split` to `temporal_split`, per compartment) — that would show whether
temporal generalization failure is spatially clustered (e.g. concentrated in one part of the
forest that changed unusually between 2021 and 2023) or spread evenly. Flagging this as an idea
for the map you'd like, not committing to build it yet — say if you want it in this pass or later.

## 4. Small fix worth bundling in

`TRANSITION_COLUMNS` in `data_processing/export_model_tables.py` currently includes `blk` but not
`cpmt` (unlike `METADATA_COLUMNS`, which already has both). If the trajectory/transition table is
going to get more use now (both here conceptually and directly in the DNN/PINN work — see
`age_only_dnn_pinn_instructions.md`), it should carry `cpmt` too for consistency with every other
exported table. One-line change, no behavioural risk.

## 5. Explicit "don't"s

- Don't change `plot_level_split` or `spatial_block_split` results — this is purely additive, a
  third split type alongside the two that already exist.
- Don't refit CR/average-by-age/linear/RF's *hyperparameters* — same models, same fitting code,
  only the train/val/test row assignment changes.
- Don't touch the test years (2023) for anything other than final evaluation, same rule as every
  other split.

## 6. Finish criteria

- `outputs/temporal/<model_name>/<cohort>/` populated for all four baselines, same file format as
  the existing `plot_level` and `spatial_block` outputs.
- Notebook section showing all three split types side by side for all four baselines.
- `TEMPORAL_YEARS` living in shared code, not duplicated between `test_splits.py` and
  `run_baselines.py`.
