# Instructions: Audit + extend `notebooks/model_diagnostics/residual_structure_comparison.ipynb`

**Status: DRAFT — not yet approved, do not execute.** This is not a new build — it's an audit of
an existing, uncommitted, actively-being-worked-on notebook (`git status` shows it modified,
last commit message "added more seeds for spatial split and am runnin baselines and pinn models
again"), plus specific additions/replacements, before trusting or extending it further.

## 0. Why this doc exists — read this first

This notebook is almost certainly the source of the "diagnostic report" pasted into this
conversation several turns ago (the plain-language RMSE/bias/compartment tables). At the time,
I flagged that report as unverified and asked to check its methodology before trusting it. This
audit is that check, done properly, on the actual notebook rather than the pasted prose.

**Headline finding, checked directly (`execution_count`/`outputs` on every code cell, not
assumed): the notebook is only partially executed.** Cells through "Compartment-Level Bias" and
"Spatial Residual Maps" have real outputs. Everything after that — **Worst-Error Concentration,
Spatial Autocorrelation, Can Environmental/Site Variables Still Predict The Leftover Residuals,
and the Model-Failure Interpretation Table** — has `execution_count: None` and zero outputs.
They've been *written*, not *run*. The interpretation table (cell 39) is a hand-authored
`pd.DataFrame` literal sitting in a code cell — it reads like a derived finding, but it is
currently prose someone typed, not a live result. Any number from that table (or attributed to
it) should be treated as an informed guess until the notebook is actually re-executed and the
table is rebuilt from real cell outputs.

This doesn't mean the notebook is bad — the opposite. The design (proper 5-fold CV via
`cross_val_predict` for residual predictability, `RidgeCV` *and* `RandomForestRegressor` compared
side by side, a real Moran's I/semivariogram check, worst-error concentration looped over every
core model rather than one) is more rigorous than several things built ad hoc this session,
including my own `env_deviation` residual-predictability check. It just hasn't been run to
completion, and one methodological issue needs fixing before it is (section 2).

## 1. What already exists (reuse, don't rebuild)

- **Model registry** (cell 3, `MODEL_RUNS`/`EXTRA_MODEL_RUNS`) — already wired up correctly,
  including `env_deviation_cr`/`env_deviation_dnn_noenv` (this session's build) and the
  `terrain_wind_extended`/physics-weight-sweep run-names from earlier in the session. Nothing to
  add here except the split-seed variants (section 3).
- **`load_predictions()`** (cell 4) — generic, handles missing paths gracefully with a warning
  rather than crashing. Reuse as-is for any new model added.
- **Worst-error concentration** (cells 22-26) — more thorough than the scratch script I ran
  earlier this session (loops over every core model, not just `dnn_noenv`; computes repeat-plot
  and repeat-compartment counts the same way). Once this notebook is re-executed, my scratch
  `outputs/spatial_block/dnn_noenv/4survey/worst_100_predictions.csv` becomes redundant — use
  this section's output as the authoritative version and drop the scratch file.
- **Spatial autocorrelation** (cell 33) — reuses `models/spatial_attribution/spatial_autocorrelation.py`
  (already-built project code, not reinvented here). Good practice, keep as-is.
- **Residual predictability** (cells 35-37) — `RidgeCV` + `RandomForestRegressor`, proper 5-fold
  CV via `cross_val_predict`. This is the CORRECT way to do what `env_deviation`'s
  `dnn_noenv`-residual variant approximated with a single val-holdout split — genuinely better
  methodology already sitting in this repo, unexecuted.
- **"Reading The Results" checklist** (cell 40) — a good, honest interpretive framework (does
  RMSE improve AND does spatial/environmental residual structure actually shrink, not just
  overall error). Keep this section's spirit; the table above it needs replacing (section 2).

## 2. What to replace, and why

### 2a. `mean_residual` is pooled per plot, averaged across years — same limitation `mean_cr_residual` has

Checked directly (cell 29): `plot_residuals = plot_df.groupby(["model","identification","cpmt"]).agg(mean_residual=("residual","mean"), ...)`.
Every downstream section (spatial maps, Moran's I, residual predictability) operates on this
POOLED residual, discarding age-varying structure exactly like `mean_cr_residual` does (the
issue already flagged and fixed for `env_deviation`'s own target construction). **Replace/add**:
build a row-level companion (`plot_df` itself, before the groupby, already has one row per
plot-survey-year with its own residual) and re-run the spatial-autocorrelation and
residual-predictability checks against it, alongside the pooled version, not instead of it —
worth knowing whether age-varying residual structure changes the answer, same as it did for
`env_deviation`.

### 2b. The `model_failure_interpretation` table (cell 39) needs to be derived, not hand-written

Once the notebook is re-executed end to end, rebuild this table's `current_evidence`/`status`
columns from the ACTUAL cell 22-37 outputs, not retype the current prose. Specific updates known
from this session's work, already more current than what's in the cell now:
- "Current environmental terrain features improve DNN spatial generalisation" — cell 39
  currently says "Mixed by seed... partly supported, not final." This is now **stale and
  understated**: the full 8-seed check (`documentation/experiment_log.md`'s 2026-08-03 entry)
  gives a 95% CI on the mean delta that excludes zero. Update to "supported, confirmed at n=8."
- "PINN is better than DNN" — cell 39 says "not supported" from the primary spatial-block
  comparison alone. Strengthen: this is now confirmed in **8 of 8** split seeds, not just the
  primary comparison — the single most robust finding of the whole investigation.
- **New row needed, not in the table at all**: PINN's own base-case accuracy (`pinn_noenv`,
  nothing to do with terrain) is ~10x more seed-volatile than DNN's (SD=0.0472 vs 0.0044,
  confirmed n=8) — a genuinely open, unexplained question surfaced this session, distinct from
  every terrain-related row already in the table.
- **New row needed**: "A leak-safe, locally-recoverable version of the neighbour-spatial-lag
  signal exists" — tested and found FALSE, not just unexplored. `spatial_block_split` holds out
  whole compartments, so 96% of held-out plots have zero legitimate train-neighbours within any
  reasonable radius — this is a structural incompatibility, not a missing feature to build.
- The existing "A few problematic compartments/plots drive many bad predictions" row (status:
  "supported") is consistent with this session's own worst-100 check (60% of worst rows from 3
  compartments, 32% repeat-offender plots) — once the notebook's own cells 22-26 are re-run,
  cross-check the exact numbers match (they should, same methodology, same underlying data).

## 3. What's genuinely new to add

### 3a. ~~A split-seed robustness section~~ — CORRECTION: already exists, already correct

Checked after re-executing the notebook fresh (section 4 step 1): cells 8-11
("Extra Experiment And Seed-Sensitivity Summary") already do exactly this — parametrized over
seeds 42-49, already handles `pinn_noenv`'s `pinn_noenv_basecase_w1` naming quirk correctly. Its
output (DNN delta mean=0.0417/SD=0.0287, PINN delta mean=0.0013/SD=0.0033) matches this
session's independently-computed numbers in `experiment_log.md` to 4 decimal places -- strong
cross-validation in both directions, nothing to add here. Original plan item was wrong; corrected
after checking rather than assumed.

### 3b. Cross-reference note for the kriging/leak-safe-spatial-lag investigation

Not a notebook section (no code was built for kriging yet) — just a markdown cell linking to
`documentation/experiment_log.md`'s 2026-08-02 entries on why the leak-safe spatial-lag feature
failed structurally, so a future reader of this notebook doesn't re-propose it without knowing
it's already been tried and diagnosed.

## 4. Order of operations

1. Re-execute the notebook top to bottom AS-IS first (no edits yet) — establish a clean baseline
   of what the existing, unmodified cells actually produce with current `outputs/` data, since
   several core model paths (`env_deviation_*`, `dnn_env_terrain_extended`) didn't exist when
   this notebook was last actually run partway through.
2. Add the row-level residual companion (section 2a) and re-run the spatial-autocorrelation /
   residual-predictability cells against both versions.
3. Add the split-seed section (section 3a).
4. Rebuild the interpretation table (section 2b) from the now-live cell outputs, not retyped
   prose.
5. Add the kriging cross-reference note (section 3b).
6. Re-execute fully via `jupyter nbconvert --to notebook --execute --inplace
   --ExecutePreprocessor.kernel_name=forest-diss` and confirm zero errors, per this project's own
   established notebook-editing convention.

## 5. Explicit "don't"s

- Don't treat cell 39's current content as a real finding in any write-up until it's rebuilt from
  live cells per section 2b — it's currently authored prose, not a derived result.
- Don't duplicate the worst-error concentration analysis with a new scratch script (already
  done once this session, redundant with cells 22-26) — re-run THIS notebook's version instead
  and retire the scratch CSV.
- Don't remove the pooled `mean_residual` version when adding the row-level one (section 2a) —
  both are informative for different questions, same as `env_deviation`'s two base-model
  variants.
- Don't add the split-seed section by hardcoding seeds 42-49 as a fixed list forever — make it a
  simple loop over whatever `_splitseed<N>` folders actually exist under
  `outputs/spatial_block/`, so it stays correct as more seeds get added later without a manual
  edit.

## 6. Finish criteria

- Every code cell in the notebook has a real `execution_count` and real outputs — zero
  `execution_count: None` cells remaining.
- The interpretation table's `current_evidence` column is derived from cells actually run in
  this notebook, cross-checked against the numbers already in `experiment_log.md`.
- A reader with no other context can open this one notebook and get the current, correct,
  seed-aware answer to every hypothesis in the interpretation table.
