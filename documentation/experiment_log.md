# Experiment Log

A factual record of what was actually run, when, and why — separate from
`documentation/model_instructions/`, which describes intent (what to build)
rather than outcome (what happened). Check here before citing a result in the
write-up; check the instructions docs before building something new.

**How to use this file**: add one row per experiment configuration the first
time it's run. If a later run reruns the exact same configuration (e.g.
re-running after a code fix), update that row's date/output location rather
than adding a new one — a new row means a genuinely different setting
(different split years, different CR fit, different model), not a rerun.

**Dates include time of day, from 2026-07-18 onward** (e.g. `2026-07-18 14:30`), not just the
date — useful once multiple entries land on the same day, which was already happening.

## Naming glossary

Two short-named choices come up repeatedly below — named for what they actually are,
not numbered, so you don't have to look up which is which:

- **`temporal_wide_gap`** vs **`temporal_narrow_gap`** — two ways to set up
  `temporal_split`'s train/val/test years. `temporal_wide_gap` (current primary)
  trains on the earliest years only and tests 11 years later (2012→2023) — the
  harder extrapolation. `temporal_narrow_gap` (run 2026-07-20, baselines + DNN +
  PINN, both cohorts — see the 2026-07-20 Findings entry) trains through 2021 and
  tests only 2023 — a 2-year gap, closer to interpolation. Confirmed: gap length,
  not "temporal prediction in general," drives most of `temporal_wide_gap`'s
  degradation (11 of 12 baseline/DNN/PINN model×cohort combinations improve under
  the shorter gap).
  **Year assignment isn't simply "wide_gap's val year moves into train"**: `2021`
  must be in train for the gap to be 2 years, but the PINN's trajectory loss also
  needs a pair of chronologically ADJACENT real surveys both labelled train (the
  transitions table only has adjacent-survey pairs, e.g. `2012→2021`, never
  `2008→2021`) — so `val_years` holds out the EARLIEST available pre-test year, not
  a middle one, keeping every later adjacency intact. A first attempt at this (val
  in the middle of the sequence) silently gave PINN 0% usable trajectory pairs and
  it failed outright, while DNN and the baselines ran "successfully" on the same
  years since neither uses trajectory pairs — see `TEMPORAL_YEARS_NARROW_GAP` in
  `models/common/splits.py` for the actual values
  (`4survey`: train=[2012,2021], val=[2008]; `6survey`: train=[2006,2008,2012,2021],
  val=[2002]; test=[2023] both cohorts).
- **`cr_pooled`** vs **`cr_matched`** — two ways to fit the PINN's frozen
  Chapman-Richards physics anchor. `cr_pooled` (currently used everywhere) fits CR on
  the plot_level split (all years, all plots pooled), regardless of which split the
  PINN itself trains under. `cr_matched` (planned, not yet run) restricts the CR fit
  to only the PINN's own training years — a stricter ablation with zero foresight
  into val/test years.

## Status legend

- **primary** — the dissertation's central result: `spatial_block_split`.
  Decided 2026-07-16: the planned environmental covariates (terrain, elevation, wind
  exposure) are static per plot, not something that varies year-to-year, so "does
  environment explain growth better than physics alone" is inherently a spatial
  question, not a temporal one. `spatial_block_split` is the split that actually
  tests that.
- **secondary** — a real, separate research question (temporal generalization,
  `temporal_split`'s SQ2 gap-length questions), worth reporting in its own right, but
  not the central test the dissertation is built around.
- **robustness-check** — run specifically to test whether a primary or secondary
  result is sensitive to a design choice, per the plan below.
- **superseded** — an earlier config later replaced by a corrected/updated
  one (kept in the table for the record, not deleted).

## Planned robustness checks (not yet run)

Recorded here before running them so the *intent* is on record even before there's a
result. `spatial_block_split` is the primary experiment (see Status legend above) —
`temporal_split` stays in as a real secondary research question (its own SQ2
gap-length questions), not something being phased out; both are worth reporting,
just not equally central.

| Planned experiment | Question it answers | Trigger to run it |
|---|---|---|
| PINN physics anchor, `cr_matched` (CR fit restricted to 2008+2012 only, matching the PINN's own training years) | Does the DNN-vs-PINN temporal comparison hold up under a stricter ablation with zero information leakage into the physics term? | Cheap to run alongside `temporal_wide_gap` results — worth doing early as a documented caveat |

**Resolved 2026-07-17** (removed from the table above, kept here for the record): "Hyperparameter variants of the primary `spatial_block` PINN run — is the result genuine convergence or a premature stop?" — yes, genuine convergence. Loosening patience/smoothing/batch size trained both models roughly 2x longer but barely moved final test RMSE (see the 2026-07-17 Findings entry) — the pre-tuning numbers were already close to real optima, except DNN/4survey's overfitting climb, which only softened, not fixed (a data-limitation, not a tuning-limitation — see the same entry).

**Resolved 2026-07-20** (removed from the table above, kept here for the record): "Is `temporal_wide_gap`'s dramatic degradation an artefact of the 11-year gap, or does it hold under a shorter gap too?" — mostly the former: gap length explains most of it. 11 of 12 baseline/DNN/PINN model×cohort combinations degrade less under `temporal_narrow_gap`'s 2-year gap (see the 2026-07-20 Findings entry) — validates `temporal_wide_gap` as the harder, more discriminating primary temporal test, not an artefact of these specific years.

**Resolved 2026-07-18** (removed from the table above, kept here for the record): "Does the same low-weight finding (`W=0.05` beats `W=1.0`) hold under `temporal_split` too, or is it `spatial_block`-specific?" — mostly yes, with a genuine nuance. Swept the same 7 values under `temporal`, both cohorts (see the 2026-07-18 Findings entry): 4survey's optimum is again `W=0.0`, same as `spatial_block`; 6survey's optimum shifts to `W≈0.1–0.2` rather than `spatial_block`'s `W=0.05`, and `W=0.05` itself is a wash (not a clear win) vs. the old `W=1.0` default for 6survey specifically under `temporal`. Kept the shared `W=0.05` default regardless — see Decisions log.

## Experiment table

| ID | Date | Split design | Years (train / val / test) | CR fit used | Models | Cohorts | Status | Output location | Result summary |
|---|---|---|---|---|---|---|---|---|---|
| `xgb_elasticnet_environmental_2026-07-29` | 2026-07-29 | `spatial_block_split` (train/val/test all used; val for feature decisions, test read once) | n/a (all years pooled per plot, mean CR residual target) | cr_pooled: plot_level CR fit (re-derived on `elev_percentile_95th`) | xgb_environmental (XGBoost+SHAP), elasticnet_environmental (ElasticNetCV) | both (4survey primary) | primary (replaces the retired-pipeline row) | `outputs/spatial_block/xgb_environmental/<feature_set>/<cohort>/`, `outputs/spatial_block/elasticnet_environmental/<feature_set>/<cohort>/` | Re-run against the new target + `yldc` removed from the 34-variable unified environmental+silviculture feature set (`Age` still excluded — circular with the CR residual, see Findings log). 4survey `all_environmental`: XGBoost val R²=0.734/test R²=0.629, Elastic Net val R²=0.700/test R²=0.671. 6survey `all_environmental`: XGBoost val R²=0.107/test R²=0.398, Elastic Net val R²=0.147/test R²=0.521 — 6survey's val R² sits well below its own test R² across every feature set for both model types (opposite of the usual overfitting direction); likely which compartments `spatial_block_split` happened to assign to val vs test for the smaller cohort, not yet investigated further. 4survey grouped permutation importance (`grouped_category_importance.ipynb`): `neighbour_spatial_lag` dominates (mean R² drop=1.177, ~10x every other category), then climate/stand_structure/terrain clustered close together (0.11-0.12), wind least (0.023). Full-model residual Moran's I=0.197 (p=0.005) -- still significant spatial autocorrelation left unexplained; removing `terrain` increases it the most (Δ=0.066), even though terrain isn't top for raw accuracy -- a genuine cross-method disagreement (matters for spatial pattern, not for prediction). |
| `baselines_rebuild_2026-07-28` | 2026-07-28 | `plot_level`, `spatial_block`, `temporal` (wide-gap), `temporal_narrow_gap` -- all four re-run | same year assignments as the retired-pipeline rows above | n/a | CR, average-by-age, linear, RF | both | primary (replaces every baseline number above) | `outputs/<split_type or nothing>/<model>/<cohort>/` | New target (`elev_percentile_95th`) + `yldc` removed from RF/linear. `plot_level`: RF best (R²=0.570) as before. `spatial_block`: RF loses its advantage to linear (R²=0.475 vs 0.512) as before -- same qualitative pattern as the retired pipeline, confirming the rebuild didn't change which baseline "wins" per split, just the absolute numbers. Chapman-Richards fit also fixed a pre-existing degeneracy (y_max was landing exactly on the observed max height under both old and new target) -- lower bound now `max_observed_height * 1.001`. Full reasoning: `progress_notes.md`'s 2026-07-28 entry |
| `dnn_pinn_epochcheck_2026-07-29` | 2026-07-29 | `spatial_block` | 4survey only, short smoke tests (max 150 epochs, patience 40) | cr_pooled | dnn_noenv, pinn_noenv, 3 PINN weight variants | 4survey only | diagnostic, not a result -- see Findings log | `outputs/spatial_block/{dnn,pinn}_noenv_epochcheck*/4survey/` | Base-case (`W=1.0`) DNN/PINN cluster jobs came back suspiciously fast (~53s, later traced to a missing rsync of `data/processed/transitions/`, fixed). Once fixed: DNN converges normally (val_loss 0.342→0.331 over ~50 epochs, patience stops at 52). PINN never beats its own epoch-1 val_loss at `W=1.0`, `W=0.0`, OR `W=0.05` (best_val_loss ≈ epoch-1's value in all three) -- ruled out physics weight as the cause. Found the real confound: `pinn_noenv.py`'s `BATCH_SIZE=128` vs `dnn_noenv.py`'s `BATCH_SIZE=512`, undocumented, never controlled for. Exposed `--batch-size`/`--pairs-batch-size` as CLI args (previously hardcoded) to test batch-size-matched. Next: rerun `physics_weight=0.0` at `--batch-size 512` on the cluster (see below) to isolate batch size from the physics-weight question properly. |
| `dnn_pinn_basecase_2026-07-30` (Stage 2) | 2026-07-30 | `spatial_block`, `temporal` | full pipeline, `batch_size=256` both models | cr_pooled | dnn_noenv, pinn_noenv (`physics_weight=trajectory_weight=1.0`, the untested default) | both | primary | `outputs/{spatial_block,temporal}/{dnn_noenv,pinn_noenv_basecase_w1}/<cohort>/` | Real base-case rebuild against the current pipeline (finally superseding the epochcheck smoke tests). Test R²: 4survey spatial_block DNN=0.633 vs PINN(w=1)=0.580; 6survey spatial_block DNN=0.750 vs PINN(w=1)=0.734; 4survey temporal DNN=0.354 vs PINN(w=1)=0.284; 6survey temporal DNN=0.284 vs PINN(w=1)=0.209. DNN beats PINN(w=1) by a real, consistent margin in all 4 cohort×split combinations -- confirms the long-believed "physics constraint hurts at full weight" finding (point 3 in `handover_2026-07-18.md`) survives the full target/yldc/batch-size rebuild, not just true under the retired pipeline. |
| `pinn_physics_weight_grid_2026-07-30` (Stage 3) | 2026-07-30 | `spatial_block` | 4survey: full 10-point `physics_weight`x`trajectory_weight` grid; 6survey: winning config + `0/0` control only | cr_pooled | pinn_noenv | 4survey (full grid), 6survey (2 configs) | primary | `outputs/spatial_block/tune_pinn_{4s,6s}_w*/<cohort>/` | Lowest `best_val_loss_smoothed` on 4survey: `pw=0.1,tw=0.0` (0.3292), essentially tied with `pw=0.01,tw=0.0` (0.3294), `pw=0.0,tw=0.01` (0.3300), `pw=0.1,tw=0.01` (0.3299), and the `0/0` control (0.3308) -- all five within the same ~0.002 noise band Stage 1's batch-size sweep already established. Any `trajectory_weight>=0.1` clearly hurts (0.338-0.350). Picked `pw=0.1,tw=0.0` as "the winner" per the pre-agreed protocol (lowest val loss) and confirmed on 6survey (0.3292-equivalent tie holds there too). **But held-out test R² does not confirm this pick is real signal**: 4survey test R² is 0.6319 (winner) vs 0.6337 (`0/0` control) -- control is *slightly better*; 6survey test R² is 0.7468 (winner) vs 0.7483 (control) -- same direction, control slightly better again. See Findings log entry for what this means before spending Stage 4 cluster time on this weight pair. |
| `dnn_pinn_final_seeded_2026-07-31` (Stage 4) | 2026-07-31 | `spatial_block`, `temporal`, `temporal_narrow_gap` | full pipeline, `batch_size=256` throughout | cr_pooled | dnn_noenv, pinn_noenv (`pw=0.1,tw=0.0` and `pw=0.0,tw=0.0`, both arms from the Stage 3 decision) | both | primary (`spatial_block`) / secondary (`temporal`, `temporal_narrow_gap`) | `outputs/{spatial_block,temporal,temporal_narrow_gap}/final_{dnn,pinn,pinn_w0}_seed{42-46}/<cohort>/` | 90/90 fits completed (5 seeds x 3 arms x 3 splits x 2 cohorts). All pass the early-stopping sanity check. `temporal`/4survey (15 runs, every seed and arm) shows 0% val-loss improvement from epoch 1 -- a genuine train/val collapse driven by that cell's narrow 2-year training window, not a bug -- see 2026-07-31 Findings entry; excluded from the seed-averaged comparison table. Evaluate loop (`evaluate_dnn_noenv.py`/`evaluate_pinn_noenv.py`) not yet run to get test-set metrics -- see `experiments_to_run.txt`. |

## Findings log (what I found → what's working / not → what it means for next steps)

This is the section an MSc examiner actually wants to see: not just "here are some
numbers" but the reasoning chain — what a result showed, whether it changed what you
believed, and what you did about it. The Experiment table above is the raw record;
this is the narrative connecting one entry to the next.

**How to use this**: add one dated entry whenever a run produces a result that changes
your understanding or your plan — not for every routine run (that's what
`outputs/run_logs/` and the Experiment table are for). Use this four-part shape every
time, even in short form:

> **[date] — [one-line headline of the finding].**
> **What I found:** the actual result, with numbers.
> **What's working:** what this confirms is sound / trustworthy.
> **What's not working / open concern:** what this exposes as a problem, risk, or
> unanswered question.
> **What this means for what's next:** the concrete decision or next step this
> directly caused — this is the line that turns a result into a research narrative.

---

**2026-08-02 — `dnn_env_terrain`/`pinn_env_terrain` E1+E2 base-case results in (all 3 splits, both
cohorts); two real bugs found and fixed along the way.**
**What I found:** All 12 base-case fits (2 models x 3 splits x 2 cohorts, `terrain_wind_solid`,
batch_size=256, dropout=0, seed=42) were already on disk. Ran the remaining evaluates
(`temporal`/`temporal_narrow_gap`, both cohorts) myself. Full test R2 table:

| Split | Cohort | dnn_env_terrain | pinn_env_terrain (w=1.0) |
|---|---|---|---|
| spatial_block | 4survey | 0.6247 | 0.5823 |
| spatial_block | 6survey | 0.7415 | 0.7329 |
| temporal | 4survey | 0.5021 | 0.2934 |
| temporal | 6survey | 0.2944 | 0.1666 |
| temporal_narrow_gap | 4survey | 0.6728 | 0.4533 |
| temporal_narrow_gap | 6survey | 0.5208 | 0.3204 |

DNN beats PINN in all 6 -- same "physics hurts at w=1.0" pattern the no-env pipeline already
found, now reproduced with terrain conditioning added.

Two real bugs surfaced and fixed:
1. `load_split_table_with_terrain()` (`models/common/torch_data.py`) hard-errored whenever
   `whcl` was requested (`terrain_wind_extended`/`broad` feature sets) -- `model_table.parquet`
   already has its OWN `whcl` column (different from the environmental-features one), so merging
   silently renamed both to `whcl_x`/`whcl_y` instead of erroring at the merge itself; the
   `KeyError` only surfaced later, far from the real cause. Fixed by dropping any pre-existing
   colliding column from `split_df` before merging -- a general fix, not whcl-specific. Also
   fixed in the same function: `ceh_twi` is genuinely missing for 39/71,766 plots (not a merge
   bug) -- was a hard `ValueError`, now drops the whole affected plot with a warning, matching
   `filter_data()`'s existing whole-plot-dropped convention.
2. `evaluate_pinn_env_terrain.py` read the global y_max anchor from the OLD unprefixed, pooled
   `outputs/chapman_richards/<cohort>/params.json` path instead of the split-matched
   `outputs/<split_type>/chapman_richards/<cohort>/params.json` path `run_pinn_env_terrain.py`
   actually trained against. Did NOT affect any height prediction or R2/MAE/RMSE/Bias metric
   (`predict()` never uses `global_y_max`) -- only the `learned_y_max` column and the printed
   anchor. Harmless for `spatial_block` (pooled and matched y_max happen to be nearly identical,
   checked directly) but wrong for `temporal`/`temporal_narrow_gap`, where they differ by 3-11%
   (e.g. 4survey temporal: pooled 51.96m vs matched 46.48m).
**What's working:** Both fixes verified directly (re-ran the loader/evaluate before and after,
confirmed the exact failure mode and the fix). All three `ENV_TERRAIN_FEATURE_SETS` now load
successfully end to end (`broad` drops far more plots than the other two -- 430 plots/1,720 rows
-- from its climate/soil columns' own missingness, not a bug).
**What's not working / open concern:** `pinn_env_terrain`/`temporal_narrow_gap`/6survey's fit ran
on CPU, not GPU (11 of 12 real fits used `cuda`, confirmed via `outputs/run_logs/`) -- a cluster
job-scheduling anomaly, not a code bug. Not yet re-submitted.
**What this means for what's next:** any `learned_y_max` analysis (e.g. comparing the PINN's
learned map against NLME/XGBoost's terrain findings) must use checkpoints fit/evaluated AFTER
this fix, especially for `temporal`/`temporal_narrow_gap`. Re-submit the one CPU-fallback job for
timing/resource consistency, not correctness (the R2 result itself isn't expected to change).

---

**2026-08-02 — Dropout/learning-rate diagnostic on `dnn_noenv` (spatial_block/4survey): null
result on both, motivating an architecture-capacity check next.**
**What I found:** Training curves for `dnn_noenv`/`pinn_noenv`/`dnn_env_terrain` all show the
same shape -- train_loss keeps decreasing smoothly through the whole run while val_loss plateaus
within the first ~10-25% of epochs, then drifts slightly worse (classic overfitting signature,
not a stuck-optimizer signature -- ruled out via direct inspection of `training_history.csv`,
e.g. `dnn_noenv`/spatial_block/6survey: val_loss drops for ~8 epochs then oscillates 0.206-0.220
for the remaining 44). Added `--dropout-rate`/`--learning-rate` CLI flags to `dnn_noenv.py` (not
previously exposed -- `learning_rate`/`dropout_rate` were hardcoded) and swept both on the
primary reporting config:

| Config | best_val_loss |
|---|---|
| baseline (dropout=0, lr=0.0001) | 0.329386 |
| dropout=0.1 | 0.330622 |
| dropout=0.2 | 0.334303 |
| dropout=0.3 | 0.338473 |
| lr=0.0003 | 0.329747 |
| lr=0.001 | 0.329485 |

**What's working:** Dropout makes best_val_loss monotonically WORSE (not better -- the
regularization hypothesis was wrong). Neither learning-rate variant moved best_val_loss beyond
noise (differences ~0.0001-0.0009, similar magnitude to the curves' own epoch-to-epoch swings).
**What's not working / open concern:** Neither of the two cheapest, most likely levers
(regularization strength, optimizer step size) explains the fast-plateau pattern. This was
checked BEFORE assuming architecture capacity was the answer, not instead of checking it.
**What this means for what's next:** ruling out dropout/LR motivated the architecture-size sweep
below, rather than jumping straight to "make the network bigger" on the training-time
observation alone.

---

**2026-08-02 — Architecture-size sweep (`--hidden-layer-sizes`, newly wired into all 4
DNN/PINN models): also a null result -- capacity is not the bottleneck on spatial_block.**
**What I found:** Added a `hidden_layer_sizes` parameter to `models/common/torch_model.py`'s
`NoEnvNetwork` (backward-compatible: `None` default preserves the exact original 3x128
structure and parameter names, so every existing checkpoint still loads with zero change --
verified directly against a real checkpoint before proceeding). Wired `--hidden-layer-sizes`
through all four models' fit/evaluate scripts. Ran 4 architecture variants (small [64,32],
medium [128,64], large [256,128,64], deeper [256,128,64,32]) on spatial_block/4survey:

| Model | Architecture | best_val_loss |
|---|---|---|
| dnn_noenv | 128x128x128 (baseline) | 0.32939 |
| dnn_noenv | small | 0.33124 |
| dnn_noenv | medium | 0.33062 |
| dnn_noenv | large | 0.33183 |
| dnn_noenv | deeper | 0.33122 |
| pinn_noenv (w=1.0) | 128x128x128 (baseline) | 0.39712 |
| pinn_noenv (w=1.0) | small | 0.39638 |
| pinn_noenv (w=1.0) | medium | 0.39549 |
| pinn_noenv (w=1.0) | large | 0.39702 |

Stopped here (user call) before the pinn_noenv/deeper run and before repeating on 6survey --
the pattern across 7/8 runs was already clear.
**What's working:** All 4 architectures are within noise of the current 128x128x128 baseline,
for both models. Confirms (doesn't just assume) that DNN-beats-PINN and terrain-not-helping on
spatial_block is a real pattern, not an artefact of an underpowered network shape.
**What's not working / open concern:** Sweep incomplete (7/8 planned runs; no 6survey; no
env_terrain models) -- not resumed, superseded in priority by the loss-weight finding below.
**What this means for what's next:** architecture size is not the lever to pursue for
`pinn_env_terrain`'s underperformance vs. its DNN control. Loss-weight tuning (already flagged
as untested for env_terrain specifically) is the more promising remaining lever -- see next
entry.

---

**2026-08-02 — `pinn_env_terrain` physics-weight sweep (spatial_block/4survey) finds a genuine
trade-off: the weight needed for the y_max sub-network to learn anything also costs accuracy.**
**What I found:** Swept `physics_weight` in {0.0, 0.01, 0.1, 1.0} with `trajectory_weight` fixed
at 0.0 (informed by the no-env pipeline's own Stage 3 finding that `trajectory_weight>=0.1`
reliably hurts), plus the existing 1.0/1.0 checkpoint from E1:

| physics_weight | trajectory_weight | best_val_loss | test R2 | learned y_max range |
|---|---|---|---|---|
| 0.0 | 0.0 | 0.3302 | 0.6333 | 51.96-51.96 (flat) |
| 0.01 | 0.0 | 0.3303 | 0.6335 | 51.96-51.96 (flat) |
| 0.1 | 0.0 | 0.3304 | 0.6323 | 51.92-51.92 (nearly flat) |
| 1.0 | 0.0 | 0.3352 | 0.6195 | 45.11-51.71 (real spread) |
| 1.0 | 1.0 | 0.3955 | 0.5823 | 52.43-61.39 (real spread) |

**What's working:** The y_max sub-network genuinely gets zero gradient at low physics_weight
(learned_y_max stays exactly at the global anchor, not just "close" -- confirms
`pinn_env_terrain.py`'s own module-docstring prediction). At physics_weight=1.0 it produces a
real ~6.6m spread, confirming the mechanism works when given enough weight.
**What's not working / open concern:** The SAME weight that makes the sub-network responsive
also costs accuracy (R2 0.633 -> 0.620 -> 0.582 as weight/trajectory_weight increase) -- this
is a real trade-off, not simply "physics doesn't help." Discussed with the user whether this is
because the physics loss forces the network's derivative to match a RIGID single-global-`k`/`p`
Chapman-Richards curve (only `y_max` varies per plot; the deviation is a static per-plot shift,
never a function of age) -- a plausible model-misspecification explanation, distinct from a
pure loss-scaling issue. Verified via web search that this maps to a real, named framework:
Universal Differential Equations (UDEs) / Universal PINNs (UPINNs), where a learned residual is
added to a known physics term and can vary continuously with the independent variable (age) and
covariates (terrain) -- not yet implemented, no code changed for this.
**What this means for what's next:** the user is taking a detailed research prompt (covering
UDEs/UPINNs, self-adaptive PINN loss weighting as a competing explanation, and relevant
forestry/plant-growth PINN literature) to a separate research session before any architecture
change is attempted. Do not implement a residual/deviation-based y_max mechanism without that
follow-up -- this entry is the state to resume from.

---

**2026-08-02 — Feature-set-parity check (`terrain_wind_extended` vs. default `terrain_wind_solid`)
on `dnn_env_terrain`/`pinn_env_terrain`: another null result -- rules out "wrong columns" as the
explanation for the DNN's own regression.**
**What I found:** `terrain_wind_solid` (the default feature set both env_terrain models use) is
only 5 of the 16 columns `grouped_category_importance.ipynb`'s terrain+wind category analysis
used to establish "terrain/wind correlates with the CR residual" -- raised as an open question
(is the DNN/PINN simply missing signal the stats notebook already found?). Checked against the
notebook's own per-variable refit ablation (Section 7.2, `per_variable_refit_ablation`, leak-free
per the 2026-07-31 fix) before assuming more columns = better: of the 16 terrain+wind columns,
only 8 have a genuinely positive refit r2_drop (removing them hurts the full 37-variable model);
the other 8 -- including `gwa_wind_speed_10m`, the actual wind-speed variable -- are net HARMFUL
(negative r2_drop, removing them *helps*). `terrain_wind_solid`'s 5 columns
(`ceh_twi, eastness, elevation, northness, topex`) already ARE 5 of the top-6 refit-confirmed
positive columns. `terrain_wind_extended` adds the other 2 refit-confirmed-positive columns
(`plan_curvature` r2_drop=0.065, `whcl` r2_drop=0.023) -- so this is the correctly-scoped test,
not "throw in all 16 columns" (which would add 8 confirmed-harmful ones). Reran `dnn_env_terrain`
and `pinn_env_terrain` (pw=0/tw=0, pw=1/tw=0) on `spatial_block`/4survey with
`--feature-set terrain_wind_extended`, locally (fast enough not to need the cluster: DNN 60s,
each PINN leg ~7min):

| Model | Feature set | Test R2 |
|---|---|---|
| dnn_noenv (no terrain at all) | n/a | 0.6330 |
| dnn_env_terrain | terrain_wind_solid (5 col, default) | 0.6247 |
| dnn_env_terrain | terrain_wind_extended (7 col) | 0.6225 |
| pinn_env_terrain, pw=0/tw=0 | terrain_wind_solid | 0.6333 |
| pinn_env_terrain, pw=0/tw=0 | terrain_wind_extended | 0.6333 |
| pinn_env_terrain, pw=1/tw=0 | terrain_wind_solid | 0.6195 |
| pinn_env_terrain, pw=1/tw=0 | terrain_wind_extended | 0.6190 |

Adding the 2 extra refit-confirmed-positive columns did not close the DNN's gap vs. `dnn_noenv`
-- if anything, R2 moved slightly further in the wrong direction (0.6247->0.6225), and the two
PINN legs are unchanged within noise.
**What's working:** the refit-ablation-informed column choice itself was sound (didn't blindly
add all 16, which would likely have hurt more given 8 of them are confirmed net-negative) --
this was a real, correctly-scoped test of the "missing columns" hypothesis, not a strawman.
**What's not working / open concern:** this rules out "the env_terrain models just have the
wrong/incomplete column list" as the explanation. Combined with today's earlier dropout/LR sweep
(null) and architecture-size sweep (null), every cheap, mechanism-preserving lever has now been
tried and nulled: not regularization, not optimizer step size, not network capacity, not feature
selection. What's left is either a genuine inductive-bias mismatch (a small MLP trained by
gradient descent may just not extract this signal as well as XGBoost's tree splits do, regardless
of which exact columns it's given), or the deviation-mechanism question already flagged in the
entry above (rigid single-global-k/p CR curve fighting a plot-specific y_max shift).
**What this means for what's next:** don't reach for another feature-set variant (e.g. `broad`)
as the next experiment -- `broad` mixes in climate/soil columns outside the terrain/wind category
the dissertation's y_max sub-network is scoped to, and this result gives no reason to expect it
would behave differently. The next real test is on the mechanism side, not the feature side --
resume from the state noted in the entry above (external research on UDE-style residuals vs.
SA-PINN self-adaptive weighting) before writing any new model code.

---

**2026-08-01 — Switched both PINN models from `cr_pooled` to a split-matched CR anchor --
confirmed a real, quantifiable leak, not just a theoretical risk.**
**What I found:** `cr_pooled` (`outputs/chapman_richards/<cohort>/params.json`, read by both
`pinn_noenv` and `pinn_env_terrain` regardless of split_type) is fit on a RANDOM 60% of plots --
verified directly: its saved `n_rows_fit` (139,472) is exactly 60.0% of the 232,448 filtered
4survey rows, matching `plot_level_split()`'s train share, not `spatial_block_split()`'s. Since
that random split was never coordinated with `spatial_block`/`temporal`, the frozen physics
anchor both PINNs used was inevitably fit on some of `spatial_block`'s own test plots. The fix
already existed on disk as a side effect of the CR baseline's own per-split fit
(`outputs/<split_type>/chapman_richards/<cohort>/params.json`, fit using ONLY that split's train
plots) -- no new fitting code needed, just pointing `load_cr_params()` at the right file.
**What's working:** both `run_pinn_noenv.py`/`run_pinn_env_terrain.py`'s `load_cr_params()` now
take `split_type` and read the matched file; confirmed via a real smoke test that values
genuinely change (6survey/spatial_block: k 0.0233->0.0201, p 1.193->1.039 -- a bigger shift than
4survey's ~1-2%, worth watching).
**What's not working / open concern:** does NOT require rerunning Stage 1 (the no-env batch-size
sweep) -- it used `physics_weight=trajectory_weight=0.0`, so the anchor's value is multiplied by
zero and never entered the loss at all, mathematically unaffected regardless of which anchor
file was read. Stage 2 (base case, w=1.0), Stage 3 (the physics-weight grid, every config except
the 0/0 control), and Stage 4 (final seeded comparison, the nonzero-weight arm) ARE affected in
principle, though given the anchor's values only shifted ~1-2% for 4survey specifically (6survey
shifted more), the qualitative conclusions already reached (DNN beats PINN(w=1); the Stage 3
"winner" is noise-level indistinguishable from the 0/0 control) are unlikely to flip -- but the
exact reported numbers would. Stage 4 was never evaluated for test-set metrics to begin with, so
re-running it fresh costs nothing extra either way.
**What this means for what's next:** Stage 1 does not need re-running. Stage 2/3 are optional
precision re-runs, not urgent. Stage 4 should simply be (re-)run fresh with the corrected anchor
rather than resumed, since its evaluate step was never completed. `pinn_env_terrain` has no real
(non-smoke) runs yet at all, so nothing there needs redoing -- the corrected anchor just applies
from its first real run onward.

---

**2026-08-01 — Code review of `models/spatial_attribution/nlme.py` found a real REML/ML bug and
a reproducibility gap; both fixed, and the 2026-07-31 NLME entry's headline number changed.**
**What I found:** `variance_explained_by_fixed_effects()` compared compartment random-effect
variance between a null (intercept-only) and full (6-fixed-effects) model, both fitted with
`reml=True` -- textbook-invalid per Pinheiro & Bates (the reference behind R's own `nlme`
package): REML variance estimates aren't comparable across models with different fixed-effects
structures, since REML profiles out the fixed effects via a transform that depends on the
fixed-effects design matrix itself. Separately: `nlme.py` was untracked by git, and NO script or
notebook anywhere in the repo actually called its functions -- the numbers in the 2026-07-31 log
entry weren't reproducible from the saved repo state. Two other suspected issues (whether
`result.resid` is conditional on the random effect as claimed, and whether `fixed_effects_table()`
misaligns `bse_fe` against `fe_params`) were checked directly against statsmodels' source and a
toy model with known coefficients -- both confirmed correct, not bugs.
**What's working:** fixed `fit_nlme()` to take a `reml` parameter (default `True`, right for its
own coefficient/p-value reporting use) and `variance_explained_by_fixed_effects()` to force
`reml=False` on both models it compares -- the correct, textbook fix, not a workaround. Built the
actual calling code as a real notebook section (`spatial_autocorrelation_terrain.ipynb`'s
Section 3, previously a "not built yet" placeholder), fit on the same train-only spatial-block
split `grouped_category_importance.ipynb` uses, so this is now reproducible end to end.
**What's not working / open concern:** the fix changed the headline number materially, not just
cosmetically -- proportion of compartment variance explained went from 2.04% (buggy REML
comparison, apparently fit on the full pooled cohort) to 5.02% (correct ML comparison, train-only,
comparable sample discipline to XGBoost's own refit ablation). The 6 fixed-effect coefficients and
the raw/fitted skew-kurtosis numbers also shifted slightly for the same train-only-vs-pooled
reason -- direction and significance unchanged, magnitudes not identical to the retracted numbers.
**What this means for what's next:** cite 5.02% (not 2.04%) if this NLME result comes up again --
the qualitative conclusion is unchanged and if anything reinforced (still tiny next to XGBoost's
+19.7% terrain category refit R2 drop; the pre-registered kurtosis-gets-worse trigger still fires,
1.36→3.19). Any other historical number in this log that was computed by uncommitted, unsaved code
should be treated with the same suspicion until it's rebuilt as a real, re-runnable cell -- this
one specific case is now closed, but the pattern (a real result exists only as prose, not code)
is worth watching for elsewhere.

---

**2026-07-31 — NLME confirms the terrain/wind relationship is real but genuinely nonlinear;
per-variable refit ablation + ALE built to give the right attribution metric given that.**
**What I found:** built `models/spatial_attribution/nlme.py`, a two-stage mixed-effects model
(fixed effects = `ceh_twi`/`eastness`/`elevation`/`northness`/`topex`/`plan_curvature` on
`mean_cr_residual`, random effect = per-compartment intercept), per the two-stage template
already scoped in `progress_notes.md`. Checked the actual distribution first, not assumed:
`mean_cr_residual_4survey` has moderate non-normality (skew=-0.48, excess kurtosis=0.74) --
decided this didn't justify GAMLSS's much heavier Python tooling cost, confirmed by this
project's own prior assessment of that tradeoff. All 6 fixed effects came back
individually significant (p≈0) with sensible directions (elevation -3.54, topex +0.66,
ceh_twi -0.55, eastness +0.38, plan_curvature -0.21, northness -0.16, all standardised).
**But the model explains almost none of the actual compartment-level structure**: only 2.04%
of the compartment random-effect variance (**correction, 2026-08-01: this 2.04% figure was a
REML/ML bug, see the entry above — the correct number is 5.02%; the qualitative conclusion below
is unchanged**), and the model's OWN post-fit residuals got WORSE,
not better (kurtosis 0.74 -> 3.24) -- the trigger condition for reconsidering GAMLSS that was
set in advance, not moved after the fact. Read together with XGBoost's much larger category-
level refit R2 drop for `terrain` (+0.197) using the same variables, the coherent explanation
is a genuinely nonlinear/interaction-driven relationship a linear mixed model cannot capture.
**What's working:** the decision to check real distributional numbers before choosing NLME vs
GAMLSS, rather than debating in the abstract; the pre-registered trigger condition (check
post-fit residuals, not just raw pre-fit ones) actually got used honestly when it fired badly,
not quietly dropped.
**What's not working / open concern:** confirmed a categorical-variable bug in the LISA
per-variable test built alongside this (`ceh_pedotope`/`ceh_subsurface_drainage`/
`ceh_textural_composition` are unordered class IDs, run through Kruskal-Wallis as if
continuous -- invalid, mirrors the exact issue `elasticnet_environmental.py` already
one-hot-encodes these three columns to avoid). Doesn't affect the final terrain/wind feature
list (none of the three were in it), but the `spatial_autocorrelation_terrain.ipynb` Section 3
table itself still has this error in it, not yet fixed.
**What this means for what's next:** confirmed nonlinearity demotes Elastic Net and NLME from
"attribution evidence" to "how much of this is simple/linear structure" cross-checks only --
the per-category and per-variable REFIT ablation (assumption-free, works on the nonlinear
XGBoost fit) is the metric to trust. Built `per_variable_refit_ablation()`
(`models/xgb_environmental/grouped_analysis.py`) and added it as Section 7.2/7.3 in
`grouped_category_importance.ipynb`, redirecting the ALE plot from SHAP's top-4 (which
included `chelsa_bio12_precip_mm`, individually r2_drop=-0.077 -- SHAP ranked as important a
variable that actively hurts) to the refit-confirmed list instead. This directly justifies
`pinn_env_terrain`'s `y_max` needing to be a genuinely flexible sub-network, not a linear
function of terrain/wind -- the linear NLME's failure to explain compartment variance despite
significant coefficients is concrete evidence for why, not just a design preference.

---

**2026-07-31 — `neighbour_mean_height`/`neighbour_height_differential` confirmed to leak
test-set ground truth into themselves; removed from `ALL_FEATURE_COLUMNS` entirely, not just
re-flagged.**
**What I found:** both features are built from every OTHER plot's own real 2023 height within a
75m radius (`aux_data_resolution_check.ipynb`, `cKDTree.query_ball_point`), computed ONCE on the
full 71,766-plot set BEFORE any train/val/test split exists. Checked directly whether this
matters given `spatial_block_split()` holds out whole compartments: for a random sample of
plots, 93.7% of a plot's own 75m-neighbours share its OWN split; for test-set plots
specifically, 95.9% of their neighbours are ALSO test-set plots, and 82.3% of test plots have
**zero** train-set neighbours within 75m at all. So for the large majority of test rows, this
"feature" is built almost entirely from other test-set plots' real ground-truth heights, not
learned from training data — the model doesn't need to find any generalisable environmental
relationship, it can just echo back a local average of the answer key. Real, already-on-disk
consequence: `models/xgb_environmental/`'s existing `all_environmental_no_neighbour` ablation
already showed this — removing the two features drops XGBoost's spatial_block test R² from
0.598→0.321 (4survey) and from 0.327→**−0.337** (6survey, worse than predicting the mean).
Checked every other one of the 37 remaining `ALL_FEATURE_COLUMNS` entries for the same
construction pattern (an aggregate of OTHER plots' own height/growth, computed pre-split) —
none share it; every other feature is either a fixed physical/geometric property of the plot's
own location, an external dataset value at that location, or the plot's own raw survey history.
**What's working:** the project's own existing `all_environmental_no_neighbour` ablation had
already surfaced the size of this effect (2026-07-29) — the fix here is upgrading it from "a
documented caveat, kept as an optional comparison" to "confirmed as a real leak, removed from
the main feature set entirely," using a direct, checkable same-split-neighbour-fraction test
rather than inference from the R² gap alone.
**What's not working / open concern:** every environmental-attribution number reported before
2026-07-31 that used `all_environmental` (XGBoost/SHAP, Elastic Net, grouped permutation
importance, the Section 7.1 refit ablation, the Moran's I before/after check, the closing
cross-check table in `grouped_category_importance.ipynb`) was computed WITH this leak present,
and needs re-reading against the fixed numbers below, not cited from before this date.
**What this means for what's next:** `neighbour_mean_height`/`neighbour_height_differential`
removed from `FEATURE_PROVENANCE`/`ALL_FEATURE_COLUMNS` (`xgb_environmental.py`) and from
`CATEGORY_GROUPS` (`grouped_analysis.py`); the now-redundant `all_environmental_no_neighbour`
feature set and `FEATURE_SETS_NEEDING_SHAP` entry removed (`all_environmental` itself is the
fixed, leak-free set now). `run_xgb_environmental.py`/`run_elasticnet_environmental.py` need
re-running for `all_environmental` (both cohorts) to regenerate the on-disk outputs under the
corrected feature set, and `grouped_category_importance.ipynb` needs re-executing so every graph
in it reflects the fix. If a genuine spatial-lag feature is wanted again for `pinn_env_terrain`,
it needs a split-aware construction (only ever averaging TRAINING-set neighbours' heights, even
for val/test rows) — not the global pre-split version used here.

---

**2026-07-31 — Stage 4 fitting complete (90/90 runs); `temporal`/`4survey` shows a genuine
train/val collapse, not a training failure — all 15 runs in that cell excluded from the
seed-averaged comparison.**
**What I found:** checked every Stage 4 fit (2 cohorts x 3 splits x 5 seeds x 3 arms [DNN,
PINN pw=0.1/tw=0.0, PINN pw=0.0/tw=0.0] = 90 runs) via `run_metadata.json`/
`training_history.csv`. All 90 pass the early-stopping sanity check already used for Stage
1-3 (`n_epochs_trained - best_epoch == patience=40` exactly, no NaNs, none hit
`max_epochs=500`) -- the short wall-clock times are genuine early stopping. A second,
different check (percent reduction in smoothed val loss from epoch 1 to the best epoch)
caught something the first check alone missed: all 15 `temporal`/4survey runs (every seed,
every arm) show exactly 0.0% reduction, `best_epoch=1`. Full per-epoch history for one of
these (`final_dnn_seed42`) shows `train_loss` declining completely normally (0.446->0.305)
while `val_loss` rises every single epoch from the start (0.610->0.732) -- the network trains
fine, it just never generalises past its random-init starting point on this split. Traced to
`TEMPORAL_YEARS` (`models/common/splits.py`): both cohorts' `temporal` split validate on
2021, 9 years after their last training year, but 4survey trains on only 2008+2012 (a single
4-year transition) vs 6survey's 2002/2006/2008/2012 (four points spanning a full decade).
Confirmed via `n_rows_fit` that this isn't a data-volume issue: `temporal`/4survey actually
has MORE rows (116,224) than `temporal`/6survey (55,076), just fewer distinct time points (2
vs 4) -- and 6survey's `temporal` split shows a completely normal 27-35% val-loss reduction
under the identical 9-year-forward val gap. The mechanism is training-window temporal
diversity, not gap length or row count.
**What's working:** the two-stage check -- early-stopping-gap check, then a separate "did val
loss actually improve from its epoch-1 starting point" check -- caught something the first
check alone would have missed. `total_epochs - best_epoch == patience` passing is necessary
but not sufficient evidence a run trained meaningfully; best_epoch could legitimately be 1.
**What's not working / open concern:** all 15 `temporal`/4survey Stage 4 runs are effectively
reporting an untrained (near-random-init) checkpoint, not a fitted model -- their test-set
metrics would be meaningless if silently averaged into the seed-averaged comparison table
alongside the other 5 cells (spatial_block and temporal_narrow_gap, both cohorts; temporal,
6survey).
**What this means for what's next:** `temporal`/4survey excluded from Stage 4's seed-averaged
comparison table (flagged here, not silently dropped) -- report it as its own separate
finding ("temporal extrapolation degrades further when the training years themselves span
little time, independent of gap length") rather than a normal result cell. This refines,
not contradicts, the existing 2026-07-20 finding that gap length drives `temporal_wide_gap`
degradation (11/12 combos improved under the narrow gap) -- gap length matters, but for
4survey specifically it's compounded by the training set spanning only one transition,
tipping this cell from "degraded" to "collapsed". `temporal_narrow_gap`/4survey is NOT
excluded -- it's only mildly affected (0.2-1.9% reduction vs 0% for wide-gap), consistent
with it training on 2012+2021 (still only 2 points, but a less extreme jump). Doesn't affect
the primary `spatial_block` result. Worth a footnote in the write-up if `temporal` (wide-gap)
results for 4survey are cited at all.
**Addendum, checked the same day:** cross-checked against the already-run baselines
(`baselines_rebuild_2026-07-28`) to see if this is DNN/PINN-specific -- it isn't. RF baseline
on `temporal`/4survey's test set scores R²=-0.041 (worse than predicting the mean), vs
R²=0.475 on `spatial_block`/4survey; linear regression drops to R²=0.287 on the same
`temporal`/4survey test set. Neither RF nor linear regression use gradient descent, early
stopping, or batch size -- so this rules out any DNN/PINN training-mechanics explanation and
confirms the collapse is a genuine, model-agnostic property of the split: 4survey's 4 total
survey years (2008/2012/2021/2023) can't structurally support a "train on early years,
extrapolate 9-11 years forward" test the way 6survey's 6 years can. This is a dataset
limitation worth stating directly in the write-up's limitations section, not something a
different architecture or more features (e.g. Env-PINN's terrain/wind) could fix.
**Correction, once evaluate scripts ran (same day):** "essentially reporting an untrained
(near-random-init) checkpoint" above overstates it -- test R² for `temporal`/4survey is
DNN=0.3535, PINN(pw=0.1)=0.3523, PINN(pw=0.0)=0.3515 (mean across 5 seeds), which is a real,
non-trivial fit, not noise. It's a 1-epoch-trained checkpoint (best_epoch=1 means the
end-of-epoch-1 weights, not literal random init), and it actually BEATS both classical
baselines on this exact split: RF scores R²=-0.041 (worse than the mean) and linear
regression R²=0.287 on `temporal`/4survey's test set. Revised interpretation: early stopping
correctly identified that further gradient steps overfit to 2008/2012-specific patterns and
hurt 2021/2023 generalisation, and preserved the least-overfit (most implicitly-regularised)
checkpoint -- which still captures a genuine coarse signal (plausibly age-driven) that
transfers better than RF's full-depth fit or linear regression's limited capacity manages to.
So: DO cite `temporal`/4survey's numbers, but they need this specific mechanistic caveat
(best result = least additional training, not full convergence) rather than being treated as
an ordinary converged fit or silently averaged alongside cells where more training did help.

---

**2026-07-30 — Stage 2 base-case rebuild confirms physics-weight=1.0 still hurts PINN; Stage 3's
"winning" physics-weight pair is a noise-level pick, not a real improvement.**
**What I found:** Stage 2 (real DNN/PINN base-case rebuild, current pipeline, `batch_size=256`,
both cohorts, `spatial_block`+`temporal`) reran clean: DNN beats `PINN(w=1.0)` on test R² in all 4
cohort×split combinations (e.g. 4survey spatial_block 0.633 vs 0.580) -- the long-standing
"physics constraint hurts at full weight" finding survives the full rebuild, not just the retired
pipeline. Stage 3 (4survey 10-point `physics_weight`x`trajectory_weight` grid) found the lowest
`best_val_loss_smoothed` at `pw=0.1,tw=0.0` (0.3292) -- but four other configs sit within 0.0016 of
it (`pw=0.01,tw=0.0`=0.3294, `pw=0.0,tw=0.01`=0.3300, `pw=0.1,tw=0.01`=0.3299, and the `0/0`
control=0.3308), all inside the same ~0.002 noise band Stage 1's pure batch-size sweep already
established (0.3286-0.3305 with zero physics weight). Checking test-set R² (not just val loss) for
the "winner" vs the `0/0` control makes this sharper: the control is *slightly better* on held-out
data on both cohorts (4survey 0.6337 vs 0.6319; 6survey 0.7483 vs 0.7468). The one clear, real
signal from the grid: `trajectory_weight>=0.1` reliably hurts (val loss 0.338-0.350 vs ~0.329-0.331
for everything at `tw<=0.01`) -- that part is not noise.
**What's working:** the selection protocol (lowest val loss) was followed correctly and is
reproducible; the diagnostic habit of also checking test-set metrics (not just the metric used to
pick the winner) is what caught this, same discipline as the 2026-07-29 epoch-check entry below.
**What's not working / open concern:** picking `pw=0.1,tw=0.0` as "the winning weight pair" for
Stage 4 would be reporting a coin-flip as if it were a tuned result -- five configs are
statistically indistinguishable on val loss, and the one with the *lowest* val loss is not the one
with the best test R². This doesn't contradict the dissertation's actual physics-weight story
(low/zero weight beats `w=1.0`, clearly and repeatably) -- it just means there's no real evidence
that fine-tuning `pw` within the `[0, 0.1]`/`tw=0` region matters at all, only that staying in that
region (vs `w=1.0` or `tw>=0.1`) does.
**What this means for what's next:** Stage 4 can honestly report either "`pw=0.1,tw=0.0`" (the
literal grid winner, matching the pre-agreed protocol) or "`pw=0.0,tw=0.0`" (the simpler, equally-
supported choice, and marginally better on test R² both cohorts) -- both are defensible, and the
dissertation's actual claim should be "physics weight in `[0,0.1]` with `tw<=0.01`", not a single
precise value. Flagged to the user before committing Stage 4 cluster time to either pick.

---

**2026-07-30 — Fixed the year-effect diagnostic's degenerate y_max bound and its missing
`--split-type` option; the survey-year confound checked under spatial_block for the first time.**
**What I found:** `year_effect_diagnostic.py` had reintroduced the exact degenerate `y_max` lower
bound already found and fixed in `chapman_richards.py`'s main fit (28-29 July 2026) -- bounded at
exactly `max_observed_height` instead of `× 1.001`, letting the optimizer land precisely on that
boundary. `run_year_effect_diagnostic.py` also had no `--split-type` argument at all -- hardwired
to `plot_level`, meaning this confound had never been checked under `spatial_block_split`, the
dissertation's stated primary split.
**What's working:** both fixed -- the bound now matches the main fit, and a first starting guess
that would otherwise sit below the new bound (`× 1.0`, now infeasible) was corrected to `× 1.01`
to match. `--split-type` now supports `plot_level`/`spatial_block`/`temporal`/
`temporal_narrow_gap`, writing to the correctly split-type-prefixed output directory; the
plain-CR comparison still always reads the plot_level CR fit regardless, matching this project's
standing convention.
**What I found, numerically:** re-ran under both splits. `plot_level` (matches the previously-
cached, now-refreshed numbers in `baseline_results.ipynb` section 4.7): 1.80% of squared error
explained by year for 4survey, 3.95% for 6survey (was cached at a stale 2.06%/4.74% from before
this fix). `spatial_block`, checked for the first time: 1.15% (4survey) / 5.83% (6survey) --
comparable magnitude to plot_level, suggesting this confound isn't sensitive to which split is
used.
**What this means for what's next:** `baseline_results.ipynb` section 4.7 no longer needs its
"not currently reproducible" caveat -- refreshed with live numbers and the new spatial_block
comparison point.

---

**2026-07-30 — HadUK-Grid extended from a single 2021 snapshot to a genuine, cohort-aware
multi-year climate feature; screened 6 candidate variables, kept 2.**
**What I found:** `haduk_tas_2021_mean` was a known, flagged limitation -- every survey year
reused the same 2021 raster rather than its own year's climate. 5 more years (2002/2006/2008/
2012/2023) are now downloaded, closing the gap for both cohorts.
**What's working:** `models/common/download_haduk_multi_year.py` downloads each year's raster,
extracts this forest's ~72,000 plots, then deletes the ~41MB raw file before the next one --
peak disk usage stayed near one file's size instead of the ~1.5GB all 42 files (7 variables x 6
years) would cost kept permanently (this machine had 14GB free). Two real bugs found and fixed
along the way: (1) the download initially used `data.ceda.ac.uk`, which silently returns an
empty 200 response for an authenticated file GET instead of erroring -- `dap.ceda.ac.uk` is the
host CEDA's own token documentation actually uses; (2) a stale "blocked" placeholder row for
HadUK-Grid in `aux_data_resolution_check.ipynb` (from before CEDA access was unblocked) was
never removed, silently colliding with the real entry's name and only surfacing as a `NaN`
crash on a genuinely fresh, full top-to-bottom run -- this notebook apparently hadn't had one in
a while.
**What I found, on the numbers**: screened `tas` (corrected) plus 6 new variables (rainfall,
tasmax, tasmin, groundfrost, sun, sfcWind) via the standard 5-check battery, all real signal
(p<0.001 vs the CR residual, none degenerate). Kept only `tas` (rho=0.273) and `groundfrost`
(rho=0.172) -- genuinely new information, low redundancy with what's already in the model.
Dropped `rainfall` (rho=-0.362, real, but rho=0.86 with the existing `chelsa_bio12_precip_mm` --
largely restating it) and `tasmax` (rho=0.325, but rho=0.91 with `chelsa_bio1_celsius`, rho=0.95
with `tas` itself). `sfcWind` is the one worth remembering: it looked like a promising real
observational wind source (potentially filling the gap WASP was meant to fill), but rho=-0.96
with `tas` against only rho=0.25 with the existing `gwa_wind_speed_10m` says it's behaving like
a smooth regional temperature proxy at 1km, not a locally-resolved wind-exposure signal.
`tasmin` was too weak/inconsistent to justify (rho=-0.051 4survey, -0.214 6survey, opposite
signs). Sentinel-2 (10m, EPSG:27700, same CEDA token) was also considered for NDMI/NDWI as a
site-moisture proxy, but only covers 2015-2025 -- can't reach 2002/2006/2008/2012, so scoped as
a future 2021/2023-only supplementary check, not a full feature (same reasoning that excluded
AlphaEarth).
**What changed, numerically**: re-ran `run_xgb_environmental.py`/`run_elasticnet_environmental.py`
for `all_environmental`/`all_environmental_no_neighbour` (unaffected: `terrain_and_wind_only`,
confirmed identical before/after). 4survey `all_environmental`: XGBoost test R² 0.629->0.598,
Elastic Net test R² 0.671->0.666. 6survey `all_environmental`:
XGBoost test R²=0.327 (was 0.398), Elastic Net test R²=0.483 (was 0.521). Real, modest changes
from correcting an approximation, not a regression to chase -- correctness was the goal, not a
higher number.
**What this means for what's next:** `FEATURE_PROVENANCE`/`CATEGORY_GROUPS` updated
(`haduk_tas_2021_mean` -> `tas_mean` + new `groundfrost_mean`), `load_plots_for_cohort()`
generalized to handle any number of cohort-suffixed columns, not just the residual target.
`grouped_category_importance.ipynb` re-run end to end against the corrected feature set.

---

**2026-07-30 — Restored the predecessor notebook's decisive refit-based ablation for
`neighbour_spatial_lag`, missing from `grouped_category_importance.ipynb` since it replaced
`env_variable_importance_RETIRED_2026-07-28.ipynb`.**
**What I found:** the retired notebook's single most decisive number was a real refit-based R2
comparison for the neighbour-features category (test R2 0.596 with them in, 0.187 without,
`FEATURE_SETS["all_environmental_no_neighbour"]`). The current notebook replaced this with
permutation importance (section 4) and Moran's I before/after (section 7) alone -- neither is as
decisive on its own, and this project's own elevation/SHAP contradiction already proved a
proxy-importance method can disagree with a real refit ablation on exactly this kind of question.
**What's working:** `category_morans_i_before_after()`/`residual_morans_i()` already refit a fresh
model per category for the Moran's I check -- the R2 from that same refit was being computed and
then discarded. Added it back to `residual_morans_i()`'s return value (`models/xgb_environmental/
grouped_analysis.py`) rather than adding a second, separate ablation fit -- one refit now answers
both questions.
**What I found, numerically** (4survey, val, spatial_block): full model R2=0.734.
`neighbour_spatial_lag` removed: R2=0.400 (drop=+0.334) -- by far the largest drop of any
category, more than 4x the next-largest (`stand_structure`, +0.069). Directly confirms the
predecessor notebook's conclusion, now via the actively-maintained notebook's own methodology
rather than a retired one.
**What this means for what's next:** `grouped_category_importance.ipynb` section 7.1 now carries
this check going forward -- no need to reference the retired notebook for it anymore.

---

**2026-07-30 — `TERRAIN_AND_WIND_COLUMNS` was mislabeled: it included 6 spatial-position/edge-effect
columns plus `dist_to_watercourse` (soil/site), not just terrain+wind.**
**What I found:** the `terrain_and_wind_only` feature set was built from a column list that also
contained `dist_to_cpmt_boundary`, `dist_to_forest_perimeter`, `dist_to_scpt_boundary`,
`dist_to_block_boundary`, `cpmt_compactness_ratio`, `dist_to_road` (all
`spatial_position_edge_effects` per `grouped_analysis.py`'s own `CATEGORY_GROUPS`), plus
`dist_to_watercourse` (`soil_site`). The in-code comment claimed this matched "the dissertation
plan's original XGB-A/B framing" -- checked directly against `Dissertation Plan v5 - 7th June.md`'s
own XGB-A/B/C table, and that framing is genuinely terrain+wind only ("terrain only" / "terrain +
WASP/GWA wind speed"), no edge-effects or hydrology at all -- so the claim didn't hold up either.
**What's working:** `grouped_analysis.py`'s `CATEGORY_GROUPS` already had the correct, carefully-
reasoned split (terrain/wind/soil_site/spatial_position_edge_effects as four distinct categories)
-- `TERRAIN_AND_WIND_COLUMNS` just wasn't built from it. Fixed to be exactly
`CATEGORY_GROUPS["terrain"] | CATEGORY_GROUPS["wind"]` (16 columns, checked programmatically to
match), so there's one consistent definition of "terrain" and "wind" across every analysis now,
not two.
**What changed, numerically:** re-ran `run_xgb_environmental.py`/`run_elasticnet_environmental.py`
for `terrain_and_wind_only` under the corrected column list (`all_environmental`/
`all_environmental_no_neighbour` are unaffected, unchanged). New numbers, `spatial_block`, test
split: 4survey XGBoost R²=0.162 (RMSE=5.123), Elastic Net R²=0.188 (RMSE=5.043); 6survey XGBoost
R²=-0.351 (RMSE=3.856), Elastic Net R²=0.096 (RMSE=3.155) -- markedly weaker than the old
(mislabeled) numbers, since the removed columns were carrying real signal, just not terrain/wind
signal. Any prior "terrain and wind explain X%" claim referencing the old `terrain_and_wind_only`
numbers needs re-reading against these.
**What this means for what's next:** the write-up's XGB-A/B/C-equivalent comparison should cite
these corrected numbers, not the ones from before 2026-07-30. `grouped_category_importance.ipynb`
(the actively-maintained category-level analysis) was never affected by this bug -- it always read
`CATEGORY_GROUPS` directly, not `TERRAIN_AND_WIND_COLUMNS`.

---

**Consolidated 29 July 2026**: the detailed dated entries that used to sit here (2026-07-13 through 2026-07-20, covering the plot_level/spatial_block/temporal baseline results, the DNN/PINN tuning and physics-weight-sweep process, the 3-seed reseed check, and the temporal_narrow_gap comparison) were all built on the now-retired `Top_Height99`+`yldc` pipeline -- not comparable to current numbers. Key results preserved in `progress_notes.md`'s "Systematic rebuild" entry (28-29 July 2026); full original narrative still in git history if ever needed.

---

**2026-07-29 — PINN's `physics_weight=0.0` test doesn't behave like DNN, exposing an
undocumented, uncontrolled batch-size difference between the two models.**
**What I found:** the real base-case (`physics_weight=trajectory_weight=1.0`) DNN/PINN cluster
jobs first came back in ~53s each -- impossible for real training, traced to
`data/processed/transitions/` never having been rsynced to the cluster (fixed). Once actually
running: DNN's `val_loss` genuinely improves (0.342→0.331 over ~50 epochs before patience=40
stops it at epoch 52). PINN's `best_val_loss` never beats its own epoch-1 value, at `W=1.0`
(0.394902), `W=0.0` (0.328968), or `W=0.05` (0.334239) -- all three runs plateau at essentially
the same value they started at. If `W=0.0` truly zeroes out the physics/trajectory terms'
gradient contribution, PINN's loss function is then identical to DNN's -- so it shouldn't stall
where DNN doesn't, and it did anyway. Reading both training loops side by side found why:
`dnn_noenv.py`'s `BATCH_SIZE=512` vs `pinn_noenv.py`'s `BATCH_SIZE=128`/`PAIRS_BATCH_SIZE=128` --
undocumented (every other hyperparameter has an explicit "kept identical to the DNN" comment;
this one doesn't), meaning ~254 optimizer steps/epoch for PINN vs. ~64 for DNN, a real, large,
previously unnoticed confound.
**What's working:** the diagnostic discipline itself -- watching per-epoch logs live (instead of
trusting `sacct`'s exit code) is what caught the missing-file bug; comparing DNN and PINN's
training curves side by side (not just their final numbers) is what caught the batch-size
mismatch, which a results-only comparison would have missed entirely.
**What's not working / open concern:** the `physics_weight` sweep's conclusions (this session's
AND the retired pipeline's) were never run with batch size held constant between DNN and PINN --
meaning "physics hurts 4survey" has an unexamined alternative explanation (batch size) that
hasn't been ruled out yet. Not resolved: is 128 a deliberate choice (e.g. GPU memory for the
extra physics/trajectory autograd graphs) or just an unexamined leftover -- no comment either way.
**What this means for what's next:** exposed `--batch-size`/`--pairs-batch-size` as CLI args on
`run_pinn_noenv.py` (previously hardcoded constants, no way to test this without permanently
editing the file) -- confirmed working via a 5-epoch local smoke test at `--batch-size 512`,
which already showed `best_val_loss` beating epoch 1 (0.344→0.337) within 5 epochs, unlike every
128-batch run. Not conclusive on its own (5 epochs, no patience exhaustion) -- the real test is
the matched-batch-size cluster run below. If that clears the epoch-1 plateau, batch size (not
physics weight) was the actual cause, and the retired pipeline's weight-sweep conclusion needs
re-examining under a fair, batch-matched comparison before being trusted further.

---

**2026-07-28 — Environmental attribution Tier 2: a circularity bug caught, spatial-CV inflation
measured directly, and three independent methods agree/disagree in informative ways.**
**What I found:** (1) Adding `Age` to the feature set pushed XGBoost test R² from 0.567 to 0.914
and made `Age` the 2nd-highest SHAP feature — checked why, and found the single global
Chapman-Richards curve has a real, non-monotonic residual bias by age bin (+0.99 at 25-32yrs,
-0.62 at 40-48yrs, +0.79 at 56-64yrs, -4.5 at 79-87yrs), which XGBoost can re-learn given `Age`
back as an input — `Age` was excluded for this reason, the other stand-structure variables
weren't. (2) The same model/data scored test R²=0.567 under `spatial_block_split` vs 0.903 under
a plain random plot-level split — +0.335 R² of pure inflation, measured directly rather than
just asserted, confirming why every result in this repo uses a spatial-aware split. (3) A
pre-existing SHAP-on-test-rows leakage bug (SHAP computed over the full plot set, including
test, and the Tier-2 notebook's ablation work was repeatedly re-using test R² for feature
decisions) was found and fixed — val is now the only split used for any feature-selection
decision, test is read once. (4) Three independent importance/effect methods (Elastic Net
coefficients, XGBoost SHAP, grouped permutation importance) mostly agree on category ranking
(neighbour/spatial-lag and stand-structure dominate; soil/site and spatial-position/edge-effects
are negligible by all three) but Moran's I before/after tells a genuinely different story:
removing `terrain` increases residual spatial autocorrelation the most (+0.297), while removing
`neighbour_spatial_lag` (the single biggest driver by every other method) DECREASES it (-0.162).
**What's working:** the cross-method agreement pattern already established for individual
variables (SHAP vs. ablation) generalises cleanly to category-level analysis — methods that
agree give real confidence (e.g. soil/site's low importance), methods that disagree (Moran's I
vs. everything else on neighbour features) are flagged as genuinely different questions, not
forced into one story.
**What's not working / open concern:** the neighbour/spatial-lag category's Moran's I result
(removing the biggest predictive driver DECREASES spatial autocorrelation) isn't fully explained
— plausibly because removing it leaves such a large, noisy residual that fine-scale spatial
structure gets swamped, but this is a hypothesis, not confirmed. `Age`'s exclusion means the
stand-structure category's real predictive contribution (permutation ΔR²=0.156) is understated
relative to what a naive "just include everything" pass would have shown, by design.
**What this means for next steps:** the same circularity check (does a candidate feature share
construction with the target) is worth applying BEFORE adding any new variable to Env-PINN's own
feature set, not just discovered after the fact. Causal SHAP, GAM, Double/Debiased ML, and BART
remain the documented path to actually answering "does X cause more or less growth" — everything
built this session (Elastic Net, grouped permutation importance, Moran's I) improves the honesty
of an associational/predictive ranking, not a causal one.

**2026-07-28/29 — Target variable and `yldc` retired across the whole pipeline; baselines
re-verified, real cluster jobs submitted.**
**What I found:** `yldc` (a real, externally-sourced FC inventory field, not computed from this
survey's own height/age) nonetheless hurts held-out generalisation in every model checked via
real ablation: RF baseline test R² 0.446→0.498 without it, DNN 0.606→0.647, `xgb_environmental`
val R² 0.649→0.729. Separately, the target changed from `Top_Height99` (=`elev_percentile_99th`)
to raw, unadjusted `elev_percentile_95th`, per an explicit decision to retire the whole "99th
percentile" family (`Vol99`, `GYCspec99` too) — `Top_Height95` (the ×1.1-adjusted version) is
kept only as an ingredient for the pre-computed `Vol95`/`GYCspec95` forestry-audit fields, never
a target or feature. While verifying the rebuild, also found and fixed a pre-existing
Chapman-Richards fitting bug: `y_max`'s lower bound was exactly the observed max height, letting
`curve_fit` land precisely on that boundary under BOTH the old and new target (confirmed by
refitting the old target with the same code) — not caused by this change, just exposed while
checking it.
**What's working:** the cleaning notebook was converted to a proper script
(`data_processing/clean_master_data.py`), and the 5 near-duplicate per-model export files
(`dnn_noenv.parquet`/`pinn_noenv.parquet` were confirmed byte-for-byte identical) were replaced
with one consolidated `model_table.parquet` per cohort. Local re-runs of all four baselines
across all four split types (`plot_level`, `spatial_block`, `temporal`, `temporal_narrow_gap`)
reproduce the exact same qualitative pattern as the retired pipeline (RF wins `plot_level`,
loses its edge to linear under `spatial_block`) — the rebuild changed the numbers, not the
underlying story. DNN/PINN smoke tests (80 epochs, both cohorts, `spatial_block` and `temporal`)
all ran clean after fixing one real bug (a renamed transition-table column,
`annual_height99_increment`→`annual_height_increment`, missed in one spot on the first pass).
**What's not working / open concern:** an accidental local mistake — the DNN/PINN smoke tests
were run without a distinct `--run-name`, overwriting the real, previously-reported full
500-epoch checkpoints/predictions at the default output paths. Confirmed recoverable (the
cluster's own copies, dated 2026-07-16, were untouched, since I have no cluster access) — no
permanent loss, but the local `outputs/` (~4GB, everything pre-dating this rebuild) was archived
wholesale to `legacy/2026-07-28/outputs/` and baselines regenerated fresh locally, rather than
attempting to restore the exact prior local state. Going forward, every exploratory/smoke run
must use a distinct `--run-name`.
**What this means for next steps:** the real full-length (`--max-epochs 500`) DNN/PINN cluster
jobs are running now (both cohorts, `spatial_block`/`temporal`/`temporal_narrow_gap`, physics
weight left at the untested default 1.0 — NOT a re-run of the physics-weight sweep or the
40-job reseed check yet). Deliberately sequenced this way: compare the plain base-case numbers
against the retired-pipeline's own base case first, and only re-invest in the expensive
sweep+reseed process if that comparison shows the target/`yldc` change moved things enough to
put the old tuning conclusion (`physics_weight=0.05`) in real doubt.

## Decisions log (the "why", chronological)

**2026-07-15 — `temporal_wide_gap` chosen over `temporal_narrow_gap` for the primary run.** `temporal_wide_gap`
(train on the two earliest years only, test 11 years later) is the harder, more discriminating
extrapolation test — physics constraints are expected to help most exactly where pure data-driven
extrapolation is hardest, so this is the test that can actually show the PINN's physics term
earning its keep. temporal_narrow_gap (train through 2021, test only 2023) is closer to interpolation and
would likely understate any generalization gap. Decision: run temporal_wide_gap as primary; temporal_narrow_gap stays
a planned robustness check specifically to test whether temporal_wide_gap's conclusions hold up under a
shorter, easier extrapolation gap — not a replacement for it.

**2026-07-15 — PINN's frozen CR anchor uses the plot_level fit (cr_pooled), not a temporal-restricted
fit (cr_matched).** Initially flagged as a possible leakage concern (the plot_level CR fit was
estimated using rows from 2021/2023, years the PINN itself never trains on). Reconsidered after
reviewing Reuben (2025)'s own stated justification for fitting CR globally: he treats
`y_max`/`k`/`p` as species-level biological constants (not a "prediction" requiring train/test
discipline of their own), explicitly accepted the same "foresight" tradeoff, and argued the fitted
values converging near expected species-level ranges was evidence against overfitting to any
particular subset. The CR curve is also identical across every plot (not plot-specific), so even
information from later years is aggregate/population-level, not a leak of any individual test
plot's label — structurally different from the network directly training on 2023 rows. Caveat
carried forward: this means the DNN-vs-PINN temporal comparison is not a perfectly clean ablation,
since the PINN's physics anchor carries a small amount of aggregate later-year information the DNN
never gets. cr_matched is recorded above as a cheap, worthwhile robustness check to quantify how much
this matters, not because cr_pooled is expected to be wrong.

**2026-07-16 — `spatial_block_split`, not `temporal_split`, is the dissertation's
primary experiment; `temporal_split` stays in as a real secondary question, not
demoted out of the write-up.** Reconsidered after clarifying what the *planned*
environmental covariates actually are: terrain (elevation, slope, TWI, TOPEX), wind
exposure — all static per plot, not something that varies year-to-year the way
climate/weather does. That means "does environment explain growth better than
physics alone" is inherently a question about *spatial* variation, not temporal
extrapolation — `spatial_block_split` is the split built to test exactly that, while
`temporal_split` tests something genuinely different (extrapolation across time,
independent of any environmental covariate) that this dissertation still cares about,
just not as the central result. Practically: the `temporal_wide_gap` vs
`temporal_narrow_gap` decision above still stands *within* the temporal question — this
new decision is one level up, about which question is the headline one. Nothing about
the temporal results gets discarded; the Experiment table's Status column for the
`temporal_wide_gap` rows changed from `primary` to `secondary` to reflect this, not
`superseded`.

**2026-07-17/18/20 — retired pipeline's physics-weight tuning: methodology precedent kept, exact
numbers not.** Under the old `Top_Height99`+`yldc` pipeline, a 7-value sweep, a `temporal`-split
recheck, and a 3-seed reseed check together settled on one shared `pw=tw=0.05` default for both
cohorts/splits (over per-cohort tuning) and then retracted an earlier "PINN wins 6survey" claim
once the reseed check showed it was noise. Full numbers preserved in `progress_notes.md`'s
"Consolidated numeric record" (29 Jul 2026); not repeated here since they're specific to the
retired target/feature set and not directly comparable to the new pipeline. What still applies
going forward: prefer one shared weight over per-cohort/per-split tuning unless the gap is large,
and don't treat a single-seed win/loss as settled — reseed before making a comparative claim. The
actual `physics_weight`/`trajectory_weight` value needs re-sweeping under the new pipeline before
being treated as decided again (deliberately deferred — see base-case-first note above).

**2026-07-20 — `temporal_narrow_gap` given a minimal pass, not a full tuning/sweep investment.**
Decided before running: this split exists to answer one question (does gap length explain
`temporal_wide_gap`'s degradation), not to become a fourth fully-tuned split. One run per model,
at settings already established elsewhere (tuned hyperparameters, shared `pw=tw=0.05`, not
re-swept) — see Findings log for the result. If a future need arises to know whether DNN or PINN
"wins" specifically under `temporal_narrow_gap`, that needs its own reseed (this pass only ran one
seed each) — not inferred from these single-seed numbers, per the exact lesson the reseed check
above just demonstrated.

## Output-path naming convention (for when new variants are actually run)

Split-type prefixing (`outputs/<split_type>/<model>/<cohort>/`) is now shared by the
baselines AND dnn_noenv/pinn_noenv (`models/common/saving.py::model_output_dir()`,
imported by both `run_baselines.py` and the DNN/PINN fit/evaluate scripts, so there
is exactly one definition of this convention, not two that could drift apart). One
difference: the baselines reserve the plain, unprefixed `outputs/<model>/<cohort>/`
path for `plot_level_split` specifically (their original default before the other two
split types existed). DNN/PINN never run `plot_level_split` at all, so their output
path is *always* prefixed — there is no unprefixed `outputs/dnn_noenv/...` or
`outputs/pinn_noenv/...` any more (the real 2026-07-16 temporal-split results were
moved to `outputs/temporal/dnn_noenv/<cohort>/` / `outputs/temporal/pinn_noenv/<cohort>/`
for exactly this reason, once `spatial_block_split` was wired in as a second option).

- **`spatial_block_split` for DNN/PINN** (wired in 2026-07-16): `--split-type spatial_block`
  on `run_dnn_noenv.py`/`run_pinn_noenv.py`/the two evaluate scripts writes to
  `outputs/spatial_block/dnn_noenv/<cohort>/` / `outputs/spatial_block/pinn_noenv/<cohort>/`
  — no separate naming decision needed, it's the same `split_type` mechanism the
  baselines already use.
- **temporal_narrow_gap** (different train/val/test years; code added 2026-07-20, `--split-type
  temporal_narrow_gap`): `outputs/temporal_narrow_gap/<model>/<cohort>/` — a distinct
  split-type-style prefix, never overwriting `outputs/temporal/...` (temporal_wide_gap). Wired into
  `model_output_dir()` (`models/common/saving.py`) the same way `spatial_block`/`temporal` are.
- **PINN cr_matched** — superseded (2026-08-01): the plan above (a distinct model-name suffix,
  `outputs/pinn_noenv_crmatched/<cohort>/`, coexisting with a `cr_pooled` path) assumed
  `cr_pooled` was a legitimate alternative worth keeping choosable. It wasn't — investigation
  confirmed `cr_pooled` was a real train/test leak (its random 60% `plot_level_split` training
  plots were never coordinated with `spatial_block`/`temporal`'s own split, so they inevitably
  overlapped with a given split's test plots). Both `pinn_noenv` and `pinn_env_terrain` now read
  the split-matched anchor unconditionally, at the plain `outputs/<split_type>/pinn_noenv/<cohort>/`
  path — no separate model name or `--cr-variant` flag, since there's no longer a second option
  worth preserving on disk. `run_metadata.json`'s `frozen_cr_params` field still records exactly
  which `y_max`/`k`/`p` values were used, for audit.
- **PINN `physics_weight`/`trajectory_weight` sweep**, same reasoning as `cr_matched` above:
  `--run-name pinn_noenv_pw<W>_tw<W>` on `run_pinn_noenv.py`/`evaluate_pinn_noenv.py` writes to
  `outputs/<split_type>/pinn_noenv_pw<W>_tw<W>/<cohort>/`, never touching the plain `pinn_noenv`
  path — see `models/pinn_noenv/run_pinn_noenv.py`'s `run_name` handling (data loading always uses
  the plain `pinn_noenv` table; only the output path and `run_logs` identity change). `W=1.0` (no
  suffix) is the historical default at the plain path; the retired pipeline's swept `pw0.05_tw0.05`
  paths no longer exist locally (that run needs re-doing under the new target/features, see
  Decisions log above) but the convention itself — non-default weights always get a suffixed path,
  never overwrite the plain one — carries forward unchanged.
- For the **baselines**, whichever configuration is primary for the write-up lives at
  the plain, unprefixed path; for **DNN/PINN**, `temporal` and `spatial_block` are
  both always prefixed, so "which one is primary" is a fact to check in this log's
  Experiment table (Status column), not something the path itself tells you.
