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

## Status legend

- **primary** — the result currently intended for the main write-up.
- **robustness-check** — run specifically to test whether a primary result
  is sensitive to a design choice, per the plan below.
- **superseded** — an earlier config later replaced by a corrected/updated
  one (kept in the table for the record, not deleted).

## Planned robustness checks (not yet run)

Recorded here before running them so the *intent* is on record even before
there's a result — this is what you described: run Design 1 as primary now,
run Design 2 later specifically to check whether Design 1's conclusions are
sensitive to the training-year cutoff, then decide whether to report one,
both, or pick per-question based on what spatial vs. temporal turns out to
need.

| Planned experiment | Question it answers | Trigger to run it |
|---|---|---|
| Temporal split, Design 2 (train 2008+2012+2021, test 2023 only) | Is Design 1's dramatic temporal degradation an artefact of the specific 11-year train/test gap, or does it hold under a shorter, easier gap too? | After Design 1 (primary) results are fully written up for baselines + DNN + PINN |
| PINN physics anchor, Route B (CR fit restricted to 2008+2012 only, matching the PINN's own training years) | Does the PINN-vs-DNN temporal comparison hold up under a stricter ablation with zero information leakage into the physics term? | Cheap to run alongside Design 1 primary results — worth doing early as a documented caveat, not necessarily gated on anything |

## Experiment table

| ID | Date | Split design | Years (train / val / test) | CR fit used | Models | Cohorts | Status | Output location | Result summary |
|---|---|---|---|---|---|---|---|---|---|
| `plot_level_v1` | 2026-07-13 | `plot_level_split` (random 60/20/20 by plot) | n/a (all years pooled) | n/a | CR, average-by-age, linear, RF | both | primary | `outputs/<model>/<cohort>/` | RF best on both cohorts (RMSE 4.65/3.86); see `baseline_results.ipynb` §2 |
| `spatial_block_v1` | 2026-07-14 | `spatial_block_split` (whole `cpmt` compartments, 60m buffer) | n/a (all years pooled) | n/a | CR, average-by-age, linear, RF | both | primary | `outputs/spatial_block/<model>/<cohort>/` | RF loses its plot_level advantage (RMSE +28.7%/+19.4%); see `baseline_results.ipynb` §8.1 |
| `temporal_design1_baselines` | 2026-07-15 | `temporal_split`, Design 1 | train=[2008,2012] (4survey) / [2002,2006,2008,2012] (6survey), val=[2021], test=[2023] | n/a | CR, average-by-age, linear, RF | both | primary | `outputs/temporal/<model>/<cohort>/` | Much larger degradation than spatial (up to +141.7% RMSE, `average_by_age`/6survey R² negative); CR most temporally robust, not RF — see `baseline_results.ipynb` §8.1 |
| `dnn_noenv_design1_smoketest` | 2026-07-15 | `temporal_split`, Design 1 | same as above | n/a (no physics term) | dnn_noenv | both | superseded (5-epoch sanity check only, not a real result) | `outputs/dnn_noenv/<cohort>/` | Pipeline verified working; 4survey R²=0.46 @ 5 epochs (not meaningful yet — real run needs full `--max-epochs` on SLURM) |
| `pinn_noenv_design1_routeA_smoketest` | 2026-07-15 | `temporal_split`, Design 1 | same as above | Route A: plot_level CR fit (`outputs/chapman_richards/<cohort>/params.json`) | pinn_noenv | both | superseded (5-epoch sanity check only) | `outputs/pinn_noenv/<cohort>/` | Pipeline verified working, both physics terms produce nonzero, non-dominating loss (physics_loss≈0.009, trajectory_loss≈0.02 @ epoch 5); not a real result yet |

## Decisions log (the "why", chronological)

**2026-07-15 — Temporal split Design 1 chosen over Design 2 for the primary run.** Design 1
(train on the two earliest years only, test 11 years later) is the harder, more discriminating
extrapolation test — physics constraints are expected to help most exactly where pure data-driven
extrapolation is hardest, so this is the test that can actually show the PINN's physics term
earning its keep. Design 2 (train through 2021, test only 2023) is closer to interpolation and
would likely understate any generalization gap. Decision: run Design 1 as primary; Design 2 stays
a planned robustness check specifically to test whether Design 1's conclusions hold up under a
shorter, easier extrapolation gap — not a replacement for it.

**2026-07-15 — PINN's frozen CR anchor uses the plot_level fit (Route A), not a temporal-restricted
fit (Route B).** Initially flagged as a possible leakage concern (the plot_level CR fit was
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
never gets. Route B is recorded above as a cheap, worthwhile robustness check to quantify how much
this matters, not because Route A is expected to be wrong.

## Output-path naming convention (for when new variants are actually run)

Follows the pattern already established for split types (`outputs/<split_type>/<model>/<cohort>/`,
vs. plain `outputs/<model>/<cohort>/` for the original plot_level default):

- **Design 2** (different train/val/test years), if run: `outputs/temporal_design2/<model>/<cohort>/`
  — a distinct split-type-style prefix, never overwriting `outputs/temporal/...` (Design 1).
- **PINN Route B** (temporal-restricted CR anchor), if run: a distinct model name,
  `outputs/pinn_noenv_crtemporal/<cohort>/` — since this isn't a different split, it's a different
  PINN configuration, so it gets a model-name suffix rather than a split-type prefix. The
  `run_metadata.json`'s `frozen_cr_params` field already records exactly which values were used
  either way, but a distinct output path is required so Route A and Route B results can coexist on
  disk rather than one overwriting the other.
- Whichever configuration is the **primary** one for the write-up should be the one living at the
  plain, unprefixed/unsuffixed path (`outputs/temporal/...`, `outputs/pinn_noenv/...`) — robustness
  checks get the longer, more specific path. This keeps "the result I'm citing in Chapter 5" always
  findable at the short, predictable path without needing to check this log first.
