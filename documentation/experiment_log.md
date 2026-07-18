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

## Naming glossary

Two short-named choices come up repeatedly below — named for what they actually are,
not numbered, so you don't have to look up which is which:

- **`temporal_wide_gap`** vs **`temporal_narrow_gap`** — two ways to set up
  `temporal_split`'s train/val/test years. `temporal_wide_gap` (current primary)
  trains on the earliest years only and tests 11 years later (2012→2023) — the
  harder extrapolation. `temporal_narrow_gap` (planned, not yet run) trains through
  2021 and tests only 2023 — a 2-year gap, closer to interpolation.
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
| `temporal_narrow_gap` (train 2008+2012+2021, test 2023 only) | Is `temporal_wide_gap`'s dramatic temporal degradation an artefact of the specific 11-year train/test gap, or does it hold under a shorter, easier gap too? A secondary-question robustness check, not gating the primary spatial result. | After `temporal_wide_gap` results are fully written up for baselines + DNN + PINN |
| PINN physics anchor, `cr_matched` (CR fit restricted to 2008+2012 only, matching the PINN's own training years) | Does the DNN-vs-PINN temporal comparison hold up under a stricter ablation with zero information leakage into the physics term? | Cheap to run alongside `temporal_wide_gap` results — worth doing early as a documented caveat |
| `physics_weight`/`trajectory_weight` sweep, applied to `temporal_split` (currently only run under `spatial_block_split`) | Does the same low-weight finding (`W=0.05` beats `W=1.0`) hold under temporal generalization too, or is it spatial_block-specific? | Cheap, same sweep mechanics already built (`--physics-weight`/`--trajectory-weight`/`--run-name` on `run_pinn_noenv.py`) |

**Resolved 2026-07-17** (removed from the table above, kept here for the record): "Hyperparameter variants of the primary `spatial_block` PINN run — is the result genuine convergence or a premature stop?" — yes, genuine convergence. Loosening patience/smoothing/batch size trained both models roughly 2x longer but barely moved final test RMSE (see the 2026-07-17 Findings entry) — the pre-tuning numbers were already close to real optima, except DNN/4survey's overfitting climb, which only softened, not fixed (a data-limitation, not a tuning-limitation — see the same entry).

## Experiment table

| ID | Date | Split design | Years (train / val / test) | CR fit used | Models | Cohorts | Status | Output location | Result summary |
|---|---|---|---|---|---|---|---|---|---|
| `plot_level_v1` | 2026-07-13 | `plot_level_split` (random 60/20/20 by plot) | n/a (all years pooled) | n/a | CR, average-by-age, linear, RF | both | primary | `outputs/<model>/<cohort>/` | RF best on both cohorts (RMSE 4.65/3.86); see `baseline_results.ipynb` §2 |
| `spatial_block_v1` | 2026-07-14 | `spatial_block_split` (whole `cpmt` compartments, 60m buffer) | n/a (all years pooled) | n/a | CR, average-by-age, linear, RF | both | primary | `outputs/spatial_block/<model>/<cohort>/` | RF loses its plot_level advantage (RMSE +28.7%/+19.4%); see `baseline_results.ipynb` §8.1 |
| `temporal_design1_baselines` | 2026-07-15 | `temporal_split`, temporal_wide_gap | train=[2008,2012] (4survey) / [2002,2006,2008,2012] (6survey), val=[2021], test=[2023] | n/a | CR, average-by-age, linear, RF | both | secondary | `outputs/temporal/<model>/<cohort>/` | Much larger degradation than spatial (up to +141.7% RMSE, `average_by_age`/6survey R² negative); CR most temporally robust, not RF — see `baseline_results.ipynb` §8.1 |
| `dnn_noenv_design1_smoketest` | 2026-07-15 | `temporal_split`, temporal_wide_gap | same as above | n/a (no physics term) | dnn_noenv | both | superseded (2-epoch sanity check only, not a real result) | `outputs/temporal/dnn_noenv/<cohort>/` | Pipeline verified working; superseded by `dnn_noenv_design1` below |
| `pinn_noenv_design1_routeA_smoketest` | 2026-07-15 | `temporal_split`, temporal_wide_gap | same as above | cr_pooled: plot_level CR fit (`outputs/chapman_richards/<cohort>/params.json`) | pinn_noenv | both | superseded (2-epoch sanity check only) | `outputs/temporal/pinn_noenv/<cohort>/` | Pipeline verified working; superseded by `pinn_noenv_design1` below |
| `dnn_noenv_design1` | 2026-07-17 (tuned rerun; first run 2026-07-16) | `temporal_split`, temporal_wide_gap | same as above | n/a (no physics term) | dnn_noenv | both | secondary | `outputs/temporal/dnn_noenv/<cohort>/` | Tuned hyperparameters (see 2026-07-17 Findings entry): `lr_scheduler_patience` 10→15, `early_stopping_patience` 20→40, `BATCH_SIZE` 128→512, added `WEIGHT_DECAY`/`GRAD_CLIP_MAX_NORM`, 5-epoch `val_loss` smoothing for the best-epoch decision. 4survey: RMSE=5.8600, R²=0.4533, trained 43 epochs, best epoch=3 (overfitting climb softened, not fixed — `val_loss` still climbs +12% vs +19% pre-tuning). 6survey: RMSE=4.9388, R²=0.2749, trained 105 epochs, best epoch=65 (healthy). Both still beat all four temporal-split baselines |
| `pinn_noenv_design1` | 2026-07-17 (tuned rerun; first run 2026-07-16) | `temporal_split`, temporal_wide_gap | same as above | cr_pooled: plot_level CR fit | pinn_noenv | both | secondary | `outputs/temporal/pinn_noenv/<cohort>/` | Same tuning as above, `physics_weight`/`trajectory_weight` still at the untested default of 1.0 (see 2026-07-17 weight-sweep entry — that sweep only covers `spatial_block` so far). 4survey: RMSE=6.0870, R²=0.4101, trained 71 epochs, best epoch=31. 6survey: RMSE=4.8857, R²=0.2904, trained 107 epochs, best epoch=67. Both still beat all four temporal-split baselines |
| `dnn_noenv_spatialblock` | 2026-07-17 (tuned rerun; first run 2026-07-16) | `spatial_block_split` | n/a (all years pooled per train-plot) | n/a (no physics term) | dnn_noenv | both | primary | `outputs/spatial_block/dnn_noenv/<cohort>/` | Same tuning as above. 4survey: RMSE=5.0185, R²=0.6091, trained 52 epochs, best epoch=12. 6survey: RMSE=3.6498, R²=0.7434, trained 52 epochs, best epoch=12 (dramatically healthier curve than pre-tuning — `val_loss` now drops -42.9% over training instead of the pre-tuning +5% drift, though final RMSE barely changed). Beats every spatial-block baseline on both cohorts |
| `pinn_noenv_spatialblock` | 2026-07-17 (tuned rerun; first run 2026-07-16) | `spatial_block_split` | same as above | cr_pooled: plot_level CR fit | pinn_noenv | both | superseded by `pinn_noenv_spatialblock_weightsweep` below | `outputs/spatial_block/pinn_noenv/<cohort>/` | Same tuning as above, `physics_weight`/`trajectory_weight` still at the untested default of 1.0. 4survey: RMSE=5.4642, R²=0.5366, trained 42 epochs, best epoch=2. 6survey: RMSE=3.7039, R²=0.7358, trained 48 epochs, best epoch=8. Beats every spatial-block baseline on both cohorts, but DNN still beats PINN on both — turned out to be because `physics_weight`/`trajectory_weight=1.0` was never actually tested against any other value, see below |
| `pinn_noenv_spatialblock_weightsweep` | 2026-07-17 | `spatial_block_split` | same as above | cr_pooled: plot_level CR fit | pinn_noenv | both | primary (this is now the reference PINN result, replacing the row above) | `outputs/spatial_block/pinn_noenv_pw<W>_tw<W>/<cohort>/` for `W` in `{0.0, 0.05, 0.1, 0.2, 0.5, 5.0}` (`W=1.0` is the row above, at the plain `pinn_noenv` path) | Swept `physics_weight`=`trajectory_weight` jointly across 7 values. Best per-cohort: 4survey at `W=0.0` (RMSE=5.0822, R²=0.5991, physics fully off); 6survey at `W=0.05` (RMSE=3.6265, R²=0.7467, beats DNN's 3.6498). Chosen shared default going forward: **`W=0.05`** (4survey RMSE=5.1209, R²=0.5930 — only 0.7% worse than 4survey's own optimum, while being 6survey's actual best). See Findings log for the full table and the val_loss cross-check confirming this isn't test-set cherry-picking |

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

**2026-07-13 — Baseline plot_level result looked strong, but that alone doesn't mean
much yet.**
**What I found:** RF was the best of the four baselines under `plot_level_split`
(68.5%→81.6% accuracy for 4survey, 83.9%→86.9% for 6survey) — expected, since it's the
most flexible model of the four.
**What's working:** all four baselines (CR, average-by-age, linear, RF) fit, evaluate,
and report the full metric suite correctly.
**What's not working / open concern:** a random 60/20/20 split doesn't test whether RF
is learning real biological signal or just exploiting spatial proximity between
train/test plots sitting near each other in the same compartment — the prior
dissertation reported exactly this failure mode (see `[[project-lynch-2025-rf-spatial-generalization]]`
in memory).
**What this means for what's next:** don't trust the plot_level number as "the"
result — test RF (and the others) under a spatial holdout that can't be gamed by
proximity, before drawing any conclusion about which model is actually best.

**2026-07-14 — Spatial holdout shows RF's plot_level advantage was partly proximity,
not generalization.**
**What I found:** under `spatial_block_split` (whole compartments held out, buffered),
RF's RMSE got +28.7% / +19.4% worse than under plot_level — a real, substantial drop;
the simpler models degraded less.
**What's working:** the buffer-distance leakage protection is verified
programmatically (KDTree nearest-neighbour re-check), so this is a genuine
generalization result, not a split-implementation artefact.
**What's not working / open concern:** RF is not the safe default "best" model once
spatial generalization is the actual question — the flexibility that won under random
splitting becomes a liability once train and test plots are truly spatially
independent.
**What this means for what's next:** spatial (and temporal) generalization, not the
plot_level number, has to be the headline comparison in the write-up. Also the first
concrete evidence for the dissertation's central premise — that an unconstrained
flexible model can look good and still generalize badly, which is the whole reason a
physics-constrained model (PINN) is worth building and comparing.

**2026-07-15 — Temporal holdout is a much harder test than spatial, and the
physically-constrained model (CR), not RF, is the most robust one.**
**What I found:** under `temporal_split` temporal_wide_gap (train on the two earliest years,
test 11 years later), degradation was far larger than under the spatial split — up to
+141.7% RMSE for some models, and `average_by_age`'s R² went negative on 6survey
(worse than predicting the mean). CR — the simplest, most constrained model —
degraded the least of the four.
**What's working:** a clean, interpretable pattern: the more flexible a model is, the
worse it generalizes across an 11-year mostly-out-of-distribution gap, while the
physically-constrained growth curve holds up comparatively well.
**What's not working / open concern:** this implies a flexible DNN is likely to fail
this exact test at least as badly as RF did, maybe worse — a real risk for the
DNN-vs-PINN comparison this dissertation is actually built around.
**What this means for what's next:** this is the direct empirical motivation for the
PINN, not just a theoretical one — if the CR constraint is what's protecting
generalization here, a network with that *same* physical constraint built in should
be more temporally robust than the plain DNN. This is why `temporal_split` temporal_wide_gap
(not `spatial_block_split`) was chosen as the primary DNN-vs-PINN test (see Decisions
log below).

**2026-07-15/16 — Forester consultation resolved an open age-filtering question;
re-verified on a real cluster run.**
**What I found:** the existing `Age >= 20` filter was too permissive at the young end
— a forester confirmed LiDAR top-height is "unrelated to age and competition" before
~30 years old (why surveyors are told to estimate Yield Class instead, around that
age) — but old-age (70+) spread is real allometric heteroskedasticity, not a data
problem, so no upper cap was needed.
**What's working:** implemented as a plot-level maturity gate (`age_2023 >= 30`)
rather than a row-level cut, which keeps 81.0%/99.1% of rows (vs. 63.3%/55.2% under a
strict row-level `Age >= 30`) — confirmed on a real cluster run removing exactly
54,616 / 768 rows as predicted, with correct 60/20/20 split counts and sensible CR
refits downstream.
**What's not working / open concern:** the plot-level rule keeps some young (<30)
rows from long-tracked plots, so it doesn't perfectly implement the forester's stated
per-measurement mechanism — documented explicitly in `progress_notes.md` as a stated
limitation, not hidden.
**What this means for what's next:** this filter was already the default inside
`filter_data()` before this conversation (used by baselines and, via
`torch_data.py`, DNN/PINN alike), so the DNN/PINN smoke tests already reflect it —
nothing needs re-running because of the filter itself. What's still outstanding is
the full-length training run (below).

**2026-07-16 — First real DNN-vs-PINN result: both massively beat every baseline, but
the physics constraint doesn't win outright — it's a split decision.**
**What I found:** real (`--max-epochs 500`, early-stopped) runs on cluster GPU, both
models, both cohorts:

| Cohort | DNN RMSE / R² | PINN RMSE / R² | Winner |
|---|---|---|---|
| 4survey | 5.8648 / 0.4524 | 6.0748 / 0.4125 | DNN |
| 6survey | 4.9182 / 0.2810 | 4.7717 / 0.3232 | PINN |

Best baseline RMSE under the same `temporal_split` was 6.3467 (linear, 4survey) and
5.3494 (RF, 6survey) — both DNN and PINN comfortably beat every one of the four
baselines on both cohorts.
**What's working:** the DNN-vs-PINN comparison is not degenerate — both models
clearly outperform the baselines that struggled under temporal generalization,
confirming a gradient-descent model *can* generalize across the 11-year gap far
better than RF/CR/linear/average-by-age did. PINN's loss curves also look healthy:
`physics_loss`/`trajectory_loss` stay small and non-dominating throughout (never
swamping `data_loss`), and best-epoch position is unremarkable (epoch 19/39 for
4survey, 89/109 for 6survey) — no sign of the physics terms destabilising training.
**What's not working / open concern:** two things, and looking at the actual loss
curves (`baseline_models_parameter_tuning.ipynb` §3) changes the read on both. (1)
The physics constraint helps on 6survey but *hurts* slightly on 4survey by final
RMSE — but DNN's 4survey curve is a textbook overfitting climb: `val_loss` rises
almost monotonically from 0.51 to 0.61 after epoch 1, while `train_loss` keeps
falling — so `best_epoch=1` means the reported 5.8648 RMSE came from essentially
untrained, epoch-1 weights, not a converged model. DNN's 6survey curve (4 training
years: 2002/2006/2008/2012) looks completely different — `val_loss` drops
substantially over ~15 epochs before plateauing, best epoch sits mid-plateau (34/54),
a normal-looking fit. The difference tracks the number of distinct training years
available (2 for 4survey vs. 4 for 6survey), not a fluke of one run. (2) PINN's
4survey `val_loss`, on the *same* narrow 2-year training set, stays roughly flat
(noisy, ~0.54–0.58) for all 39 epochs — no overfitting climb — while
`physics_loss`/`trajectory_loss` stay small and non-dominating throughout (never
distorting the fit). That looks like the physics terms acting as a regularizer
against overfitting on a narrow training set, even though it doesn't show up as a
better final RMSE than DNN's (because DNN's number isn't really a trained model's
number at all).
**What this means for what's next:** (a) don't write up "DNN beats PINN on 4survey"
from the RMSE table alone — it's closer to "an undertrained DNN happened to score
better than a properly-trained PINN," a materially different claim once the curves
are read alongside the numbers. (b) DNN/4survey needs an actual fix, not just a
caveat: try a lower initial learning rate or stronger `L1_COEFFICIENT`/dropout so it
doesn't overfit two years of data within one epoch, then re-evaluate before citing
its RMSE at all. (c) the flat-vs-climbing `val_loss` contrast between DNN and PINN on
the *same* 4survey data is itself a citable qualitative result about the physics
term's regularizing effect, independent of whichever model wins on RMSE — worth
stating explicitly in the write-up rather than only reporting the RMSE table. (d)
cr_matched (temporal-restricted CR anchor) is still worth running to check whether
4survey's PINN result is sensitive to the CR anchor being fit on 2021/2023 data it
never trains on.

**2026-07-16 — Realized `temporal_split` was never meant to be the dissertation's
main experiment for DNN/PINN — `spatial_block_split` was, and it wasn't wired in.**
**What I found:** all the framing above (the temporal-gap and CR-anchor choices) is
about which *temporal* setup to prioritize — but per the dissertation's actual research question
(spatial/environmental attribution, not primarily temporal extrapolation),
`spatial_block_split` is the more central test, with `temporal_split` an important
secondary one. Checked `models/common/torch_data.py::load_split_table()`: it
hardcoded `temporal_split()` with no split-type parameter at all, so DNN/PINN
literally could not have been run under `spatial_block_split` even if asked to.
**What's working:** `run_baselines.py` already had this exact split-type-parameter
pattern for the four sklearn/CR models — reused directly rather than inventing a
second convention. Also caught (before it caused a real bug): DNN/PINN's fit/evaluate
scripts hardcoded the string `"temporal"` in every `run_logging` call — harmless
while only temporal ran, but would have silently mislabeled every spatial-block run
in `outputs/run_logs/` if left in place.
**What's not working / open concern:** the PINN's trajectory-consistency loss needed
a real generalization, not just a parameter passthrough — `load_trajectory_pairs()`
used to filter pairs by an explicit `train_years` list, which only makes sense when
train years are fixed ahead of time (temporal). Under `spatial_block_split`, whole
plots (not years) move together, so the correct rule is "both endpoints of a pair
must themselves be labelled train" — verified this produces byte-identical behavior
to the old code under `temporal_split` (same 58,112-row-per-year training counts) and
correct behavior under `spatial_block_split` (all four survey years present in
training data, since a train-plot keeps every year) via an isolated smoke test
before touching any real output.
**What this means for what's next:** `--split-type spatial_block` now works on both
fit scripts (`run_dnn_noenv.py`/`run_pinn_noenv.py`) and both evaluate scripts;
existing real temporal-split results were moved to `outputs/temporal/{dnn,pinn}_noenv/<cohort>/`
to match the baselines' path convention (DNN/PINN never run `plot_level_split`, so
their path is now always split-type-prefixed, never bare). Next real step: run
`--max-epochs 500` under `spatial_block_split` for both models, both cohorts — the
first result that actually speaks to this dissertation's central question, not just
its secondary one.

**2026-07-16 — First `spatial_block_split` result for DNN/PINN: both beat every
baseline, and both beat their own `temporal_split` numbers.**
**What I found:** DNN and PINN both outperform all four sklearn/CR baselines on both
cohorts under `spatial_block_split` (DNN: RMSE 5.07/3.62 vs. best baseline 5.82/3.89;
PINN: RMSE 5.49/3.72). Both also score noticeably better than their own
`temporal_split` numbers from the previous entry (e.g. DNN/4survey R² 0.60 here vs.
0.45 under temporal) — despite training for far fewer epochs (21-30, vs. up to 109
under temporal) and `best_epoch` landing very early (1-10).
**What's working:** the early stopping isn't a sign of a broken/undertrained model —
`spatial_block`'s training set spans all four survey years for every train-plot
(whole plots move together), vs. `temporal`'s narrow 2-year 4survey training set, so
there's simply a richer, more learnable signal here, and a small network converges to
a *better* result faster. This is the first `(model, cohort)` combination where DNN
beats PINN on *both* cohorts (under `temporal_split` it was a 1-1 split) — worth
noting as a real pattern, not just cohort noise.
**What's not working / open concern:** fast convergence + a long flat tail before
`early_stopping_patience=20` triggers means it's not yet confirmed whether these are
genuinely converged optima or premature stops driven by noisy per-epoch `val_loss`
(`lr_scheduler_patience=10` is tight enough that the LR barely gets to step down
before stopping cuts in). Can't yet tell "already as good as it gets" from "stopped
before it could do better" from this data alone.
**What this means for what's next:** run the hyperparameter-tuning checklist below
before treating these numbers as final — cheap to test given each run only takes
15-30 minutes at this epoch count. If loosening patience/smoothing early stopping
doesn't change the result, that confirms genuine convergence and these numbers stand
as the primary `spatial_block` result.

### Hyperparameter-tuning plan (next — applies to both split types, not just spatial_block)

Widened after checking all 8 real runs so far, not just the `spatial_block` ones —
**every single one stops well short of the 500-epoch budget**, `temporal` included:

| Split | Model | Cohort | Total epochs | Best epoch | `val_loss` start→end |
|---|---|---|---|---|---|
| temporal | dnn_noenv | 4survey | 21 | 1 | 0.51 → 0.61 (**+19%, climbs**) |
| temporal | dnn_noenv | 6survey | 54 | 34 | 0.65 → 0.49 (improves, healthy) |
| temporal | pinn_noenv | 4survey | 39 | 19 | 0.56 → 0.54 (flat, healthy) |
| temporal | pinn_noenv | 6survey | 109 | 89 | 0.65 → 0.50 (improves, healthy) |
| spatial_block | dnn_noenv | 4survey | 30 | 10 | 0.28 → 0.30 (mild drift) |
| spatial_block | dnn_noenv | 6survey | 23 | 3 | 0.21 → 0.22 (mild drift) |
| spatial_block | pinn_noenv | 4survey | 22 | 2 | 0.36 → 0.34 (flat) |
| spatial_block | pinn_noenv | 6survey | 21 | 1 | 0.21 → 0.23 (mild drift) |

**DNN is the unstable one of the two, not PINN.** DNN's results are bimodal: its best
case (`temporal`/6survey) is the healthiest-looking curve of all eight, genuinely
improving over 34 epochs — but its worst case (`temporal`/4survey) is the single
worst curve of all eight, a textbook overfitting collapse (`val_loss` +19%, monotonic,
never recovers). PINN never produces a collapse anywhere near that severe — its worst
drift (`spatial_block`/6survey, +12%) is still under DNN's best-case improvement
in magnitude. Across settings, PINN is the more *consistent* model even where neither
model is dramatically better on final RMSE — the physics terms appear to be doing
real regularizing work, not just adding compute cost.

**Decision: focus tuning effort on PINN, not DNN, going forward.** DNN's instability
makes it a less reliable target to spend tuning budget on — a config that fixes
4survey's collapse might just be masking the same instability elsewhere, whereas
PINN's physics constraint is both the actual research contribution of this
dissertation and the more stable starting point to tune from. DNN stays in as the
baseline comparison (that comparison is still the point), but isn't the priority for
hyperparameter search itself.

Cheapest-to-test first — each of these is a single changed constant, not a code
redesign, and applies to both split types:

1. **Loosen both patience knobs.** `lr_scheduler_patience=10`,
   `early_stopping_patience=20` (`models/dnn_noenv/dnn_noenv.py`,
   `models/pinn_noenv/pinn_noenv.py`) are tight relative to how noisy `val_loss` looks
   epoch-to-epoch, and relative to a 500-epoch budget essentially none of the eight
   runs came close to using. Try `lr_scheduler_patience=15`, `early_stopping_patience=40`.
2. **Smooth the early-stopping criterion.** Currently a single noisy epoch counts
   toward the patience counter (`fit()`'s `is_new_best` check compares the raw
   per-epoch `val_loss`). Tracking a short moving average (e.g. 3-5 epoch window)
   instead would stop one noisy epoch from triggering/counting toward a stop.
3. **Increase `BATCH_SIZE`** (currently 128) for smoother, less noisy gradient
   estimates — combined with #2, should clarify whether more training actually helps.
4. **Try SGD+Nesterov momentum as an alternate optimizer to Adam.** Not an addition
   to what's already there — Adam already is a momentum method (β1=0.9 default first-
   moment estimate) — this is a genuinely different optimizer, worth an A/B test
   after #1-3, to see if it finds a different (possibly better-generalizing) minimum.

If none of these meaningfully change the result, that itself is useful evidence the
current numbers reflect genuine convergence, not an artefact of stopping too early.

**Forward-looking note**: whatever combination of these actually fixes/improves PINN's
training dynamics here (no-environment, age + thinning only) should carry over
directly once terrain/wind features are added for the full Env-PINN — same
architecture, same optimizer, same loss structure, just a wider feature set. Worth
re-checking the same instability/convergence questions again at that point rather
than assuming they're automatically solved, but this round of tuning isn't throwaway
work specific to the no-environment version.

**2026-07-17 — Ran the hyperparameter-tuning plan above: confirms genuine convergence
for 3 of 4 configs, and shows DNN's overfitting collapse is a data limit, not a
tuning limit.**
**What I found:** re-ran all four DNN/PINN × split-type combinations with the tuned
settings (`lr_scheduler_patience` 10→15, `early_stopping_patience` 20→40, `BATCH_SIZE`
128/32→512/128, added `WEIGHT_DECAY`/`GRAD_CLIP_MAX_NORM`, 5-epoch `val_loss`
smoothing for the best-epoch decision). Final test RMSE/R² barely moved anywhere
despite training roughly 2x longer (see the Experiment table rows above for exact
numbers) — except DNN/4survey's `temporal` overfitting climb, which only softened
(`best_epoch` 1→3, climb +19%→+12%), it didn't go away.
**What's working:** for `spatial_block` (both models, both cohorts) and 3 of 4
`temporal` configs, the pre-tuning numbers were already close to real convergence —
not premature stops. `dnn_noenv`/`temporal`/6survey's curve got dramatically
healthier mid-training (`val_loss` -42.9% instead of the pre-tuning -24.6%) even
though its final RMSE barely changed, confirming more training years (not more
patience) is what actually drives a healthy curve.
**What's not working / open concern:** DNN/4survey's collapse persisting even after
loosening every training-dynamics knob available means it isn't an optimization
problem — `temporal`'s 4survey split has only 2 distinct training years (2008, 2012),
which is underdetermined for learning a trajectory almost by definition, independent
of how carefully you train on it.
**What this means for what's next:** don't spend more tuning budget trying to fix
DNN/4survey's collapse with training-dynamics knobs — it needs `temporal_narrow_gap`
(a genuinely different, less extreme train/test gap) to actually test whether the
problem is "any temporal split breaks this" or specifically "an 11-year extrapolation
from 2 years breaks this." These tuned numbers replace the pre-tuning ones in the
Experiment table above (same rows, updated in place per this file's convention).

**2026-07-17 — Physics/trajectory weight sweep: the untested default (`W=1.0`) was
quietly hurting PINN the whole time; properly weighted, PINN beats DNN on 6survey.**
**What I found:** `PHYSICS_WEIGHT`/`TRAJECTORY_WEIGHT` had been fixed at `1.0` since
the PINN was first built — never tuned, never tested against any other value.
Swept both jointly across `{0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0}` under
`spatial_block`, both cohorts:

| W | 4survey RMSE / R² | 6survey RMSE / R² |
|---|---|---|
| 0.0 | 5.0822 / 0.5991 | 3.6402 / 0.7448 |
| 0.05 | 5.1209 / 0.5930 | **3.6265 / 0.7467** |
| 0.1 | 5.1562 / 0.5873 | 3.6296 / 0.7463 |
| 0.2 | 5.1319 / 0.5912 | 3.6302 / 0.7462 |
| 0.5 | 5.3522 / 0.5554 | 3.6918 / 0.7375 |
| 1.0 (old default) | 5.4642 / 0.5366 | 3.7039 / 0.7358 |
| 5.0 | 5.7476 / 0.4872 | 3.7601 / 0.7277 |
| DNN (reference) | 5.0185 / 0.6091 | 3.6498 / 0.7434 |

4survey's best result is physics fully **off** (`W=0.0`) — every value above 0 makes
it monotonically worse. 6survey wants a small nonzero weight (`W=0.05`), which beats
DNN outright (3.6265 vs. 3.6498).
**What's working:** cross-checked against `val_loss_smoothed` (computed during
training, never touches the test set) — it shows the exact same ranking for both
cohorts (4survey monotonically best at 0.0, 6survey best at 0.05), so this isn't
test-set cherry-picking; the same choice would follow from validation data alone.
**What's not working / open concern:** the cohorts genuinely disagree on whether the
physics constraint helps at all. Most likely explanation, consistent with earlier
findings: 6survey has more distinct training years, so its trajectory pairs carry a
real, learnable signal the physics term can anchor to; 4survey's pairs are thinner
and the same constraint mostly adds noise instead of biological signal.
**What this means for what's next:** chose **`W=0.05`** as a single shared default
for both cohorts, rather than tuning per-cohort — it's 6survey's actual optimum and
only 0.7% worse than 4survey's own optimum (5.1209 vs. 5.0822), a small price for
keeping "one PINN configuration" a clean, defensible claim (same reasoning as
keeping architecture fixed across splits). This becomes the new primary PINN
`spatial_block` result (see `pinn_noenv_spatialblock_weightsweep` row above). Two
things this does NOT settle: (a) whether the same weight is right for `temporal`
(not yet swept there — added to the Planned robustness checks table), and (b)
whether `W=0.05` still holds once Env-PINN adds terrain/wind features and the
sub-network changes what `y_max` means — worth re-sweeping there rather than
assuming it carries over.

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

**2026-07-17 — `physics_weight`/`trajectory_weight=0.05` (jointly) chosen as the
PINN's shared default, replacing the never-tested `1.0`.** Swept 7 values under
`spatial_block`; 4survey's true optimum is `0.0` (physics fully off) and 6survey's
is `0.05`. Picked one shared value for both cohorts rather than tuning per-cohort,
since `0.05` costs 4survey almost nothing (0.7% RMSE) while being 6survey's actual
best and letting PINN beat DNN there for the first time. A per-cohort-tuned pair of
hyperparameters would be harder to defend as "the PINN" in the write-up than one
number that's very close to optimal everywhere. Not yet decided: whether to change
`PHYSICS_WEIGHT`/`TRAJECTORY_WEIGHT`'s code defaults in `pinn_noenv.py` from `1.0` to
`0.05` (so future runs don't need `--physics-weight 0.05 --trajectory-weight 0.05`
passed explicitly) — flagged for a decision, not done automatically as part of this
logging update.

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
- **temporal_narrow_gap** (different train/val/test years), if run: `outputs/temporal_narrow_gap/<model>/<cohort>/`
  — a distinct split-type-style prefix, never overwriting `outputs/temporal/...` (temporal_wide_gap).
- **PINN cr_matched** (temporal-restricted CR anchor), if run: a distinct model name,
  `outputs/pinn_noenv_crmatched/<cohort>/` — since this isn't a different split, it's a different
  PINN configuration, so it gets a model-name suffix rather than a split-type prefix. The
  `run_metadata.json`'s `frozen_cr_params` field already records exactly which values were used
  either way, but a distinct output path is required so cr_pooled and cr_matched results can coexist on
  disk rather than one overwriting the other.
- **PINN `physics_weight`/`trajectory_weight` sweep** (2026-07-17), same reasoning as
  `cr_matched` above: `--run-name pinn_noenv_pw<W>_tw<W>` on `run_pinn_noenv.py`/
  `evaluate_pinn_noenv.py` writes to `outputs/<split_type>/pinn_noenv_pw<W>_tw<W>/<cohort>/`,
  never touching the plain `pinn_noenv` path — see `models/pinn_noenv/run_pinn_noenv.py`'s
  `run_name` handling (data loading always uses the plain `pinn_noenv` table; only the
  output path and `run_logs` identity change). `W=1.0` has no suffix (it's the
  historical default, living at the plain path); the chosen `W=0.05` result lives at
  `outputs/spatial_block/pinn_noenv_pw0.05_tw0.05/<cohort>/`, not at the plain path —
  promoting it there would mean overwriting a checkpoint, which this file's own
  2026-07-16 near-miss (see progress notes) is reason enough to avoid doing casually.
- For the **baselines**, whichever configuration is primary for the write-up lives at
  the plain, unprefixed path; for **DNN/PINN**, `temporal` and `spatial_block` are
  both always prefixed, so "which one is primary" is a fact to check in this log's
  Experiment table (Status column), not something the path itself tells you.
