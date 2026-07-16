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
| `dnn_noenv_design1_smoketest` | 2026-07-15 | `temporal_split`, Design 1 | same as above | n/a (no physics term) | dnn_noenv | both | superseded (2-epoch sanity check only, not a real result) | `outputs/dnn_noenv/<cohort>/` | Pipeline verified working; superseded by `dnn_noenv_design1` below |
| `pinn_noenv_design1_routeA_smoketest` | 2026-07-15 | `temporal_split`, Design 1 | same as above | Route A: plot_level CR fit (`outputs/chapman_richards/<cohort>/params.json`) | pinn_noenv | both | superseded (2-epoch sanity check only) | `outputs/pinn_noenv/<cohort>/` | Pipeline verified working; superseded by `pinn_noenv_design1` below |
| `dnn_noenv_design1` | 2026-07-16 | `temporal_split`, Design 1 | same as above | n/a (no physics term) | dnn_noenv | both | primary | `outputs/dnn_noenv/<cohort>/` | First real (`--max-epochs 500`, cluster GPU) run. 4survey: RMSE=5.8648, R²=0.4524, early-stopped epoch 21 with **best epoch=1** (val_loss never improved after epoch 1 — flagged, see Findings log). 6survey: RMSE=4.9182, R²=0.2810, early-stopped epoch 54, best epoch=34 (normal pattern). Both beat all four temporal-split baselines |
| `pinn_noenv_design1` | 2026-07-16 | `temporal_split`, Design 1 | same as above | Route A: plot_level CR fit | pinn_noenv | both | primary | `outputs/pinn_noenv/<cohort>/` | First real run. 4survey: RMSE=6.0748, R²=0.4125, early-stopped epoch 39, best epoch=19 (normal). 6survey: RMSE=4.7717, R²=0.3232, early-stopped epoch 109, best epoch=89 (normal). Both beat all four temporal-split baselines |

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
**What I found:** under `temporal_split` Design 1 (train on the two earliest years,
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
be more temporally robust than the plain DNN. This is why `temporal_split` Design 1
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
Route B (temporal-restricted CR anchor) is still worth running to check whether
4survey's PINN result is sensitive to the CR anchor being fit on 2021/2023 data it
never trains on.

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
