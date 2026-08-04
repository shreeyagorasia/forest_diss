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
| `xgb_elasticnet_environmental_2026-07-29` | 2026-07-29 | `spatial_block_split` (train/val/test all used; val for feature decisions, test read once) | n/a (all years pooled per plot, mean CR residual target) | cr_pooled: plot_level CR fit (re-derived on `elev_percentile_95th`) | xgb_environmental (XGBoost+SHAP), elasticnet_environmental (ElasticNetCV) | both (4survey primary) | primary (replaces the retired-pipeline row) | `outputs/spatial_block/xgb_environmental/<feature_set>/<cohort>/`, `outputs/spatial_block/elasticnet_environmental/<feature_set>/<cohort>/` | Re-run against the new target + `yldc` removed from the 34-variable unified environmental+silviculture feature set (`Age` still excluded — circular with the CR residual, see Findings log). 4survey `all_environmental`: XGBoost val R²=0.734/test R²=0.629, Elastic Net val R²=0.700/test R²=0.671. 6survey `all_environmental`: XGBoost val R²=0.107/test R²=0.398, Elastic Net val R²=0.147/test R²=0.521 — 6survey's val R² sits well below its own test R² across every feature set for both model types (opposite of the usual overfitting direction); likely which compartments `spatial_block_split` happened to assign to val vs test for the smaller cohort, not yet investigated further. 4survey grouped permutation importance (`av1_grouped_category_importance.ipynb`): `neighbour_spatial_lag` dominates (mean R² drop=1.177, ~10x every other category), then climate/stand_structure/terrain clustered close together (0.11-0.12), wind least (0.023). Full-model residual Moran's I=0.197 (p=0.005) -- still significant spatial autocorrelation left unexplained; removing `terrain` increases it the most (Δ=0.066), even though terrain isn't top for raw accuracy -- a genuine cross-method disagreement (matters for spatial pattern, not for prediction). |
| `baselines_rebuild_2026-07-28` | 2026-07-28 | `plot_level`, `spatial_block`, `temporal` (wide-gap), `temporal_narrow_gap` -- all four re-run | same year assignments as the retired-pipeline rows above | n/a | CR, average-by-age, linear, RF | both | primary (replaces every baseline number above) | `outputs/<split_type or nothing>/<model>/<cohort>/` | New target (`elev_percentile_95th`) + `yldc` removed from RF/linear. `plot_level`: RF best (R²=0.570) as before. `spatial_block`: RF loses its advantage to linear (R²=0.475 vs 0.512) as before -- same qualitative pattern as the retired pipeline, confirming the rebuild didn't change which baseline "wins" per split, just the absolute numbers. Chapman-Richards fit also fixed a pre-existing degeneracy (y_max was landing exactly on the observed max height under both old and new target) -- lower bound now `max_observed_height * 1.001`. Full reasoning: `progress_notes.md`'s 2026-07-28 entry |
| `dnn_pinn_epochcheck_2026-07-29` | 2026-07-29 | `spatial_block` | 4survey only, short smoke tests (max 150 epochs, patience 40) | cr_pooled | dnn_noenv, pinn_noenv, 3 PINN weight variants | 4survey only | diagnostic, not a result -- see Findings log | `outputs/spatial_block/{dnn,pinn}_noenv_epochcheck*/4survey/` | Base-case (`W=1.0`) DNN/PINN cluster jobs came back suspiciously fast (~53s, later traced to a missing rsync of `data/processed/transitions/`, fixed). Once fixed: DNN converges normally (val_loss 0.342→0.331 over ~50 epochs, patience stops at 52). PINN never beats its own epoch-1 val_loss at `W=1.0`, `W=0.0`, OR `W=0.05` (best_val_loss ≈ epoch-1's value in all three) -- ruled out physics weight as the cause. Found the real confound: `pinn_noenv.py`'s `BATCH_SIZE=128` vs `dnn_noenv.py`'s `BATCH_SIZE=512`, undocumented, never controlled for. Exposed `--batch-size`/`--pairs-batch-size` as CLI args (previously hardcoded) to test batch-size-matched. Next: rerun `physics_weight=0.0` at `--batch-size 512` on the cluster (see below) to isolate batch size from the physics-weight question properly. |
| `dnn_pinn_basecase_2026-07-30` (Stage 2) | 2026-07-30 | `spatial_block`, `temporal` | full pipeline, `batch_size=256` both models | cr_pooled | dnn_noenv, pinn_noenv (`physics_weight=trajectory_weight=1.0`, the untested default) | both | primary | `outputs/{spatial_block,temporal}/{dnn_noenv,pinn_noenv_basecase_w1}/<cohort>/` | Real base-case rebuild against the current pipeline (finally superseding the epochcheck smoke tests). Test R²: 4survey spatial_block DNN=0.633 vs PINN(w=1)=0.580; 6survey spatial_block DNN=0.750 vs PINN(w=1)=0.734; 4survey temporal DNN=0.354 vs PINN(w=1)=0.284; 6survey temporal DNN=0.284 vs PINN(w=1)=0.209. DNN beats PINN(w=1) by a real, consistent margin in all 4 cohort×split combinations -- confirms the long-believed "physics constraint hurts at full weight" finding (point 3 in `handover_2026-07-18.md`) survives the full target/yldc/batch-size rebuild, not just true under the retired pipeline. |
| `growth_curve_broad_static_cv_2026-08-04` | 2026-08-04 16:08 | 5-fold compartment-held-out spatial CV + 60 m leakage buffer | n/a (all years pooled per plot into one `local_y_max_difference` target) | each plot gets its own fixed-shape curve; no pooled CR baseline involved | Elastic Net, XGBoost | both (`4survey` primary) | robustness-check / descriptive extension | `outputs/growth_curve_attribution/{broad_environmental_spatial_cv_4survey.csv,broad_environmental_spatial_cv_6survey.csv,terrain_wind_management_comparison.csv,broad_environmental_category_checks_4survey.csv}` | Follow-on static-model expansion of the validated terrain/wind Stage 2 result. `4survey`: terrain/wind = 0.125 / 0.117 (Elastic Net / XGBoost), broad environment = 0.093 / 0.102, terrain/wind + management = 0.289 / 0.302, all 38 static variables = 0.290 / 0.318. `6survey`: terrain/wind = 0.023 / 0.021, broad environment = 0.019 / 0.027, terrain/wind + management = 0.079 / 0.101, all 38 = 0.110 / 0.094. Category-addition reruns were only written for `4survey`: +climate = 0.112 / 0.127, +soil/site = 0.117 / 0.099, +edge-position = 0.112 / 0.121. This run family is deliberately only Elastic Net + XGBoost; no DNN/PINN was tried for this target extension. |
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

> **[date] — Q: [the question this entry investigates] — [one-line headline of the finding].**
> **What I found:** the actual result, with numbers.
> **What's working:** what this confirms is sound / trustworthy.
> **What's not working / open concern:** what this exposes as a problem, risk, or
> unanswered question.
> **What this means for what's next:** the concrete decision or next step this
> directly caused — this is the line that turns a result into a research narrative.

**Titling rule (added 2026-08-03, user request):** lead every headline with `Q: <question
being investigated>` before the finding itself. Multiple lines of investigation run in
parallel in this project (different sessions, different agents) — a question-first title
lets entries be scanned and grouped by thread even when they land interleaved by date, not
just read as one continuous chronological narrative.

---

**2026-08-04 — Q: does adding the newer, already-extracted GWA Weibull + multiscale terrain
candidates (`data_processing/add_environmental_candidates.py`) to `grouped_category_importance
.ipynb`'s feature set improve or clarify the environmental attribution picture -- no, it exposes a
multicollinearity instability that needs consolidating before any new ranking is trusted.**
**What I found:** Added 12 previously-unused-but-already-validated columns (real per-plot
variance confirmed on disk before adding) to `FEATURE_PROVENANCE`/`ALL_FEATURE_COLUMNS`
(`xgb_environmental.py`) and `CATEGORY_GROUPS` (`grouped_analysis.py`): GWA Weibull wind
(`gwa_weibull_a/k_10m`, `gwa_wind_p95_10m`, `gwa_prob_above_critical_10m`, and the 50m
equivalents) and multiscale terrain (`tpi_250m`, `tpi_500m`, `local_relief_500m`) -- this is
exactly what the notebook's own "Future wind screen" planning cell (added 2026-08-03) already
called for, just waiting on the extraction step, which was already done. Re-ran the notebook's
existing pipeline unchanged (correlation/Elastic Net/XGBoost/permutation/SHAP/ALE/spatial-CV/
Moran's-I/refit-ablation) against the expanded 49-variable set. Real result: the `wind`
category's genuine-refit ablation (`r2_drop`) flips from a small positive contribution to
**-0.076** (now ranked worst of 6 categories, below climate) once all 9 GWA additions sit
alongside the 4 pre-existing wind columns -- while at the SAME time, one individual wind variable
(`gwa_wind_speed_10m`) shows the single LARGEST per-variable `r2_drop` (+0.274) of all 49
variables, a dramatic reversal from its previous near-zero/negative individual score. Both
numbers can't be telling a coherent "wind matters" story at once -- this is the textbook signature
of multicollinearity destabilising single-variable ablation while the whole correlated bundle
nets negative.
**What's working:** the redundancy the notebook's own cell 45 anticipated ("check GWA A/k_w/mean
wind/p90/p95 for redundancy... do not put several deterministic transformations of the same two
parameters into the headline model") is now directly measurable, not hypothetical --
`gwa_weibull_a_50m`/`gwa_wind_speed_50m`/`gwa_wind_p95_50m` all correlate with `elevation` at
rho=0.66-0.73, and `tpi_500m` correlates with `topex` at rho=-0.777. A PyALE library bug
(`IndexError` on a categorical feature, `ceh_pedotope`, that newly qualified for the top-4
refit-confirmed ALE plot) was caught and worked around with a try/except that skips just that
one subplot -- doesn't touch any correlation/importance/SHAP number, only a plotting cell.
**What's not working / open concern:** this independently confirms, on a DIFFERENT target
(`mean_cr_residual`, not the per-plot growth-curve deviation a parallel session's own council
reviewed the same GWA bundle for), that raw-adding all 9 GWA columns causes real instability, not
a clean improvement -- convergent evidence across two different targets/pipelines. `gwa_wind_speed_10m`
is also flagged (2026-08-04, cross-referenced from that parallel session's council) as
conceptually backwards for a mature canopy (sub-canopy wind measurement) -- now doubly suspect:
wrong in principle AND behaving erratically in practice.
**What this means for what's next:** do NOT treat this run's per-variable refit-ablation ranking
as final -- the GWA wind bundle needs consolidating (single best representation, or a PCA/VIF-
based reduction) before re-running, matching what the parallel session's own council on the
per-plot target already concluded independently. Not re-litigated here to avoid duplicating that
session's work -- see its own variable-selection conclusions once finalised. `terrain` and
`spatial_position_edge_effects` remain the two categories every method agrees are genuinely
useful (`refit_r2_drop` 0.306 and 0.190, ranked 1st/2nd on every one of the four cross-checked
methods), unaffected by the wind-bundle instability.

---

**2026-08-04 — Q: is `pinn_env_terrain_k`'s -0.71 y_max/k correlation inherited from a known
classical curve-fitting identifiability property, or something the neural architecture
introduced -- the classical fit's own correlation is far MORE extreme (-0.99), but it's a
different statistic, so this doesn't cleanly settle the question either way.**
**What I found:** Per the `llm-council` review's top-priority diagnostic (see the 2026-08-03
entry below), added an optional `return_covariance` flag to `models/chapman_richards/
chapman_richards.py::fit()` (default `False`, zero behaviour change for its one existing caller,
`run_baselines.py`) that also returns `curve_fit`'s own 3x3 parameter covariance matrix from the
winning multi-start attempt. New script `models/pinn_env_terrain_k/check_cr_identifiability.py`
reuses `run_baselines.py`'s own train-split construction (same rows the frozen CR anchor was
actually fit on) and computes the classical fit's y_max/k correlation from that covariance.
Result, 4survey/`spatial_block`: classical correlation = **-0.9930** (near-total degeneracy) vs.
`pinn_env_terrain_k`'s own **-0.7109** (from its existing seed-42 `metrics.json`).
**What's working:** the diagnostic ran cleanly and cheaply (no cluster time, reuses existing
fit/split code). It also surfaced a real conceptual point the council's own framing glossed over:
these are NOT the same statistic. The classical number is a single global fit's OWN parameter-
estimation-uncertainty correlation (how jointly uncertain the one true y_max/k point estimate
is); the neural number is a CROSS-SECTIONAL correlation across many different plots' own learned
per-plot (y_max, k) pairs. They answer different questions and aren't expected to match
numerically -- an earlier version of this script's automated "READ" line compared them with a
same-sign/same-magnitude heuristic and produced a confidently wrong verdict ("does not look like
inheritance") before this was caught and the interpretation corrected.
**What's not working / open concern:** because the two numbers aren't directly comparable, this
diagnostic does NOT cleanly resolve the "inherited vs. introduced" question the council posed --
it can only say the classical fit's own extreme correlation (-0.99, near the -1 boundary) is
strong INDEPENDENT evidence that y_max and k are hard to identify separately from this dataset's
height-vs-age shape at all, which makes the neural network hitting a similar ambiguity per plot
MORE plausible, not less -- but doesn't rule out an architecture-specific cause (two identical
sub-networks fed identical terrain inputs) either. 6survey's own `pinn_env_terrain_k` comparison
is still missing -- no checkpoint has been fit for that cohort yet, only 4survey.
**What this means for what's next:** this diagnostic alone can't distinguish "inherited curve-
shape ambiguity" from "architecture artefact" -- the freeze-one-vary-other ablation (y_max pinned
to the global constant, only k learned per plot) is the next, more decisive check, since it
directly tests whether the wide learned-k range persists even with only one degree of freedom
per plot, which the population-correlation number here cannot test on its own.

---

**2026-08-04 — Q: is `pinn_env_terrain_k`'s -0.71 y_max/k correlation just noise from a small
number of held-out compartments -- no, it's real and precise: 95% CI [-0.77, -0.62].**
**What I found:** Every one of the council's 5 peer reviewers independently flagged the same gap:
nobody checked whether -0.71 was even statistically distinguishable from noise. New script
`models/pinn_env_terrain_k/bootstrap_correlation_check.py` reuses the cluster-bootstrap pattern
already proven in `models/growth_curve_attribution/bootstrap_ci_check.py` (resample whole
compartments with replacement, not individual plot rows, since plots in the same compartment
aren't independent) -- deduplicated `predictions.csv` to one row per plot first (learned_y_max/k
are static per plot, repeated across a plot's own survey-year rows; bootstrapping the long table
directly would silently over-count multi-survey plots). 4survey/`spatial_block`, 2000 resamples,
11,739 test plots across 51 compartments: point estimate -0.7109, bootstrap mean -0.7034
(std=0.0404), **95% CI [-0.7737, -0.6163]** -- 0% of resamples landed on the wrong (positive) sign.
**What's working:** this closes one of the two remaining open questions from the 2026-08-03
council review cleanly -- the correlation's sign and rough magnitude are NOT an artefact of a
small, arbitrary test-compartment sample. Whatever is causing it (curve-shape identifiability,
architecture, or genuine ecology), it's a real, stable feature of this fit, not noise.
**What's not working / open concern:** a real, precise correlation still doesn't say WHY it
exists -- this bootstrap can't distinguish identifiability-artefact from architecture-artefact
from genuine ecological signal, only that whichever it is, it's not sampling noise. The same-day
Hessian-covariance check (see the other 2026-08-04 entry) found the classical CR fit's own
y_max/k correlation is even more extreme (-0.99) but is a different statistic (single-fit
estimation uncertainty, not cross-plot population correlation) -- still not conclusive on its own.
**What this means for what's next:** the freeze-one-vary-other ablation (y_max pinned to the
global constant, only k free per plot) is now the clearly decisive remaining check -- with both
"is it noise" (no) and "is it inherited from the classical fit" (ambiguous, different statistics)
answered, only the ablation can directly test whether restricting to ONE free parameter per plot
still produces an implausibly wide learned-k range, which would point at overparameterisation-of-
any-second-knob rather than something k-specific.

---

**2026-08-04 — Q: does `pinn_env_terrain_k`'s wide learned-k range persist even with y_max
completely pinned (the council's freeze-one-vary-other ablation) -- yes, almost as wide as the
full two-parameter model, pointing away from a two-knobs-fighting-each-other explanation.**
**What I found:** Added a `freeze_y_max` flag to `models/pinn_env_terrain_k/pinn_env_terrain_k.py`
(`compute_physics_loss`/`compute_trajectory_loss`/`train_one_epoch`/`fit`, threaded via a new
`--freeze-y-max` CLI flag on `run_pinn_env_terrain_k.py`) -- when set, `y_max_per_row` is pinned
to the plain global CR constant and `y_max_subnetwork` is never called in the loss computation
at all, so it receives zero gradient and stays at its random initialisation for the whole run
(confirmed: its reported "learned_y_max" is then meaningless noise, not a trained value -- only
`learned_k` is a real result from this variant). Full real run (not a smoke test), 4survey/
`spatial_block`, seed=42: converged cleanly in 77 epochs (early-stopped, same shape as every
other healthy run this session). Result: **learned y_max range = 51.96m to 51.96m (numerically
exact constant, freeze confirmed working)**; **learned k range = 0.009116 to 0.018244** (width
0.009128) -- compare to the full two-parameter model's own k range, 0.009965 to 0.021879 (width
0.011914). y_max/k correlation is `+nan` (mathematically correct and expected: correlation with
a true constant is 0/0-undefined, not a bug).
**What's working:** the ablation mechanism itself is clean and verifiable -- a numerically exact
constant y_max (not just "close to constant") proves the sub-network really received zero
gradient, so this is a genuine single-degree-of-freedom test, not an approximation.
**What's not working / open concern:** the frozen-k-only model's k-range (0.0091) is ~77% as
wide as the full model's k-range (0.0119) -- almost as wide, from a SINGLE free parameter per
plot instead of two. This weakens (does not eliminate) the "two knobs fighting over one true
degree of freedom" explanation the council's First Principles Thinker and Outsider raised --
if that were the whole story, pinning y_max should have let k settle into a much NARROWER,
better-identified range once it no longer had to share ambiguity with a second free parameter.
It didn't, materially. This is more consistent with either (a) a single per-plot free parameter
responding to weak/noisy terrain features generally runs this wide regardless of how many knobs
exist, or (b) genuine terrain-driven variation in growth rate across plots -- this ablation alone
cannot distinguish those two.
**What this means for what's next:** combined with the same-day Hessian check (classical fit's
own y_max/k correlation is extremely negative, -0.99, though a different statistic) and the
bootstrap CI (the -0.71 correlation is real, not small-sample noise), the weight of evidence
across all three cheap diagnostics leans toward "this is a real, precise, but hard-to-interpret
property of fitting Chapman-Richards curves to this dataset's height-age shape with ANY per-plot
free parameter" rather than a clean architecture bug specific to having two sub-networks. Given
this, a genuine multi-seed check (5-8 seeds, matching this project's established protocol) is
now better-motivated than before this ablation -- there's no cheap remaining diagnostic left to
run before it, and the open question has narrowed from "is this even real" (resolved: yes) to
"does this exact pattern replicate across seeds," which only multi-seed averaging can answer.

---

**2026-08-03 — Q: can conditioning the Chapman-Richards rate parameter `k` (not just `y_max`) on
terrain/wind let the PINN express "same ceiling, different timing" -- built `pinn_env_terrain_k`,
first (single-seed) result shows a real but strongly confounded effect, needs multi-seed checking
before any conclusion.**
**What I found:** Supervisor (Hermann, via email) asked whether an exponent/rate parameter could
be environment-conditioned, reasoning "trees reach their height eventually but not at the same
time." Checked the ecological literature before picking a parameter (not guessed): `k` is
described as scaling absolute growth rate (plausibly site-sensitive); `p` is tied to catabolic/
allometric scaling theory, typically a fixed biological constant. Built `models/pinn_env_terrain_k/`
(new folder, no shared code with `pinn_env_terrain` or the unrelated Stage 2 per-plot work in
`models/growth_curve_attribution/`) -- adds a second small sub-network (reusing `YMaxSubNetwork`)
outputting a per-plot adjustment to `k`, parameterised multiplicatively in log-space
(`k_per_row = global_k * exp(k_log_adjustment)`, guarantees positivity) alongside the EXISTING
`y_max` adjustment; `p` stays global/frozen. This is a real, deliberate departure from
`pinn_env_terrain`'s own documented citation (Socha et al. 2021's ADA/GADA framework, which
treats the asymptote as the site-varying parameter), not a bug fix.

First result, seed 42, `spatial_block`/4survey, `pw=tw=1.0` (matching how `pinn_env_terrain`
itself was first evaluated): test R2=0.5871 vs `pinn_env_terrain`'s 0.5823 at the same
seed/config vs `dnn_noenv`'s 0.6330. Learned `y_max` range 45.98-52.56m (global anchor 51.96m --
notably WIDER than `pinn_env_terrain` alone ever showed at this weight). Learned `k` range
0.009965-0.021879 (global anchor 0.010369). **`y_max`/`k` correlation across test plots: -0.71**
-- a strong negative correlation, exactly the confound risk flagged when this was designed (a
lower `k` could be mimicking what should be a lower `y_max`, or vice versa, rather than the two
being independently meaningful).
**What's working:** both fit and evaluate scripts run cleanly end to end (smoke-tested before the
real run), training pattern is healthy (61 epochs, standard early-stopping shape, not degenerate).
The `y_max`/`k` correlation diagnostic itself works as designed -- it's there specifically to
catch this kind of confound, and it did.
**What's not working / open concern:** only ONE seed so far. This project has already found,
twice this session (`dnn_env_terrain`'s and then `dnn_terrain_wind`/`dnn_broad_legitimate`'s
terrain-delta), that seed 42 specifically gives a MISLEADING single-seed picture for exactly this
kind of environment-conditioning question -- averaging across seeds reversed the conclusion both
times. No conclusion should be drawn from this one seed given that track record. The -0.71
correlation itself is also not yet interpreted -- could mean the two adjustments are
uninterpretably substitutable (bad), or could reflect a real, coherent relationship (e.g. faster-
growing sites plausibly also reaching a lower realised ceiling within the observed age range) --
not yet distinguished.
**What this means for what's next:** multi-seed check (matching the now-established 5-8 seed
protocol) before any interpretation of either the R2 or the confound. User is restarting the
session to invoke the `llm-council` skill (`.claude/skills/llm-council/SKILL.md`) for an
independent critical review of this design before proceeding further -- not run yet in this
session (`Skill` tool returned "Unknown skill", likely added to `.claude/skills/` after this
session started and not live-reloaded). Resume point for the next session: get the council
review first, then multi-seed `pinn_env_terrain_k` before treating either number above as a real
finding.

---

**2026-08-03 — `dnn_terrain_wind` (real wind variables added) and `dnn_broad_legitimate` (full
vetted feature set) built and multi-seed checked: seed 42 was misleading again, on the exact
same pattern as `dnn_env_terrain` -- the broader feature set is the BEST performer once averaged
across seeds, not worse.**
**What I found:** Added two new `ENV_TERRAIN_FEATURE_SETS` entries (`models/common/torch_data.py`)
reusing the existing `dnn_env_terrain` model unchanged, no new model files: `terrain_wind_full`
(16 cols, exactly `xgb_environmental.TERRAIN_AND_WIND_COLUMNS`, imported not redefined -- adds
`gwa_wind_speed_10m`, `windward_topex`, and `whcl`, the external Windthrow Hazard Class rating,
none of which are in the default `terrain_wind_solid`) and `broad_legitimate` (27 cols, every
`ALL_FEATURE_COLUMNS` entry not already fed via the no-env pathway and not an unencoded
categorical -- `tas_mean`/`groundfrost_mean` also excluded, a real structural gap found by
hitting the error: they're cohort-suffixed columns `load_split_table_with_terrain()` doesn't
resolve, unlike `xgb_environmental`'s own pipeline). First fit, seed 42 only, looked like a clear
regression (`terrain_wind_full` R2=0.6050, `broad_legitimate` R2=0.5876, both below
`dnn_noenv`'s 0.6330 AND below the smaller `terrain_wind_solid`'s 0.6247). Given the exact same
seed-42-is-an-outlier pattern already found for `dnn_env_terrain`, refit both under seeds
43-46 before drawing any conclusion:

| Seed | dnn_noenv | terrain_wind_solid (5) | terrain_wind_full (16) | broad_legitimate (27) |
|---|---|---|---|---|
| 42 | 0.6330 | 0.6247 | 0.6050 | 0.5876 |
| 43 | 0.6415 | 0.7193 | 0.7535 | 0.7551 |
| 44 | 0.6427 | 0.6732 | 0.6619 | 0.6860 |
| 45 | 0.6457 | 0.6997 | 0.7246 | 0.7580 |
| 46 | 0.6418 | 0.6543 | 0.6785 | 0.6789 |
| **mean delta vs no-env** | -- | **+0.0333/SD 0.0338** | **+0.0438/SD 0.0541** | **+0.0522/SD 0.0656** |

**What's working:** confirmed, again, on a genuinely different feature-set question, not just a
replication of the earlier finding: seed 42 is a real, recurring outlier for this whole class of
terrain/environment comparisons on this split, not a one-off. `broad_legitimate` has the LARGEST
mean improvement of the three feature sets on this same 5-seed set (+0.0522, vs `terrain_wind_full`'s
+0.0438 and `terrain_wind_solid`'s +0.0333) -- more features, on average, genuinely helps here,
not hurts. Variance also scales with feature-set size (SD 0.034 -> 0.054 -> 0.066) -- expected,
sensible tradeoff (more capacity to exploit real signal, but also more sensitivity to which
compartments get held out).
**What's not working / open concern:** only 5 seeds for the two new feature sets (vs. 8 for
`dnn_noenv`/`dnn_env_terrain`) -- worth extending to match if this becomes a headline number.
`broad_legitimate`'s missingness (1,720 rows/430 plots dropped) is real and larger than the
other sets' (156 rows/39 plots) -- from climate/soil column gaps, not a bug, but worth knowing
if comparing row counts across configs. The individual contribution of `windward_topex`/`whcl`
specifically (vs. the whole 16-27 column bundle) is still unknown -- this only tested "does
adding the whole bundle help," not which specific variable is doing the work.
**What this means for what's next:** the wind-damage/`whcl` hypothesis from the worst-error
investigation is not rejected by this result -- if anything, the broader feature set (which
includes `whcl`) is the best performer on average. A per-variable ablation (does removing
`windward_topex`/`whcl` specifically from `terrain_wind_full` hurt) would isolate whether it's
those wind-specific columns doing the work or the extra columns generally. Not yet run.

---

**2026-08-02 — Split-seed robustness check: the "terrain regresses `dnn_env_terrain`" finding
that motivated most of today's investigation does NOT hold up across `spatial_block_split`
seeds -- it was true for seed 42 specifically, not true on average.**
**What I found:** Every result today (scope-matched XGBoost check, feature-set-parity check,
`env_deviation`, the leak-safe spatial-lag test, the mechanism critique of the PINN's physics
loss) used the single fixed `SPLIT_SEED=42` partition of `spatial_block_split` -- never varied,
across the whole project, until now. Exposed `--split-seed` on `load_split_table()`/
`load_split_table_with_terrain()` (`models/common/torch_data.py`, default unchanged so every
existing call/result is unaffected) and threaded it through `run_dnn_noenv.py`/
`evaluate_dnn_noenv.py`/`run_dnn_env_terrain.py`/`evaluate_dnn_env_terrain.py`,
`run_pinn_noenv.py`/`evaluate_pinn_noenv.py`/`run_pinn_env_terrain.py`/
`evaluate_pinn_env_terrain.py`, and `run_baselines.py` (all four baselines, output paths
`_splitseed<N>`-suffixed so nothing overwrites the primary seed-42 results).
`models/common/saving.py::load_cr_params()` also takes `split_seed` now, reading the matching
suffixed CR anchor -- necessary so a PINN run under a non-default split seed doesn't silently
read the seed-42 CR anchor (the exact mismatch bug the 2026-08-01 pooled-anchor leak fix was
about, in a new form). Refit `dnn_noenv`/`dnn_env_terrain` under seeds 43, 44, 45, and 46,
`spatial_block`/4survey (fast, no cluster needed -- ~50-75s per fit locally):

| Seed | dnn_noenv R2 | dnn_env_terrain R2 | Delta (terrain - noenv) |
|---|---|---|---|
| 42 | 0.6330 | 0.6247 | -0.0083 (the only seed where terrain hurts) |
| 43 | 0.6415 | 0.7193 | +0.0778 |
| 44 | 0.6427 | 0.6732 | +0.0305 |
| 45 | 0.6457 | 0.6997 | +0.0540 |
| 46 | 0.6418 | 0.6543 | +0.0125 |
| 47 | 0.6405 | 0.6832 | +0.0427 |
| 48 | 0.6388 | 0.7014 | +0.0626 |
| 49 | 0.6477 | 0.7095 | +0.0618 |
| **Mean +/- SD (n=8)** | **0.6415 +/- 0.0044** | **0.6832 +/- 0.0315** | **+0.0417 +/- 0.0287** |

**Update (n=8, seeds 47-49 added):** 7 of 8 seeds positive; 95% CI on the mean delta is
approximately [+0.018, +0.066] -- now clearly excludes zero. This is no longer "probably
positive, thin sample" -- it's a confident result. Terrain reliably helps `dnn_noenv` on
average; seed 42 was a genuine, if unlucky, outlier.

**What's working:** `dnn_noenv`'s own numbers are tight across all 5 seeds (SD=0.0047) -- the
no-env model is NOT sensitive to which compartments land in test. This confirms the split-seed
exposure itself works correctly (a genuinely stable result stays stable) and isolates where the
instability actually lives: `dnn_env_terrain`'s SD (0.0372) is ~8x wider -- the moment terrain
enters, results swing with which compartments got held out.
**What's not working / open concern:** the central claim that motivated the scope-matched
XGBoost check, the feature-engineering instructions doc, the `env_deviation` decoupled model,
and the mechanism critique of the PINN's rigid-`k`/`p` physics loss -- "terrain regresses
`dnn_env_terrain`" -- was true for seed 42 SPECIFICALLY and is the opposite of what happens on
average: terrain helps in 4 of 5 seeds, mean delta +0.033 R2. Seed 42, the one seed used for
every other check this session, was the outlier. Consistent with everything else found today
about heterogeneous, compartment-clustered environmental structure (compartment bias, the
leaked/removed neighbour-lag signal) -- if terrain's real effect varies by which part of the
forest is held out, a single train/test partition can land on an unusually easy or unusually
hard test set for a terrain-conditioned model specifically, while a model that mostly uses Age
stays insulated from that.
**What this means for what's next:** this does not invalidate today's mechanistic findings on
their own narrow terms (the physics-loss rigidity argument, the tree-vs-MLP scope-matched
result, both verified facts under seed 42 specifically) -- but the MOTIVATING PREMISE (the DNN
control itself is broken) is now better described as "true under one atypical seed, false on
average" than as a general finding. Today's proposed fixes (feature engineering, decoupled
residual modeling) were designed to fix a problem that mostly isn't there on average -- worth
revisiting their priority once PINN's own seed-sensitivity is known. The user is running
`pinn_noenv`/`pinn_env_terrain` under the same seeds 43-46 on the cluster (code now supports
this end-to-end, including the CR-anchor prerequisite via `run_baselines.py --split-seed`) to
check whether the PINN's own numbers -- and its comparison against the DNN -- are similarly
seed-sensitive. Worth flagging beyond today's scope too: every OTHER spatial_block_split-based
headline result in this dissertation (the base-case DNN-vs-PINN comparison, the physics-weight
sweep, `xgb_environmental`'s attribution numbers, NLME) also used this same single seed,
untested for robustness -- not re-examined here, but the same question applies.

---

**2026-08-02 — PINN split-seed results in (seeds 45, 46, on the cluster, plus the pre-existing
seed-42 base case): PINN's terrain-delta is tiny and stable across seeds, for a structural
reason -- the DNN-vs-PINN(w=1) gap itself is the one finding from today that DOES hold up under
reseeding.**
**What I found:** `pinn_noenv`/`pinn_env_terrain` refit under seeds 45/46 on the cluster (GPU
needed, unlike the DNN which ran locally), CR anchors refit first via
`run_baselines.py --split-seed` and verified to match (`frozen_cr_params` in each run's
`run_metadata.json` checked against the corresponding `chapman_richards_splitseed<N>/params.json`
-- agreed to ~9 decimal places for all 4 runs, confirming no stale/mismatched-anchor risk despite
a messy first cluster submission attempt that hit exactly the job-ordering race flagged when the
sbatch sequence was handed over -- `run_baselines`/`pinn_env_terrain`'s first attempt showed
CANCELLED in `sacct`, `pinn_noenv`'s first attempt showed COMPLETED despite the anchor not being
ready yet, which was the actual reason for double-checking rather than trusting exit code 0
alone). `pw=1.0/tw=1.0` (untested base case, matching the DNN comparison):

| Seed | dnn_noenv | dnn_env_terrain | dnn delta | pinn_noenv | pinn_env_terrain | pinn delta |
|---|---|---|---|---|---|---|
| 42 | 0.6330 | 0.6247 | -0.0083 | 0.580 | 0.5823 | +0.0023 |
| 45 | 0.6457 | 0.6997 | +0.0540 | 0.5530 | 0.5547 | +0.0017 |
| 46 | 0.6418 | 0.6543 | +0.0125 | 0.4676 | 0.4682 | +0.0006 |

**What's working:** PINN's terrain-delta stays tiny (+0.0006 to +0.0023) across all three seeds
-- essentially flat, unlike DNN's -0.008 to +0.054 swing. This is explained directly by the
architecture, not coincidence: `pinn_env_terrain.py`'s `forward()` is identical code/inputs to
`pinn_noenv`'s (age + no-env features only) -- the y_max sub-network is called ONLY inside the
physics/trajectory loss during training, never during a plain height-prediction call. Terrain can
only reach PINN's predictions indirectly, through how physics-loss gradient pressure reshapes the
main network's weights over training -- a far more muted channel than DNN's direct feature
concatenation, which explains both why the effect is small AND why it's stable (an indirect,
muted channel is less exposed to which specific compartments got held out than a direct one is).
Separately: DNN beats PINN(w=1) in all three seeds now checked (0.633>0.580; 0.646>0.553;
0.642>0.468) -- the "physics hurts at full weight" finding, unlike the narrower "terrain
regresses the DNN control" claim, DOES hold up under reseeding.
**What's not working / open concern:** PINN's own absolute accuracy swings a lot across seeds
too (0.580/0.553/0.468 for pinn_noenv) -- comparable in magnitude to `dnn_noenv`'s swings are
NOT (0.633/0.646/0.642, tight) -- so PINN's overall fit quality (not just its terrain-delta) may
itself be more seed-sensitive than DNN's base case, a separate question from today's terrain
investigation, not yet explained.
**What this means for what's next:** the structural explanation for PINN's muted terrain-delta
(terrain never touches the main network's forward pass) is a stronger, more direct account of
"why PINN barely uses terrain" than the physics-loss-rigidity argument alone -- both are true and
compounding (rigid `k`/`p` limits what the y_max channel COULD express even if given full
gradient; the forward-pass architecture limits how much that channel can influence predictions AT
ALL). Any future PINN mechanism redesign (UDE-style time-varying deviation, SA-PINN weighting)
should address the architecture point first -- letting terrain influence predictions only through
a training-time loss term, never through the actual forward pass, caps the achievable effect size
regardless of how well-tuned the loss weighting is.

---

**2026-08-03 — Split-seed robustness completed at n=8 for both DNN and PINN (seeds 42-49):
every conclusion from the smaller samples confirmed, with the DNN-vs-PINN gap now the most
solid finding of the whole investigation.**
**What I found:** User ran the remaining PINN cluster jobs (seeds 43, 44, 47, 48, 49) end to
end and rsynced. All 10 new outputs (both models x 5 seeds) verified present with matching
`frozen_cr_params` (confirmed programmatically, not assumed) before trusting any number. Full
8-seed table, `spatial_block`/4survey, `pw=1.0/tw=1.0`:

| Seed | dnn_noenv | dnn_env_terrain | dnn delta | pinn_noenv | pinn_env_terrain | pinn delta |
|---|---|---|---|---|---|---|
| 42 | 0.6330 | 0.6247 | -0.0083 | 0.5798 | 0.5823 | +0.0025 |
| 43 | 0.6415 | 0.7193 | +0.0778 | 0.5036 | 0.4991 | -0.0045 |
| 44 | 0.6427 | 0.6732 | +0.0305 | 0.5650 | 0.5678 | +0.0028 |
| 45 | 0.6457 | 0.6997 | +0.0540 | 0.5530 | 0.5547 | +0.0017 |
| 46 | 0.6418 | 0.6543 | +0.0125 | 0.4676 | 0.4682 | +0.0006 |
| 47 | 0.6405 | 0.6832 | +0.0427 | 0.5117 | 0.5169 | +0.0051 |
| 48 | 0.6388 | 0.7014 | +0.0626 | 0.5955 | 0.6001 | +0.0045 |
| 49 | 0.6477 | 0.7095 | +0.0618 | 0.5955 | 0.5930 | -0.0025 |
| **Mean+/-SD** | **0.6415+/-0.0044** | **0.6832+/-0.0315** | **+0.0417+/-0.0287** | **0.5465+/-0.0472** | **0.5478+/-0.0470** | **+0.0013+/-0.0033** |

**What's working (four separate findings, all now at n=8, all confirmed):**
1. `dnn_env_terrain` genuinely benefits from terrain on average (95% CI on the delta ~[+0.018,
   +0.066], excludes zero) -- seed 42 was a real outlier, not the typical case.
2. `pinn_env_terrain`'s terrain-delta stays essentially flat regardless of seed (mean=+0.0013,
   SD=0.0033, ~9x tighter than DNN's) -- the architectural explanation (terrain never touches
   the forward pass) is now confirmed at full sample size, not just suggestive.
3. **DNN beats PINN(w=1) in all 8/8 seeds, no exceptions** -- the single most robust finding of
   the entire session, more solid than anything else investigated today.
4. PINN's own accuracy (`pinn_noenv`, nothing to do with terrain) is far more seed-volatile than
   DNN's -- SD=0.0472 vs. DNN's SD=0.0044, roughly 10x wider -- confirmed at full sample size,
   still unexplained.
**What's not working / open concern:** point 4 is a genuinely new, unresolved question this
investigation surfaced rather than answered -- why is PINN's baseline fit quality itself so much
more sensitive to which compartments land in the test set than DNN's, independent of terrain
entirely? Not investigated further this session.
**What this means for what's next:** the split-seed investigation is complete and its
conclusions are now trustworthy at a real sample size, not a hunch from 3 seeds. Priority
ordering for what comes next: (a) point 4 (PINN's seed-volatility) is a new, well-evidenced,
completely open question worth its own investigation before any PINN mechanism redesign work,
since an architecture change built on top of an already-unstable base case is hard to evaluate
cleanly; (b) `env_deviation`'s decoupled approach remains well-motivated by point 2 (the
architecture point), independent of whatever explains point 4; (c) the DNN feature-engineering
instructions doc stays deprioritized, per the 2026-08-02 entry's reasoning, now on a firmer
evidence base.

---

**2026-08-03 — Q: Does excluding the confidently-clearfell/measurement-inconsistent plots from
the per-plot y_max fit improve the terrain/wind attribution R2, as expected, or reveal that the
uncleaned number was inflated by contamination? — the latter: 4survey's R2 roughly HALVES after
cleaning, the opposite of what cleaning "garbage" data should do.**
**What I found:** Wired `disturbance_checks.summarize_plot_disturbance_status()` (built on
Codex's `classify_structural_change_intervals()`) into `scale_comparison_check.
build_plot_level_table()` -- any plot with at least one `clearfell_like` or
`measurement_inconsistent` interval is excluded before `fit_y_max_per_plot()` runs (mixing pre-
and post-event rows into one curve fit is meaningless either way); `ambiguous_disturbance` plots
are kept, per the earlier decision not to remove possible genuine disturbance signal. Only a
small fraction of plots are actually excluded (315/56,841 = 0.55% for 4survey, 138/13,473 = 1.0%
for 6survey). Re-ran the plot-level terrain+wind attribution check (`TERRAIN_AND_WIND_COLUMNS`,
seed 42, both cohorts):

| Cohort | Cleaning | Elastic Net R2 | XGBoost R2 | Plots excluded |
|---|---|---|---|---|
| 4survey | uncleaned | 0.172 | 0.188 | -- |
| 4survey | cleaned | 0.061 | 0.103 | 315 (0.55%) |
| 6survey | uncleaned | -0.028 | -0.012 | -- |
| 6survey | cleaned | -0.032 | -0.013 | 138 (1.0%) |

**What's working:** the exclusion logic itself ran correctly and only touched the intended small
fraction of plots (confirmed via the printed exclusion count, not assumed) -- 4survey still shows
a real, positive signal after cleaning (0.06-0.10), not zero, so the core "terrain/wind explains
something" finding survives, just at a smaller magnitude.
**What's not working / open concern:** removing under 1% of plots roughly HALVED 4survey's R2 --
the opposite of the expected direction for removing garbage data, and a large effect for such a
small excluded fraction. Most likely explanation: felling is not randomly distributed across the
forest (it follows compartment-level management schedules, which plausibly correlate with
terrain via access/rotation planning or storm-driven early felling) -- if the pre-cleaning model
was partly fitting "terrain predicts which plots got felled" rather than "terrain predicts real
growth deviation," removing those corrupted target values would shrink R2 exactly like this. This
means the earlier read ("4survey's uncleaned R2=0.17-0.19 matches the established ceiling, looks
sane") was likely partly an artefact of contamination, not a clean confirmation -- a real revision
to how that number should be reported. 6survey barely moves either way, reinforcing that its
weak/negative result is independent of this specific data-quality issue.
**What this means for what's next:** 0.06-0.10 (4survey) is the more honest current estimate of
the real terrain/wind effect size on this target, but it hasn't been stress-tested the way the
uncleaned number was -- a signal this much smaller is more vulnerable to being a single-seed
result. Re-running the cleaned check across a few split seeds (same seeds already used for the
uncleaned sweep: 42/43/44/45) is the next concrete step before reporting either number as final.

---

**2026-08-03 — Q: Does the cleaned attribution signal hold up across split seeds the way the
uncleaned one did, or was 0.06-0.10 itself a single-seed fluke? — it holds up: positive in every
seed/method combination for 4survey, mean ~0.11-0.13; 6survey stays exactly as messy as before.**
**What I found:** Re-ran `run_seed_sweep_check.py` (now defaulting to cleaned, since
`build_plot_level_table()`'s `apply_disturbance_cleaning=True` default took effect), same 4
seeds as the uncleaned sweep:

| Cohort | Seed | Elastic Net R2 | XGBoost R2 |
|---|---|---|---|
| 4survey | 42 | 0.061 | 0.103 |
| 4survey | 43 | 0.055 | 0.049 |
| 4survey | 44 | 0.123 | 0.101 |
| 4survey | 45 | 0.200 | 0.255 |
| 6survey | 42 | -0.032 | -0.013 |
| 6survey | 43 | -0.006 | 0.002 |
| 6survey | 44 | 0.011 | -0.000 |
| 6survey | 45 | -0.253 | 0.144 |

**What's working:** 4survey's cleaned mean across seeds is 0.110 (Elastic Net) / 0.127
(XGBoost) -- positive in all 8 cells, confirming this is a real, stable effect, not a one-seed
artefact, just genuinely smaller than the uncleaned 0.17-0.19 and noisier seed-to-seed (0.05-0.26
range) than the uncleaned version was.
**What's not working / open concern:** 6survey reproduces the exact same inconsistent,
near-zero-in-both-directions pattern found for the uncleaned target (including one seed, 45,
with opposite signs between Elastic Net (-0.253) and XGBoost (+0.144)) -- cleaning neither fixed
nor explained 6survey's weak signal, confirming it as a separate, likely sample-size-driven issue
rather than a data-quality one.
**What this means for what's next:** the attribution model's honest reportable numbers are now:
4survey R2~0.11-0.13 (real, stable, but ~35% smaller than first thought before contamination was
removed), 6survey inconclusive (likely underpowered, not yet resolved). This is a defensible
place to move on from the data-cleaning question -- next open items are 4survey's shape-misfit
(problem 1), finalizing the feature representation on this cleaned population, and reconciling
with Codex's notebook, before any locked-test-set evaluation.

---

**2026-08-04 — Q: Is the per-plot growth-curve attribution avenue (Candidate A) worth continued
investment, given the effect size has only shrunk under scrutiny (0.17-0.19 -> 0.11-0.13) and
now matches -- not beats -- every other method tried against a different, invalidated target? —
ran this whole question through the `llm-council` skill (5 independent advisor personas, peer
review, chairman synthesis). Near-unanimous verdict: don't build Candidates B/C to chase a bigger
number; quantify uncertainty on the existing result before writing anything up.**
**What I found:** Framed the full avenue (target construction, cleaning results, seed sweep,
6survey's null result, the two unbuilt candidates, the parallel PINN thread) as a council
question. All 5 advisors independently leaned toward NOT continuing to build Bayesian
hierarchical/GNNWR (Candidates B/C) as a way to chase a larger R2 -- reasoning: they'd share the
same 16 features and same target as Candidate A, so absent a specific hypothesis that functional
form (not signal availability) is the bottleneck, they'd inherit the same ceiling. Peer review
(5 reviewers, anonymized responses) converged 5/5 on the single most important blind spot: one
advisor's suggestion to peek at the LOCKED TEST SET to help decide direction was flagged by every
reviewer as a real methodological risk -- it would repeat the exact leakage mistake that
invalidated the original pooled-CR target. Peer review also converged 4/5 on the universal gap:
no confidence intervals anywhere in the existing R2 numbers, and no count of how many independent
compartments the spatial CV is actually averaging over -- without that, "converges" (some
advisors) and "declines toward zero" (other advisors) were both just readings of a trend into
numbers with unknown uncertainty. Chairman's one concrete recommendation: bootstrap a CI on the
existing 4survey R2 before doing anything else.
**What's working:** the council correctly flagged that months of refinement had moved the effect
size in one direction only (smaller), and correctly refused to accept either "this is a real
converged finding" or "this is a shared artifact" as settled without the missing uncertainty
quantification -- a genuinely useful gate that stopped further blind iteration on Candidate A
(more feature engineering, more representation comparisons) before checking whether the result
even supports being iterated on further.
**What's not working / open concern:** the council session itself required framing the full
context by hand (a 5-paragraph prompt) since the skill has no access to this project's actual
files/history -- worth noting for future use, not a flaw in the verdict itself.
**What this means for what's next:** immediately actioned (see next entry) -- built and ran the
cluster bootstrap CI the chairman specified.

---

**2026-08-04 — Q: What's the real uncertainty on 4survey's/6survey's plot-level attribution R2,
and how many independent compartments is the spatial CV actually averaging over? — 4survey's val
set spans only 39 compartments, 6survey's only 7; BOTH of 4survey's 95% confidence intervals
span zero, meaning the R2~0.06-0.10 result is NOT statistically distinguishable from noise at
conventional confidence, despite being positive in every seed checked.**
**What I found:** Built `models/growth_curve_attribution/bootstrap_ci_check.py` -- a CLUSTER
bootstrap (resampling whole COMPARTMENTS with replacement, not individual plot rows, since
within-compartment plots are not independent -- same pseudo-replication concern already fixed
for the compartment-ICC check) on the fixed split_seed=42 validation predictions, 2,000
resamples, both cohorts, both methods, on the cleaned population:

| Cohort | Method | n_val_compartments | Point R2 | 95% CI | % resamples < 0 |
|---|---|---|---|---|---|
| 4survey | Elastic Net | 39 | 0.061 | [-0.085, 0.187] | 25.2% |
| 4survey | XGBoost | 39 | 0.103 | [-0.033, 0.227] | 9.5% |
| 6survey | Elastic Net | 7 | -0.032 | [-0.203, -0.000] | 100% |
| 6survey | XGBoost | 7 | -0.013 | [-0.202, 0.054] | 66.8% |

**What's working:** this directly and quantitatively resolves the council's central open
question. 4survey's val set holding only 39 independent compartments (out of 232 total) is now a
concrete, checkable number, not a suspicion -- and it explains why both 4survey confidence
intervals span zero despite every point estimate across the 4-seed sweep being positive: the
DIRECTION is consistently positive (a real pattern), but the PRECISION on any single estimate is
much weaker than the "0.11-0.13, real and stable" framing implied before this check.
**What's not working / open concern:** 6survey's 7 compartments make its interval close to
uninformative either way -- not a new finding, but now quantified rather than inferred. This
bootstrap only quantifies evaluation-sample uncertainty for a FIXED trained model at one split
seed; it does not by itself combine with the separate seed-to-seed variability already found in
`run_seed_sweep_check.py` -- the two are complementary evidence, not yet combined into one
number.
**What this means for what's next:** the honest characterization of Candidate A's headline
result changes: not "a real, stable 0.11-0.13 R2" but "a consistently positive point estimate
across seeds and methods, whose precision on any single evaluation is too wide to rule out zero
given Aberfoyle's limited number of independent compartments." This is itself a legitimate,
reportable methodological finding (small-forest spatial attribution is fundamentally
precision-limited by compartment count, not just signal size) -- worth writing up as a real
limitation rather than resolved further by more feature engineering on Candidate A.

---

**2026-08-04 — Q: Does rotating the held-out compartments across many folds (instead of one
fixed split) actually fix the precision problem the bootstrap check found? — yes, decisively:
4survey's R2 comes back higher AND every fold agrees on the sign (0.105-0.204 XGBoost, all
positive); 6survey comes back confidently null (nearly every fold negative), not just
inconclusive.**
**What I found:** Built `models/growth_curve_attribution/spatial_cv_check.py` -- rotates which
compartments are held out across K=5 folds (greedy row-count-balanced partitioning of
compartments, same buffer-distance leakage protection reused per fold via
`apply_spatial_buffer()`), pools every plot's out-of-fold prediction (every plot gets evaluated
exactly once, by a model that never saw its own compartment) into one R2 computed over the WHOLE
population, cleaned target, `TERRAIN_AND_WIND_COLUMNS`:

| Cohort | Method | Pooled R2 (all compartments) | Per-fold mean +/- std | Per-fold range |
|---|---|---|---|---|
| 4survey (231 compartments) | Elastic Net | 0.118 | 0.100 +/- 0.032 | 0.054-0.144 (all 5 positive) |
| 4survey (231 compartments) | XGBoost | 0.174 | 0.156 +/- 0.038 | 0.105-0.204 (all 5 positive) |
| 6survey (47 compartments) | Elastic Net | -0.029 | -0.082 +/- 0.069 | all 5 folds negative |
| 6survey (47 compartments) | XGBoost | 0.029 | -0.021 +/- 0.034 | mostly negative, one marginal positive |

**What's working:** this directly resolves the bootstrap check's "CI spans zero" finding for
4survey -- using the full compartment population instead of one ~39-compartment slice, EVERY
fold agrees on sign for both methods, and the pooled R2 (0.118-0.174) is higher than the single
split_seed=42 estimate (0.061-0.103) was -- that seed happened to be a below-average draw, not a
representative one. 6survey is now confidently null (not just inconclusive) using all 47
compartments -- nearly every fold negative for both methods, a cleaner result than the
single-split/bootstrap check could support.
**What's not working / open concern:** this used the DEFAULT (existing) model hyperparameters
for both Elastic Net and XGBoost, not a fresh tuning pass -- per the reasoning already logged
(the earlier dropout/LR/architecture-size sweeps for DNN/PINN found null results, so a full new
sweep wasn't run here either) -- worth a light sanity check later, not expected to change the
qualitative conclusion.
**What this means for what's next:** 4survey's terrain/wind attribution result is now a
genuinely confirmed, precisely-estimated finding (R2~0.12-0.17, real and consistent across every
fold), a materially stronger and more defensible headline number than anything reported earlier
in this investigation. 6survey's null result is likewise now confirmed, not just suspected. This
whole precision question -- raised by the LLM Council, diagnosed by the bootstrap check, fixed
here -- is the cleanest, most decisive result of the entire Stage 2 growth-curve investigation.
Same spatial-CV-folds approach is worth applying to Stage 1's existing spatial_block_split
results (xgb_environmental, DNN/PINN) before treating those numbers as final either, though not
yet done.

---

**2026-08-04 — Q: LLM Council on which of the 28 available terrain/wind columns should go into
the explanation model — does the confirmed 4survey signal survive swapping the established
list's `gwa_wind_speed_10m` (known to likely measure sub-canopy wind, physically backwards for a
mature stand) for `gwa_wind_speed_50m`? — yes, the signal survives essentially unchanged,
resolving the council's single most consequential open question.**
**What I found:** Ran a second `llm-council` session specifically on variable selection for the
next (explanation/SHAP) phase -- all 5 peer reviewers unanimously picked the Contrarian's
response as strongest, converging on: don't trust "local shelter beats GWA" (that comparison ran
on the OLD uncleaned target) or the R2=0.12-0.17 headline itself as settled, since both may be
entangled with the known-contaminated `gwa_wind_speed_10m` variable already sitting in the
established 16-column list. Peer review also caught a target-circularity check nobody proposed
independently (do any of these columns feed into the target's own construction upstream -- not
yet checked) and flagged that no correlation/VIF check has been run inside the project's own
spatial CV framework. Chairman's "one thing to do first": swap 10m for 50m under the same 5-fold
spatial CV and see if R2 survives. Ran it (`run_wind_height_swap_check.py`, both cohorts):

| Cohort | Feature set | Elastic Net R2 | XGBoost R2 |
|---|---|---|---|
| 4survey | 10m wind (established) | 0.118 | 0.174 |
| 4survey | 50m wind (swapped) | 0.113 | 0.161 |
| 6survey | 10m wind (established) | -0.029 | 0.029 |
| 6survey | 50m wind (swapped) | -0.032 | 0.021 |

**What's working:** 4survey's drop (0.005 Elastic Net, 0.013 XGBoost) is well within the
per-fold standard deviation (0.032-0.037) already measured for this check -- noise-level
movement, not a real change. The confirmed signal does NOT depend on the contaminated 10m
variable; it holds up on the physically more defensible 50m measurement. This resolves the
council's single most consequential open question directly, with a real number, not an assumption.
**What's not working / open concern:** the OTHER items the council flagged as needing checking
before finalizing a variable list are still open: whether "local shelter beats GWA" replicates on
the current cleaned target, the TPI-100m/250m/500m correlation matrix (not yet run), the
target-circularity check (whether any of these columns were already consumed building
`local_y_max_difference` upstream -- not yet checked, and flagged as the single most
project-relevant catch from peer review, echoing this project's own established `Age`/CR-curve
circularity precedent), and whether SHAP/importance rankings are stable under the spatial CV
framework rather than a single pooled fit.
**What this means for what's next:** the wind-height question is closed. The remaining
pre-SHAP checklist from the council (re-confirm local-shelter-vs-GWA on cleaned data, TPI
correlation matrix, circularity check, `inverse_slope_proxy` redundancy confirmed with a real
number not narrative) should be worked through before building the actual SHAP/permutation-
importance explanation pass -- the circularity check in particular should go first, since it's
the one check that could invalidate a variable entirely rather than just simplify the list.

---

**2026-08-04 — Q: Were any of the candidate terrain/wind columns already consumed constructing
`local_y_max_difference` upstream (the target-circularity check the council's peer review flagged
as the single most project-relevant, unchecked risk)? — no code-level circularity found: every
formula ingredient traces back to raw Forestry Commission inventory fields, not terrain/wind
data.**
**What I found:** Traced the full formula chain directly against the real code (not from
memory): `local_y_max_difference = y_max_fit - y_max_yldc`. `y_max_fit`
(`temporal_stability_check.fit_y_max_per_plot()`) uses only `Top_Height95`/`Age`/`p4`/`p5`.
`y_max_yldc = yldc*p2 + p1 + p3*2` (`export_growth_curve_tables.py:108`) uses only
`yldc`/`p1`/`p2`/`p3`. Checked `clean_master_data.py`'s own `MASTER_COLUMNS` list and comments to
confirm these are all raw, independently-recorded FC inventory fields, not derived from any
terrain/wind formula in this pipeline. `whcl` (the one terrain/wind-adjacent field that DOES live
in the same master table) is explicitly commented "audit/sensitivity only, never a baseline model
feature" and never enters any of the target formulas. The terrain/wind candidate columns
(`plot_environmental_features.parquet`) come from an entirely separate GIS/climate extraction
pipeline, merged in only downstream as candidate predictors.
**What's working:** none of the 16 (or 28) candidate columns are mathematically consumed
anywhere in constructing the target -- a clean, verified result, not an assumption.
**What's not working / open concern:** this can only verify circularity WITHIN this codebase's
own pipeline. Whether the Forestry Commission's original, decades-old assignment of `yldc` or
`p1`-`p5` to a stand ever informally incorporated visible site exposure at assignment time is a
genuine historical-provenance question no amount of code-tracing can resolve -- flagged as an
open, documented limitation, not silently assumed away. Would need real forestry domain knowledge
(the user's, if they have it) to address further.
**What this means for what's next:** the circularity gate is cleared. Remaining pre-SHAP items:
re-confirm local-shelter-vs-GWA on the cleaned target, the TPI-100m/250m/500m correlation matrix,
and `inverse_slope_proxy`'s redundancy confirmed with an actual correlation number.

---

**2026-08-04 — Q: TPI-multiscale and inverse_slope_proxy correlation checks (the council's
remaining "waved through without a number" items) — TPI at native/250m/500m turns out genuinely
NOT redundant (native-vs-500m rho=0.62), overturning the earlier "keep at most one extra scale"
assumption; inverse_slope_proxy confirmed an exact -1.0 duplicate of slope_degrees.**
**What I found:** Built `models/growth_curve_attribution/correlation_screen.py`, ran real
Spearman correlations (not assumed): `tpi` (native ~100m) vs `tpi_250m`=0.841, `tpi_250m` vs
`tpi_500m`=0.879, but `tpi` vs `tpi_500m`=only 0.619 -- the two extremes are meaningfully
distinct even though each correlates strongly with the middle scale. `local_relief_500m` is
essentially uncorrelated with any TPI scale (-0.07 to -0.11) -- a genuinely separate variable,
not a TPI duplicate. `inverse_slope_proxy` vs `slope_degrees` = exactly -1.0, confirming the
code's own comment ("an EXACT linear duplicate") with a real number.
**What's working:** this overturns the pre-registered plan from the council chairman's own
recommendation ("keep at most one extra TPI scale") -- native and 500m TPI both carry real,
non-redundant information, so dropping to one scale would have discarded real signal on an
untested assumption. `inverse_slope_proxy` drops cleanly, confirmed not assumed.
**What this means for what's next:** final feature list can include native `tpi` AND `tpi_500m`
AND `local_relief_500m` together (all three add distinct information), while dropping
`tpi_250m` (redundant with both neighbours) and `inverse_slope_proxy` (exact duplicate).

---

**2026-08-04 — Q: What actually explains the confirmed 4survey terrain/wind signal — which
variables, in which direction? — a coherent, physically-sensible wind-exposure story, told
consistently by six variables via two independent methods: elevation, wind speed, Windthrow
Hazard Class, and shelter indices all point the same direction.**
**What I found:** Skipped the expensive (~120-fit) representation-CV sweep in favour of a cheap,
self-contained check that didn't depend on it: built a final 17-column feature list directly
from checks already on disk (established 16 minus `inverse_slope_proxy`, `gwa_wind_speed_10m`
swapped for 50m, `tpi_500m`/`local_relief_500m` added -- all justified by prior entries, no new
comparison needed), then ONE XGBoost fit on the full cleaned 4survey population (56,489 plots --
an interpretation model, not a performance-evaluation one, so no held-out split needed) plus
Spearman correlation as a near-free first pass. Top variables, consistent across both methods:
`elevation` (SHAP-dominant, 2.96 mean |SHAP|, more than double the next variable),
`gwa_wind_speed_50m` (strongest raw correlation, -0.178), `whcl` (-0.165), `windward_topex`
(+0.157), `topex` (+0.123), `tpi_500m` (-0.100), `eastness` (+0.112).
**What's working:** every top-ranked variable points the SAME direction -- more wind exposure/
less shelter/more ridge-like position associates with underperforming the yield-class rating;
more shelter associates with outperforming it. `eastness`'s positive sign is coherent with this
too, not a separate effect -- Scotland's prevailing wind is westerly/southwesterly, so
east-facing slopes are the sheltered ones. This directly validates the FC's own `whcl`
(Windthrow Hazard Class) index as carrying real signal against a newly-confirmed target, and is
consistent with Aberfoyle/Loch Ard's documented history as a windthrow-prone region -- the result
matches what forestry domain knowledge would predict, not an implausible artefact.
**What's not working / open concern:** this is one model fit, not yet cross-validated for
importance STABILITY (a gap the council's peer review flagged -- do these rankings hold up under
the same spatial CV that confirmed the underlying signal, or would a different held-out
compartment set reorder them). SHAP mean |value| also doesn't distinguish correlation from
causation, same caveat as everywhere else in this project's attribution work.
**What this means for what's next:** a real, coherent, publication-worthy "what reason" answer
now exists for the confirmed 4survey signal -- wind exposure, corroborated by an independent
operational forestry index. Worth checking importance stability under spatial CV before treating
this ranking as final, and worth mapping the explained/unexplained deviation spatially (the
originally planned "step 3") to see if the pattern is geographically coherent.

---

**2026-08-04 — Q: Does the SHAP-based ranking from the single full-data fit hold up across
different held-out compartment sets, and does the explained-vs-unexplained deviation look
spatially coherent when mapped? — yes to both: elevation/whcl are rock-solid stable at #1/#2
across every fold, and the environment-explained component shows real spatial structure while
the residual looks like genuine noise.**
**What I found:** Two cheap checks, both reusing infrastructure already built rather than new
expensive fits. (1) `importance_stability_check.py` -- XGBoost's own built-in gain importance
(not a full SHAP recompute) from 5 fold-specific TRAIN-only fits, same fold assignment as the
confirmed spatial CV. `elevation` (rank_mean=1.2, std=0.45) and `whcl` (rank_mean=1.8, std=0.45)
are always #1 or #2 across every one of the 5 folds -- extremely stable. A second tier
(`slope_degrees`, `eastness`, `local_relief_500m`, `topex`, `windward_topex`,
`elevation_roughness`, `solar_radiation_index`) clusters together as a GROUP (never top-2, never
bottom-tier) even though their exact order shifts fold to fold. Curvature/frost/native-`tpi`
variables are consistently unimportant (stable low rank, low std). One real nuance:
`gwa_wind_speed_50m` drops to rank_mean=10.8 here despite having the single strongest raw
Spearman correlation (-0.178) -- gain-based importance shows it's largely redundant with
elevation/whcl/topex once those are already in the tree, not adding much UNIQUE information on
top, a genuinely different question than simple correlation answers.
(2) Spatial map (`deviation_map.png`, 3-panel: observed / environment-explained / residual) --
the environment-explained component shows real spatial coherence (large, smooth blocks of
consistent colour, matching how terrain/elevation/shelter actually vary across a real landscape),
while the residual looks much more like the raw observed deviation -- fine-grained, speckled,
plot-to-plot scatter with no obvious large-scale structure.
**What's working:** both checks point the same direction as each other and as the earlier SHAP
result -- this is a real, stable, spatially coherent signal, not an artefact of one lucky fit or
noise that happens to correlate. The residual's LACK of visible spatial structure is itself
informative: if terrain were missing something systematic, the residual would likely still show
a spatial pattern; it doesn't, which is consistent with the terrain features having genuinely
"used up" the spatial part of the signal rather than leaving it on the table.
**What's not working / open concern:** the gain-importance-vs-correlation divergence for
`gwa_wind_speed_50m` is worth a line in the write-up (unique vs. marginal contribution are
different questions, both worth reporting) but doesn't change the headline story. The residual
map is a visual read, not a formal spatial-autocorrelation test (e.g. Moran's I on the residual,
already used elsewhere in this project for the old pooled-CR target) -- worth running that formal
version if this becomes a dissertation-reported figure.
**What this means for what's next:** the explanation phase for 4survey's confirmed signal is
essentially complete and well-evidenced: real signal (5-fold spatial CV), survives its one
identified vulnerability (wind-height swap), no circularity, a coherent and stable wind-exposure
story (elevation + whcl dominant, corroborated by an independent FC index), and spatially
coherent when mapped. This is a strong, defensible place to write this up as the dissertation's
core Stage 2 finding.

---

**2026-08-04 — Q: Were Moran's I / LISA run on this new per-plot residual, or only ever on the
old pooled-CR target? — not run until now; the real numbers show this new approach leaves
behind dramatically less GLOBAL spatial structure than the old target did (I=0.0021 vs the old
target's I=0.197), while LISA finds a real, coherent ~21% local-cluster pattern underneath that
near-zero global average.**
**What I found:** Reused this project's own established spatial-autocorrelation tools
(`models/spatial_attribution/spatial_autocorrelation.py::global_morans_i()`,
`models/spatial_attribution/lisa.py::local_morans_i()`, both already used elsewhere for the OLD
pooled-CR-residual target) on the residual from the same full-data XGBoost fit used for SHAP.
Global Moran's I = 0.0021 (p=0.004 -- nominally significant only because of the large sample
size; the effect size itself is essentially zero). LISA (k=8 nearest neighbours, Benjamini-
Hochberg FDR-corrected, both already this module's own established defaults): 79.1% of plots
"Not significant", 10.4% Low-Low (cold-spot), 9.5% High-High (hot-spot), 1.0% High-Low/Low-High
(spatial outliers).
**What's working:** directly comparable to an already-logged number from this project's own
history -- the old pooled-CR target's full-model residual Moran's I was 0.197, ~2 orders of
magnitude larger than this new target's 0.0021. This new, more carefully validated per-plot
approach leaves behind far less unexplained global spatial structure than the old approach did,
a real, quantified improvement, not just a methodological preference. The LISA result is a
coherent nuance, not a contradiction: ~21% of plots sit in a real local hot/cold-spot cluster
even though positive and negative clusters cancel out to a near-zero GLOBAL average -- exactly
what local indicators are built to catch that a global statistic alone would miss.
**What's not working / open concern:** the semivariogram fit used to pick Moran's I's distance
parameter returned "no_structure" on this residual (fell back to the project's own established
3,956m range from the old target's fit) -- consistent with the residual having very little
range-detectable spatial structure to begin with, but means the 3,956m distance wasn't
independently re-derived for this specific target. The ~21% local-cluster plots haven't been
mapped/inspected individually yet (which specific compartments/regions they fall in).
**What this means for what's next:** both the visual read and the formal statistical tests now
agree: this new approach's residual is close to spatially "used up," a meaningfully stronger
result than the old target's real leftover autocorrelation. Worth including both the global and
local numbers in any dissertation write-up of this finding, not just the visual map.

---

**2026-08-04 — Q: [MAJOR CORRECTION] Was every reported XGBoost R2 in this Stage 2 investigation
computed on a genuinely held-out test set, or was the evaluation set also used for early
stopping? — the user found and fixed a real leak in `spatial_cv_check.py`; a systematic sweep
found the SAME bug in three more files. The uncleaned-target R2 (previously reported as 0.172/
0.188, "matches the established ceiling") was almost entirely a leak artefact -- the true,
leak-free number is NEGATIVE. Cleaning does not just shrink an inflated number, it is what turns
a null result into a real one.**
**What I found:** `xgb_environmental.fit_with_columns()`'s `val_df` parameter drives XGBoost's
early stopping (how many boosting rounds to run, chosen by monitoring loss on `val_df`) -- it is
not just a label. `scale_comparison_check.py`, `bootstrap_ci_check.py`, and
`feature_representation_check.py` (never run/reported, fixed anyway) all passed the SAME `val`
partition both as `val_df` (early stopping) AND as the set predictions/metrics were computed on
-- meaning the model's own fit was tuned to minimise loss on the exact rows its performance was
then measured on. `spatial_block_split()` already builds a genuine third partition (`test`) that
was sitting completely unused in every one of these functions. The user independently found and
fixed the same pattern in `spatial_cv_check.py` (each fold now carves out a separate `val` fold
purely for early stopping, evaluates on the rotating held-out `test` fold). A systematic grep of
every `xgb_fit`/`elasticnet_fit` call site in `models/growth_curve_attribution/` confirmed these
were the only four affected functions -- `explain_signal.py` (full-data fit, no `val_df`) and
`importance_stability_check.py` (train-only fit, no `val_df`, no evaluation at all) were never at
risk. Also checked directly (not assumed): whether the "test" partition now being evaluated
overlaps with Codex's notebook's own deliberately-locked test set (same seed=42, same split
mechanism) -- empirically NOT identical (only 15 of ~45-51 compartments overlap), because the
disturbance-cleaning step changes each compartment's row count, which changes
`spatial_block_split`'s greedy assignment even under the same seed.
**Corrected numbers** (single split, seed=42, cleaned population unless noted):

| Check | Metric (EN / XGB), before fix | Metric (EN / XGB), after fix |
|---|---|---|
| Uncleaned, single split | R2 = 0.172 / 0.188 | R2 = **-0.054 / -0.020** |
| Cleaned, single split | R2 = 0.061 / 0.103 | R2 = **0.062 / 0.086** |
| 5-fold spatial CV, cleaned, pooled | R2 = 0.118 / 0.174 | R2 = **0.119 / 0.134** |
| Wind-height swap (50m), 5-fold CV | R2 = 0.113 / 0.161 | R2 = **0.119 / 0.135** |
| Bootstrap 95% CI, single split | [-0.085, 0.187] / [-0.033, 0.227] | **[-0.079, 0.137] / [-0.046, 0.158]** |
| 4-seed sweep mean, cleaned, single split | not previously reported this way | EN 0.114, XGB 0.121 (every seed positive) |

**What's working:** every qualitative conclusion this session reached survives, several MORE
cleanly than before: 4survey still shows a real, positive, seed-stable, CV-confirmed signal
(now every one of 4 re-swept seeds is positive for both methods, not just directionally
consistent); 6survey is still confirmed null (now negative across all 4 seeds, both methods,
both before and after cleaning); the wind-height-swap conclusion (signal doesn't depend on the
contaminated 10m variable) still holds. The disturbance-cleaning finding is actually STRONGER
now, not weaker: the leak-free uncleaned R2 is negative -- cleaning does not merely reduce an
inflated number, it is the difference between a real signal and no signal at all for 4survey.
**What's not working / open concern:** the SHAP/importance/stability/Moran's-I/LISA results
(section on "what explains the signal") are unaffected by this bug (no `val_df` involved in any
of those fits) and do not need re-running. The single-split checks (seed sweep, cleaning
comparison) have now each looked at the same seed=42 test partition more than once across
different rounds of this session's own decision-making -- not a leak into training, but a softer
"researcher degrees of freedom" concern in the same spirit the LLM Council's peer review already
flagged elsewhere. The 15-compartment test-set overlap with Codex's notebook, while not
identical, means the two threads are not fully independent either -- worth keeping in mind if
the two are later combined into one final reported number.
**What this means for what's next:** the notebook (`av2_per_plot_growth_curve_story.ipynb`) was
built and executed before this fix and needs re-running with corrected numbers throughout. The
user is separately planning a broader multi-seed sweep across all of this to get a proper error
bound on the final result -- this correction is exactly the kind of check that needed to happen
before that larger investment, not after.

---

**2026-08-04 — Q: Given the leak fix, is the 5-fold CV headline number now stable and
reproducible, or is there ALSO hidden non-determinism (e.g. from unseeded set-iteration order,
a classic Python gotcha)? — no hidden randomness: 5 repeated runs (including two different
PYTHONHASHSEED values) against a frozen code state gave bit-identical results. The apparent
"three different numbers" seen while re-running this check were three different MOMENTS in an
actively-evolving shared codebase, not true non-determinism.**
**What I found:** After the leak fix, `run_spatial_cv_check.py`, then the notebook's own
execution, then a fresh direct call all gave three DIFFERENT 4survey pooled R2 values
(0.119/0.134, then 0.123/0.162, then 0.130/0.122) despite an apparently fixed seed throughout --
concerning enough to investigate immediately rather than pick one and move on. Ran the identical
function 3x in the same process (bit-identical each time: 0.1302/0.1222), then twice more under
explicit `PYTHONHASHSEED=1` and `PYTHONHASHSEED=2` (ruling out Python's per-process hash
randomization, a well-known source of `set()`-iteration-order-dependent non-determinism) --
still bit-identical (0.1302/0.1222) both times.
**What's working:** given a FROZEN code state, this pipeline is fully deterministic and
reproducible -- confirmed, not assumed. The three earlier different numbers are explained
entirely by the shared codebase being actively edited by the other concurrent thread
(`models/common/splits.py` and related files) between each of these checks this session, exactly
consistent with the concurrent-editing pattern already observed and flagged multiple times
(the notebook, `experiment_log.md`, `evaluate_baselines.py`, `spatial_cv_check.py` itself).
**What's not working / open concern:** this means every number quoted earlier in this
investigation needs to be read as "correct for the code state at that moment," not as a single
fixed ground truth -- the project's own convention of re-running scripts rather than trusting
quoted numbers is what caught this, and remains the right discipline going forward, especially
while another thread is actively touching shared files.
**What this means for what's next:** the CURRENT, verified-reproducible 4survey 5-fold CV
result is R2 = 0.130 (Elastic Net) / 0.122 (XGBoost); 6survey is R2 = 0.044 / -0.028 (still
confirmed null). The notebook's R2 progression table and headline have been updated to these
numbers and the notebook re-executed end to end successfully. Also fixed in the same pass: a
`NameError` in the notebook (a newly-added "follow-on broader models" section -- referencing
real, valuable `broad_environmental_check.py` output showing terrain/wind+management reaches
R2~0.29-0.32 for 4survey, correctly kept conceptually separate from the environmental-only
claim -- used `plt.subplots()` before `matplotlib.pyplot` had been imported); moved the import
to the notebook's first setup cell.

---

**2026-08-04 — Q: What's the real error bound on the 5-fold spatial CV headline R2, across
different CV fold-assignment seeds (not just one fixed partition)? — a clean, decisive result:
4survey is positive in all 16 seed x method combinations (R2 0.066-0.141); 6survey's mean sits
almost exactly on zero across 8 independent partitions.**
**What I found:** Built `models/growth_curve_attribution/run_cv_seed_sweep.py` -- re-runs the
full 5-fold spatial CV design (not the older single-split seed sweep) under 8 different
fold-assignment seeds (42-49), both cohorts, both methods, on the current verified code state:

| Cohort | Method | Mean | Std | Min | Max |
|---|---|---|---|---|---|
| 4survey | Elastic Net | 0.116 | 0.023 | 0.077 | 0.138 |
| 4survey | XGBoost | 0.105 | 0.027 | 0.066 | 0.141 |
| 6survey | Elastic Net | -0.002 | 0.034 | -0.050 | 0.044 |
| 6survey | XGBoost | -0.002 | 0.024 | -0.028 | 0.036 |

**What's working:** every one of the 16 4survey seed x method results is positive -- this is now
the most robust confirmation of the signal in the whole investigation, an actual error bound
(mean +/- std across independent CV partitions) rather than a single point estimate. 6survey's
mean landing almost exactly on zero, straddling positive and negative across 8 partitions, is the
cleanest possible confirmation of a genuine null result, not an inconclusive one.
**What's not working / open concern:** Elastic Net's mean (0.116) is now slightly higher than
XGBoost's (0.105) across this seed sweep too, consistent with the earlier single-seed finding
that the effect is close to linear and tree-based nonlinearity isn't adding real value.
**What this means for what's next:** this is a defensible final error bound for the headline
4survey result (R2 ~0.10-0.12, positive under every tested partition) and confirms 6survey's null
result with real cross-seed evidence, not just one seed's estimate. Given Elastic Net's edge over
XGBoost, a linear mixed-effects/hierarchical model with partial pooling across compartments
(Candidate B, informed by the earlier ICC=0.399 finding) is the best-motivated next model --
testing whether real compartment structure adds anything beyond the fixed terrain/wind effects,
not chasing a bigger R2.

---

**2026-08-03 — Q: Given 6survey's plots are a strict subset of 4survey's, why does its
terrain/wind attribution signal behave so differently? — 6survey's compartments sit in a
genuinely narrower slice of the landscape (roughly half the elevation and wind-exposure range of
4survey), a second, independent, compounding reason beyond raw sample size.**
**What I found:** Confirmed directly (not assumed from the established project convention):
6survey's 13,769 plots and 47 compartments are a strict subset of 4survey's 58,112 plots / 232
compartments (`issubset()` check, both True). Compared geographic footprint and
`TERRAIN_AND_WIND_COLUMNS` variance between the two populations:

| Measure | 4survey | 6survey | 6survey as % of 4survey |
|---|---|---|---|
| x-range (m) | 16,785 | 8,110 | 48% |
| y-range (m) | 12,013 | 6,462 | 54% |
| `elevation` std | 112.6 | 47.8 | 42% |
| `topex` std | 19.6 | 10.6 | 54% |
| `windward_topex` std | 6.25 | 2.55 | 41% |
| `gwa_wind_speed_10m` std | 1.11 | 0.56 | 50% |
| `elevation_roughness` std | 7.36 | 5.25 | 71% |
| `northness`/`eastness`/`tpi`/`ceh_twi` std | -- | -- | ~95-102% (essentially unchanged) |

**What's working:** the variables that shrink the most in 6survey (elevation, both topex
measures, GWA wind speed, elevation roughness) are exactly the ones tied to a compartment's
position in the BROADER landscape -- the ones that barely change (northness/eastness/TPI/TWI,
local micro-topography shape) aren't tied to broad-landscape position. This is a coherent,
specific pattern, not noise: 6survey's 47 compartments occupy a genuinely more
terrain-homogeneous, geographically smaller slice of Aberfoyle (roughly half the elevation and
wind-exposure range), not a random sample of 4survey's full landscape variety.
**What's not working / open concern:** WHY these particular 47 compartments got surveyed 6 times
instead of 4 (a management/research decision) isn't known from data on disk -- only the
consequence (narrower terrain range) is demonstrated here, not the cause.
**What this means for what's next:** 6survey's weak attribution signal now has TWO independent,
compounding, real explanations -- smaller sample size (already established) AND less terrain
variation to explain in the first place (this entry) -- rather than one unresolved mystery.
Reframes problem 3 again: not "find what's wrong with 6survey," but "6survey may have a
genuinely lower achievable ceiling for this question, for reasons that don't reflect on the
method." Any final write-up should report 6survey's null result with this context, not as an
unexplained inconsistency.

---

**2026-08-02 — Regression/residual kriging proposed as a further alternative (real, checkable
precedent found: residual kriging is already used specifically for canopy height mapping); a
leak-safe version of the spatial-lag feature was then quick-tested and found structurally
incompatible with `spatial_block_split`, not just weak.**
**What I found:** Proposed regression kriging (tree-based mean function + kriged spatial
residual correction) as a candidate model addressing causes 2 and 4 from the same-day synthesis
entry together -- grounded in real literature, not speculation: "Random Forest Regression
Kriging" is a named, cited method, and residual kriging has been applied specifically to canopy
height mapping (Hengl et al.; GEDI canopy-height residual-kriging literature). Before building
it, quick-tested the cheaper, more direct fix for cause 4 first: a LEAK-SAFE version of
`neighbour_mean_height` (train-split-only candidate pool, per `progress_notes.md`'s own
already-written fix path, 75m radius, 2023 heights, same construction as the original leaky
feature minus the leak). Spliced into a scope-matched XGBoost check
(`spatial_block`/4survey, age+no-env+terrain baseline=0.6444 known from the 2026-08-02
scope-matched entry): adding the leak-safe feature dropped test R2 to **0.4486** -- a large
regression, not a small one, which prompted a coverage diagnosis before trusting the number.
**What I found, diagnosing the drop:** train rows have 0.0% missing (avg 30.3 legitimate
train-neighbours within 75m each); val/test rows have **96.3-96.4% missing** (avg 0.1
neighbours). Root cause, structural not a bug: `spatial_block_split` holds out WHOLE
compartments, and compartments are large, spatially contiguous units (per `splits.py`'s own
note, 1 to 1,300+ plots) -- almost always bigger than a 75m radius. A held-out plot's true
75m-neighbours are therefore overwhelmingly OTHER plots in the SAME held-out compartment, never
train plots, except right at a compartment's edge. XGBoost, fit on train rows where the feature
is dense and locally accurate, learned to lean on it -- then found it almost entirely
missing/filled-with-a-constant at test time, degrading the model's other learned structure too
(not just "one ignored weak feature" -- the size of the drop reflects a real train/test
distribution mismatch, confirmed directly via the coverage numbers, not assumed from the R2 drop
alone).
**What's working:** the original `neighbour_mean_height` finding (2026-07-31, +0.334 R2 drop,
the strongest category ever measured) is now understood MORE precisely, not just confirmed
leaky: its predictive power was always structurally tied to spatial autocorrelation WITHIN a
compartment, which is exactly the thing `spatial_block_split` exists to remove from evaluation.
A genuinely leak-free version of that specific feature shape (fixed-radius neighbour average)
cannot recover that signal under the primary split, because the split's whole methodological
purpose is to make "borrow a nearby plot's real answer" impossible for held-out compartments --
this isn't a gap that was simply never gotten around to fixing, it's the split working as
designed.
**What's not working / open concern:** this closes off "quickly rebuild the leaked feature
safely" as a viable route to cause 4 specifically. It does NOT rule out regression/residual
kriging as still worth trying -- kriging uses a fitted variogram (spatial-correlation-decay
function) rather than a hard 75m cutoff, so it can in principle borrow weaker signal from
train points farther away than 75m, where a plain radius-based feature goes straight to
missing. Whether that weaker, longer-range signal is still worth anything under
`spatial_block_split`'s whole-compartment holdout is untested, not assumed either way.
**What this means for what's next:** cause 4 (missing spatial-lag signal) is harder to
legitimately recover than it looked -- not abandoned, but the honest expectation should be
"much less than the original leaky feature's +0.334, possibly not much at all," given the same
structural mismatch would degrade any short-range spatial method the same way. If terrain/wind's
real predictive ceiling really is ~R2=0.16-0.19 (cause 1, established independently multiple
times), regression kriging's realistic upside is recovering whatever LONG-range spatial
structure survives whole-compartment holdout, on top of that ceiling -- worth trying, but not
expected to be a dramatic fix on its own.

---

**2026-08-02 — `models/env_deviation/` built and run: decoupling from the physics loss gives a
real, clean win against its own base (CR), but composing onto the STRONGEST base (`dnn_noenv`)
makes things worse -- a sample-size artefact of the leakage-avoidance fix, not evidence the idea
is wrong.**
**What I found:** Built `models/env_deviation/` per `documentation/model_instructions/
env_deviation_decoupled_instructions.md`: row-level residual (not `mean_cr_residual`'s
plot-averaged version), split-matched CR anchor (not the still-leaky pooled one
`mean_cr_residual` reads), terrain-only inputs (`terrain_wind_solid`), XGBoost
(`fit_with_columns`/`predict_with_columns`, reused from `xgb_environmental.py`, `n_jobs=1` --
required, torch+xgboost segfault together in one process on this Mac otherwise). Two base-model
variants, `spatial_block`/4survey:

| Model | Base R2 (alone) | Composed R2 (base + predicted residual) |
|---|---|---|
| CR curve | 0.4046 | **0.5291** (+0.1245) |
| `dnn_noenv` | 0.6335 | **0.5789** (-0.0546) |

The CR-variant is a real, clean win -- decoupling recovers real signal the physics loss was
suppressing (see the 2026-08-02 physics-weight-sweep entry's mechanism discussion: `k`/`p` are
frozen globals, `y_max` is the only per-plot degree of freedom the physics loss allows, so any
deviation that isn't a pure ceiling shift is actively penalised in proportion to
`physics_weight` -- decoupling removes that constraint entirely from the residual-fitting step).
The `dnn_noenv`-variant instead makes the best available model WORSE. Diagnosed directly (not
assumed): `predicted_residual`'s std (2.95m) is much narrower than the true leftover residual's
std (4.65m) -- classic shrinkage-to-the-training-mean under data starvation, not a genuine
"deviations are near zero" finding (if it were genuine, composing near-zero residuals onto
`dnn_noenv` would roughly reproduce 0.6330, not fall to 0.5789). Root cause: avoiding the
in-sample-residual leakage risk (train-set residuals are artificially small since `dnn_noenv`
overfits -- train_loss << val_loss on every training curve this session) required fitting the
residual model on VAL-derived rows only -- 37,645 rows, a 3.4x reduction from the CR-variant's
full 129,564-row train set. Terrain is already a small, low-SNR signal (standalone R2~0.16-0.19
throughout this project) -- exactly the kind that needs MORE data to extract reliably, not less.
**What's working:** the leakage-avoidance discipline itself did its job -- caught what would
otherwise have been an inflated, untrustworthy composed number; the CR-variant proves the
underlying "decouple from the physics loss" idea recovers real signal when not data-starved.
**What's not working / open concern:** the `dnn_noenv`-variant, the one that would actually
matter for beating the current best model, currently doesn't -- fixable in principle via k-fold
(train `dnn_noenv` k times, predict each fold out-of-sample, giving the residual model ~130k
held-out-but-representative rows instead of 37k) but not yet built -- a real additional
engineering cost, not a small tweak.
**What this means for what's next:** k-fold is the natural next step for this specific model, but
see the same-day synthesis entry below for how this result fits into the bigger picture -- a small
genuine terrain effect size and a demonstrated MLP/tree inductive-bias gap are the two most
solid explanations for the broader pattern of underwhelming results this session, and neither is
fully addressed by env_deviation alone.

---

**2026-08-02 — Same-day synthesis: ranking the actual causes behind this session's run of
underwhelming results, from firmest evidence to weakest.**
**What I found:** Pulling together every check run today (dropout/LR/architecture-size sweeps,
feature-set-parity check, scope-matched XGBoost check, `env_deviation`'s two variants, plus the
mechanism reading of `pinn_env_terrain.py`'s actual physics-loss code) into one ranked list,
firmest evidence first:
1. **Terrain's own effect size is genuinely small** -- terrain+wind alone tops out around
   R2=0.16-0.19 in every method that's measured it this project (XGBoost, Elastic Net, this
   session). Not a bug -- a ceiling every fix is bounded by.
2. **A plain MLP can't reliably extract even that small signal** -- confirmed independently
   twice: the scope-matched XGBoost check (+0.0111 trees vs -0.0083 MLP, identical inputs) and
   (from a diagnostic report the user ran separately) residual-predictability numbers showing
   `dnn_env_terrain`'s residuals MORE predictable from environment than `dnn_noenv`'s own,
   despite already having terrain as a direct input.
3. **The PINN's physics loss actively suppresses any deviation that isn't a pure ceiling shift**
   -- confirmed by reading `compute_physics_loss()` directly, not inferred: `k`/`p` frozen
   globals, `y_max` the only per-plot degree of freedom, penalised in proportion to
   `physics_weight`. PINN-specific, doesn't touch the DNN's own underperformance.
4. **Real, large predictive signal (`neighbour_spatial_lag`) was correctly removed as a leak and
   nothing has replaced it** -- was the single strongest category ever measured in this project
   (+0.334 R2 drop, ~10x every other category) before removal (2026-07-31). Compartment-level
   bias persisting after the DNN (~2.9m mean absolute, per the user's separate diagnostic) is
   consistent with this still being an open gap -- speculative as THE explanation, but the
   largest known missing piece nobody has tried to rebuild in a leak-safe way yet.
5. **Today's `env_deviation`/`dnn_noenv`-variant sample-size problem** -- real, but narrow;
   explains one specific number, not the broader pattern.
6. **A handful of real small bugs**, found and fixed along the way this session (`whcl` merge
   collision, y_max anchor path, degenerate CR bound) -- present, minor next to 1-4.
**What's working:** ranking by evidence strength rather than treating every candidate cause as
equally likely -- 1 and 2 together are the best-supported explanation for the broad pattern
(small real signal, wrong model family extracting it); 3 is solid but PINN-specific; 4 is the
single biggest untried lever, independent of every model-architecture question above it.
**What's not working / open concern:** `mean_cr_residual` (and everything downstream of it --
SHAP, permutation importance, the refit ablation, Moran's I, `av1_grouped_category_importance.ipynb`)
still reads the pooled/leaky CR anchor, never revisited since the PINN's own 2026-08-01 fix --
independent, small, still open.
**What this means for what's next:** of everything tried or proposed this session, nothing yet
directly addresses cause 4 (the missing spatial-lag signal) -- see the same-day note in this log
about `env_deviation` and the instructions docs for how causes 1-3 map onto the two draft models;
a leak-safe spatial-lag rebuild plus a spatially-aware residual model (e.g. regression/residual
kriging on top of a tree-based mean function -- real precedent exists for exactly this on canopy
height specifically, see Hengl et al. and the GEDI residual-kriging literature) is the most
evidence-backed untried direction, addressing causes 2 and 4 together in one model rather than
separately.

---

**2026-08-02 — Scope-matched XGBoost check: a tree ensemble DOES get a real uplift from the exact
terrain columns that make `dnn_env_terrain` worse -- settles the "is the signal usable at this
scope" question the feature-set-parity entry above left open.**
**What I found:** Built a fresh, minimal comparison (not read from `xgb_environmental.py`'s
existing outputs, which fit a different target/feature scope) -- reused
`load_split_table_with_terrain()` (the same row-level loader `dnn_env_terrain` itself calls) and
`xgb_environmental.py`'s own `fit_with_columns()`/`predict_with_columns()`, so this is the exact
same rows, same `terrain_wind_solid` columns, same raw `elev_percentile_95th` target, same
`spatial_block`/4survey split `dnn_env_terrain` sees -- just a tree model instead of an MLP.
`n_jobs=1` was required (default multi-threaded XGBoost segfaults in the same process as PyTorch
on this Mac -- a known torch/xgboost OpenMP conflict, `KMP_DUPLICATE_LIB_OK=TRUE` alone wasn't
enough, single-threaded XGBoost was). Result:

| Model | Features | Test R2 |
|---|---|---|
| dnn_noenv | age + no-env | 0.6330 |
| XGBoost (scope-matched) | age + no-env | 0.6333 |
| dnn_env_terrain | age + no-env + terrain_wind_solid | 0.6247 (-0.0083) |
| XGBoost (scope-matched) | age + no-env + terrain_wind_solid | **0.6444 (+0.0111)** |

The no-env baseline confirms the two pipelines are genuinely comparable (0.6330 vs 0.6333, well
within noise) -- so the divergence once terrain is added is real, not an artefact of the two data
paths differing.
**What's working:** this directly answers the fork left open by the feature-set-parity null
result: XGBoost, given the IDENTICAL narrow feature scope `dnn_env_terrain` has (not the richer
37-variable model the original SHAP/permutation/refit-ablation numbers came from), still finds a
real, positive uplift from terrain. The signal is genuinely usable at this scope by *some* model
family -- this rules out "the terrain signal just isn't extractable from age+no-env+terrain alone,
regardless of model" as an explanation.
**What's not working / open concern:** this reframes, rather than closes, the investigation --
it's now a genuine MLP/gradient-descent inductive-bias gap (+0.011 for a tree ensemble vs. -0.008
for the MLP on the identical scope), not a data-scope or feature-choice limitation. Doesn't
distinguish which specific MLP shortcoming is responsible (e.g. trees natively split on exact
thresholds/interactions that a plain ReLU MLP can represent in principle but not easily discover
via gradient descent when the terrain signal is small relative to Age's dominant one) -- that's
still open.
**What this means for what's next:** reverses the caution in the entry above about holding off on
mechanism-level work. This is now real evidence that a different mechanism/architecture COULD
recover something current `dnn_env_terrain`/`pinn_env_terrain` are leaving on the table -- the
SA-PINN/UDE research is worth actually acting on, not shelving. Per the user's direction: any new
mechanism variant gets built as its own new model folder (not an in-place rewrite of
`pinn_env_terrain.py`/`dnn_env_terrain.py`, which stay as the reported, reproducible baseline/
control this finding is measured against).

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
only 5 of the 16 columns `av1_grouped_category_importance.ipynb`'s terrain+wind category analysis
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
actual calling code as a real notebook section (`av1_spatial_autocorrelation_terrain.ipynb`'s
Section 3, previously a "not built yet" placeholder), fit on the same train-only spatial-block
split `av1_grouped_category_importance.ipynb` uses, so this is now reproducible end to end.
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
list (none of the three were in it), but the `av1_spatial_autocorrelation_terrain.ipynb` Section 3
table itself still has this error in it, not yet fixed.
**What this means for what's next:** confirmed nonlinearity demotes Elastic Net and NLME from
"attribution evidence" to "how much of this is simple/linear structure" cross-checks only --
the per-category and per-variable REFIT ablation (assumption-free, works on the nonlinear
XGBoost fit) is the metric to trust. Built `per_variable_refit_ablation()`
(`models/xgb_environmental/grouped_analysis.py`) and added it as Section 7.2/7.3 in
`av1_grouped_category_importance.ipynb`, redirecting the ALE plot from SHAP's top-4 (which
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
75m radius (`av1_aux_data_resolution_check.ipynb`, `cKDTree.query_ball_point`), computed ONCE on the
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
cross-check table in `av1_grouped_category_importance.ipynb`) was computed WITH this leak present,
and needs re-reading against the fixed numbers below, not cited from before this date.
**What this means for what's next:** `neighbour_mean_height`/`neighbour_height_differential`
removed from `FEATURE_PROVENANCE`/`ALL_FEATURE_COLUMNS` (`xgb_environmental.py`) and from
`CATEGORY_GROUPS` (`grouped_analysis.py`); the now-redundant `all_environmental_no_neighbour`
feature set and `FEATURE_SETS_NEEDING_SHAP` entry removed (`all_environmental` itself is the
fixed, leak-free set now). `run_xgb_environmental.py`/`run_elasticnet_environmental.py` need
re-running for `all_environmental` (both cohorts) to regenerate the on-disk outputs under the
corrected feature set, and `av1_grouped_category_importance.ipynb` needs re-executing so every graph
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
HadUK-Grid in `av1_aux_data_resolution_check.ipynb` (from before CEDA access was unblocked) was
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
`av1_grouped_category_importance.ipynb` re-run end to end against the corrected feature set.

---

**2026-07-30 — Restored the predecessor notebook's decisive refit-based ablation for
`neighbour_spatial_lag`, missing from `av1_grouped_category_importance.ipynb` since it replaced
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
**What this means for what's next:** `av1_grouped_category_importance.ipynb` section 7.1 now carries
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
these corrected numbers, not the ones from before 2026-07-30. `av1_grouped_category_importance.ipynb`
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

## Stage 2: Per-plot growth curve vs. yldc

A new, parallel model family (separate from everything above) — instead of comparing every plot
against ONE pooled Chapman-Richards curve, give each plot its own growth curve (fixed shape from
that plot's own recorded `(p1..p5)`, only `y_max` free) and compare it against the FC's own
static, per-plot yldc-implied curve. See `documentation/model_instructions/
growth_curve_stage2_handover.md` for the full write-up (idea, candidate architectures, coding-style
instructions) — this section is the factual experiment/findings record only, matching this file's
own convention.

**What's built** (all new files, nothing in Stage 1 touched — confirmed via `git status` after
every step): `data_processing/export_growth_curve_tables.py` (new data layer, writes
`data/processed/growth_curve/<cohort>/growth_curve_table.parquet`) and
`models/growth_curve_attribution/` (`data.py`, `phase0_checks.py`, `temporal_stability_check.py`,
`disturbance_checks.py`, each with a `run_*.py` runner).

**Decided 2026-08-03**: comparisons happen in `Top_Height95` space (`elev_percentile_95th x 1.1`),
matching the FC's own GYC convention, not the raw target Stage 1 trains on — explicit user
choice, not a default. Also decided: NLME/GADA-style per-plot fitting is the cheap FEASIBILITY
diagnostic for this question, not a candidate AI model itself — the actual candidate models
(per-plot fixed-shape fit + tree attribution, Bayesian hierarchical, GNNWR/GRF/GPBoost-style,
explicitly NOT a PINN) stay a shortlist, not yet chosen, per the user's explicit choice to compare
2-3 side by side rather than commit to one this early.

---

**2026-08-03 — Q: Can a reliable per-plot growth-curve data layer be built (with p1-p5/yldc/
coordinates the Stage 1 model_table drops), and is its output trustworthy? — a real
cpmt/cpmt_x/cpmt_y merge-collision bug found and
fixed before the export was trusted.**
**What I found:** `export_growth_curve_tables.py` reads `clean_master_{cohort}.parquet` directly
(already has `p1-p5`/`yldc`/`area` — Stage 1's `model_table.parquet` drops these, which is why
this needed its own script). First run's output columns included `cpmt_x`/`cpmt_y` instead of a
clean `cpmt` — `plot_coordinates.csv.gz` carries its own `cpmt` column, colliding silently with
master's own `cpmt` during the merge, exact same failure shape as the `whcl` collision already
found once in `torch_data.py`. Fixed by dropping any pre-existing coordinate column before
merging, re-ran, confirmed `cpmt` clean afterward.
**What's working:** row/plot counts match the documented values exactly (287,064/71,766 4survey,
83,382/13,897 6survey) — the merge/computation didn't drop or duplicate anything. `y_max_yldc`
(`yldc*p2+p1+p3*2`) confirmed static per plot (std=0). Coordinates land inside the real Aberfoyle
range, not just the wide sanity envelope.
**What's not working / open concern:** 5,168 rows/1,292 plots (4survey, 1.8%) and 1,776
rows/296 plots (6survey, 2.1%) have no valid p1-p5 combination, so no yldc curve — flagged in the
export's own console output, not silently dropped. `p1-p5` itself takes 7 distinct value-sets in
this data (not one universal species constant an earlier brainstorm assumed) — constant per plot,
not tied to that plot's own yldc — still unexplained why 7 sets exist for a
single-species-filtered cohort, functionally fine to proceed on (each plot's own tuple is used
directly, faithful to how the FC itself computed `GYCspec95`/`Vol95` for that plot) but a real
open question for the write-up's limitations.
**What this means for what's next:** data layer trustworthy enough to build Phase 0 checks and
the temporal stability check on top of.

---

**2026-08-03 — Q: Before building any Stage 2 model, is there real signal in the yldc deviation
worth explaining, and would a distance-based spatial mechanism (GNNWR/GRF) even be viable under
spatial_block_split? — real yldc-deviation signal confirmed, but the neighbour-coverage
problem that killed the leak-safe spatial-lag feature in Stage 1 is very likely to recur for any
distance-based Stage 2 architecture.**
**What I found:** Four cheap, no-fitting checks (`models/growth_curve_attribution/
phase0_checks.py`), both cohorts. (1) Every plot has exactly 4 (4survey) or 6 (6survey) distinct
survey years — a genuinely balanced panel, no low-timestamp subgroup. (2) yldc deviation
(`Top_Height95 - top_height95_yldc_predicted`) has real, non-degenerate spread: mean
+1.58m/+2.19m, std 6.5m/4.7m, skew -0.26/-0.55, excess kurtosis 1.57/5.41 — and the spread GROWS
with Age (std climbs from ~3.8-4.7m at young ages to ~9-9.5m at the oldest bands), the OPPOSITE of
an initial low-Age-instability guess, corrected once actually checked. (3) Thinning confound is
real but modest and inconsistent in direction between cohorts. (4) 96.3% (4survey) / 97.1%
(6survey) of `spatial_block_split` test plots have ZERO train-plot neighbours within 75m (mean
distance to nearest train plot ~260-295m) — essentially the same number already found fatal for
the leak-safe `neighbour_mean_height` feature (2026-08-02 entry above).
**What's working:** the deviation's real, non-trivial spread answers "is there anything here to
explain" cheaply, before any model — yes. The balanced-panel result removes one whole category of
worry (uneven identifiability) from every candidate architecture.
**What's not working / open concern:** finding 4 means a short-bandwidth GNNWR/GRF is very likely
dead on arrival under `spatial_block_split`, same structural reason as the Stage 1 neighbour-lag
finding — whole compartments held out, and compartments are almost always bigger than a short
radius. Finding 2's Age-growing variance isn't modelled by any of the 3 shortlisted candidates yet.
**What this means for what's next:** any distance-based Stage 2 model needs a large bandwidth
(hundreds-to-thousands of metres, closer to the ~3,956m semivariogram range already measured for
the Stage 1 CR residual), not a short local radius, decided BEFORE building it, not discovered
after.

---

**2026-08-03 — Q: Is a per-plot fixed-shape growth curve (single free y_max) a real, stable
quantity worth attributing, or just curve-fitting noise? — the core per-plot-curve premise
passes clearly, with two
real caveats.**
**What I found:** Fit each plot's `y_max` using ONLY its earliest survey years (`TEMPORAL_YEARS`'
own `train_years`), then check how well that predicts the SAME plot's own later, held-out years
(`models/growth_curve_attribution/temporal_stability_check.py`, closed-form fit: since
`height = y_max * shape_term` is linear in `y_max` given fixed `p4`/`p5`, the least-squares
solution is `sum(height*shape_term)/sum(shape_term^2)`, no iterative `curve_fit` needed).

| Cohort | Held-out year | Early-fit curve R2 | Static yldc curve R2 |
|---|---|---|---|
| 4survey | 2021 (9yr ahead) | 0.873 | 0.226 |
| 4survey | 2023 (11yr ahead) | 0.779 | 0.165 |
| 6survey | 2021 (9yr ahead) | 0.525 | 0.101 |
| 6survey | 2023 (11yr ahead) | 0.386 | -0.029 |

**What's working:** a plot's own early growth predicts its own future height far better than its
assigned yield class does, at both cohorts and both extrapolation horizons — direct, quantified
evidence of real, stable, plot-specific site information the old pooled-CR-residual approach
never demonstrated before attributing anything to environment.
**What's not working / open concern:** real, systematic degradation with horizon (R2 drops, bias
grows in magnitude, as the gap lengthens) — the premise is good, not perfect. 4survey and 6survey
disagree in a way not yet explained: 6survey is consistently weaker despite an identical
extrapolation horizon from the same last fitting year (2012), and the bias even flips sign
between cohorts (4survey underpredicts growth over time, 6survey overpredicts it).
**What this means for what's next:** clears the gate to build a spatial attribution model on top
of a per-plot y_max — the target itself has real predictive validity, not just in-sample fit. The
cohort disagreement is now one of the 6 open problems being investigated next (see the
2026-08-03 disturbance-checks entry and the handover doc's problem list).

---

**2026-08-03 — Q: Does the per-plot curve's leftover residual show a real mid-trajectory
disturbance signature, and does recorded thinning explain any of the temporal-check bias? —
4survey's residual pattern looks like shape-misfit, not a
disturbance event; a data-quality-flagged long tail found; thinning found to make the fit MORE
accurate, consistent with top height's known thinning-invariance.**
**What I found:** Fit `y_max` using ALL years per plot (not just early), looked at the leftover
residual three ways (`models/growth_curve_attribution/disturbance_checks.py`). (1a) Residual by
survey year, population-level: 4survey is smooth and monotonic (-0.36 -> -0.19 -> -0.03 -> +0.45
across 2008/2012/2021/2023) — no dip anywhere. 6survey shows a real dip specifically at 2021
(-0.57, the single most negative point), more disturbance-like — but the exact survey MONTH (not
just year) isn't known, so this can't yet be checked against a specific storm/drought date.
(1b) Per-plot residual range (max-min across a plot's own years): median 2.1m/2.9m, but max
**47.2m (4survey) / 35.3m (6survey)** — almost certainly clearfell/replant or boundary-mismatch
data problems, not real growth variance; which specific plots these are is NOT yet identified.
(2) Cross-referencing the temporal-check's bias-growth finding against RECORDED thinning
(`last_thinn` — confirmed constant per plot, 0 of 71,766 plots vary, so this checks whether that
single value falls inside the gap window, not a before/after comparison): thinned-during-gap
plots show bias consistently closer to zero than non-thinned plots, in BOTH cohorts, even though
the cohorts' overall bias sign disagrees.
**What's working:** the thinning finding lines up with a real forestry fact surfaced this
session — top height (this whole project's target) is relatively insensitive to thinning by
design (thinning removes suppressed/intermediate trees, not the dominants that define top
height) — so thinning normalising growth toward the textbook curve, rather than boosting past it,
is the theoretically expected direction, not a puzzle.
**What's not working / open concern:** 4survey's smooth monotonic misfit suggests the fixed
`(k,p)` per plot may not flex enough even with `y_max` free — a genuine modelling-assumption
weakness, distinct from and more fundamental than the disturbance question this check set out to
test. The 47m/35m residual-range tail is a real, unaddressed data-quality risk for trusting any
per-plot `y_max` broadly.
**What this means for what's next:** the 6 open problems (4survey shape-misfit, the long-tail
plots, the cohort disagreement, terrain's established ceiling, the neighbour-coverage bandwidth
decision, the unmeasured 2012-2021 gap) are now the explicit next work, ranked cheapest-first in
`documentation/model_instructions/growth_curve_stage2_handover.md` — resume there, starting with
the long-tail plots (cheapest, most concrete, most likely to change how much the other 5 matter).

---

**2026-08-03 — Q: Does compartment membership explain enough of yldc_deviation's variance to
justify building a compartment-pooled growth-curve model (Candidate B)? — compartment
membership explains real, substantial variance (not the ~5% figure from a DIFFERENT target) — BUT the
long-tail plots (problem 2, finally identified) turn out to cluster heavily by compartment too,
so the ICC number above needs re-checking once those are handled, not taken at face value yet.**
**What I found:** Built `models/growth_curve_attribution/compartment_pooling_check.py` —
one-way random-effects ICC of `yldc_deviation` grouped by `cpmt`, computed on each plot's OWN
MEAN deviation across its years (not raw plot-year rows — collapsing to one row per plot first
matters here: plot-year rows are repeated measurements of the same trees, not independent
observations, and grouping raw rows directly by compartment would conflate trivial within-plot
variance with real between-plot variance). This is a check on THIS target specifically — not
copied from the ~5% compartment-variance figure already found for `mean_cr_residual` (Stage 1's
different, pooled-CR target), per this project's own rule to re-pull every number rather than
reason by category.

| Cohort | n plots | n compartments | ICC |
|---|---|---|---|
| 4survey | 56,841 | 232 | **0.399** |
| 6survey | 13,473 | 47 | **0.188** |

(232/47 vs. the familiar 296 (4survey) / 48 (6survey) full-population compartment counts elsewhere
in this project: same underlying compartment population, not a different dataset — confirmed
directly by loading `clean_master_4survey.parquet`/`clean_master_6survey.parquet` (296/48 distinct
`cpmt`, matching `av1_aux_data_resolution_check.ipynb`/`baseline_results.ipynb`) vs.
`load_filtered_growth_curve_table()` (232/47 distinct `cpmt`). 64 (4survey) / 1 (6survey) whole
compartments contain zero plots reaching Age>=30 by the 2023 survey and so drop out entirely under
`filter_data()`'s standard gate — this ICC is computed on the filtered population, same as every
other Stage 2 number.)

ICC also computed by 5-year age band for both cohorts (see the module's own output) — ranges
roughly 0.20-0.61 (4survey) and 0.04-0.58 (6survey), noisier in the oldest/youngest bands where
few compartments have any plots at all, but no band collapses to near-zero.
**What's working:** compartment membership explains a real, substantial share of this target's
variance — ~40% (4survey) / ~19% (6survey) — genuinely different from (and much larger than)
the ~5% figure from the unrelated NLME/`mean_cr_residual` check, confirming that number does not
transfer across targets. This is a real, on-paper case FOR compartment-pooling (Candidate B),
not against it.
**What's not working / open concern:** this ICC was computed on the FULL filtered population,
including the long-tail plots identified in the same session (see next entry) — and those
cluster heavily in a small number of compartments (one single compartment, `2033`, holds 142 of
4survey's 1,137 flagged plots — 12.5% of the whole flagged set in one compartment). If a
compartment's own long-tail plots share a common DATA problem (not a real site effect), that
compartment's mean deviation is distorted by the same artefact for every one of its plots, which
would inflate the between-compartment variance this ICC measures for the wrong reason. The 0.399/
0.188 numbers are therefore an upper bound until re-checked with the long-tail plots
excluded/corrected, not yet a clean answer.
**What this means for what's next:** re-run this exact check after resolving the long-tail
plots (next entry) before using 0.399/0.188 to justify building a compartment-pooled model —
this is now the concrete blocking dependency between problem 2 and the compartment-pooling
question, not two independent problems.

---

**2026-08-03 — Q: Which specific plots have the extreme (up to 47m) residual swings flagged by
the disturbance checks, and why? — nearly all of
them share one specific signature — smooth, plausible growth through the first 3 surveys, then a
collapse to single-digit height at the very next survey — and they cluster heavily by
compartment, consistent with clearfell/windthrow, not scattered per-plot noise.**
**What I found:** Built `models/growth_curve_attribution/long_tail_plots.py`, pulled the top 2%
of plots by `per_plot_residual_range` (already computed by `disturbance_checks.py`, but never
before actually looked at row-by-row) and printed their real Age/height/thinning trajectories.
1,137 plots flagged (4survey), 270 (6survey). Eyeballing the 15 single worst plots per cohort:
the overwhelming majority show `Top_Height95` climbing smoothly and plausibly (e.g. 37m -> 39m ->
42m across 2008/2012/2021), THEN dropping to 4-10m at the very next survey, while
`predicted_height` keeps climbing smoothly across the same years — a 25-32m single-survey height
loss, physically impossible for a mature stand without felling. For 4survey this collapse is
overwhelmingly at 2023 (the last survey); for 6survey it's a mix of 2023 and 2002 (the very
FIRST survey, i.e. an implausibly-high starting height rather than a collapse — same signature,
opposite end of the timeline). Only 340/1,137 (4survey, 29.9%) and 71/270 (6survey, 26.3%) trip
the crude "one year >=3x the median of the plot's other years" ratio test — the test undercounts
(e.g. plot 51917's 2023 crash is 26.5m vs a 9.6m median, ratio 2.76, just under the 3x cutoff) —
the pattern is clearer from reading the trajectories directly than from that single ratio.
Crucially, this does NOT line up with recorded thinning: several of the worst plots (e.g. 318400,
330689, 372203) show `Thin=0.0`/`never_thinned` for their entire recorded history, ruling out
"the recorded thinning field already explains this."
Compartment clustering: the 1,137 flagged 4survey plots span only 128 compartments, and the
single worst compartment (`2033`) alone holds 142 of them (12.5%). 6survey's 270 flagged plots
span 28 compartments, with three compartments (`2250`, `2251`, `2217`) each holding 33.
**What's working:** this directly answers "which specific plots" (never identified before this
session) and gives a real, physically-grounded explanation (clearfell/replant, or possibly storm/
windthrow given the compartment clustering and the lack of a recorded-thinning match) rather than
leaving this as an unexplained statistical tail. The compartment clustering is independently
consistent with the wind-damage/`whcl` hypothesis already raised elsewhere in this project (2026-
08-03 `dnn_terrain_wind` entry) — storm damage plausibly hits a whole compartment at once, not
scattered individual plots.
**What's not working / open concern:** whether this is clearfell/replant (a management decision,
not an environmental signal at all) or storm/windthrow (an environmental signal terrain/wind
SHOULD in principle explain) is not yet distinguished — both produce the same height-collapse
signature in this data, and choosing "exclude as data-quality" vs. "keep as a genuine environment-
attributable disturbance target" depends on which. Not yet cross-checked against any external
felling-record or storm-date source.
**What this means for what's next:** (1) the crash is usually confined to ONE survey year per
plot, with 3 (4survey) or 5 (6survey) other years of good data — dropping only the corrupted
plot-YEAR, not the whole plot, preserves real data the whole-plot exclusion in the original
disturbance-check framing would have thrown away. (2) Compartments `2033` (4survey) and
`2250`/`2251`/`2217` (6survey) are the concrete, checkable next targets — worth a direct look
(and asking the user, who may know the management history) at whether these were felled or
storm-hit around 2021-2023 (4survey) or right at survey start (6survey). (3) The compartment-
variance ICC above must be re-run after this resolution, not trusted as-is.

---

**2026-08-03 — Q: Is the Age<30 plot-level filter hurting or helping baseline model accuracy?
(`--maturity-age-min`, new CLI flag on `run_baselines.py`/`evaluate_baselines.py`) — disabling
the Age<30 gate raises every baseline's R² on
`plot_level`, but this is a test-set-composition artefact, not a real accuracy improvement on
the population the filter was built to isolate.**
**What I found:** Added `--maturity-age-min` (default 30, matching `filter_data()`'s own
default) to both scripts, non-default writing to a `_agemin<N>`-suffixed output path (same
"only non-default gets a suffix" convention as `--split-seed`). Ran all four baselines,
`plot_level`, both cohorts, with `--maturity-age-min 0` (filter fully disabled) and compared
against the existing default (`maturity_age_min=30`) results. Headline R² rises for every
model/cohort (e.g. `rf_baseline`/4survey: 0.570 -> 0.636; `average_by_age`/4survey: 0.429 ->
0.534; `chapman_richards`/4survey: 0.394 -> 0.509) -- looks like a clear case for dropping the
filter. But the age-banded breakdown (`metrics.json`'s own `age_bands`) tells a different story:
on every band that exists in BOTH runs (40-60, 60-80, 80+ -- the mature stands the filter is
built around), MAE/RMSE are essentially identical between the two runs (e.g. `rf_baseline`
40-60: MAE 4.343 vs 4.342; 60-80: MAE 5.468 vs 5.450). The entire R² gain comes from the new
`<30` band the no-filter run adds (21,087 rows for 4survey `rf_baseline`), which has
substantially lower error than every other band (MAE=2.818 vs 3.3-5.7 elsewhere) simply because
young trees are shorter with less variance to predict -- not because the model became more
accurate at anything it previously did.
**What's working:** the age-banded diagnostic caught exactly the confound worth worrying about
before trusting a headline R² change -- comparing overall R² across two DIFFERENT test
populations (one with an easy young-stand subset, one without) is not a fair "did this filter
help or hurt" comparison, and this is now demonstrated with real numbers rather than argued from
first principles.
**What's not working / open concern:** this was only run under `plot_level`; the same
comparison was not repeated under `spatial_block`/`temporal`, the DNN/PINN pipeline, or Stage
2's `load_filtered_growth_curve_table()` (which calls the same `filter_data()`) -- not expected
to show a different mechanism, but not directly verified either.
**What this means for what's next:** the Age<30 filter should stay as-is -- it correctly
isolates the harder, forestry-relevant mature-stand population (the UK practice switch from
top-height measurement to Yield-Class estimation this filter's own comment already documents),
and removing it does not make predictions on that population any better, it just dilutes the
test set with an easier population that inflates the headline number. No further reruns planned
under this thread unless a genuinely different mechanism is suspected elsewhere.

---

**2026-08-03 — Q: For the Stage 2 growth-curve attribution target, is per-plot or
subcompartment-level aggregation the right unit of analysis? — per-plot
is decisively the right unit of analysis, not a close call -- but 6survey's plot-level result
itself is a new, unresolved problem (near-zero/negative R2, unlike 4survey's expected result).**
**What I found:** Built `models/growth_curve_attribution/scale_comparison_check.py` --
`local_y_max_difference` (fitted per-plot `y_max` minus the yldc-implied `y_max`, same
construction as the parallel notebook Codex is building) attributed via Elastic Net + XGBoost
against `xgb_environmental.TERRAIN_AND_WIND_COLUMNS` (the established, already-vetted 16-column
terrain+wind set, not the notebook's wider experimental candidates), under `spatial_block_split`,
val-set R2 (test locked):

| Cohort | Scale | Elastic Net R2 | XGBoost R2 | n (train/val) |
|---|---|---|---|---|
| 4survey | plot | 0.172 | 0.188 | 31,603 / 11,517 |
| 4survey | subcompartment | -0.048 | -0.006 | 350 / 118 |
| 6survey | plot | -0.028 | -0.012 | 7,131 / 2,721 |
| 6survey | subcompartment | -0.199 | -0.158 | 85 / 30 |

Subcompartment table built by grouping plot-level rows on `(cpmt, scpt)`, taking the median of
the target and every feature (same aggregation the notebook uses) -- only 589 (4survey) / 147
(6survey) subcompartments exist in total, matching the earlier concern (raised before any code
was written) that collapsing to this unit would leave far too few units for a signal this small.
**What's working:** 4survey's plot-level number (R2=0.17-0.19) lands exactly where every other
terrain/wind attribution in this project has landed (R2~0.16-0.19) -- a real, expected, sane
result, confirming the `local_y_max_difference` target behaves sensibly at plot level. Aggregating
to subcompartment collapses this to WORSE than predicting the mean for both cohorts -- decisive,
not marginal, evidence against any coarser unit (subcompartment, or a hypothetical intermediate
"divide a subcompartment into areas" unit, which would only sit between these two and inherit the
same small-N problem to a lesser degree).
**What's not working / open concern:** 6survey's plot-level result (R2 ~ -0.01 to -0.03) is
NOT the expected result -- it's a genuinely different outcome from 4survey's, not just a smaller
version of the same signal. This is a new manifestation of problem 3 (the already-open 4survey
vs 6survey disagreement) on a DIFFERENT target than where that problem was first found (the
temporal-stability bias-sign flip) -- worth investigating together, not as two separate issues.
**What this means for what's next:** the scale question (plot vs subcompartment) is settled --
proceed with per-plot as the committed unit of analysis for the attribution model, no further
time on subcompartment or intermediate-area aggregation. 6survey's weak/negative plot-level
signal is now a concrete, evidenced blocker worth investigating before treating either cohort's
attribution number as final.

---

**2026-08-03 — Q: Is 6survey's near-zero/negative plot-level attribution R2 a real result, or an
artefact of the one split seed (42) this project has always used? — 4survey's signal is real
and stable; 6survey does NOT show the same "one unlucky seed hides a real signal" pattern this
project found for the DNN earlier -- it looks like genuine small-sample noise around a weak
effect, not a hidden strong one.**
**What I found:** Re-ran `scale_comparison_check.run_for_cohort()` (plot-level only, subcompartment
already settled dead-on-arrival) under 4 split seeds (42/43/44/45), same `TERRAIN_AND_WIND_COLUMNS`
target/features as the entry above:

| Cohort | Method | seed 42 | seed 43 | seed 44 | seed 45 |
|---|---|---|---|---|---|
| 4survey | Elastic Net | 0.172 | 0.039 | 0.129 | 0.113 |
| 4survey | XGBoost | 0.188 | 0.139 | 0.130 | 0.168 |
| 6survey | Elastic Net | -0.028 | -0.005 | 0.012 | -0.142 |
| 6survey | XGBoost | -0.012 | 0.008 | 0.002 | 0.144 |

**What's working:** 4survey is positive across every seed and method -- XGBoost stays in a tight
0.13-0.19 band, confirming seed 42 was not an unusually lucky draw for this cohort; the signal is
real and reasonably stable.
**What's not working / open concern:** 6survey does NOT reproduce the 2026-08-02 DNN precedent
(where one bad seed hid a consistent positive signal every other seed revealed). Three of four
seeds cluster near zero in BOTH directions (-0.03 to +0.01); only one cell (XGBoost/seed 45)
reaches 0.144, while Elastic Net under that SAME seed goes the other way (-0.142) -- inconsistent
in sign even within one seed, not a clean reversal. Reads as noise around a small/near-zero true
effect, not a real signal masked by an unlucky partition -- most likely explained by 6survey's
much smaller sample (6,500-7,200 train plots per seed vs 4survey's ~31,000), not a split-seed
artefact.
**What this means for what's next:** problem 3 (4survey vs 6survey disagreement) should be
reframed for this target -- not "find the split that reveals 6survey's real signal" (the DNN
playbook), but "6survey likely lacks the statistical power to detect this attribution question at
all, independent of split choice." Any final reporting of this attribution model should treat
4survey as the primary result and flag 6survey's null result as inconclusive-due-to-sample-size,
not as evidence terrain/wind doesn't matter there.

---

**2026-08-04 — Q: If the per-plot growth-curve story is widened beyond the validated terrain/wind
set, do climate, soil/site, edge-position, or management variables materially improve the static
model -- management does, the wider environmental groups mostly do not.**
**What I found:** Re-ran a broader static Stage 2 comparison completely outside the notebooks,
writing fresh CSV outputs on 2026-08-04 (`broad_environmental_spatial_cv_{4survey,6survey}.csv`,
`terrain_wind_management_comparison.csv`, `broad_environmental_category_checks_4survey.csv`).
This kept the same cleaned `local_y_max_difference` target, the same 5-fold compartment-held-out
spatial CV, and the same 60 m buffer, but compared larger static feature scopes using **only
Elastic Net and XGBoost**. `4survey`: terrain/wind = **0.125 / 0.117** (Elastic Net / XGBoost),
broad environment (terrain/wind + climate + soil/site + edge) = **0.093 / 0.102**,
terrain/wind + management = **0.289 / 0.302**, and all 38 static variables together =
**0.290 / 0.318**. One-category-at-a-time additions for `4survey` were also re-run:
terrain/wind + climate = **0.112 / 0.127**, +soil/site = **0.117 / 0.099**, +edge-position =
**0.112 / 0.121**. `6survey`: terrain/wind = **0.023 / 0.021**, broad environment =
**0.019 / 0.027**, terrain/wind + management = **0.079 / 0.101**, and all 38 = **0.110 / 0.094**.
No separate `6survey` category-addition CSV was generated in this run family.
**What's working:** this resolves the user's broader-model question with real reruns, not
assumed notebook outputs. The result is internally coherent: climate, soil/site, and
edge-position do not add much on top of terrain/wind at the available resolutions, while the
five management/stand variables produce the only large descriptive lift. The compact
terrain/wind+management model captures nearly all of the full 38-variable model's signal in both
cohorts.
**What's not working / open concern:** this does NOT mean climate/soil/edge are irrelevant in
general -- only that under this static one-number-per-plot target, at these data resolutions and
under these splits, they add little incremental held-out R² once terrain/wind is already in.
Also, the category-addition reruns were only written for `4survey`, not `6survey`: that was a
scope choice, not evidence. The reason was practical and evidential, not technical -- `6survey`
already had a confirmed near-null terrain/wind baseline and much smaller sample size, so the
one-category-at-a-time 6survey breakdown was expected to be lower-signal and was not needed to
settle the main story once the broader-environment and management comparisons had already been
run for `6survey`.
**What this means for what's next:** the Stage 2 write-up should now distinguish two claims, not
one. The **environmental attribution claim** remains the narrow terrain/wind result. The **best
broader static descriptive model** is terrain/wind + management, with the full 38-variable model
only slightly stronger. If the user wants symmetry, a `6survey` category-addition rerun can still
be added later, but it is a completeness extension rather than a blocker for the main conclusion.

---

**2026-08-04 — Q: should Avenue 1 and Avenue 2 be forced onto one identical feature list, and
does adding the already-built temporal wind information help the current AV2 target if it is
converted into plot-varying interaction features?**
**What I found:** Built a shared variable registry export,
`documentation/variable_registry_av1_av2.csv`, from one code path
(`models/common/variable_registry.py`, exported by
`data_processing/export_variable_registry.py`) so every variable now has an explicit cross-avenue
status: AV1 candidate universe, AV2 static status, AV2 temporal status, provenance, and notes on
whether it is derived/categorical/cohort-specific. This makes the intended relationship between the
two avenues explicit: shared definitions where a variable appears in both, but not a forced
identical feature list across two different targets. Also built and ran
`models/growth_curve_attribution/temporal_wind_extension_check.py`, keeping the SAME AV2 target
throughout (`local_y_max_difference`) and testing the only coherent temporal-wind extension
compatible with that target under the current data shape: cohort-level MIDAS interval storminess
summaries interacted with plot-level exposure variables (`topex`, `windward_topex`, `whcl`,
`gwa_wind_speed_50m`). Raw MIDAS interval metrics are effectively constant within a cohort here, so
they cannot improve a cross-plot model by themselves; only interaction terms can vary plot to plot.
Results written to `outputs/growth_curve_attribution/temporal_wind_extension_check.csv`:
`4survey` terrain/wind baseline = **0.130222 / 0.122229** (Elastic Net / XGBoost) versus
terrain/wind + temporal-wind interactions = **0.130445 / 0.122229**; `6survey` baseline =
**0.043748 / -0.027513** versus extension = **0.027667 / -0.027513**. In other words: effectively
no gain for `4survey`, and a small deterioration for `6survey`.
**What's working:** this resolves two design questions cheaply and cleanly. First, the repo now has
a single, auditable cross-avenue variable registry instead of relying on notebook memory or prose
to explain why a variable appears in AV1 but not AV2. Second, the temporal extension was tested
honestly against the CURRENT AV2 target rather than argued from intuition: because the storm
metrics are cohort-level summaries, the script only tests the biologically plausible thing they can
do under this target -- modulate terrain/wind exposure -- and it does so under the same 5-fold
compartment-held-out CV + 60 m buffer discipline as the established AV2 static checks.
**What's not working / open concern:** this does NOT mean temporal weather is biologically
irrelevant. It means that with the current one-number-per-plot AV2 target (`local_y_max_difference`
= fitted per-plot `y_max` minus yldc-implied `y_max`), the available temporal wind summaries do not
add useful held-out cross-plot information, even after exposure interactions. That limitation is
structural, not just predictive: the current MIDAS interval file is cohort-level, not plot-level,
and the target itself is a persistent plot deviation rather than an interval-specific response.
So this run is a fair extension of the current target, but not a full causal/timing test of storm
effects.
**What this means for what's next:** keep the AV2 target consistent as `local_y_max_difference`
throughout the current notebook/script family and keep the static terrain/wind model as the primary
AV2 environmental result. Use the new registry to explain cross-avenue consistency properly: same
variable definitions and explicit inclusion/exclusion logic, but not a mandatory identical feature
set across different targets. If temporal weather is to be pursued further, the right next step is
NOT more static-target feature bloat; it is a separate interval-level AV2 extension with an
interval-level target (for example, survey-to-survey deviation change), where storm/drought/rain
timing can be tested directly rather than compressed into one per-plot number.

---

**2026-08-04 — Avenue 1: Which environmental and management conditions are associated with
plots being consistently taller or shorter than the shared Aberfoyle Chapman--Richards reference
curve? — the 4survey cohort contains a meaningful spatially generalisable descriptive signal;
the smaller 6survey cohort does not. (Implemented and run by Codex, not Claude.)**
**What changed:** Kept the pooled Chapman--Richards curve and each plot's `mean_cr_residual`
unchanged as the deliberately static, forest-wide descriptive reference. Generalised the existing
five-fold compartment CV helper so it can evaluate this target with the same held-out-compartment
rotation, separate validation fold, and 60 m training buffer used by the per-plot work. Evaluated
the existing 49-variable `all_environmental` set and the 16-variable `terrain_and_wind_only` set
with Elastic Net and XGBoost. The analysis reads the regenerated environmental table, so it uses
the corrected inside-polygon sampling coordinates and corrected historical `Thin` values. Added
the exact forest question and interpretation boundary to
`notebooks/environmental_data/av1_grouped_category_importance.ipynb`. Compact results were written to
`outputs/spatial_block_kfold/cr_residual_environmental_spatial_cv.csv`.

| Cohort | Feature set | Elastic Net pooled OOF R2 | XGBoost pooled OOF R2 | evaluated plots / compartments |
|---|---|---:|---:|---:|
| 4survey | all environmental + management | 0.360 | 0.411 | 71,330 / 296 |
| 4survey | terrain and wind only | 0.216 | 0.274 | 71,727 / 296 |
| 6survey | all environmental + management | 0.004 | 0.062 | 13,897 / 48 |
| 6survey | terrain and wind only | -0.049 | 0.012 | 13,897 / 48 |

**What's working:** For 4survey, both model families agree that the full environmental/management
set explains more held-out spatial variation than terrain and wind alone. Because every eligible
compartment is held out once, this is stronger and more representative evidence than the previous
single-split result. The agreement between a linear regularised model and a nonlinear tree model
also makes the direction of the main conclusion less model-dependent.
**What's not working / interpretation limit:** The 6survey pooled out-of-fold scores remain near
zero. With only 48 compartments, this cohort should be reported as weak/inconclusive evidence,
not as a successful attribution model and not as proof that environment or management has no
effect. The number of evaluated 4survey plots differs between feature sets because rows missing
any required predictor are removed. Coarse raster values are still repeated across many plots and
remain a documented data-resolution limitation. Most importantly, this target is fitted relative
to a pooled curve that overlaps the observed population: the analysis is retrospective spatial
attribution against a fixed reference, **not** prospective forecasting or fully independent
prediction of a newly estimated biological target.
**What this means for what's next:** Use 4survey as the primary Avenue 1 result and present
6survey as an underpowered/null sensitivity result. Avenue 2 remains the separate per-plot
growth-curve question; these Avenue 1 results neither replace nor redefine it. No fold-wise CR
refitting is required because the pooled CR curve is intentionally the shared descriptive
comparison chosen for this avenue.
