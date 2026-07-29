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
- **PINN cr_matched** (temporal-restricted CR anchor), if run: a distinct model name,
  `outputs/pinn_noenv_crmatched/<cohort>/` — since this isn't a different split, it's a different
  PINN configuration, so it gets a model-name suffix rather than a split-type prefix. The
  `run_metadata.json`'s `frozen_cr_params` field already records exactly which values were used
  either way, but a distinct output path is required so cr_pooled and cr_matched results can coexist on
  disk rather than one overwriting the other.
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
