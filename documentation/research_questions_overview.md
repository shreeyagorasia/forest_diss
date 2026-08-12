# Research questions — overview and current approach

**Living document, not a fixed spec.** This is a stable reference for "what is each research
question and how are we answering it right now" — not a locked plan. More metrics, evaluations,
sets, and models can be (and likely will be) added as the work continues; update this file rather
than letting it go stale, but don't treat its current contents as final.

**Where the rest of this lives**:
- Execution/roadmap (which jobs to run, in what order, on which machine): `jobs/rq123_methodology/README.md`.
- Environmental feature-set construction methodology (Set1–5, VIF screening, rank-aggregate procedure): `documentation/methodlogy_env_setpick.md`.
- Actual current numbers, with real headline findings: `TEMP_results/TEMP_*.tex` (dated at the top of each file — always prefer these over anything summarized here for citable numbers).
- Plot/map ideas for each research question (tiered simple → complex, with example encodings): the published "RQ1–RQ3 Results & Plot Atlas" artifact.

---

## RQ1 — raw top-height prediction

**Question**: how accurately can a DNN/PINN predict raw top height (`elev_percentile_95th`) once
conditioned on tiers of environmental information — and does the physics constraint still earn
its keep once that information is present?

**Approach**: 3 models (`dnn_env_terrain`, `pinn_env_terrain`, `pinn_env_terrain_k`) x 3
environmental feature tiers (`nested_set2_top10`/`nested_set3_gated_terrain_wind_vif`/
`nested_set4_gated_all_vif`, VIF-screened per `documentation/methodlogy_env_setpick.md`) x 2
cohorts x 5-fold spatial CV, single seed (42) first as a comparison sweep, then a 5-seed reseed of
whichever wins. **Current status (2026-08-11)**: complete, including the reseed. Seed-42 sweep:
`TEMP_results/TEMP_rq1_sweep_results_2026-08-11.tex`. **Winner: DNN + Set3**
(`nested_set3_gated_terrain_wind_vif`) — picked on non-overlapping 95% CIs vs. every PINN variant
on 4survey (a real, statistically distinguishable gap, not noise) and a statistical tie (not a
loss) on 6survey, confirmed by RMSE/MAE and by DNN's own tighter per-fold stability. Re-checked
against RMSE/MAE directly (not just R2) — same conclusion, the metric choice doesn't change it.
5-seed reseed (`TEMP_results/TEMP_rq1_winner_reseed_results_2026-08-11.tex`): confirms the win is
not a lucky single seed (4survey R2 varies only ±0.0035 across 5 seeds) and surfaces a real,
seed-independent finding — DNN systematically underpredicts height on 6survey (negative bias,
every one of 5 seeds).

**Physics-weight ablation — done and answered (2026-08-11)**:
`TEMP_results/TEMP_rq1_physicsablation_results_2026-08-11.tex`. Ran on Set3 (both cohorts), two
tiers: cheap single-split exploratory pass across `physics_weight` ∈ {0, 1, 2}, plus a rigorous
5-fold pass at `physics_weight=0` directly comparable to the existing `physics_weight=1` numbers.
**Finding**: R2 decreases monotonically as physics weight increases, every model, every cohort,
no exceptions — the physics loss is actively counterproductive at this project's data scale, not
just neutral (w=0 beats w=1 by +0.05 R2 on 4survey, a real margin outside both configurations'
CIs). This directly answers why PINN doesn't beat DNN: removing the physics loss closes most of
the gap on 4survey and nearly all of it on 6survey. **But** it comes with a real
accuracy-vs-interpretability tradeoff: PINN_k's `y_max`/`k` parameters become far less
identifiable without the constraint (correlation -0.93 at w=0 vs. -0.45 at w=1 on 4survey,
sign-flipped on 6survey) — worth stating both sides, not just "physics hurts, remove it."

**Split-type integrity check — plot_level vs. spatial_block_kfold (found 2026-08-11, not yet
current-methodology)**: real data already exists showing the "easy" split badly overstates
performance and the effect is very different by model — DNN: 0.799 (unprefixed/easy split) vs.
0.663 (`spatial_block_kfold` pooled) = a 0.136 gap; PINN: 0.589 vs. 0.579 = only a 0.010 gap. DNN
inflates dramatically on the easy split; PINN's physics constraint appears to make it far less
able to exploit whatever leakage that split allows, even though PINN's honest spatial R2 is
lower — a real secondary finding, not just a justification for using spatial CV. **Caveat**: this
is from 2026-08-08, the old `terrain_wind_solid` tier, predates the VIF-screened `nested_set*`
tiers, and was sitting undocumented until now. Re-running this specific check under the current
tiers (cheap — DNN/PINN both fit fast) would make it a citable current-methodology result instead
of background evidence.

**Scope decision — temporal split (resolved 2026-08-11): out of scope, future work only.** Not
pursued further as an active research question, for two different reasons depending on the RQ:

- **RQ2b/RQ3**: not a weak result, a structural non-starter. Confirmed directly (2026-08-11):
  `plot_environmental_features.parquet` (the source for every RQ2b/RQ3 candidate feature, and for
  RQ2b's `mean_cr_residual_{cohort}` target) is exactly one row per plot — 71,766 rows, 71,766
  unique plot IDs, zero year-varying columns. There is no temporal axis in either RQ's data at
  all, so a temporal train/test split is undefinable for them, not just untested. No further
  discussion needed beyond stating this once.
- **RQ1**: **now confirmed under the current methodology too** (`TEMP_results/TEMP_rq1_temporalcheck_results_2026-08-11.tex`,
  Set3, all 3 models, both cohorts, single-split `spatial_block` vs `temporal`) — the old-tier
  evidence undersold it. Current tier: every model, both cohorts, shows a large spatial→temporal
  drop, no exceptions — PINN loses more than 3/4 of its R2 on 6survey (0.73 → 0.16-0.19). This is
  sufficient, current-methodology, directly citable evidence for choosing `spatial_block_kfold` as
  the primary split — the scope decision is fully closed, not just motivated by old background
  evidence anymore. Further temporal work (e.g. multi-seed, other sets) remains future work.

  **Why the drop is this large, not just that it happens (added 2026-08-12)**: confirmed directly
  against the methodology doc — RQ1's environmental conditioning (`nested_set2/3/4`, the entire
  point of this methodology) is drawn from the same static, one-row-per-plot export as RQ2b/RQ3's,
  and is explicitly documented as "static per plot (do not vary by survey year)". Only the target
  itself, `Age`, and the 4 row-level baseline stand-structure columns actually vary temporally in
  RQ1's input. A model conditioned mostly on time-invariant covariates has little to draw on for
  temporal extrapolation beyond `Age` and thinning history — this is a structural reason the drop
  is expected to be large, complementing (not replacing) the empirical finding above. Distinct
  from RQ2b/RQ3's case: there the split was *undefinable* (no temporal axis in the data at all);
  here it was well-posed and worth actually running, which is why it was run.

## RQ2a — does environmental conditioning shrink the departure from the shared growth curve?

**Question**: does giving a DNN/PINN environmental information reduce its residual from the
shared Chapman-Richards curve — and does that reduction concentrate specifically on the plots the
shared curve already does worst on?

**Approach**: no new fitting — reuses RQ1's own predictions against the Chapman-Richards
baseline's predictions, joined row-for-row, split into CR-residual quartiles. Runs once per RQ1
sweep combination (cheap, seconds each). **Current status (2026-08-11)**: complete for the full
seed-42 sweep — see `TEMP_results/TEMP_rq2a_residual_reduction_results_2026-08-11.tex`. Headline:
the "hurts where CR is already accurate, helps most where CR is worst" pattern replicates across
all 18 (model x set x cohort) combinations without exception — a robust, not accidental, finding.

## RQ2b — which environmental variables explain the departure, and in which direction?

**Question**: which specific environmental variables are associated with `mean_cr_residual`, in
which direction, and do three genuinely different model families (linear mixed-effects, penalised
linear, tree ensemble) agree?

**Approach**: NLME + Elastic Net + XGBoost, each fit directly on `mean_cr_residual`, 3
VIF-screened feature tiers, 5-fold spatial CV, 4survey only (6survey's 47 compartments are too
few for stable coefficient estimates here). **Current status (2026-08-11)**: complete for all 3
tiers — see `TEMP_results/TEMP_rq2_attribution_results_2026-08-11.tex`. Headline: `CanopyCover`
and the two thinning-history baseline columns dominate every set for both models, with tight
cross-fold stability — the single most consistent finding in this whole project so far.
**CanopyCover confound — checked 2026-08-13** (see the same TEMP file's own section): the
project's data cheat-sheet flags `CanopyCover` as a possible reverse-causation confound (same ALS
flight/pipeline as the height target, though not algebraically inside any of the 3 target formulas
— not the same category as the proven `Age` circularity). A baseline-only (Set1, the 4
stand-structure columns alone) fit — never previously run — shows baseline alone reaches only
R2=0.19-0.22, with environmental variables adding a real +0.10-0.15 R2 on top, and NLME's
between-compartment variance explained is ~0 for baseline alone (0.016+/-0.078) vs. 0.05-0.20 for
the full sets. Read: `CanopyCover` is a genuine row-level predictor but explains almost none of
the spatial pattern RQ2 is attributing — supports presenting it as a baseline stand-structure
control in the write-up, with environmental variables' contribution beyond that baseline as the
actual attribution headline, rather than treating "CanopyCover wins" itself as the finding. XGBoost
SHAP was considered for this RQ but deliberately pointed at RQ3 instead (see below) — RQ2b already
has three converging global-coefficient views; SHAP's per-plot detail adds less value here than it
does for RQ3's outlier-diagnosis goal.

## RQ3 — attribution of plot-specific curve deviation

**Question**: does letting a plot's environmental relationship vary spatially (GNNWR) explain its
`local_y_max_difference` better than one global relationship (Elastic Net/XGBoost) — and for
which variables, and why do specific outlier plots deviate the way they do?

**Approach**: Elastic Net + XGBoost + GNNWR, same 3 VIF-screened feature tiers (Set4 additionally
carries one-hot soil-category dummies with the reference level dropped), both cohorts, 5-fold
spatial CV. **Current status (2026-08-11)**: EN/XGBoost complete for all 3 tiers x both cohorts —
see `TEMP_results/TEMP_rq3_en_xgb_results_2026-08-11.tex`. Headline: same `CanopyCover`
dominance as RQ2b; 6survey performs far worse than 4survey for this specific target (near-zero or
negative R2 in several cells) — a real finding, not a bug. GNNWR still has jobs in flight on the
cluster, not yet evaluated. Per-plot XGBoost SHAP values (the one thing in RQ3's toolkit that can
explain an individual outlier plot's prediction, not just "which variable matters on average")
were added 2026-08-11 — currently run for Set4/4survey only, not yet used for an actual
outlier-diagnosis pass.

**Known nuance**: Set4's rare one-hot soil dummies (e.g. `ceh_pedotope=2.0`) are absent from
6survey's smaller population. Zero-filled rather than dropped for Elastic Net/XGBoost (keeps
Set4's column count identical to 4survey's), but GNNWR's own pre-existing zero-variance-column
guard strips them back out — so GNNWR's Set4x6survey trains on 17 columns, not 20 like
Set4x4survey. Not a bug, just not the column-count parity the EN/XGBoost path has.

---

## Condensed results-inventory table

For the full version (every available file, every plot/map idea, tiered simple → complex), see
the published artifact. This is the condensed available-results / what-matters / variants view
only, kept here because it changes less often than the plot-idea brainstorm does.

| RQ | All available results | What matters (essential) | Experiment variants |
|---|---|---|---|
| **RQ1** raw height | `predictions.csv` per model/set/cohort/fold; `training_history.csv` per epoch; pooled + per-fold R2/RMSE with bootstrap CI; x/y joinable via `load_plot_coordinates()` | Pooled 5-fold R2 per (model x set x cohort) — done, see TEMP note; physics-weight ablation at the winning set — not done, winner not yet chosen; seed variance on the winner — not done | 3 models x 3 sets x 2 cohorts x 5 folds (seed 42) — **done**. Winner x 5 seeds + zero-physics control — not started |
| **RQ2a** residual reduction | `rq2_residual_reduction.csv` per row with x/y; overall + by-CR-quartile summary stats | Mean reduction & % improved, overall and by quartile — **done**, all 18 combos; whether help concentrates in the worst-CR quartile — **done**, confirmed robust | 3 models x 3 sets x 2 cohorts x 5 folds (90 calls) — **done**. Winner x 5 seeds — not started, contingent on RQ1's winner |
| **RQ2b** attribution | NLME `nlme_fixed_effects.json`; `elastic_net_coefficients.csv` + EN/XGBoost checkpoints; pooled + per-fold R2 per set | Which variables agree in sign/magnitude across NLME/EN/XGBoost — **done**, see TEMP note; fold-to-fold coefficient stability — **done** | 3 sets (VIF'd) x 5 folds, 4survey only — **done** (15 fits) |
| **RQ3** local deviation | EN/XGB `predictions.csv` (with x/y), `metrics.csv`, `elastic_net_coefficients_by_fold.csv`, `xgboost_gain_importance_by_fold.csv`; **new**: `xgboost_shap_values.csv` (per-plot, per-fold, Set4/4survey only so far); GNNWR per-row local coefficients + x/y + predictions | Does GNNWR's spatially-varying fit beat global EN/XGBoost — GNNWR still pending; global vs. local variable importance — partial (EN/XGBoost done, GNNWR pending); 5-fold vs. single-fold R2 gap — not built | EN/XGB x 3 sets x 2 cohorts — **done** (including Set4x6survey, fixed 2026-08-11). GNNWR x 3 sets x 2 cohorts x 5 folds — in progress on cluster |
