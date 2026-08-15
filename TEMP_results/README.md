# TEMP_results — index

**What this folder is**: dated, real-numbers holding notes for each experiment as it completes —
not the final dissertation write-up. Each file states how its numbers were generated (exact
commands/functions/inputs), the real results, and a headline finding, so nothing here needs to be
taken on trust or re-derived later. Delete/fold a file into the real results artifact once it's
been formally written up.

**This table is a living index, not a fixed plan** — more experiments, evaluations, and files will
be added as the work continues; update this table when they land rather than letting it drift.

**Related documents**: `documentation/research_questions_overview.md` (what each RQ asks and the
current approach — read that first for context), `jobs/rq123_methodology/README.md` (operational
run order/job counts).

**Plot/map ideas** below are condensed from the published "RQ1–RQ3 Results & Plot Atlas" artifact
— that's the full tiered (simple → complex) brainstorm with example encodings; this table only
gives the headline idea per row, and the GNNWR-dependent ones stay "TBD" until GNNWR's jobs
actually finish and sync (not done as of 2026-08-11).

| RQ | All available results | What matters (essential) / what this table answers | Experiment variants | Good plot/map idea |
|---|---|---|---|---|
| **RQ1 main sweep** | `TEMP_rq1_sweep_results_2026-08-11.tex`: `predictions.csv`, `training_history.csv`, pooled+per-fold R2/RMSE/MAE/bias with bootstrap CI, x/y joinable | **Best model × set × cohort** — **done, winner picked**: DNN+Set3 — non-overlapping 95% CIs vs. every PINN variant on 4survey, statistically tied (not beaten) on 6survey, confirmed by RMSE/MAE/per-fold stability | 3 models (`dnn_env_terrain`/`pinn_env_terrain`/`pinn_env_terrain_k`) × 3 sets (`nested_set2_top10`/`nested_set3_gated_terrain_wind_vif`/`nested_set4_gated_all_vif`) × 2 cohorts × 5 folds, seed 42 — **done**, 90/90 | Bar+error: pooled R2 by model×set, faceted by cohort |
| **RQ1 winner reseed** | `TEMP_rq1_winner_reseed_results_2026-08-11.tex`: pooled R2 per seed + across-seed mean/SD/bias | **Is the DNN+Set3 win a lucky single seed?** — **done, answered**: no. 4survey R2 varies only ±0.0035 across 5 seeds; 6survey SD 0.0078, one mild outlier seed, nothing overturns the original read. 6survey's negative bias is consistent across all 5 seeds — a real, seed-independent finding | DNN × Set3 × 2 cohorts × 5 folds × 4 new seeds (43-46; 42 already existed) — **done**, 40/40 | Bar+error: R2 across 5 seeds, one bar per cohort |
| **RQ1 baseline comparison** | `TEMP_rq1_baseline_comparison_2026-08-11.tex` (the actual head-to-head, Set3, now also includes `average_by_age`) + `TEMP_baseline_env_results_2026-08-09.tex`/`TEMP_baseline_env_new_sets_results_2026-08-10.tex` (raw baseline numbers) | **Does DNN/PINN's complexity beat simple baselines given the same information?** — **CORRECTED 2026-08-14**: RQ1's XGBoost was running on unfair raw defaults (`n_est=100/depth=6/lr=0.3`, no tuning) vs. RQ3's own fixed config (`n_est=500/depth=4/lr=0.04`, early stopping) — refit with the fixed config: **XGBoost now beats DNN on BOTH cohorts** (0.675 vs 0.655 on 4survey, 0.718 vs 0.684 on 6survey), overturning the old "DNN wins" headline. RF is unaffected (never used XGBoost's settings) and still loses to DNN/PINN on 6survey. `average_by_age` (naive Age-only lookup) added as a floor reference: worst on 4survey, beats Linear on 6survey | Linear/RF/XGBoost × 3 sets × 2 cohorts × (1 plot_level + 5 spatial_block_kfold folds) — **done**. `average_by_age` — reused existing run. **XGBoost hyperparameter sensitivity check — done**, 10/10 fits (Set3 × 2 cohorts × 5 folds), CPU-only local | Grouped bar: baseline vs. DNN/PINN pooled R2, same set/cohort |
| **RQ1 physics-weight ablation** | `TEMP_rq1_physicsablation_results_2026-08-11.tex`: both tiers complete | **Does the physics loss itself help, not just the y_max/k architecture?** — **done, answered**: R2 decreases monotonically as weight increases (0→1→2), every model, every cohort, no exceptions; w=0 beats w=1 by +0.05 R2 on 4survey (outside both CIs, real) and +0.01 on 6survey. But w=0 makes PINN_k's y_max/k parameters far LESS identifiable (correlation -0.93 vs -0.45 at w=1) — a real accuracy-vs-interpretability tradeoff, not a clean win | Single-split: 3 weights (0/1/2) × 2 models × 2 cohorts — **done**, 12/12. 5-fold: PINN+PINN_k × 2 cohorts × 5 folds at w=0 — **done**, 20/20 | Loss curves by epoch, physics weight as the colour; bar+error R2 across the weight range |
| **RQ1 temporal check (current tier)** | `TEMP_rq1_temporalcheck_results_2026-08-11.tex`: `predictions.csv` per model/cohort/split-type | **Does spatial_block_kfold actually generalize better than temporal forecasting?** — **done**, 12/12, all 3 models both cohorts. Every model/cohort shows a large drop, PINN loses >3/4 of its R2 on 6survey (0.73→0.16-0.19) — sufficient to formally close the temporal scope question under the current methodology | DNN/PINN/PINN_k × Set3 × 2 cohorts × {spatial_block, temporal}, single split each — **done**, 12/12 | Dumbbell: spatial_block R2 → temporal R2, one row per model |
| **RQ1 plot_level integrity check** | `TEMP_rq1_plotlevel_check_results_2026-08-12.tex` — current tier | **Does spatial_block_kfold actually matter?** — **done, confirmed and sharpened**: DNN 0.831 (easy) vs 0.634 (spatial_block) = 0.197 gap, even bigger than the old tier; PINN/PINN_k show ~0 inflation on either cohort (±0.006) | DNN/PINN/PINN_k × Set3 × 2 cohorts × plot_level, single split — **done**, 6/6 | Dumbbell: easy-split R2 → spatial_block R2, one row per model |
| **RQ1 architecture-size sweep** | `TEMP_rq1_architecture_sweep_results_2026-08-13.tex` | **Was 128x128x128 the wrong architecture for the current env-conditioned models?** — **done, answered: no, for all 3 models**. DNN: no architecture wins on both cohorts (deeper best on 4survey, worst-ish on 6survey; small best on 6survey, worst on 4survey) — the signature of noise, val_loss ranking doesn't even track test-R2 ranking. **PINN/PINN_k extension**: checks the sharper concern "does PINN's combined data+physics loss need different capacity than DNN's" — answered no, and more clearly than DNN: spread across architectures is 5-10x smaller than DNN's, sits at/below the known training-seed noise floor. No limitation needs flagging for RQ2a's PINN numbers | DNN + Set3 × 4 architectures × 2 cohorts — **done**, 8/8. PINN + PINN_k × 4 architectures × 2 cohorts — **done**, 16/16. All single spatial_block split (coarse screen, matches 2026-08-02 convention), evaluate run locally, no cluster round-trip needed for any of it | Not needed — null result, reported as a table |
| **RQ2a residual reduction** | `TEMP_rq2a_residual_reduction_results_2026-08-11.tex`: `rq2_residual_reduction.csv` per row with x/y, overall + by-CR-quartile stats, Step 4a 5-seed section, XGBoost comparison (added 2026-08-14) | **Does environment help most where the shared curve is already worst?** — done, confirmed robust across all 18 combos + 5-seed pass. **XGBoost's own version checked (2026-08-14)**: does NOT give a clean "point for PINN" — XGBoost shows the same quartile-concentrated shrinkage pattern, comparable/larger average reduction than DNN; the one real difference is stability (XGBoost's fold-to-fold SD is 5-14x DNN's). Changes the dissertation's argument structure — "PINN shrinks the residual more cleanly" isn't supportable, the case now rests on reproducibility + structural (physics-parametrized `y_max`/`k`) arguments instead | Reuses RQ1's 90 sweep fits directly × CR baseline per fold — **done**, all 90 + 5-seed pass. **XGBoost env vs no-env, Set3 only, 2 cohorts × 5 folds × 2 arms — done**, 20/20 fits, local | Quartile bar chart: mean reduction by CR-error quartile, colour = model |
| **RQ2b attribution** | `TEMP_rq2_attribution_results_2026-08-11.tex`: `nlme_fixed_effects.json`, `elastic_net_coefficients.csv`, EN/XGBoost checkpoints, pooled+fold R2 per set (all 4, incl. Set1), `xgboost_shap_values.csv` (all 3 env sets, recomputed) | **Which variables explain the CR-residual, and do NLME/EN/XGBoost agree?** — done: `CanopyCover`+thinning dominate every set, every method including SHAP, essentially unchanged by the fix. **CanopyCover reverse-causation concern checked**: supports reframing CanopyCover as a baseline control. **XGBoost bug FIXED IN PRODUCTION 2026-08-15** (`run_rq2_attribution.py` edited, not just a sensitivity check): pooled R2 rises 0.04-0.06 per set (e.g. Set4: 0.338→0.395), **XGBoost now beats EN on every set**. Set1 baseline-only: XGBoost (0.222) and EN (0.220) now essentially tied — the old apparent EN advantage there was a hyperparameter artifact | 4 sets (Set1 baseline + 3 VIF'd) × 5 folds, **4survey only** — **done**, 20/20 fits + 20/20 evaluates through the corrected pipeline. SHAP — **done**, 15/15, recomputed against corrected checkpoints | Coefficient forest plot: NLME/EN side by side; SHAP beeswarm, data ready |
| **RQ3 EN/XGBoost + SHAP** | `TEMP_rq3_en_xgb_results_2026-08-11.tex` + `TEMP_rq3_outlier_diagnosis_results_2026-08-12.tex` | **Best set, which variables matter, per-plot outlier explanation** — **all done, and corrected**: the same ~10 plots recur as the worst residual across nearly all 6 (set × cohort) combos (outlier-ness is a plot property, not a feature-set artifact). Cross-referenced against the disturbance-classification system — **0 of 10 carry any disturbance/measurement flag, and 0 of 10 clear the top-1% trajectory-instability cutoff either**, overturning the earlier "almost certainly disturbance/data-quality artifacts" read. Better-supported split: 5 plots show real (sub-threshold) trajectory instability; 5 show a smooth, internally consistent trajectory offset far from the official yldc benchmark — pointing to a yldc lookup mismatch, not disturbance. No basis found for excluding these plots from RQ3's numbers | EN/XGBoost: 3 sets × 2 cohorts — **done**, 6/6. SHAP: **done**, 6/6. Outlier diagnosis: **done**, all 6/6 combos + disturbance cross-reference | Grouped bar: EN vs. XGBoost pooled R2; SHAP beeswarm |
| **RQ3 GNNWR** | `TEMP_rq3_gnnwr_results_2026-08-11.tex` (now includes the 6survey reseed correction) | **Does GNNWR's spatially-varying fit beat global EN/XGBoost?** — **done, and corrected**: yes on 4survey (stable, reliable). 6survey's original "GNNWR loses" reading was wrong — the reseed shows R2 sign-flips across seeds for every set, mean ≈0, within the seed-to-seed SD — the honest read is "underpowered, no reliable direction," not "loses" | 3 sets × 2 cohorts × 5 folds — **done**, 30/30. 6survey reseed (2 more seeds × 3 sets × 5 folds) — **done**, 30/30, zero failures | Local-coefficient map (data ready, not yet plotted); 3-panel EN/XGB/GNNWR residual comparison |
| **RQ3 category attribution** | `TEMP_rq3_category_attribution_results_2026-08-11.tex` (current, VIF-screened Set4) — supersedes `TEMP_rq3_category_attribution_results_2026-08-10.tex` (pre-VIF, kept as historical record) | **Which category of variable matters** — **done, and the rerun changed the finding**: `stand_structure` still dominates, but the earlier "every other category removal improves R2" pattern is gone — VIF screening fixed what looks like collinearity-driven noise; `climate`'s contribution roughly doubled; `soil_site` remains ~0 (pure noise) in both versions | Single spatial_block split, Set4/4survey only, current VIF-screened membership — **done** | Category-level permutation-importance bar chart |

---

## Discussion, analysis, and the results-chapter draft

The ranked discussion primer and the results/evaluation chapter draft itself now live in
`documentation/august_draft/5_Chapter_results_evaluation/a_draft_plan_14th_aug.md` — that's the
actively-edited file for turning these numbers into an argument. This folder (`TEMP_results/`)
stays scoped to what it says at the top: raw, dated, real-numbers holding notes per experiment,
plus the method notes below — no discussion/ranking/prose interpretation content here, so there's
one place, not two, for that.

**Methodology → Results content migration table** (specific sentences flagged as living in the
wrong chapter — methodology currently states them as findings) also moved there, since it
cross-references the primer's item numbers directly.

---

## Reproducing the ad-hoc TEMP checks (2026-08-15)

Several checks this session were run as one-off local snippets rather than saved as a dedicated
script under `models/`. In every case the actual statistic is computed by an existing, already-used
project function — only the *driving loop* (which files/sets/cohorts to feed it) was throwaway. All
read already-saved `predictions.csv`/`test_predictions.csv` files — none of them retrain a model.
Each has its own short method file (what was evaluated + pseudocode for how + pointer to the
numbers), so nothing here needs digging through chat history to reproduce:

| Method file | RQ | What it covers |
|---|---|---|
| `TEMP_rq2b_rq3_method_bootstrap_ci_2026-08-15.tex` | RQ2b + RQ3 | Cluster-bootstrap 95% CI on pooled R2 (EN/XGBoost, GNNWR) |
| `TEMP_rq2b_method_vif_2026-08-15.tex` | RQ2b | VIF on the SHAP-important attribution variables |
| `TEMP_rq2b_rq3_method_morans_i_2026-08-15.tex` | RQ2b + RQ3 | Residual Moran's I (EN/XGBoost/GNNWR) |
| `TEMP_rq3_method_boundary_ymaxfit_mechanism_2026-08-15.tex` | RQ3 | Why some outlier plots' `y_max_fit` is implausible; boundary-distance pull |
| `TEMP_rq3_method_broader_scan_sensitivity_2026-08-15.tex` | RQ3 | Tukey-fence scan for implausible `y_max_fit` beyond the known 10; exclusion sensitivity check |
| `TEMP_rq3_method_typology_crossmodel_2026-08-15.tex` | RQ3 | Cross-model (EN/XGBoost/GNNWR) outlier agreement; sub-compartment spatial clustering |
| `TEMP_rq3_method_2008_artifact_2026-08-15.tex` | RQ3 | Wind exposure, storm-year signature, 2008 survey-boundary artifact test |
