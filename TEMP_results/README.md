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
| **RQ1 main sweep** | `TEMP_rq1_sweep_results_2026-08-11.tex`: `predictions.csv`, `training_history.csv`, pooled+per-fold R2/RMSE/MAE/bias with bootstrap CI, x/y joinable | **Best model × set × cohort** — done: DNN wins 4survey, PINN/PINN_k win 6survey, no universal winner | 3 models (`dnn_env_terrain`/`pinn_env_terrain`/`pinn_env_terrain_k`) × 3 sets (`nested_set2_top10`/`nested_set3_gated_terrain_wind_vif`/`nested_set4_gated_all_vif`) × 2 cohorts × 5 folds, seed 42 — **done**, 90/90 | Bar+error: pooled R2 by model×set, faceted by cohort |
| **RQ1 baseline comparison** | `TEMP_baseline_env_results_2026-08-09.tex` (old tiers) + `TEMP_baseline_env_new_sets_results_2026-08-10.tex` (current tiers) | **Does DNN/PINN's complexity beat simple baselines given the same information?** — baselines done on both old and current tiers; the actual head-to-head against RQ1's sweep **not yet pulled together** even though both sides now exist | Linear/RF/XGBoost × 3 sets × 2 cohorts × (1 plot_level + 5 spatial_block_kfold folds) — **done**, both tier generations | Grouped bar: baseline vs. DNN/PINN pooled R2, same set/cohort |
| **RQ1 physics-weight ablation** | Not yet run — no file | **Does the physics loss (not just the y_max sub-network architecture) actually help?** DNN vs. PINN(w=0) vs. PINN(w=1) | Planned: `pinn_env_terrain` + `pinn_env_terrain_k` at `physics_weight=0.0`, Set3 only (shared set, avoids confounding weight with feature richness), 2 cohorts × 5 folds = 20 new jobs. `physics_weight=1.0` already exists from the main sweep | Loss curves by epoch, physics weight as the color; bar+error R2 at w=0 vs w=1 |
| **RQ1 plot_level integrity check** | Real numbers found 2026-08-11 (in `documentation/research_questions_overview.md`, no dedicated TEMP file yet) — **old tier only** | **Does spatial_block_kfold actually matter, or would the easy split give the same answer?** DNN: 0.799 (easy) vs 0.663 (spatial_block_kfold) = 0.136 gap. PINN: 0.589 vs 0.579 = 0.010 gap — DNN inflates far more than PINN | Old `terrain_wind_solid` tier only, both split types, 2026-08-08 fit date. **Not yet rerun on current `nested_set*` tiers** | Dumbbell: easy-split R2 → spatial_block R2, one row per model |
| **RQ2a residual reduction** | `TEMP_rq2a_residual_reduction_results_2026-08-11.tex`: `rq2_residual_reduction.csv` per row with x/y, overall + by-CR-quartile stats | **Does environment help most where the shared curve is already worst?** — done, confirmed robust across all 18 combos, no exceptions | Reuses RQ1's 90 sweep fits directly (no new fitting) × CR baseline per fold — **done**, all 90. 5-seed final pass not started (contingent on RQ1's winner) | Quartile bar chart: mean reduction by CR-error quartile, colour = model |
| **RQ2b attribution** | `TEMP_rq2_attribution_results_2026-08-11.tex`: `nlme_fixed_effects.json`, `elastic_net_coefficients.csv`, EN/XGBoost checkpoints, pooled+fold R2 per set | **Which variables explain the CR-residual, and do NLME/EN/XGBoost agree?** — done: `CanopyCover`+thinning dominate every set, both models, tight cross-fold stability (biggest, most consistent finding in the project so far) | 3 sets (VIF'd) × 5 folds, **4survey only** (design choice — 6survey's 47 compartments too few) — **done**, 15/15 | Coefficient forest plot: NLME/EN side by side, sorted by \|effect\| |
| **RQ3 EN/XGBoost + SHAP** | `TEMP_rq3_en_xgb_results_2026-08-11.tex`: `predictions.csv` (x/y), `metrics.csv`, `elastic_net_coefficients_by_fold.csv`, `xgboost_gain_importance_by_fold.csv`, `xgboost_shap_values.csv` (Set4/4survey only) | **Best set for local deviation, which variables matter, and per-plot outlier explanation** — done for the first two (same `CanopyCover` dominance as RQ2b; 6survey much weaker for this target); SHAP built but not yet used for an actual outlier pass | EN/XGBoost: 3 sets × 2 cohorts — **done**, 6/6 (including Set4×6survey, fixed 2026-08-11). SHAP: Set4/4survey only so far, not Set2/Set3/6survey | Grouped bar: EN vs. XGBoost pooled R2 per set/cohort; SHAP beeswarm once run for all sets |
| **RQ3 GNNWR** | Not yet synced — no file | **Does GNNWR's spatially-varying fit beat global EN/XGBoost?** — cannot answer yet | 3 sets × 2 cohorts × 5 folds — jobs submitted, **still running/queued on the cluster as of 2026-08-11**, nothing synced | TBD once results exist — local-coefficient map is the signature idea |
| **RQ3 category attribution** | `TEMP_rq3_category_attribution_results_2026-08-10.tex` | **Which category of variable (terrain/wind/soil/climate/edge/stand) matters** — **stale**, built on the pre-VIF `nested_set4_gated_all` (no `_vif` suffix) | Single spatial_block split, Set4/4survey only, pre-VIF membership. **Not yet rerun on the current, VIF-screened Set4** | Category-level permutation-importance bar chart |
