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

## Discussion primer — ranked, per RQ (2026-08-15)

Not a draft, not exhaustive — a prompt for you to react to. Per RQ, the results most worth
building discussion around, ranked by how much they matter for actually answering that RQ and how
genuine/non-obvious the finding is (not just "biggest number"). Pull whichever ones you want to go
deeper on.

### RQ1 — raw height prediction, model comparison

1. **XGBoost hyperparameter correction** (`TEMP_rq1_baseline_comparison`). The single biggest open
   item: RQ1's XGBoost baseline was running on unfair defaults; fixed fairly, it beats DNN on both
   cohorts. Directly threatens the "DNN is the winner" framing everything else was built on — worth
   deciding how to resolve before writing the RQ1 conclusion.
2. **5-seed reseed of the winner** (`TEMP_rq1_winner_reseed`). Catches a real overclaim: the
   original "PINN wins on 6survey" read doesn't survive — DNN's 6survey CI fully contains PINN's.
   Good example of a claim checked and corrected with evidence, not just asserted.
3. **Physics-weight ablation** (`TEMP_rq1_physicsablation`). Quantifies what the physics constraint
   actually costs (real R2 drop at w=1 vs w=0 on 4survey) against what it buys (`y_max`/`k`
   identifiability collapses without it) — the clearest accuracy-vs-interpretability tradeoff
   number in the whole project.
4. **Temporal check** (`TEMP_rq1_temporalcheck`). PINN loses over 3/4 of its R2 forecasting
   forward on 6survey, far more than DNN — a real, structural finding about physics-constraint
   fragility under distribution shift, not just "the split matters."
5. **`average_by_age` floor baseline** (`TEMP_rq1_baseline_comparison`). Cheap but genuinely
   informative: on 6survey, a zero-parameter age lookup beats fitted Linear regression — a useful,
   non-obvious calibration point for how much any model's complexity is actually buying.
6. **Plot_level vs. spatial_block_kfold asymmetry** (`TEMP_rq1_plotlevel_check`). DNN inflates
   hugely under an easy split (+0.197 R2); PINN/PINN_k barely move (±0.006) — DNN and PINN don't
   just differ in accuracy, they differ in how exploitable an easy split is for them.
7. **Architecture-size sweep** (`TEMP_rq1_architecture_sweep`). Clean null result across all three
   models — rules out "wrong architecture" as an explanation for anything else found. Useful as a
   ruled-out alternative, lower standalone interest.

### RQ2a — does environmental conditioning shrink the departure from the shared curve

1. **XGBoost's own version of the quartile check** (`TEMP_rq2a_residual_reduction`). The most
   important corrective finding in RQ2: the "shrinks most where CR is worst" pattern is NOT
   PINN-specific — XGBoost shows it too, with comparable or larger average reduction. Forces the
   "why PINN" argument onto different (stability + structural) ground.
2. **The core quartile pattern itself** (`TEMP_rq2a_residual_reduction`). The actual RQ2a answer:
   Q1 (CR already good) gets *worse* under environmental conditioning, Q4 (CR worst) improves a lot
   — a real trade-off, not a uniform improvement. This is the finding the whole RQ hinges on.
3. **Stability gap between DNN and XGBoost** (`TEMP_rq2a_residual_reduction`). The one thing that
   *does* survive as evidence for DNN specifically: its reduction is 5-14x more reproducible
   fold-to-fold than XGBoost's — worth building the "why PINN" argument around this, not accuracy.
4. **5-seed robustness of the quartile pattern** (`TEMP_rq2a_residual_reduction`, Step 4a). Confirms
   the Q1-hurts/Q4-helps shape isn't a single-seed fluke — tight SDs across all 5 seeds.
5. **Cohort asymmetry** (`TEMP_rq2a_residual_reduction`). 6survey's effect is real but much smaller
   than 4survey's (0.174 vs 1.501 mean reduction) — fits the broader 6survey-noisiness pattern
   across the whole project, worth one general sentence rather than re-explaining per RQ.

### RQ2b — attribution of the CR-residual to environmental/stand-structure variables

1. **CanopyCover baseline-only ablation** (`TEMP_rq2_attribution`, Set1 section). Resolves a real
   methodological concern (reverse-causation) with actual evidence rather than assumption:
   environment adds real R2 beyond baseline, and — more decisively — explains real spatial variance
   the baseline explains almost none of. Strong material for "justification of choices."
2. **CanopyCover/thinning dominance, converging across NLME/EN/XGBoost-SHAP** (`TEMP_rq2_attribution`).
   The actual headline: three independent methods agree, tightly, on the same answer — the single
   most consistent finding in the project. Needs the Set1 framing above to not read as circular.
3. **XGBoost hyperparameter fix, adopted in production** (`TEMP_rq2_attribution`). Real, not just
   documented: XGBoost now beats EN on every set; Set1's earlier apparent EN advantage was purely
   a hyperparameter artifact, not a real EN-vs-XGBoost difference on stand-structure-only data.
4. **`topex`/`slope_degrees` as the most stable real environmental signals** (`TEMP_rq2_attribution`).
   The actual environmental (non-baseline) attribution answer, distinct from the baseline-dominance
   story — both NLME and EN agree on direction/magnitude.
5. **Variables that don't survive scrutiny** (`TEMP_rq2_attribution`). `chelsa_gdd5_degc`, `tas_mean`,
   `soilgrids_ph` flip sign or have SD comparable to their mean — honest "what we can't claim" list,
   useful for critical evaluation rather than overselling every coefficient.
6. **Set3 as the weakest set** (`TEMP_rq2_attribution`). Lowest R2, lowest NLME variance explained —
   ties directly to Set3's terrain-heavy, thin-wind, zero-soil/climate composition; a clean
   set-composition-to-outcome link worth stating explicitly.

### RQ3 — plot-specific curve deviation

1. **GNNWR 6survey reseed, correcting the original finding** (`TEMP_rq3_gnnwr_results`). The
   clearest self-correction in the project: "GNNWR loses on 6survey" was wrong — R2 sign-flips
   across seeds, mean near zero. Strong, concrete example of checking a suspicious result rather
   than reporting it at face value.
2. **RQ3 outlier diagnosis + disturbance cross-reference** (`TEMP_rq3_outlier_diagnosis`). A
   hypothesis ("these are disturbance artifacts") generated, then tested against real data, then
   found NOT supported (0/10 flagged) — a genuine hypothesis-test-and-revise example, good material
   for demonstrating critical evaluation, not just reporting a number.
3. **GNNWR beats EN/XGBoost on 4survey — the one reliable positive result** (`TEMP_rq3_gnnwr_results`).
   Answers RQ3's actual question directly: yes, a spatially-varying fit beats a global one, on the
   cohort where the comparison is well-powered.
4. **Same outlier plots recur across nearly every (set, cohort) combo** (`TEMP_rq3_outlier_diagnosis`).
   Outlier-ness is a property of the plot, not an artifact of which feature set was used — a clean,
   robust, non-obvious cross-check most projects wouldn't bother running.
5. **Category attribution rerun changed the finding after VIF screening** (`TEMP_rq3_category_attribution`).
   The earlier "every category removal improves R2" pattern disappeared once VIF screening fixed
   collinearity — shows the VIF fix mattered in practice, not just in principle.
6. **Unexplained Moran's I pattern** (`TEMP_rq3_category_attribution`). Removing `terrain` *lowers*
   residual spatial clustering rather than raising it — flagged as a genuine open question, not
   force-fit an explanation. Good honest material for a limitations/further-work paragraph.
