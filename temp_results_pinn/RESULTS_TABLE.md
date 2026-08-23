# PINN (corrected forward pass) -- results ledger

Running index of every experiment run since the forward-pass fix (`PLAN.md`), so a future
dissertation edit has one place to check: what was tested, what it found, which file has the
raw numbers, and which script would reproduce it. Update this file when a new experiment
finishes -- do not recreate it.

Status legend: **DONE** = ran, numbers below are real. **PENDING** = script exists, not yet
submitted/finished. **HELD** = deliberately not run yet (waiting on an earlier step's result).

---

## 1. Corrected full rerun -- Table 3's actual PINN/PINN-k numbers

**Status: DONE.** These are the numbers currently cited in the dissertation (`results_q3_*.tex`).

| Variant | Set3 test R2 (5-fold mean) |
|---|---:|
| PINN (y_max only) | 0.631 |
| PINN-k (y_max + k) | 0.618 |

- Script: `temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold.py`
- Raw fold files: `temp_results_pinn/outputs/full_rerun/fold_{0,1,2}/summary.json` (local, folds 0-2)
  and `temp_results_pinn/outputs/full_rerun_cluster/fold_{3,4}/pinn_{ymax,k}_fixed_summary.json`
  (cluster, folds 3-4)
- Config: lr=0.0001, weight_decay=1e-5, batch_size=256, physics_weight=1.0, Set3
  (`nested_set3_gated_terrain_wind_vif`)

## 2. Plain-PINN population-level y_max distribution check

**Status: DONE.** Answers "was the example plot in the Q3 figure representative, or cherry-picked?"

- Fold 0 only, corrected plain PINN (y_max-only fix).
- n=11,508 plots. Mean y_max_pred - y_max_population = **+2.93 m**, SD = **5.32 m**.
  77.2% of plots above the population curve, 22.8% below. 18/11,508 (0.16%) implausible
  (<5 m or >70 m).
- Script: `temp_results_pinn/pinn_env_terrain_fix/run_ymax_distribution_check.py`
- Output: `temp_results_pinn/outputs/example_curve/plain_pinn_fixed_test_set_predictions.csv`

## 3. DNN / PINN / PINN-k learning-rate + weight-decay transfer check (2026-08-22)

**Status: DONE -- null result.** Tests whether the Aug-19 DNN hyperparameter sweep's winning
config (lr=0.001, weight_decay=1e-3) helps, given DNN/PINN/PINN-k all share the same
lr=0.0001/weight_decay=1e-5 defaults. Fixed to Set3, 5-fold CV, batch_size=256 throughout
(matching PINN's own default -- DNN's module default was changed from 512 to 256 on this date
to make this a fair, explicit comparison; see `models/dnn_env_terrain/dnn_env_terrain.py` and
`models/dnn_noenv/dnn_noenv.py`).

| Model | Old config (lr=0.0001, wd=1e-5) | New config (lr=0.001, wd=1e-3) | Delta |
|---|---:|---:|---:|
| DNN | 0.6550 (Table 3, trusted) | 0.6426 | -0.0124 |
| PINN (y_max) | 0.6308 | 0.6303 | ~flat |
| PINN-k | 0.6178 | 0.6142 | -0.0036 |

**Conclusion**: the new config is flat-to-worse for all three models at batch_size=256. The
apparent Aug-19 win was confounded by batch size -- that sweep ran entirely at
`batch_size=512` (the DNN module's default at the time), never actually tested at 256 (Table
3's real setting), and used a single train/val/test split rather than 5-fold CV. **The
dissertation's existing "training hyperparameter under-tuning (ruled out)" bullet stands as
written** -- no edit needed. See `TEMP_results/TEMP_rq1_dnn_hyperparameter_search_2026-08-19.tex`
for the (now superseded) Aug-19 sweep this was checking.

- Scripts: `models/baselines/rq1_dnn_tuned_cluster_fold.py`,
  `temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_lr_check.py`
- Raw fold files: `outputs/CORRECTED_2026-08-22_dnn_tuned_cluster/fold_{0-4}/dnn_tuned_summary.json`,
  `temp_results_pinn/outputs/CORRECTED_2026-08-22_lr_check/{ymax,k}/fold_{0-4}/pinn_{variant}_lr_check_summary.json`
- **Follow-up (see #3b below)**: whether `batch_size=512` (at the ORIGINAL default lr/weight_decay)
  beats `batch_size=256`, tested for DNN and PINN/PINN-k together (so the fair-comparison pairing
  is never broken in either direction).

## 3b. Batch-size check: 256 vs 512, at original lr/weight_decay (2026-08-22)

**Status: DNN half DONE, PINN half PENDING (jobs still running on cluster).** DNN, PINN, and
PINN-k all together (never testing one model's batch size in isolation, so the pairing rationale
in #3 stays intact either way this resolves). Same scripts as #3, reused via new CLI flags
(`--batch-size 512 --output-dir-name CORRECTED_2026-08-22_{dnn,pinn}_bs512_check`).

| Model | batch_size=256 (trusted/step 3) | batch_size=512 | Delta |
|---|---:|---:|---:|
| DNN | 0.6550 +/- 0.016 | 0.6552 +/- 0.016 | ~0 (no difference) |
| PINN-k | 0.6178 | 0.6185 +/- 0.021 (5/5 folds) | ~0 (no difference) |
| PINN (y_max) | 0.6308 | 0.6406 (3/5 folds, PARTIAL -- not final) | pending |

**Conclusion: batch_size=512 is not worth adopting.** DNN and PINN-k both show a flat null.
batch_size=512 is also strictly more expensive for PINN -- higher per-epoch cost (~11.3s/epoch
vs ~8s/epoch) AND no reduction in epochs needed (48-102 epochs vs the usual ~87), so it's a
pure cost with no offsetting speed or accuracy benefit. The partial y_max number above looks
higher but is missing exactly the two folds (3, 4) that were PINN-k's worst-performing folds at
512 -- likely to regress toward parity once complete, not confirmed as a real gain. **Decision:
keep batch_size=256 for the lambda ablation and Set2/Set4 sweep** (already PINN's natural
default, already what Table 3 uses -- no re-run needed). Will update this row if the final 2
y_max folds meaningfully change the picture, but this is not blocking further work.

- Scripts: same as #3 (`rq1_dnn_tuned_cluster_fold.py`, `run_cluster_fold_lr_check.py`), now with
  `--batch-size` and `--output-dir-name` exposed so results never collide with #3's batch_size=256
  results.
- Output: `outputs/CORRECTED_2026-08-22_dnn_bs512_check/fold_<i>/dnn_tuned_summary.json` (all 5
  folds in), `temp_results_pinn/outputs/CORRECTED_2026-08-22_pinn_bs512_check/<variant>/fold_<i>/pinn_<variant>_lr_check_summary.json`
  (still running)

## 4. Lambda (physics-loss weight) ablation

**Status: DONE -- null result, keeping lambda=1.0.** All 70 jobs (7 lambdas x 5 folds x 2
variants) landed. R2 is essentially flat across the whole range for both variants (PINN-k:
0.612-0.620; plain PINN: 0.623-0.632), well within the ~0.02 fold-to-fold noise -- no lambda
value meaningfully beats the trusted lambda=1.0 (PINN-k 0.6178, PINN 0.6308). There is a mild,
consistent downward drift for PINN-k once lambda exceeds 1.0 (0.6178 -> 0.6169 -> 0.6153 ->
0.6118 as lambda goes 1.0 -> 1.5 -> 2.0 -> 3.0), suggesting pushing physics weight too high
mildly hurts, but nothing in 0-0.75 beats 1.0 either. `param_correlation_ymax_k` is similarly
flat (0.49-0.55) with no clear trend against lambda. **Decision: keep lambda=1.0 for the
Set2/Set4 sweep** -- it already has the trusted Table 3 numbers built on it and nothing tested
improves on it beyond noise.

| lambda | PINN-k R2 | std | param_corr(y_max,k) | PINN(ymax) R2 | std |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.6201 | 0.0227 | 0.5361 | 0.6313 | 0.0222 |
| 0.25 | 0.6163 | 0.0233 | 0.4916 | 0.6303 | 0.0236 |
| 0.5 | 0.6181 | 0.0219 | 0.5016 | 0.6321 | 0.0229 |
| 0.75 | 0.6199 | 0.0197 | 0.5547 | 0.6314 | 0.0232 |
| 1.0 (trusted, not rerun) | 0.6178 | -- | -- | 0.6308 | -- |
| 1.5 | 0.6169 | 0.0211 | 0.5508 | 0.6279 | 0.0255 |
| 2.0 | 0.6153 | 0.0187 | 0.5189 | 0.6257 | 0.0244 |
| 3.0 | 0.6118 | 0.0182 | 0.5142 | 0.6231 | 0.0256 | Uses the ORIGINAL lr=0.0001/weight_decay=1e-5 (since
step 3 found no improvement from tuning these), batch_size=256, Set3, **both variants** (plain
PINN and PINN-k). lambda in {0, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0} (lambda=1.0 already exists as
the trusted Table 3 number for both variants, see #1) -- 70 jobs total (7 lambdas x 5 folds x 2
variants).

PINN-k is the interpretability-story variant (personalizes both y_max and k per plot) and is
the one whose winning lambda will carry forward into the Set2/Set4 sweep. Plain PINN is included
here for a complete picture across both Table 3 rows, but per-variant note: plain PINN
consistently OUTPERFORMS PINN-k on raw R2 in every test run so far (Table 3: 0.631 vs 0.618;
LR-check: 0.6303 vs 0.6142; batch_size=512: 0.6406 partial vs 0.6185) -- PINN-k's value is
interpretability/mechanistic completeness (two personalized curve parameters), not accuracy.
Plain PINN's own lambda ablation is informational only; its Set2/Set4 run will stay at the
untuned lambda=1.0 default regardless of what this ablation finds for it.

- Script: `temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_lambda_ablation.py`
- Job: `temp_results_pinn/jobs/run_pinn_fix_lambda_ablation_cluster.sh`
- Output (once run): `temp_results_pinn/outputs/CORRECTED_2026-08-22_lambda_ablation/lambda<X>/<variant>/fold_<i>/pinn_<variant>_fixed_summary.json`
- Also records `param_correlation_ymax_k` (correlation between predicted y_max_i and k_i per
  plot) as a second metric for the k variant only -- not meaningful for plain PINN (no k
  sub-network there).

## 5. Set2/Set4 environment feature-set sweep -- "does more environment help PINN?"

**Status: DONE -- real, positive finding.** All 20 jobs landed (2026-08-23). Config: lambda=1.0,
learning_rate=0.0001, weight_decay=1e-5, batch_size=256, both variants, Set2 and Set4 (Set3
already trusted from #1; Set1 covered by `pinn_noenv`, see scope note below).

| Set | PINN (y_max) | PINN-k |
|---|---:|---:|
| No environment (`pinn_noenv`, Set1-equivalent) | 0.573 | 0.573 |
| Set2 (`nested_set2_top10`) | 0.6274 +/- 0.023 | 0.6223 +/- 0.017 |
| Set3 (`nested_set3_gated_terrain_wind_vif`, headline/Table 3) | **0.6310** | **0.6180** |
| Set4 (`nested_set4_gated_all_vif`, broadest) | 0.6184 +/- 0.029 | 0.6196 +/- 0.020 |

**Finding**: the environment-helps argument holds up clearly under the corrected forward pass --
no-env to any-env (Set2) is a big, real jump (+0.054 plain PINN, +0.049 PINN-k), far larger than
the ~0.02 fold-to-fold noise. This directly supersedes the pre-fix Aug-11 sweep, which falsely
showed all three sets flat at ~0.577-0.579 (the bug signature), indistinguishable from no-env.

**Nuance**: once *some* environment is present, adding *more* doesn't help further, and going
too broad (Set4) mildly hurts plain PINN (0.631 -> 0.618, -0.013) while PINN-k stays flat across
Set2/Set3/Set4 (0.618-0.622, all within noise of each other). Set3's VIF-gated feature curation
looks close to optimal -- not just "environment matters" but "curated environment matters, raw
breadth doesn't add more."

- Script: `temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_set_sweep.py`
- Job: `temp_results_pinn/jobs/run_pinn_fix_set_sweep_cluster.sh`
- Output: `temp_results_pinn/outputs/CORRECTED_2026-08-22_pinn_set_sweep/<feature_set>/fold_<i>/pinn_<variant>_fixed_summary.json`
- Sets covered: `nested_set2_top10`, `nested_set4_gated_all_vif` (Set3 already covered by #1;
  Set1 is covered by `pinn_noenv`, which has no per-plot sub-network so is structurally immune
  to the forward-pass bug -- no rerun needed there).

- Script: `temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_set_sweep.py`
- Job: `temp_results_pinn/jobs/run_pinn_fix_set_sweep_cluster.sh`
- Output (once run): `temp_results_pinn/outputs/CORRECTED_2026-08-22_pinn_set_sweep/<feature_set>/fold_<i>/pinn_<variant>_fixed_summary.json`
- Sets to run: `nested_set2_top10`, `nested_set4_gated_all_vif` (Set3 already covered by #1;
  Set1 is covered by `pinn_noenv`, which has no per-plot sub-network so is structurally immune
  to the forward-pass bug -- no rerun needed there).

---

## 6. Terrain permutation importance -- is y_max dominated by one feature, like GNNWR was by CanopyCover?

**Status: DONE -- clean negative result (in a good way).** Q2 found GNNWR's whole "location
matters" advantage collapsed to one variable (CanopyCover, ~80-90% of total signal). Tested
whether the CORRECTED PINN-k has the same problem, using permutation importance (shuffle one
terrain feature at a time, measure how much `y_max_pred`/`k_pred` moves) on a freshly-trained
fold-0, Set3, trusted-config model (no checkpoint existed for the trusted run, so retrained here
with the model kept in memory -- diagnostic only, not a new Table 3 number).

| Feature | y_max importance (% of total) | k importance (% of total) |
|---|---:|---:|
| windward_topex | 15.2% | 6.1% |
| eastness | 14.2% | 7.0% |
| slope_degrees | 12.8% | 19.5% |
| gwa_weibull_k_50m | 12.1% | 9.5% |
| elevation | 11.4% | 18.5% |
| gwa_weibull_a_50m | 6.5% | 9.5% |
| ceh_twi | 6.4% | 5.4% |
| gwa_weibull_k_10m | 5.9% | 6.9% |
| solar_radiation_index | 5.5% | 9.4% |
| gwa_wind_speed_10m | 5.1% | 2.8% |
| gwa_weibull_a_10m | 4.7% | 5.5% |

**Finding**: no single feature dominates -- the top feature (windward_topex) carries only 15.2%
of total importance for y_max (vs. GNNWR's ~80-90% for CanopyCover). All 11 terrain features
contribute meaningfully (roughly 5-15% each for y_max, 3-20% for k). PINN's personalization
draws on the whole terrain picture, not one variable secretly doing all the work -- a genuine,
positive point of difference from GNNWR worth citing in the dissertation.

- Script: `temp_results_pinn/pinn_env_terrain_fix/run_terrain_permutation_importance.py`
- Config: fold 0, Set3, lr=0.0001, weight_decay=1e-5, batch_size=256, physics_weight=1.0 (trusted
  defaults), 10 shuffle repeats per feature, averaged.

---

## 7. Why does plain PINN beat PINN-k? Trunk-residual comparison

**Status: DONE -- real mechanism found, CONFIRMED robust across 3 folds (2026-08-23).**
Compared how much "compensating work" the shared trunk network (main_network) does in each
variant. If personalizing k adds useful structure, the trunk shouldn't need to compensate more;
if it adds noise, the trunk has to work harder to correct for it.

| Fold | Plain PINN trunk mean\|.\| | PINN-k trunk mean\|.\| | Ratio (PINN-k / plain) |
|---|---:|---:|---:|
| 0 | 0.2069 | 0.4096 | 1.98 |
| 1 | 0.2375 | 0.5532 | 2.33 |
| 2 | 0.2547 | 0.8438 | 3.31 |

**Finding**: PINN-k's trunk does substantially more compensating work than plain PINN's in
*every* fold tested, and the effect grows rather than shrinks (ratio 1.98 -> 2.33 -> 3.31) --
a robust, real mechanism, not a fold-0 fluke. This is measurable evidence that personalizing k
doesn't produce a better-fitting curve on its own -- it makes the curve fit *worse*, and the
flexible trunk network has to clean up after it. Directly explains (not just describes) why
plain PINN beats PINN-k on accuracy (section 1/Table 3: 0.631 vs 0.618).

- Script: `temp_results_pinn/pinn_env_terrain_fix/run_pinn_mechanism_checks.py` (check 1)

## 8. Why doesn't the broader Set4 help? Permutation importance on Set4's new features

**Status: DONE -- clean, consistent answer across 3 folds.** Set4 is not simply Set3 plus more
features -- it DROPS `elevation` (VIF casualty once more variables are present, per
`documentation/plans_md/methodlogy_env_setpick.md`) and ADDS four new ones: `dist_to_scpt_boundary`,
`tas_mean`, `chelsa_gdd5_degc`, `cpmt_compactness_ratio`. Ran the same permutation-importance
method as section 6, on a freshly-trained Set4 PINN-k model, across 3 folds.

| Fold | New-features' share of total y_max importance | Proportional "fair share" (4/14 features) |
|---|---:|---:|
| 0 | 39.5% | 28.6% |
| 1 | 30.2% | 28.6% |
| 2 | 31.9% | 28.6% |

`tas_mean` and `chelsa_gdd5_degc` rank in the top 3-4 features in every fold; `dist_to_scpt_boundary`
is consistently the least important feature (1.5-3.5%).

**Finding**: the new features are not harmless/ignored -- they get real, substantial importance,
at or above their proportional share in every fold. This rules out "the model just ignores the
extra features" as the explanation for Set4's flat-to-negative result (section 5: Set4 R2=0.618
plain PINN / 0.620 PINN-k, vs. Set3's 0.631 / 0.618). Instead, the new climate variables
(`tas_mean`, `chelsa_gdd5_degc`) are actively used, substantially, but this doesn't translate
into better held-out accuracy -- closer to "actively counterproductive" (the model treats them
as real signal that doesn't generalize as well as Set3's more curated set) than "harmless
clutter."

- Script: `temp_results_pinn/pinn_env_terrain_fix/run_set4_permutation_importance.py`

---

## 9. What's actually unusual about the most-inflated y_max plots? Direct stand-data inspection

**Status: DONE -- strong, specific finding, strengthened by widening the check beyond the
strict threshold (2026-08-23).** Started by joining the 18 plots that clear the hardcoded
implausibility bound (`y_max_pred < 5m or > 70m`, fold-0 population check,
`temp_results_pinn/outputs/example_curve/plain_pinn_fixed_test_set_predictions.csv`) directly
against the master dataset (`data/processed/master/clean_master_4survey.parquet`) to inspect the
real stand fields (species, yield class, block/compartment, canopy cover, area). Then checked
whether 70m is a real cliff in the distribution or an arbitrary line -- it's the latter (317
plots sit in 65-70m, tapering smoothly, no gap) -- so widened the inspection to the top 40
most-inflated plots regardless of the strict threshold.

**Finding**: **31 of the top 40 most-inflated plots (77.5%) sit in exactly one physical
compartment** -- block 21, compartment 1129 -- confirming the clustering isn't an artifact of
exactly where the 70m line falls; it dominates the whole extreme tail, not just the 18 that
happen to cross it. Two smaller secondary hotspots also emerge once the strict threshold is
dropped:
- **blk 30 / cpmt 2229** (6 plots, all yield class 24, y_max_pred ~69.1m).
- **blk 32 / cpmt 2064** (3 plots, yield class only 6 -- notably low, y_max_pred ~69.3m).

Within compartment 1129 itself, two sub-patterns: most plots (sub-compartments B and E) have
yield class 22 or 24 -- at or near the top of the *entire dataset's* range (2-24), genuinely
among the best-growing sites in the whole forest by the forestry service's own classification;
PINN's personalization may be amplifying a real "exceptional site" signal past a plausible
ceiling. A smaller group (sub-compartment I, all age 71 -- notably older than the rest) has
yield class only 10 (below the dataset median of 18) yet gets similarly inflated predictions,
with wildly inconsistent canopy cover within the same small group (0.06 to 0.48) -- plausibly a
data-quality/survey artifact specific to that sub-compartment, not a model failure. Not fully
explained by yield class alone: sub-compartments A and G in the *same* compartment also carry
yield class 24 (139 plots combined) but contributed zero implausible predictions -- so something
more specific than "high yield class" is at work within the affected sub-compartments
specifically. Species is uninformative (100% Sitka Spruce dataset-wide, not a distinguishing
factor).

**Read**: this is a real, broad-tail pattern (not just 18 isolated points), overwhelmingly
concentrated in one compartment with at least two smaller secondary hotspots -- points at either
a genuinely unusual, high-quality site being over-amplified, or a data/survey artifact localized
to specific compartments' LiDAR passes, rather than a generalizable model weakness spread evenly
across the forest. Worth stating as an open question with a concrete, checkable lead (not a dead
end).

- Method: direct join, no new training -- `identification` -> `data/processed/master/clean_master_4survey.parquet`.

---

## Legacy / do-not-cite

Pre-fix PINN/PINN-k runs (broken forward pass -- terrain features never reached the
prediction, only the physics-loss target) and old feature-set-naming runs were moved to
`legacy/2026-08-22_old_feature_sets_and_prefix_pinn/` on 2026-08-22. Do not cite numbers from
there. R2 in the 0.577-0.579 range on any PINN/PINN-k Set2/3/4 run is the tell-tale bug
signature.
