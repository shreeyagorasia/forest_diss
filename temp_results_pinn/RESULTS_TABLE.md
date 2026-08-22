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

**Status: PENDING** -- not yet submitted. Uses the ORIGINAL lr=0.0001/weight_decay=1e-5 (since
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

**Status: HELD** -- waiting on #4 (winning lambda) before submitting. This is the core
"environment helps" argument for the dissertation. The earlier (Aug 11) version of this sweep
used the pre-fix architecture and is invalid (all three sets landed at ~0.577-0.579,
indistinguishable from no-environment -- the known bug signature).

- Script: `temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_set_sweep.py`
- Job: `temp_results_pinn/jobs/run_pinn_fix_set_sweep_cluster.sh`
- Output (once run): `temp_results_pinn/outputs/CORRECTED_2026-08-22_pinn_set_sweep/<feature_set>/fold_<i>/pinn_<variant>_fixed_summary.json`
- Sets to run: `nested_set2_top10`, `nested_set4_gated_all_vif` (Set3 already covered by #1;
  Set1 is covered by `pinn_noenv`, which has no per-plot sub-network so is structurally immune
  to the forward-pass bug -- no rerun needed there).

---

## Legacy / do-not-cite

Pre-fix PINN/PINN-k runs (broken forward pass -- terrain features never reached the
prediction, only the physics-loss target) and old feature-set-naming runs were moved to
`legacy/2026-08-22_old_feature_sets_and_prefix_pinn/` on 2026-08-22. Do not cite numbers from
there. R2 in the 0.577-0.579 range on any PINN/PINN-k Set2/3/4 run is the tell-tale bug
signature.
