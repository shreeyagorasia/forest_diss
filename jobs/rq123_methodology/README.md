# RQ1/RQ2/RQ3 essential experiment matrix — roadmap

**This file is operational (run order, job counts, machine labels) only.** For what each
research question actually asks and the current state of its results, see
`documentation/research_questions_overview.md` — a living reference, not fixed, updated as more
metrics/evaluations/sets get added.

DEADLINE: 2026-08-17. Everything here is scoped to the **essential** rows from this session's
methodology review — Set5 (every RQ) is cut entirely, not "if time permits" (see the plan file
for the full reasoning). All four smoke tests passed (one real bug found and fixed along the
way — RQ2's XGBoost checkpoint was silently degraded by joblib pickling across the cluster/local
boundary; fixed to use XGBoost's own `save_model`/`load_model`).

Every script here is labelled with the ONE machine it's meant to run on, same discipline as the
rest of this project: `sbatch` only exists on the cluster, evaluation is always local. Naming:
every new run this batch creates starts with `rq1_`/`rq2_`/`rq3_`, using the `nested_set*` names
from `documentation/env_feature_sets_manifest.csv` — never the bare "Set 2"/"Set 3" the old
ledger already uses for something else (see `documentation/experiment_log.md`'s 2026-08-10 entry).

**Updated 2026-08-10 (`documentation/methodlogy_env_setpick.md`)**: the manifest was regenerated
with VIF-screening added (threshold 5.0) and reference-level dropping for RQ3's categoricals —
set names changed (`nested_set2_top5` → `nested_set2_top10`; RQ3's `nested_set3/4/5` gained a
`_vif` suffix, since VIF is now applied to RQ3 too, reversing the earlier "deliberately not
VIF-screened" decision). All scripts/paths below already reflect the new names — no old-named run
exists yet, so nothing needs re-running for this rename alone.

## RQ2 is two complementary pieces, not one — read this before Step 1

**RQ2a (primary): does giving DNN/PINN environmental information shrink the residual from the
shared CR curve?** This is your own framing, and it needs **no new fitting at all** — it reuses
RQ1's own predictions.csv (identical schema for CR baseline and every DNN/PINN model:
`identification, LiDAR_year, observed_top_height, predicted_top_height, residual, split`), joins
model vs. CR residual on the same rows, and checks whether the model's error shrinks — especially
on the plots CR does *worst* on, not just on average. Script:
`models/spatial_attribution/rq2_residual_reduction.py`. Already tested on real, already-existing
local data: DNN+environment cut CR's mean |residual| from 3.04m to 1.93m overall, but the effect
concentrates heavily in the quartile where CR was worst (+3.92m reduction, 92% of rows improved)
versus barely helping — even hurting slightly — where CR was already accurate (Q1: −0.90m). A
real, interpretable finding, not just a sanity check.

**RQ2b (secondary, interpretability layer): NLME + Elastic Net + XGBoost fit directly on
`mean_cr_residual`.** This is Step 1 below. Different question from RQ2a: not "can a model reduce
the departure," but "*which specific* environmental variables are associated with it, and in
which direction" (NLME/EN give a directly interpretable coefficient per variable; XGBoost's SHAP
values do the nonlinear equivalent). This is what bridges cleanly into RQ3, which uses the exact
same EN/XGBoost/attribution toolkit on a different target.

RQ2a can run as soon as RQ1's fits exist (see below — first pass right after Step 3, before the
winner is even picked; a second, final pass after Step 4's reseed). RQ2b (Step 1) has zero
dependency on RQ1 at all and can run immediately.

## The order, and why

```
Step 1 (RQ2b: NLME+EN+XGBoost)  ─┐
Step 2 (RQ3)                     ─┼─  run in parallel, no dependencies between them or on Step 3
Step 3 (RQ1 tier sweep)          ─┘
                         │
                         ├──► Step 3a (RQ2a, first pass) -- run on ALL 9 sweep combinations,
                         │    no need to wait for a winner; shows how the residual-reduction
                         │    effect varies by model/feature-set/cohort
                         ▼
              Step 3 finishes → pick the winning (model, feature-set) pair
                         │
                         ▼
              Step 4 (RQ1 winner reseed + physics-weight check)
                         │
                         ▼
              Step 4a (RQ2a, final pass -- the winner's reseeded, robustness-checked numbers)
```

**Steps 1-3 have zero dependencies on each other** — submit all three at once rather than
waiting on one to finish before starting the next. RQ2 and RQ3's EN/XGBoost pieces are CPU-only
and fast (minutes), so submitting them first means real results exist almost immediately, while
RQ1's GPU sweep (the slowest piece) and RQ3's GNNWR (also GPU) churn through the queue in the
background.

**Step 4 must wait for Step 3** — you can't reseed "the winner" before you know which
(model, feature-set) pair actually won.

## Step 1 — RQ2b (NLME + Elastic Net + XGBoost)

**Already fully built and smoke-tested.** 3 sets × 5 folds = 15 fit jobs (cluster, CPU, fast) +
15 local evaluate calls. 4survey only (see the cohort-justification entry in
`documentation/experiment_log.md` — 6survey's 47 compartments are too few for this).

```bash
# on the cluster:
bash jobs/rq123_methodology/step1_rq2_fit.sh
# on your Mac, once squeue shows those 15 jobs COMPLETED:
bash jobs/rq123_methodology/step1_rq2_sync.sh
bash jobs/rq123_methodology/step1_rq2_evaluate.sh
```

## Step 2 — RQ3 (Elastic Net + XGBoost, then GNNWR)

**Already fully built and smoke-tested.** EN/XGBoost: 3 sets × 2 cohorts = 6 jobs (cluster, CPU,
fit+evaluate combined in one call — see the plan file for why this one can't get the same
fit/evaluate split as RQ2b). GNNWR: 3 sets × 2 cohorts × 5 folds = 30 jobs (cluster, GPU,
fit+evaluate combined on the cluster by design — the model itself is too large to sync down,
only the small `*_test_predictions.csv` results come back).

```bash
# on the cluster:
bash jobs/rq123_methodology/step2_rq3_en_xgb.sh
bash jobs/rq123_methodology/step2_rq3_gnnwr.sh
# on your Mac, once both finish -- results only, GNNWR's models/ and runs/ never sync down:
bash jobs/rq123_methodology/step2_sync_results.sh
```

## Step 3 — RQ1 (env-conditioned DNN/PINN tier sweep)

**Already fully built and smoke-tested.** 3 models (`dnn_env_terrain`/`pinn_env_terrain`/
`pinn_env_terrain_k`) × 3 sets × 2 cohorts × 5 folds = 90 fit jobs (cluster, GPU) — single seed
(42), matching this project's own established E6→E9 pattern (cheap single-seed sweep first,
THEN reseed only the winner, not every combination at full seed count). Same scale as the old
E6 sweep (also 90 jobs), so this is a known, already-survived cluster load.

```bash
# on the cluster:
bash jobs/rq123_methodology/step3_rq1_sweep_fit.sh
# on your Mac, once squeue shows all 90 COMPLETED:
bash jobs/rq123_methodology/step3_rq1_sweep_sync.sh
bash jobs/rq123_methodology/step3_rq1_sweep_evaluate.sh
```

**Winner = highest pooled 5-fold R², averaged across both cohorts** (your call from earlier —
speed over a significance-test gate). Once `step3_rq1_sweep_evaluate.sh` finishes, its own
printed summary table is the thing to read to pick the winner by hand — deliberately not
auto-selected, so you can sanity-check it before committing cluster time to the reseed.

## Step 3a — RQ2a, first pass (run right after Step 3, no winner needed yet)

Local only, no fitting -- reads Step 3's freshly-synced predictions against the CR baseline's
own (already-existing) predictions for the matching split/fold. Run once per (model, set)
combination from the sweep:

```bash
for MODEL in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for SET_NAME in nested_set2_top10 nested_set3_gated_terrain_wind_vif nested_set4_gated_all_vif; do
    for COHORT in 4survey 6survey; do
      for FOLD in 0 1 2 3 4; do
        RUN_NAME="rq1_${MODEL}_${SET_NAME}_seed42"
        .venv/bin/python -m models.spatial_attribution.rq2_residual_reduction \
          --model-name "$RUN_NAME" --cr-run-name chapman_richards \
          --cohort "$COHORT" --split-type spatial_block_kfold --fold-index "$FOLD"
      done
    done
  done
done
```

(90 calls, matching Step 3's own 90 fits -- cheap, seconds each, since it's just reading two
already-computed CSVs and joining them.)

## Step 4 — RQ1 winner reseed + physics-weight check (fill in after Step 3)

Not pre-built — the exact jobs depend on which (model, set) wins Step 3. Two cases:

- **A PINN variant wins**: 5-seed reseed of that exact (model, set) pair (4 more seeds — seed 42
  already exists from Step 3), 2 cohorts × 5 folds = 40 more fit jobs. Plus a matched
  zero-physics-control run (`physics_weight=0`) at the SAME winning set, same seed count, to
  directly test whether the physics constraint helps once environmental conditioning is added —
  another ~40 jobs.
- **DNN wins**: DNN itself doesn't have a physics weight, so the reseed is just DNN's 4 more
  seeds (40 jobs) — but the RQ1 chapter's physics question still needs an answer *at the winning
  feature set*, so also 5-seed reseed the corresponding `pinn_env_terrain`(w=1) run at that same
  set (40 jobs) plus its zero-physics-control counterpart (40 jobs) for the comparison.

`step3_rq1_sweep_fit.sh`'s own per-job `sbatch` line is the template to copy for whichever case
applies — same command shape, just swap in the winning `--feature-set` and loop seeds 43-46
instead of a fixed 42, plus `--physics-weight 0.0 --trajectory-weight 0.0` for the control.

## Step 4a — RQ2a, final pass

Rerun `rq2_residual_reduction.py` (same command shape as Step 3a) against Step 4's final,
5-seed-reseeded winner — the robustness-checked numbers that actually go in the chapter, not
just the single-seed sweep's preview.

## Job count summary

| Step | Jobs | Machine | Approx. cost |
|---|---|---|---|
| 1 (RQ2b) | 15 fit + 15 evaluate | cluster (CPU) + local | minutes total |
| 2 (RQ3 EN/XGB) | 6 | cluster (CPU) | minutes total |
| 2 (RQ3 GNNWR) | 30 | cluster (GPU) | the slower of the two RQ3 pieces -- real training, 200 epochs/run |
| 3 (RQ1 sweep) | 90 fit + 90 evaluate | cluster (GPU) + local | same scale as the old E6 sweep -- already-survived cluster load |
| 3a (RQ2a, first pass) | 90 local analysis calls | local | seconds each |
| 4 (RQ1 reseed + physics) | ~80-120, contingent | cluster (GPU) + local | fill in after Step 3 |
| 4a (RQ2a, final pass) | a handful | local | seconds each |

No hard ETA promised here either, same reasoning as before (this project's own logs show
identical code/seed swinging 0.10 R2 across different physical nodes — per-job timing on this
cluster is genuinely variable) -- these are planning numbers, not a schedule.
