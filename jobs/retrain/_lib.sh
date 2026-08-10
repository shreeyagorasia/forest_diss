#!/bin/bash
#
# Shared setup for every jobs/retrain/tierN_*.sh script. Each tier is now THREE files, always
# run in this order:
#   tierN_fit.sh       -- submits GPU fit jobs to the CLUSTER (sbatch). Baselines (tier 1) are
#                          the one exception: CPU-only and fast, so they just run locally,
#                          no cluster/sync step needed for them at all.
#   tierN_sync.sh       -- run LOCALLY, after squeue shows tierN_fit.sh's jobs COMPLETED. Pulls
#                          down ONLY the specific run folders that tier just fit -- never the
#                          whole outputs/ tree. This is the fix for the actual problem: syncing
#                          all of outputs/ wholesale is what let a local and a cluster run
#                          silently overwrite each other with no record of which one won.
#   tierN_evaluate.sh  -- run LOCALLY, after tierN_sync.sh finishes. Evaluation NEVER runs on
#                          the cluster for this project -- it's a cheap CPU forward pass over an
#                          already-fit checkpoint, always done on this Mac.
#
# ============================================================================================
# NAMING CONVENTION -- old (inconsistent) vs. new (this batch)
# ============================================================================================
#
#   OLD PATTERN                                  MEANS                                FIELD ORDER
#   final_dnn_seed42                             5-seed no-env DNN rerun              purpose, model, seed
#   final_pinn_w1_anchorfix_seed42                5-seed no-env PINN, corrected anchor purpose, model, weight, fix-tag, seed
#   e7winner_drop0.0_lr0.0001_pinn_env_terrain    E7's winning hyperparameter combo     purpose, hp, hp, model
#   stage1_terrain_dnn_env_terrain                E6 tier sweep                        featureset, model
#   arch_dnn_noenv_large                          architecture-size sweep              purpose, model, size
#   dnn_noenv_splitseed43                         split-seed robustness check          model, hp
#   env_terrain_batchfix_seed42                   most recent 5-seed reseed check       purpose, seed (model prefix separate)
#
#   Four different field orders, three different ways of saying "model", no shared prefix to
#   grep for "just this batch" without also matching unrelated folders.
#
#   NEW PATTERN (this batch): clean_<model>_<featuretag>_<split>[_fold<N>]_seed<N>
#   Always the same 5 fields, always in this order. <featuretag> uses the ledger's OWN Set
#   numbers (set2/set3a/set3b/set4 -- see the feature-sets table at the top of the results
#   ledger) instead of inventing another tag, and is omitted for the no-env models (their only
#   feature set IS "no-env", nothing to disambiguate).
#
#   OLD                                          NEW
#   final_dnn_seed42                             clean_dnn_noenv_spatial_block_seed42
#   final_pinn_w1_anchorfix_seed42                clean_pinn_noenv_spatial_block_seed42
#   stage1_terrain_dnn_env_terrain                clean_dnn_env_terrain_set3a_spatial_block_kfold_fold0_seed42
#
#   "clean_" is a NEW prefix, distinct from final_/e7winner_/stage*_/arch_/diag_/
#   env_terrain_batchfix_ -- grep outputs/ for "clean_" to find only this batch's runs, and
#   nothing here can collide with or overwrite a prior run's folder.
#
# ============================================================================================

set -e
PREFIX="clean"
SEEDS="42 43 44 45 46"
COHORTS="4survey 6survey"
VENV_PYTHON=".venv/bin/python"

# NOTE on sync/push: tierN_sync.sh and tier1_push.sh do NOT source this file or call any
# function below this point -- they're fully explicit, standalone rsync commands (literal
# filter rules written out, not computed at runtime) so you can read exactly what a sync will
# touch before running it. This file only covers FIT and local EVALUATE.

# Ledger Set-number tags, mapped to the real --feature-set string each model script expects.
# terrain_wind_solid (Set 2) is every model's current default feature tier.
featureset_string_for_tag () {
  case "$1" in
    set2) echo "terrain_wind_solid" ;;
    set3a) echo "stage1_terrain" ;;
    set3b) echo "stage2_terrain_wind" ;;
    set4) echo "stage4_all_environmental" ;;
    *) echo "UNKNOWN_TAG_$1" >&2; exit 1 ;;
  esac
}

_noenv_run_name () {
  # $1 = model, $2 = split, $3 = seed, $4 = fold (optional, kfold only)
  local model="$1" split="$2" seed="$3" fold="${4:-}"
  local split_tag="$split"
  [ -n "$fold" ] && split_tag="${split}_fold${fold}"
  echo "${PREFIX}_${model}_${split_tag}_seed${seed}"
}

_env_terrain_run_name () {
  # $1 = model, $2 = feature_tag (set2/set3a/set3b/set4), $3 = split, $4 = seed, $5 = fold (optional)
  local model="$1" tag="$2" split="$3" seed="$4" fold="${5:-}"
  local split_tag="$split"
  [ -n "$fold" ] && split_tag="${split}_fold${fold}"
  echo "${PREFIX}_${model}_${tag}_${split_tag}_seed${seed}"
}

# ---- FIT (cluster, sbatch) ----

submit_noenv_fit () {
  # $1 = model (dnn_noenv or pinn_noenv), $2 = cohort, $3 = seed, $4 = split_type,
  # $5 = fold_index (spatial_block_kfold only, blank otherwise).
  local model="$1" cohort="$2" seed="$3" split="$4" fold="${5:-}"
  local run_name; run_name=$(_noenv_run_name "$model" "$split" "$seed" "$fold")
  local fold_index="${fold:-0}"
  if [ "$model" == "dnn_noenv" ]; then
    sbatch jobs/dnn_noenv/run_dnn_noenv.sh "$cohort" 500 40 "$split" "$seed" "$run_name" 256 42 5 "$fold_index"
  else
    sbatch jobs/pinn_noenv/run_pinn_noenv.sh "$cohort" 500 40 "$split" 1.0 1.0 "$run_name" "$seed" 256 42 5 "$fold_index"
  fi
}

# ---- BASELINES (CPU-only, always local -- never the cluster, no sync step needed) ----
# CR/average-by-age/Linear/RF fit in seconds on this Mac (confirmed directly this session) --
# no reason to spend a GPU cluster job or a sync round-trip on them. This is also the PINN
# CR-anchor prerequisite: run before any PINN fit for a split/fold you haven't touched yet.

local_fit_baselines_for_split () {
  local split="$1"
  if [ "$split" == "spatial_block_kfold" ]; then
    for fold in 0 1 2 3 4; do
      "$VENV_PYTHON" -m models.baselines.run_baselines --split-type "$split" --n-folds 5 --fold-index "$fold"
    done
  else
    "$VENV_PYTHON" -m models.baselines.run_baselines --split-type "$split"
  fi
}

local_evaluate_baselines_for_split () {
  local split="$1"
  if [ "$split" == "spatial_block_kfold" ]; then
    for fold in 0 1 2 3 4; do
      "$VENV_PYTHON" -m models.baselines.evaluate_baselines --split-type "$split" --n-folds 5 --fold-index "$fold"
    done
  else
    "$VENV_PYTHON" -m models.baselines.evaluate_baselines --split-type "$split"
  fi
}

submit_env_terrain_fit () {
  # $1 = model, $2 = cohort, $3 = seed, $4 = split_type, $5 = feature_tag (set2/set3a/set3b/set4),
  # $6 = fold_index (kfold only, blank otherwise).
  local model="$1" cohort="$2" seed="$3" split="$4" tag="$5" fold="${6:-}"
  local feature; feature=$(featureset_string_for_tag "$tag")
  local run_name; run_name=$(_env_terrain_run_name "$model" "$tag" "$split" "$seed" "$fold")
  local fold_index="${fold:-0}"
  case "$model" in
    dnn_env_terrain)
      sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh "$cohort" 500 40 "$split" "$seed" "$run_name" 256 "$feature" 0.0 42 5 "$fold_index" 0.0001
      ;;
    pinn_env_terrain)
      sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh "$cohort" 500 40 "$split" 1.0 1.0 "$run_name" "$seed" 256 "$feature" 0.0 42 5 "$fold_index" 0.0001
      ;;
    pinn_env_terrain_k)
      sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh "$cohort" 500 40 "$split" 1.0 1.0 "$run_name" "$seed" 256 "$feature" 0.0 42 5 "$fold_index" "" 0.0001
      ;;
  esac
}

# ---- EVALUATE (local, never the cluster -- --cohort omitted runs both cohorts in one call) ----

local_evaluate_noenv () {
  local model="$1" split="$2" seed="$3" fold="${4:-}"
  local run_name; run_name=$(_noenv_run_name "$model" "$split" "$seed" "$fold")
  local fold_index="${fold:-0}"
  local module="models.dnn_noenv.evaluate_dnn_noenv"
  [ "$model" == "pinn_noenv" ] && module="models.pinn_noenv.evaluate_pinn_noenv"
  "$VENV_PYTHON" -m "$module" --split-type "$split" --run-name "$run_name" --split-seed 42 --n-folds 5 --fold-index "$fold_index"
}

local_evaluate_env_terrain () {
  local model="$1" split="$2" tag="$3" seed="$4" fold="${5:-}"
  local run_name; run_name=$(_env_terrain_run_name "$model" "$tag" "$split" "$seed" "$fold")
  local fold_index="${fold:-0}"
  local module="models.${model}.evaluate_${model}"
  "$VENV_PYTHON" -m "$module" --split-type "$split" --run-name "$run_name" --split-seed 42 --n-folds 5 --fold-index "$fold_index"
}
