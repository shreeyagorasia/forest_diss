#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh [cohort] [scope] [max_epoch] [early_stop] [reference_set_size] [split_seed] [held_out_fold] [k_folds]
#
# Examples:
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind 300 30
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind_plus_management 300 30
#   # 5-fold spatial CV at the cheap 6,000-row reference-set size (submit once per fold):
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind 200 20 6000 42 0 5
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind 200 20 6000 42 1 5
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind 200 20 6000 42 2 5
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind 200 20 6000 42 3 5
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind 200 20 6000 42 4 5
#   # then pool the 5 resulting CSVs:
#   python -m models.growth_curve_attribution.pool_gnnwr_kfold_results --cohort 4survey --scope terrain_wind --reference-set-size 6000
#
# Arguments:
#   cohort               4survey or 6survey. Defaults to 4survey.
#   scope                terrain_wind (17 features, the headline comparison against the
#                        established Elastic Net/XGBoost result of 0.125/0.117 pooled OOF R2) or
#                        terrain_wind_plus_management (22 features, the best-performing static
#                        scope found by broad_environmental_check.py, EN/XGB 0.289/0.302).
#                        Defaults to terrain_wind. See SCOPES in
#                        models/growth_curve_attribution/gnnwr_check.py.
#   max_epoch            Maximum training epochs. Defaults to 200.
#   early_stop           Stop if validation R2 hasn't improved for this many epochs. -1 disables
#                        early stopping. Defaults to 20.
#   reference_set_size   Caps GNNWR's reference/training set to this many plots
#                        (compartment-stratified) -- see gnnwr_check.py's module docstring.
#                        Defaults to 0 (the FULL ~31,117-plot population) -- the memory-cost model
#                        that correctly predicted the real observed OOM at full scale on a 10.57
#                        GiB GPU (est. ~12.9 GB vs the actual crash) also shows the full population
#                        only needs ~12.8 GB, which comfortably fits the --gres below (an 18 GiB
#                        H200 MIG slice), so there is no need to shrink the reference set at all
#                        on this GPU type -- GNNWR sees the SAME training population as EN/XGBoost/
#                        the DNN baselines, not a disclosed-but-real handicap.
#                        FALLBACK, if the MIG slice also turns out hard to schedule (untested --
#                        only a specific RTX A6000 request has actually been confirmed to sit PD
#                        for a full day on this cluster's queue; MIG slices are a reasonable bet
#                        to be more available since they let several modest jobs share one
#                        physical GPU, but that is not yet verified empirically): switch --gres
#                        back to the generic "gpu:1" pool and pass 16000 here instead (est. ~4.1 GB,
#                        safely fits 10.57 GiB), or 6000 for an even cheaper, already-tested value.
#   split_seed           Seed for the compartment-based spatial_block_split. Defaults to 42 (this
#                        project's standard SPLIT_SEED), matching every other
#                        growth-curve-attribution check so results are directly comparable.
#   held_out_fold        Leave BLANK for the original single train/val/test split (unchanged
#                        default behaviour). Pass 0..k_folds-1 to instead run ONE fold of a
#                        proper K-fold spatial CV -- the same kind of split Elastic Net/XGBoost's
#                        own headline numbers are pooled across. Submit once per fold value (see
#                        examples above), then pool the resulting CSVs with
#                        pool_gnnwr_kfold_results.py for a genuinely comparable headline R2.
#   k_folds              Number of folds, only used when held_out_fold is set. Defaults to 5
#                        (DEFAULT_K_FOLDS), matching Elastic Net/XGBoost's own spatial CV.
#
# Why an 18 GiB H200 MIG slice instead of the generic pool or a specific bigger card (see
# gnnwr_check.py's module docstring for the full investigation, including a second, separate
# memory bottleneck in gnnwr's own diagnostic code that is patched inside gnnwr_check.py itself):
# GNNWR's spatial-weighting sub-network takes each plot's full distance-to-every-reference-plot
# vector as input, so its first layer's width equals the reference-set size. At the full
# ~31,000-plot population that is roughly 500 million parameters (~12.9 GB total with the
# accompanying distance tensors) -- confirmed to OOM the cluster's generic "gpu:1" allocation
# (10.57 GiB VRAM) in an earlier run. Rather than shrink the reference set (the earlier fix, still
# available as a fallback -- see reference_set_size above) or request a specific high-VRAM card
# like an RTX A6000 (48 GiB, confirmed to sit PD/"ReqNodeNotAvail" for a full day on this
# cluster's queue), this instead requests a MIG slice of the cluster's H200 -- just 18 GiB is
# already comfortably enough to fit the FULL training population with real margin, and MIG slices
# are a more plausible bet for fast scheduling than a whole scarce high-end card, since several
# modest jobs can share one physical GPU. If this GRES also proves hard to schedule in practice,
# fall back to --gres=gpu:1 with reference_set_size=16000 (or 6000).
#
# --mem=32G (2026-08-05, found via a real oom_kill on the first full-population attempt at
# --mem=16G): a SEPARATE, HOST-RAM-side bottleneck, nothing to do with the GPU numbers above.
# gnnwr's distance computation (BasicDistance(), called from _init_gnnwr_distance()) runs on the
# CPU via scipy.spatial.distance.cdist() BEFORE anything reaches the GPU -- and despite the
# function explicitly casting its inputs to float32, cdist() always returns float64 output
# regardless of input dtype (confirmed directly: cdist() on float32 arrays still returns a
# float64 array). At the full ~31,117-plot population, the three float64 distance matrices
# (train<->train, val<->train, test<->train) total ~13.5 GB in host RAM alone, before pandas/
# Python overhead and the subsequent scaling step's own temporary arrays -- comfortably over the
# previous 16G request, hence the OOM-kill. Smaller reference_set_size values (6000/16000) never
# hit this, since their distance matrices are proportionately smaller even in float64.
#
# Logs:
#   stdout -> logs/growth_curve_attribution/gnnwr_<jobid>.out
#   stderr -> logs/growth_curve_attribution/gnnwr_<jobid>.err
#
# Results:
#   outputs/growth_curve_attribution/gnnwr/<run_name>_test_predictions.csv
#   outputs/growth_curve_attribution/gnnwr/models/<run_name>.pkl (best-on-validation checkpoint)
#
# Prerequisite:
#   Reads the terrain/wind/management columns from
#   data/processed/environmental/plot_environmental_features.parquet and the growth-curve target
#   tables gnnwr_check.py builds from -- make sure both are on the cluster before submitting.

#SBATCH --job-name=gnnwr
#SBATCH --output=logs/growth_curve_attribution/%x_%j.out
#SBATCH --error=logs/growth_curve_attribution/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --mem=32G

cd ~/forest_diss

mkdir -p logs/growth_curve_attribution outputs/growth_curve_attribution/gnnwr

# Toolchain lives in a shared TA home directory under a dated folder name that gets replaced
# periodically -- hardcoding one date breaks silently (a confusing torch/CUDA import error deep
# inside the Python job, not an obvious "toolchain missing" message) the next time it rotates.
# Finds whatever toolchain-* currently exists instead, picks the most recently modified one, and
# fails loudly with a clear message immediately if none exist at all.
TOOLCHAIN_RC=$(ls -1t /home/htang2/toolchain-*/toolchain.rc 2>/dev/null | head -1)
if [ -z "${TOOLCHAIN_RC}" ]; then
  echo "ERROR: no toolchain.rc found under /home/htang2/toolchain-*/ -- ask a TA if this has moved." >&2
  exit 1
fi
. "${TOOLCHAIN_RC}"
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
SCOPE=${2:-terrain_wind}
MAX_EPOCH=${3:-200}
EARLY_STOP=${4:-20}
REFERENCE_SET_SIZE=${5:-0}
SPLIT_SEED=${6:-42}
HELD_OUT_FOLD=${7:-}
K_FOLDS=${8:-5}

echo "--- GNNWR job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Scope: ${SCOPE}"
echo "Max epoch: ${MAX_EPOCH}"
echo "Early stop patience: ${EARLY_STOP}"
echo "Reference set size: ${REFERENCE_SET_SIZE}"
echo "Split seed: ${SPLIT_SEED}"
echo "Held-out fold: ${HELD_OUT_FOLD:-(none -- single split)}"

# Only pass --held-out-fold when actually set -- omitting it entirely (not passing 0/blank)
# keeps gnnwr_check.py's original single-split behaviour, which is its own default too.
FOLD_ARGS=()
if [ -n "${HELD_OUT_FOLD}" ]; then
  FOLD_ARGS=(--held-out-fold "${HELD_OUT_FOLD}" --k-folds "${K_FOLDS}")
fi

python -u -m models.growth_curve_attribution.gnnwr_check \
  --cohort "${COHORT}" \
  --scope "${SCOPE}" \
  --max-epoch "${MAX_EPOCH}" \
  --early-stop "${EARLY_STOP}" \
  --reference-set-size "${REFERENCE_SET_SIZE}" \
  --split-seed "${SPLIT_SEED}" \
  --use-gpu \
  "${FOLD_ARGS[@]}"

echo "--- GNNWR job end ---"
