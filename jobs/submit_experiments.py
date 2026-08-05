# Run this from the project root, ON THE CLUSTER:
#
#   python jobs/submit_experiments.py --list
#   python jobs/submit_experiments.py E1 fit
#   python jobs/submit_experiments.py E1 evaluate
#   python jobs/submit_experiments.py E1 fit --dry-run
#
# WHAT THIS SCRIPT IS FOR:
# documentation/experiments_to_run.txt used to be the only place the sbatch commands for each
# experiment stage lived -- you had to copy each line out by hand and paste it into the
# terminal one at a time. This script does the same job, but as real Python: each STAGE below
# is a list of jobs, and running one command submits every job in that stage for you.
#
# It still follows the same "sections, done in steps" idea as experiments_to_run.txt:
#   - "fit" submits the SBATCH training jobs for a stage (these need the cluster's GPU).
#   - "evaluate" runs the evaluate scripts for that stage LOCALLY (these are cheap CPU-only
#     scripts -- see each job's own .sh file for why no sbatch is needed for these).
# Run "fit" first, wait for those cluster jobs to actually finish, THEN run "evaluate" --
# evaluate reads the checkpoint files "fit" produces, so it will fail if you run it too early.
#
# Adding a new stage later: copy one of the STAGES entries below and edit the cohort/split_type/
# hyperparameter values. Nothing else in this file needs to change.

import argparse
import shutil
import subprocess

# ---------------------------------------------------------------------------
# STAGE DEFINITIONS
#
# Each stage is a dictionary with three parts:
#   "description"        -- a one-line reminder of what this stage is testing.
#   "fit_jobs"            -- a list of sbatch jobs to submit (these run ON the cluster).
#   "evaluate_commands"  -- a list of plain "python -m ..." commands to run AFTER the fit jobs
#                            finish (these run locally, no sbatch needed).
#
# Each entry in "fit_jobs" is itself a list of strings: the job script path, followed by the
# positional arguments that script's own .sh file expects (cohort, max_epochs, patience,
# split_type, ...). See jobs/dnn_env_terrain/run_dnn_env_terrain.sh and
# jobs/pinn_env_terrain/run_pinn_env_terrain.sh for what each position means.
# ---------------------------------------------------------------------------

STAGES = {

    "E1": {
        "description": (
            "dnn_env_terrain / pinn_env_terrain base-case fit+evaluate, spatial_block (primary "
            "split), both cohorts, single seed (42), default terrain_wind_solid feature set, "
            "no dropout. See documentation/experiments_to_run.txt's 2026-08-01 entry."
        ),
        # batch_size=256 is passed explicitly on every job here (not left at each script's own
        # default) -- dnn_env_terrain.sh defaults to 512, pinn_env_terrain.sh defaults to 128,
        # and leaving those defaults in place would silently recreate the exact uncontrolled
        # DNN-vs-PINN batch-size mismatch that Stage 1 (see experiments_to_run.txt) deliberately
        # fixed for the no-env models. "" is a placeholder for run_name (left blank, uses the
        # default output path) -- it has to be there so batch_size lands in the right position.
        "fit_jobs": [
            ["jobs/dnn_env_terrain/run_dnn_env_terrain.sh", "4survey", "500", "40", "spatial_block", "42", "", "256"],
            ["jobs/dnn_env_terrain/run_dnn_env_terrain.sh", "6survey", "500", "40", "spatial_block", "42", "", "256"],
            ["jobs/pinn_env_terrain/run_pinn_env_terrain.sh", "4survey", "500", "40", "spatial_block", "1.0", "1.0", "", "42", "256"],
            ["jobs/pinn_env_terrain/run_pinn_env_terrain.sh", "6survey", "500", "40", "spatial_block", "1.0", "1.0", "", "42", "256"],
        ],
        "evaluate_commands": [
            ["python", "-m", "models.dnn_env_terrain.evaluate_dnn_env_terrain", "--cohort", "4survey", "--split-type", "spatial_block"],
            ["python", "-m", "models.dnn_env_terrain.evaluate_dnn_env_terrain", "--cohort", "6survey", "--split-type", "spatial_block"],
            ["python", "-m", "models.pinn_env_terrain.evaluate_pinn_env_terrain", "--cohort", "4survey", "--split-type", "spatial_block"],
            ["python", "-m", "models.pinn_env_terrain.evaluate_pinn_env_terrain", "--cohort", "6survey", "--split-type", "spatial_block"],
        ],
    },

    "E2": {
        "description": (
            "Same 4 base-case configs as E1, but on temporal and temporal_narrow_gap -- gives "
            "env_terrain the same 3-split coverage dnn_noenv/pinn_noenv already have. Only run "
            "this once E1 has landed and looks sane."
        ),
        # Same explicit batch_size=256 reasoning as E1 above.
        "fit_jobs": [
            ["jobs/dnn_env_terrain/run_dnn_env_terrain.sh", "4survey", "500", "40", "temporal", "42", "", "256"],
            ["jobs/dnn_env_terrain/run_dnn_env_terrain.sh", "6survey", "500", "40", "temporal", "42", "", "256"],
            ["jobs/dnn_env_terrain/run_dnn_env_terrain.sh", "4survey", "500", "40", "temporal_narrow_gap", "42", "", "256"],
            ["jobs/dnn_env_terrain/run_dnn_env_terrain.sh", "6survey", "500", "40", "temporal_narrow_gap", "42", "", "256"],
            ["jobs/pinn_env_terrain/run_pinn_env_terrain.sh", "4survey", "500", "40", "temporal", "1.0", "1.0", "", "42", "256"],
            ["jobs/pinn_env_terrain/run_pinn_env_terrain.sh", "6survey", "500", "40", "temporal", "1.0", "1.0", "", "42", "256"],
            ["jobs/pinn_env_terrain/run_pinn_env_terrain.sh", "4survey", "500", "40", "temporal_narrow_gap", "1.0", "1.0", "", "42", "256"],
            ["jobs/pinn_env_terrain/run_pinn_env_terrain.sh", "6survey", "500", "40", "temporal_narrow_gap", "1.0", "1.0", "", "42", "256"],
        ],
        "evaluate_commands": [
            ["python", "-m", "models.dnn_env_terrain.evaluate_dnn_env_terrain", "--cohort", "4survey", "--split-type", "temporal"],
            ["python", "-m", "models.dnn_env_terrain.evaluate_dnn_env_terrain", "--cohort", "6survey", "--split-type", "temporal"],
            ["python", "-m", "models.dnn_env_terrain.evaluate_dnn_env_terrain", "--cohort", "4survey", "--split-type", "temporal_narrow_gap"],
            ["python", "-m", "models.dnn_env_terrain.evaluate_dnn_env_terrain", "--cohort", "6survey", "--split-type", "temporal_narrow_gap"],
            ["python", "-m", "models.pinn_env_terrain.evaluate_pinn_env_terrain", "--cohort", "4survey", "--split-type", "temporal"],
            ["python", "-m", "models.pinn_env_terrain.evaluate_pinn_env_terrain", "--cohort", "6survey", "--split-type", "temporal"],
            ["python", "-m", "models.pinn_env_terrain.evaluate_pinn_env_terrain", "--cohort", "4survey", "--split-type", "temporal_narrow_gap"],
            ["python", "-m", "models.pinn_env_terrain.evaluate_pinn_env_terrain", "--cohort", "6survey", "--split-type", "temporal_narrow_gap"],
        ],
    },

}


# ---------------------------------------------------------------------------
# E3: the spatial_block_kfold sweep (2026-08-05) -- built with loops instead of hardcoded lists
# like E1/E2 above, purely because of scale: 5 folds x 2 cohorts x 4 models is 40 fit jobs and
# 40 evaluate commands, not the 4-8 of E1/E2. The JOB ORDER MATTERS here in a way E1/E2 didn't:
#   1. Run E3_baselines_kfold FIRST and wait for all 5 jobs to finish. These fit the
#      fold-MATCHED Chapman-Richards anchor (and the other three baselines, fit as a side
#      effect) for every fold -- pinn_env_terrain/pinn_env_terrain_k's load_cr_params() reads
#      this file and will fail with a clear "No such file" error if it doesn't exist yet.
#   2. THEN run "E3_kfold fit", wait for those 40 cluster jobs to finish.
#   3. THEN run "E3_kfold evaluate" (runs locally, cheap CPU-only, one call per model/cohort/fold).
#   4. Once every fold's predictions.csv exists, pool them with
#      models/common/kfold_summary.py (not part of this script -- see its own module docstring).
# dnn_noenv/dnn_env_terrain don't actually need step 1 (no physics anchor), but there's no harm
# in waiting for it anyway -- simpler to always do steps in the same order than to special-case.
def build_kfold_stage_jobs():
    n_folds = 5
    cohorts = ["4survey", "6survey"]

    baseline_fit_jobs = []
    for fold_index in range(n_folds):
        # run_baselines.py always fits BOTH cohorts in one call (see its own main()) -- one job
        # per fold, not one per (cohort, fold) pair.
        baseline_fit_jobs.append(
            ["jobs/baselines/run_baselines.sh", "spatial_block_kfold", "42", str(n_folds), str(fold_index)]
        )

    model_fit_jobs = []
    model_evaluate_commands = []
    for cohort in cohorts:
        for fold_index in range(n_folds):
            # batch_size=256 for both dnn_env_terrain and the two PINN models, matching E1's own
            # reasoning above (avoids an uncontrolled DNN-vs-PINN batch-size mismatch) -- left at
            # dnn_noenv.sh's own default (512) since there's no PINN-equivalent batch size it
            # needs to match here (dnn_noenv has no physics loss).
            model_fit_jobs.append([
                "jobs/dnn_noenv/run_dnn_noenv.sh", cohort, "500", "40", "spatial_block_kfold",
                "42", "", "512", "42", str(n_folds), str(fold_index),
            ])
            model_evaluate_commands.append([
                "jobs/dnn_noenv/evaluate_dnn_noenv.sh", cohort, "spatial_block_kfold", "", "42",
                str(n_folds), str(fold_index),
            ])

            model_fit_jobs.append([
                "jobs/dnn_env_terrain/run_dnn_env_terrain.sh", cohort, "500", "40", "spatial_block_kfold",
                "42", "", "256", "terrain_wind_solid", "0.0", "42", str(n_folds), str(fold_index),
            ])
            model_evaluate_commands.append([
                "jobs/dnn_env_terrain/evaluate_dnn_env_terrain.sh", cohort, "spatial_block_kfold", "", "42",
                str(n_folds), str(fold_index),
            ])

            model_fit_jobs.append([
                "jobs/pinn_env_terrain/run_pinn_env_terrain.sh", cohort, "500", "40", "spatial_block_kfold",
                "1.0", "1.0", "", "42", "256", "terrain_wind_solid", "0.0", "42", str(n_folds), str(fold_index),
            ])
            model_evaluate_commands.append([
                "jobs/pinn_env_terrain/evaluate_pinn_env_terrain.sh", cohort, "spatial_block_kfold", "", "42",
                str(n_folds), str(fold_index),
            ])

            model_fit_jobs.append([
                "jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh", cohort, "500", "40", "spatial_block_kfold",
                "1.0", "1.0", "", "42", "256", "terrain_wind_solid", "0.0", "42", str(n_folds), str(fold_index),
            ])
            model_evaluate_commands.append([
                "jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh", cohort, "spatial_block_kfold", "", "42",
                str(n_folds), str(fold_index),
            ])

    return baseline_fit_jobs, model_fit_jobs, model_evaluate_commands


_BASELINE_KFOLD_FIT_JOBS, _MODEL_KFOLD_FIT_JOBS, _MODEL_KFOLD_EVALUATE_COMMANDS = build_kfold_stage_jobs()

STAGES["E3_baselines_kfold"] = {
    "description": (
        "Fold-matched Chapman-Richards anchors (+ average-by-age/linear/RF as a side effect) "
        "for all 5 folds, both cohorts -- MUST be run and finished before E3_kfold, since "
        "pinn_env_terrain/pinn_env_terrain_k read these files. No separate evaluate step -- "
        "the baselines' own accuracy under k-fold isn't the point here, just their output files."
    ),
    "fit_jobs": _BASELINE_KFOLD_FIT_JOBS,
    "evaluate_commands": [],
}

STAGES["E3_kfold"] = {
    "description": (
        "5-fold spatial_block_kfold sweep for dnn_noenv, dnn_env_terrain, pinn_env_terrain, "
        "pinn_env_terrain_k, both cohorts -- the precision fix for the single-slice spatial_block "
        "numbers everywhere else in this project (see documentation/experiment_log.md's 2026-08-04 "
        "entries). Run E3_baselines_kfold FIRST and wait for it to finish. Once this stage's "
        "'evaluate' step is done, pool each model's 5 folds with models/common/kfold_summary.py."
    ),
    "fit_jobs": _MODEL_KFOLD_FIT_JOBS,
    "evaluate_commands": _MODEL_KFOLD_EVALUATE_COMMANDS,
}

STAGES["E4_pinn_env_terrain_k_6survey_base"] = {
    "description": (
        "The one missing single-seed base-case cell: pinn_env_terrain_k has never been run for "
        "6survey at all (only 4survey, seed 42, spatial_block -- the council's original result). "
        "Plain spatial_block, not k-fold -- reads the existing (non-fold) CR anchor, no "
        "dependency on E3_baselines_kfold."
    ),
    "fit_jobs": [
        ["jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh", "6survey", "500", "40", "spatial_block", "1.0", "1.0", "", "42", "256"],
    ],
    "evaluate_commands": [
        ["jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh", "6survey", "spatial_block"],
    ],
}


STAGES["E5_pinn_env_terrain_k_temporal"] = {
    "description": (
        "pinn_env_terrain_k has never been run under temporal or temporal_narrow_gap at all -- "
        "it didn't exist yet when E2 covered this for dnn_env_terrain/pinn_env_terrain. Both "
        "cohorts, both temporal split types. Frozen CR anchors already exist for both "
        "(outputs/{temporal,temporal_narrow_gap}/chapman_richards/<cohort>/params.json), so no "
        "baseline dependency here, unlike E3_kfold."
    ),
    "fit_jobs": [
        ["jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh", "4survey", "500", "40", "temporal", "1.0", "1.0", "", "42", "256"],
        ["jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh", "6survey", "500", "40", "temporal", "1.0", "1.0", "", "42", "256"],
        ["jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh", "4survey", "500", "40", "temporal_narrow_gap", "1.0", "1.0", "", "42", "256"],
        ["jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh", "6survey", "500", "40", "temporal_narrow_gap", "1.0", "1.0", "", "42", "256"],
    ],
    "evaluate_commands": [
        ["jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh", "4survey", "temporal"],
        ["jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh", "6survey", "temporal"],
        ["jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh", "4survey", "temporal_narrow_gap"],
        ["jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh", "6survey", "temporal_narrow_gap"],
    ],
}


def list_stages():
    # Just prints out what's available, without running anything -- so you can check what a
    # stage will do before actually submitting it.
    print("Available stages:\n")
    for stage_name, stage in STAGES.items():
        print(f"{stage_name}: {stage['description']}")
        print(f"  {len(stage['fit_jobs'])} fit job(s), {len(stage['evaluate_commands'])} evaluate command(s)")
        print()


def run_fit_jobs(stage_name, dry_run):
    stage = STAGES[stage_name]
    print(f"===== Submitting fit jobs for stage {stage_name} =====")
    print(stage["description"])
    print()

    # sbatch only exists on the cluster. If it's not installed (e.g. testing this script on a
    # laptop), print the commands instead of crashing, so you can still see exactly what would
    # have been submitted.
    sbatch_is_available = shutil.which("sbatch") is not None

    for job in stage["fit_jobs"]:
        command = ["sbatch"] + job
        print("  " + " ".join(command))
        if dry_run:
            continue
        if not sbatch_is_available:
            print("    (sbatch not found on this machine -- not actually submitted)")
            continue
        subprocess.run(command, check=True)

    print()
    print(f"Done. Once these finish on the cluster, run: python jobs/submit_experiments.py {stage_name} evaluate")


def run_evaluate_commands(stage_name, dry_run):
    # Two shapes of "evaluate command" live in this file, and each is handled differently:
    #   - a plain ["python", "-m", ...] list (E1/E2's original shape) -- run directly, right
    #     here, in whatever shell called this script. Fine ONLY if that shell is your own
    #     laptop or an interactive compute-node allocation, NEVER the cluster's shared login
    #     node -- even a "cheap CPU-only" script is still real compute, and the login node is
    #     shared by every user on the cluster (2026-08-05 fix: this was silently assumed
    #     "locally" always meant a laptop, which isn't true for everyone).
    #   - a [".../evaluate_*.sh", ...] list (E3/E4/E5's shape) -- submitted via sbatch, exactly
    #     like a fit job, so it actually runs on a real (CPU-only, no --gres=gpu) compute node
    #     instead of wherever this script itself happens to be running.
    stage = STAGES[stage_name]
    print(f"===== Running evaluate commands for stage {stage_name} =====")
    print(stage["description"])
    print()

    sbatch_is_available = shutil.which("sbatch") is not None

    for command in stage["evaluate_commands"]:
        is_sbatch_job = command[0].endswith(".sh")
        if is_sbatch_job:
            full_command = ["sbatch"] + command
            print("  " + " ".join(full_command))
            if dry_run:
                continue
            if not sbatch_is_available:
                print("    (sbatch not found on this machine -- not actually submitted)")
                continue
            subprocess.run(full_command, check=True)
        else:
            print("  " + " ".join(command))
            if dry_run:
                continue
            subprocess.run(command, check=True)

    print()
    print(f"Done evaluating stage {stage_name}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=list(STAGES.keys()), help="Which stage to run, e.g. E1")
    parser.add_argument("action", nargs="?", choices=["fit", "evaluate"], help="fit = submit sbatch jobs, evaluate = run evaluate scripts locally")
    parser.add_argument("--list", action="store_true", help="List all available stages and exit, without running anything")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands that would run, without actually running them")
    args = parser.parse_args()

    if args.list:
        list_stages()
        return

    if args.stage is None or args.action is None:
        parser.print_help()
        return

    if args.action == "fit":
        run_fit_jobs(args.stage, args.dry_run)
    elif args.action == "evaluate":
        run_evaluate_commands(args.stage, args.dry_run)


if __name__ == "__main__":
    main()
