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
    stage = STAGES[stage_name]
    print(f"===== Running evaluate commands for stage {stage_name} =====")
    print(stage["description"])
    print()

    for command in stage["evaluate_commands"]:
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
