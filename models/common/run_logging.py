# One JSON file per run EVENT (not one file per run) -- every run writes a
# "started" entry the moment it begins, then a "success" or "failed" entry
# when it finishes, both sharing the same attempt_id so they can be paired
# up later. If a job is killed from OUTSIDE Python (SLURM OOM-kill, walltime
# exceeded, node failure), no Python exception ever fires, so the "success"/
# "failed" entry never gets written -- but the "started" entry survives, and
# its absence of a matching completion entry is itself the signal that
# something died externally. Check the run's SLURM job_id (recorded below)
# against the matching SLURM stderr file under logs/<job family>/ for what
# actually happened in that case.
#
# Deliberately a plain, append-only folder of small files (not a database
# or one growing log file), so a run that's no longer useful can just be
# deleted by hand -- the filename itself says whether a run started,
# succeeded, failed, or was a quick test, so you don't need to open every
# file to decide what to keep.
#
# This module only writes log entries. It never reads or filters them -- a
# separate script can do that later once there's enough history to be worth
# filtering (see documentation/experiment_log.md).

import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_LOGS_DIR = PROJECT_ROOT / "outputs" / "run_logs"

# A "real" training run uses max_epochs=500 (the default in
# run_dnn_noenv.py / run_pinn_noenv.py); a quick local sanity check uses
# something like --max-epochs 5. This threshold sits comfortably between
# the two, so a run is auto-tagged as a test run purely from how many
# epochs it was configured for -- no separate --test-run flag to remember
# to pass. Only meaningful for the two PyTorch models; the four sklearn/
# scipy baselines have no concept of epochs and are never test runs.
TEST_RUN_MAX_EPOCHS_THRESHOLD = 50


class RunTimer:
    # Small stopwatch: start() when the run begins, elapsed_seconds() to
    # read back how long it took for the log's runtime_seconds field.
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        return self

    def elapsed_seconds(self):
        if self.start_time is None:
            return None
        return round(time.time() - self.start_time, 2)


def make_attempt_id(model_name, cohort, split_type, run_phase):
    # Shared between a run's "started" entry and its later "success"/
    # "failed" entry, so the two can be matched up even though they are
    # written at different timestamps (and so get different filenames).
    # Microsecond precision avoids collisions if two attempts somehow start
    # in the same second.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{model_name}_{cohort}_{split_type}_{run_phase}_{timestamp}"


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def get_slurm_info():
    # None for every field when not running under SLURM (e.g. a local Mac
    # run) -- these are only set inside an actual SLURM job. job_id is what
    # lets a run_logs/*.json entry be matched back to its raw
    # logs/<job family>/<job_name>_<job_id>.out / .err files, which have
    # the full print output this summary doesn't include.
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
    }


def get_versions():
    # Worth recording alongside the git commit: if a run behaves
    # differently between the Mac and the cluster, a library version
    # mismatch is one of the first things to rule out.
    versions = {"python": platform.python_version()}

    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        pass

    try:
        import sklearn
        versions["sklearn"] = sklearn.__version__
    except ImportError:
        pass

    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except ImportError:
        pass

    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        pass

    return versions


def format_error(exception):
    # Kept as a small dict (not just str(exception)) so the log is useful
    # even without re-running -- the traceback usually shows exactly which
    # line failed.
    return {
        "type": type(exception).__name__,
        "message": str(exception),
        "traceback": traceback.format_exc(),
    }


def write_started_marker(model_name, cohort, split_type, run_phase, is_test_run, device, hyperparameters):
    # Call this FIRST, before any real work happens, and hang onto the
    # attempt_id it returns -- pass that same attempt_id into write_run_log()
    # at the end so the two entries can be paired up.
    attempt_id = make_attempt_id(model_name, cohort, split_type, run_phase)
    _write_log_entry(
        attempt_id=attempt_id,
        model_name=model_name, cohort=cohort, split_type=split_type, run_phase=run_phase,
        status="started", is_test_run=is_test_run, device=device,
        hyperparameters=hyperparameters, metrics=None, error=None,
        output_dir=None, runtime_seconds=None, n_rows_fit=None,
    )
    return attempt_id


def write_run_log(
    attempt_id, model_name, cohort, split_type, run_phase,
    status, is_test_run, device,
    hyperparameters, metrics, error,
    output_dir, runtime_seconds, n_rows_fit=None,
):
    # Call this once a run finishes, whether it succeeded or failed.
    # run_phase is "fit", "evaluate", or "fit_and_evaluate" -- the four
    # sklearn baselines fit and evaluate in two separate script runs
    # (run_baselines.py then evaluate_baselines.py), so those get one log
    # entry per phase; the DNN/PINN scripts do both in one call, so they
    # log a single "fit_and_evaluate" entry instead.
    return _write_log_entry(
        attempt_id=attempt_id,
        model_name=model_name, cohort=cohort, split_type=split_type, run_phase=run_phase,
        status=status, is_test_run=is_test_run, device=device,
        hyperparameters=hyperparameters, metrics=metrics, error=error,
        output_dir=output_dir, runtime_seconds=runtime_seconds, n_rows_fit=n_rows_fit,
    )


def _write_log_entry(
    attempt_id, model_name, cohort, split_type, run_phase,
    status, is_test_run, device,
    hyperparameters, metrics, error,
    output_dir, runtime_seconds, n_rows_fit,
):
    RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    timestamp_for_filename = timestamp.strftime("%Y%m%dT%H%M%SZ")

    if status == "started":
        filename_tag = "STARTED"
    elif status == "failed":
        filename_tag = "FAILED"
    elif is_test_run:
        filename_tag = "TEST"
    else:
        filename_tag = "success"

    run_id = f"{timestamp_for_filename}_{model_name}_{cohort}_{split_type}_{run_phase}_{filename_tag}"

    log_entry = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "timestamp_utc": timestamp.isoformat(),
        "model_name": model_name,
        "cohort": cohort,
        "split_type": split_type,
        "run_phase": run_phase,
        "status": status,
        "is_test_run": is_test_run,
        "device": device,
        "hostname": socket.gethostname(),
        "slurm": get_slurm_info(),
        "runtime_seconds": runtime_seconds,
        "n_rows_fit": n_rows_fit,
        "git_commit": get_git_commit(),
        "versions": get_versions(),
        "command": " ".join(sys.argv),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "error": error,
        "output_dir": str(output_dir) if output_dir is not None else None,
    }

    log_path = RUN_LOGS_DIR / f"{run_id}.json"
    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    return log_path
