# Run as: python -m documentation.scripts.audit_ledger_provenance
#
# READ-ONLY audit. This script does not train anything, does not re-evaluate anything, and does
# not write any file. It only reads outputs/run_logs/*.json (one JSON file per fit/evaluate event,
# written by models/common/run_logging.py) and checks file timestamps under outputs/.
#
# Why this script exists: work on this project happens both on the SLURM cluster (GPU) and
# locally on a Mac. Code syncs via git, but the outputs/ folder (checkpoints, metrics.json,
# predictions.csv) syncs via a wholesale rsync of the whole folder -- no merge logic. If a local
# run and a cluster run ever wrote to the SAME output folder independently, one silently
# overwrites the other on the next sync, and there is no record of which one "won". This script
# checks every real result shown on the results ledger against its own run_logs history to catch
# exactly that kind of silent mix-up, plus a few related staleness problems:
#
#   1. LOCAL/CLUSTER CONFLICT: both a local (Mac) and a cluster (SLURM) run exist for the same
#      output folder, with a different git commit or different hyperparameters. Whichever file is
#      on disk right now might not be the one you expect.
#   2. STALE VS KNOWN FIX: the run that produced this result happened BEFORE a bug fix that is
#      known to matter for this exact model family (see KNOWN_FIXES below).
#   3. MTIME MISMATCH: the metrics/predictions file's own last-modified time doesn't line up with
#      the newest matching run_logs entry -- a sign the file on disk might not be what the logs
#      say was last computed (a possible sync clobber).
#   4. MISSING: there's no run_logs entry at all for a result the ledger shows a real number for.
#
# It also records two things every ledger cell should show going forward:
#   - seed_n: how many different seeds this exact configuration has actually been run with
#     (1 means single-seed -- shown as "n/a (single seed)" in the report, not a bare "1").
#   - last_evaluated: the timestamp of the newest matching run_logs entry, so staleness is
#     visible at a glance from now on, not just discovered by an audit like this one.

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_LOGS_DIR = PROJECT_ROOT / "outputs" / "run_logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Split types that get their own outputs/<split_type>/ subfolder. plot_level is the one split
# type with NO prefix -- matches models/common/saving.py's model_output_dir() exactly, so this
# script resolves paths the same way the real training/evaluation code does.
PREFIXED_SPLIT_TYPES = {"spatial_block", "spatial_block_kfold", "temporal", "temporal_narrow_gap"}


# ---------------------------------------------------------------------------------------------
# STEP 1: load every run_logs "success" entry once. A "started" entry with no matching "success"
# entry means that run crashed or got killed -- not usable as provenance for a real number, so we
# only load the success ones here.
# ---------------------------------------------------------------------------------------------

def load_success_logs():
    logs = []
    for path in RUN_LOGS_DIR.glob("*_success.json"):
        try:
            with open(path) as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        entry["_log_path"] = str(path)
        logs.append(entry)
    return logs


def logs_matching(all_logs, model_name, cohort, split_type):
    # Every run_logs entry already records model_name/cohort/split_type as plain fields (see the
    # sample entry printed in this project's own documentation) -- an exact-match filter, not a
    # guess based on folder name.
    return [
        entry for entry in all_logs
        if entry.get("model_name") == model_name
        and entry.get("cohort") == cohort
        and entry.get("split_type") == split_type
    ]


def is_local(entry):
    # A cluster (SLURM) run always has a non-null job_id; a Mac run never does.
    slurm = entry.get("slurm") or {}
    return slurm.get("job_id") is None


# ---------------------------------------------------------------------------------------------
# STEP 2: known fix commits. A run trained BEFORE one of these dates is suspect, but only for the
# model families the fix actually touched -- narrowly scoped on purpose, so this script doesn't
# cry wolf on unrelated models that happen to have an old git_commit for an unrelated reason.
# ---------------------------------------------------------------------------------------------

KNOWN_FIXES = [
    {
        "commit": "3ffc5fc",
        "date_utc": "2026-08-04T18:04:51+00:00",
        "description": "XGBoost eval-set leak fix in growth-curve spatial CV/bootstrap checks",
        "model_names": {
            "xgb_environmental_all_environmental",
            "xgb_environmental_all_environmental_no_neighbour",
            "xgb_environmental_terrain_and_wind_only",
            "elasticnet_environmental_all_environmental",
            "elasticnet_environmental_all_environmental_no_neighbour",
            "elasticnet_environmental_terrain_and_wind_only",
        },
    },
    {
        "commit": "b8884a4",
        "date_utc": "2026-08-08T23:04:44+00:00",
        "description": "HadUK-Grid tas_mean/groundfrost_mean cohort-suffix fix (stage4_all_environmental tier only)",
        "model_names": {"dnn_env_terrain", "pinn_env_terrain", "pinn_env_terrain_k"},
        "output_dir_must_contain": "stage4_all_environmental",
    },
]


def check_known_fixes(model_name, output_dir_name, newest_entry):
    problems = []
    for fix in KNOWN_FIXES:
        if model_name not in fix["model_names"]:
            continue
        if "output_dir_must_contain" in fix and fix["output_dir_must_contain"] not in output_dir_name:
            continue
        run_commit_date = newest_entry.get("timestamp_utc")
        if run_commit_date is None:
            continue
        if run_commit_date < fix["date_utc"]:
            problems.append(
                f"trained {run_commit_date}, BEFORE the {fix['commit']} fix "
                f"({fix['date_utc']}) -- {fix['description']}"
            )
    return problems


# ---------------------------------------------------------------------------------------------
# STEP 3: one audit function per cell group. Each returns a small report dict rather than
# printing directly, so the caller can format everything consistently at the end.
# ---------------------------------------------------------------------------------------------

def resolve_output_dir(run_name, cohort, split_type, fold=None, fold_style=None):
    # Two different fold-numbering conventions exist in this codebase for spatial_block_kfold,
    # confirmed by directly listing outputs/spatial_block_kfold/ rather than assuming:
    #   - baselines (CR/Linear/RF): a separate top-level folder per fold, "<run_name>_fold<N>",
    #     with cohort nested underneath (models/baselines/run_baselines.py's own convention).
    #   - DNN/PINN models (including all E6 tier-sweep variants): one folder per run_name/cohort,
    #     with "fold_<N>" (underscore) nested underneath that
    #     (models.common.saving.model_output_dir(model_name, cohort, f"fold_{held_out_fold}", ...)
    #     -- exactly what evaluate_dnn_noenv.py itself calls).
    if fold is not None:
        if fold_style == "baseline":
            run_name = f"{run_name}_fold{fold}"
        elif fold_style == "torch":
            if split_type in PREFIXED_SPLIT_TYPES:
                return OUTPUTS_DIR / split_type / run_name / cohort / f"fold_{fold}"
            return OUTPUTS_DIR / run_name / cohort / f"fold_{fold}"
    if split_type in PREFIXED_SPLIT_TYPES:
        return OUTPUTS_DIR / split_type / run_name / cohort
    return OUTPUTS_DIR / run_name / cohort


def audit_one_cell(label, model_name, run_name, cohort, split_type, all_logs, metrics_filename="metrics.json", fold=None, fold_style=None):
    output_dir = resolve_output_dir(run_name, cohort, split_type, fold=fold, fold_style=fold_style)
    metrics_path = output_dir / metrics_filename

    result = {
        "label": label,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "flags": [],
        "seed_n": 0,
        "last_evaluated": None,
        "status": "OK",
    }

    if not metrics_path.exists():
        result["status"] = "MISSING"
        result["flags"].append(f"no {metrics_filename} at this path")
        return result

    # run_logs' own "model_name" field is whatever --run-name was passed on the command line
    # (falls back to the model's plain MODEL_NAME constant only when no run_name is given) --
    # confirmed by reading models/dnn_noenv/run_dnn_noenv.py directly: it logs
    # model_name=output_model_name, and output_model_name = run_name if run_name else MODEL_NAME.
    # So the log-matching key is always run_name, never the base model family name.
    matching_logs = logs_matching(all_logs, run_name, cohort, split_type)
    # Only keep the logs that actually point at THIS run's output folder -- a plain model_name
    # like "dnn_env_terrain" can appear in logs for several different run_name folders (e.g. the
    # tier-sweep variants), so filter by output_dir too, not just model/cohort/split.
    matching_logs = [e for e in matching_logs if e.get("output_dir", "").endswith(str(output_dir.relative_to(PROJECT_ROOT)))
                      or Path(e.get("output_dir", "")) == output_dir]

    if not matching_logs:
        result["status"] = "MISSING run_logs"
        result["flags"].append("metrics.json exists on disk, but no matching run_logs entry -- can't verify provenance")
        return result

    matching_logs.sort(key=lambda e: e.get("timestamp_utc", ""))
    newest = matching_logs[-1]
    result["last_evaluated"] = newest.get("timestamp_utc")

    seeds_seen = {e.get("hyperparameters", {}).get("seed") for e in matching_logs}
    seeds_seen.discard(None)
    result["seed_n"] = len(seeds_seen) if seeds_seen else 1

    # Local vs cluster conflict check -- FIT phase only. Evaluate-phase runs are SUPPOSED to be
    # local (this project's own documented workflow: train on the cluster GPU, evaluate locally
    # on the Mac CPU afterwards -- see documentation/progress_notes.md) at a possibly-later git
    # commit than the fit, since code keeps evolving between the two steps. That's normal, not a
    # conflict. The real risk is two FIT attempts (one local, one cluster) writing to the SAME
    # output folder -- whichever gets rsynced down last silently wins, with no merge and no
    # warning, which is exactly the scenario this whole audit exists to catch.
    fit_entries = [e for e in matching_logs if e.get("run_phase") == "fit"]
    local_fits = [e for e in fit_entries if is_local(e)]
    cluster_fits = [e for e in fit_entries if not is_local(e)]
    if local_fits and cluster_fits:
        newest_local_fit = max(local_fits, key=lambda e: e.get("timestamp_utc", ""))
        newest_cluster_fit = max(cluster_fits, key=lambda e: e.get("timestamp_utc", ""))
        if newest_local_fit.get("git_commit") != newest_cluster_fit.get("git_commit") or \
                newest_local_fit.get("hyperparameters") != newest_cluster_fit.get("hyperparameters"):
            result["flags"].append(
                f"LOCAL/CLUSTER FIT CONFLICT: a local FIT @ {newest_local_fit.get('timestamp_utc')} "
                f"(commit {str(newest_local_fit.get('git_commit'))[:10]}) AND a cluster FIT @ "
                f"{newest_cluster_fit.get('timestamp_utc')} (commit {str(newest_cluster_fit.get('git_commit'))[:10]}) "
                f"both wrote to this exact folder -- whichever synced down LAST is what's on disk "
                f"now, not necessarily the newer or more-correct one"
            )

    # Known-fix check.
    fix_problems = check_known_fixes(model_name, run_name, newest)
    result["flags"].extend(fix_problems)

    # mtime-vs-log-timestamp check: the file on disk should be roughly as new as the newest log
    # entry says it should be. A big gap (file much OLDER than the log claims) suggests the file
    # was overwritten by an older sync afterwards.
    try:
        file_mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=None).astimezone()
        log_time = datetime.fromisoformat(newest.get("timestamp_utc"))
        gap_hours = (file_mtime - log_time).total_seconds() / 3600.0
        if gap_hours < -1.0:
            # File is MORE than an hour OLDER than the log says -- something else touched this
            # folder after the logged run, without leaving its own log entry.
            result["flags"].append(
                f"MTIME MISMATCH: {metrics_filename} is {abs(gap_hours):.1f}h OLDER than the newest "
                f"matching run_logs entry -- file may have been overwritten by an older sync"
            )
    except (OSError, TypeError, ValueError):
        pass

    if result["flags"]:
        result["status"] = "FLAG"
    return result


# ---------------------------------------------------------------------------------------------
# STEP 4: the actual list of cell groups on the ledger. Written out explicitly rather than
# derived cleverly from a formula -- this project has a LOT of differently-named experiment
# families (arch_*, diag_*, splitseed*, epochcheck*, final_*...), and guessing wrong here would
# silently audit the wrong folder. Every run_name below was confirmed against a real `ls` of
# outputs/ before being written in.
# ---------------------------------------------------------------------------------------------

COHORTS = ["4survey", "6survey"]

# Main RESULTS grid: split_type -> model_name -> run_name (run_name usually == model_name, except
# where the ledger's own footer already documents a special seed-averaged folder).
BASELINE_RUN_NAMES = {"chapman_richards": "chapman_richards", "linear_baseline": "linear_baseline", "rf_baseline": "rf_baseline"}
ENV_TERRAIN_MODELS = ["dnn_env_terrain", "pinn_env_terrain", "pinn_env_terrain_k"]

MAIN_GRID_CELLS = []
for split_type in ["plot_level", "spatial_block", "spatial_block_kfold", "temporal"]:
    for model_name, run_name in BASELINE_RUN_NAMES.items():
        MAIN_GRID_CELLS.append((split_type, model_name, run_name))
    # dnn_noenv/pinn_noenv: the 5-seed corrected folders only exist for spatial_block/temporal
    # (per the ledger footer) -- plot_level and spatial_block_kfold still use the plain single-run
    # folder name.
    if split_type in ("spatial_block", "temporal"):
        for seed in range(42, 47):
            MAIN_GRID_CELLS.append((split_type, "dnn_noenv", f"final_dnn_seed{seed}"))
            MAIN_GRID_CELLS.append((split_type, "pinn_noenv", f"final_pinn_w1_anchorfix_seed{seed}"))
    else:
        MAIN_GRID_CELLS.append((split_type, "dnn_noenv", "dnn_noenv"))
        MAIN_GRID_CELLS.append((split_type, "pinn_noenv", "pinn_noenv"))
    for model_name in ENV_TERRAIN_MODELS:
        MAIN_GRID_CELLS.append((split_type, model_name, model_name))

# E6 tier sweep, spatial_block_kfold: <tier>_<model>, folds 0-4.
E6_TIERS = ["stage1_terrain", "stage2_terrain_wind", "stage4_all_environmental"]
E6_KFOLD_CELLS = [
    ("spatial_block_kfold", model_name, f"{tier}_{model_name}")
    for tier in E6_TIERS for model_name in ENV_TERRAIN_MODELS
]

# E6 plot_level extension: <tier>_<model>, no split prefix.
E6_PLOTLEVEL_CELLS = [
    ("plot_level", model_name, f"{tier}_{model_name}")
    for tier in E6_TIERS for model_name in ENV_TERRAIN_MODELS
]

# AV2 Elastic Net / XGBoost -- these DO have run_logs entries, under names that don't match the
# ledger's scope labels directly (confirmed by grepping outputs/run_logs/*.json for model_name).
AV2_EN_XGB_MODEL_NAMES = [
    "elasticnet_environmental_terrain_and_wind_only",
    "elasticnet_environmental_all_environmental_no_neighbour",
    "elasticnet_environmental_all_environmental",
    "xgb_environmental_terrain_and_wind_only",
    "xgb_environmental_all_environmental_no_neighbour",
    "xgb_environmental_all_environmental",
]


def audit_main_grid(all_logs):
    print("\n=== Main RESULTS grid (RQ1) ===")
    print("split_type / model / cohort -> status")
    for split_type, model_name, run_name in MAIN_GRID_CELLS:
        for cohort in COHORTS:
            # spatial_block_kfold is a pooled result across 5 folds -- check every fold's own
            # folder, since a single bad fold would corrupt the pooled number too.
            if split_type == "spatial_block_kfold":
                fold_flags = []
                fold_seed_ns = []
                fold_timestamps = []
                any_missing = False
                fold_style = "baseline" if model_name in BASELINE_RUN_NAMES else "torch"
                for fold in range(5):
                    r = audit_one_cell(
                        f"{model_name}/{cohort}/{split_type} fold{fold}",
                        model_name, run_name, cohort, split_type, all_logs,
                        fold=fold, fold_style=fold_style,
                    )
                    if r["status"] == "MISSING":
                        any_missing = True
                    fold_flags.extend(f"fold{fold}: {f}" for f in r["flags"])
                    fold_seed_ns.append(r["seed_n"])
                    if r["last_evaluated"]:
                        fold_timestamps.append(r["last_evaluated"])
                status = "MISSING" if any_missing else ("FLAG" if fold_flags else "OK")
                report_line(status, f"{split_type}/{model_name}/{cohort}",
                            seed_n=max(fold_seed_ns) if fold_seed_ns else 0,
                            last_evaluated=max(fold_timestamps) if fold_timestamps else None,
                            flags=fold_flags)
            else:
                r = audit_one_cell(f"{model_name}/{cohort}/{split_type}", model_name, run_name, cohort, split_type, all_logs)
                report_line(r["status"], f"{split_type}/{model_name}/{cohort}",
                            seed_n=r["seed_n"], last_evaluated=r["last_evaluated"], flags=r["flags"])


def audit_e6(all_logs):
    print("\n=== E6 tier sweep, spatial_block_kfold (RQ1, env-conditioned prediction) ===")
    for split_type, model_name, run_name in E6_KFOLD_CELLS:
        for cohort in COHORTS:
            fold_flags = []
            any_missing = False
            timestamps = []
            for fold in range(5):
                r = audit_one_cell(f"{run_name}/{cohort} fold{fold}", model_name, run_name, cohort, split_type, all_logs,
                                    fold=fold, fold_style="torch")
                if r["status"] == "MISSING":
                    any_missing = True
                fold_flags.extend(f"fold{fold}: {f}" for f in r["flags"])
                if r["last_evaluated"]:
                    timestamps.append(r["last_evaluated"])
            status = "MISSING" if any_missing else ("FLAG" if fold_flags else "OK")
            report_line(status, f"{run_name}/{cohort}", seed_n=1,
                        last_evaluated=max(timestamps) if timestamps else None, flags=fold_flags)

    print("\n=== E6 tier sweep, plot_level extension (appendix) ===")
    for split_type, model_name, run_name in E6_PLOTLEVEL_CELLS:
        for cohort in COHORTS:
            r = audit_one_cell(f"{run_name}/{cohort}", model_name, run_name, cohort, split_type, all_logs)
            report_line(r["status"], f"{run_name}/{cohort}", seed_n=r["seed_n"],
                        last_evaluated=r["last_evaluated"], flags=r["flags"])


def audit_av2_en_xgb(all_logs):
    print("\n=== AV2 Elastic Net / XGBoost (RQ3) -- via run_logs ===")
    print("These use models/growth_curve_attribution/broad_environmental_check.py's own pipeline,")
    print("which writes to outputs/growth_curve_attribution/*.csv rather than a per-model folder --")
    print("checking git_commit of the LATEST matching run_logs entry against the XGBoost leak fix only.")
    for model_name in AV2_EN_XGB_MODEL_NAMES:
        matches = [e for e in all_logs if e.get("model_name") == model_name]
        if not matches:
            report_line("MISSING run_logs", model_name, seed_n=0, last_evaluated=None, flags=["no run_logs entry found"])
            continue
        matches.sort(key=lambda e: e.get("timestamp_utc", ""))
        newest = matches[-1]
        flags = check_known_fixes(model_name, "", newest)
        seeds = {e.get("hyperparameters", {}).get("seed") for e in matches}
        seeds.discard(None)
        report_line("FLAG" if flags else "OK", model_name, seed_n=len(seeds) if seeds else 1,
                     last_evaluated=newest.get("timestamp_utc"), flags=flags)


def audit_av2_no_provenance_log():
    print("\n=== AV2 GNNWR / Simple DNN / Compartment-mixed DNN (RQ3) -- NO run_logs coverage ===")
    print("These three don't call write_run_log() at all (confirmed by grep) -- there is no")
    print("git_commit/hostname/seed record for them, only the output CSV files' own mtimes.")
    print("This is a real provenance gap worth noting in the report, not something this audit can")
    print("verify the same way as everything else.")
    candidates = {
        "gnnwr": OUTPUTS_DIR / "growth_curve_attribution" / "gnnwr",
        "simple_dnn": OUTPUTS_DIR / "growth_curve_attribution" / "simple_dnn",
        "compartment_mixed_dnn": OUTPUTS_DIR / "growth_curve_attribution" / "compartment_mixed_dnn",
    }
    for name, folder in candidates.items():
        if not folder.exists():
            report_line("MISSING", name, seed_n=0, last_evaluated=None, flags=["folder not found"])
            continue
        pooled_files = sorted(folder.glob("*kfold_pooled_summary.csv"))
        if not pooled_files:
            report_line("MISSING", name, seed_n=0, last_evaluated=None, flags=["no *_kfold_pooled_summary.csv found"])
            continue
        newest_mtime = max(f.stat().st_mtime for f in pooled_files)
        newest_iso = datetime.fromtimestamp(newest_mtime).astimezone().isoformat()
        report_line("NO PROVENANCE LOG (mtime only)", name, seed_n=0, last_evaluated=newest_iso,
                     flags=[f"{len(pooled_files)} pooled summary file(s) found, newest mtime {newest_iso}"])


def report_line(status, label, seed_n, last_evaluated, flags):
    seed_display = "n/a (single seed)" if seed_n in (0, 1) else f"{seed_n} seeds"
    print(f"[{status}] {label} -- seed_n={seed_display}, last_evaluated={last_evaluated}")
    for f in flags:
        print(f"    - {f}")


def main():
    all_logs = load_success_logs()
    print(f"Loaded {len(all_logs)} successful run_logs entries.")
    audit_main_grid(all_logs)
    audit_e6(all_logs)
    audit_av2_en_xgb(all_logs)
    audit_av2_no_provenance_log()


if __name__ == "__main__":
    main()
