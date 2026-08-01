# Run as: python -m models.common.build_results_table
#
# Reads every JSON file in outputs/run_logs/ (see run_logging.py -- one file per run EVENT:
# started/success/failed/a quick test) and flattens them all into ONE csv, one row per event,
# so it's easy to open in a spreadsheet or pandas and filter by whatever matters: is_test_run
# (real runs vs a quick --max-epochs 5 sanity check), model_name, cohort, split_type, or any
# specific hyperparameter/metric.
#
# Deliberately simple: no filtering logic lives here at all -- every event, real and test alike,
# goes into the csv exactly as logged. Filtering (e.g. "only real, successful fit runs") is left
# to whoever reads the csv, matching run_logging.py's own stated design ("a separate script can
# do that later" -- this is that script, but it still doesn't decide what counts as
# interesting, it just makes everything visible in one place).

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_LOGS_DIR = PROJECT_ROOT / "outputs" / "run_logs"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs" / "run_logs_summary.csv"

# Columns worth seeing FIRST when the csv is opened -- everything else (every hyperparameter,
# every metric) still ends up in the csv too, just further right, alphabetically ordered by
# pandas.json_normalize() rather than hand-picked one by one.
PRIORITY_COLUMNS = [
    "timestamp_utc", "model_name", "cohort", "split_type", "run_phase", "status", "is_test_run",
    "runtime_seconds", "n_rows_fit", "run_id",
]


def load_all_run_logs():
    # One dict per run EVENT, flattened with pandas' own json_normalize -- a nested field like
    # hyperparameters -> physics_weight becomes one column named "hyperparameters.physics_weight"
    # (sep="." is pandas' own default, kept as-is rather than picking a different separator, so
    # this matches what anyone already familiar with json_normalize would expect).
    all_rows = []
    for log_path in sorted(RUN_LOGS_DIR.glob("*.json")):
        with open(log_path) as f:
            log_entry = json.load(f)
        all_rows.append(log_entry)
    return all_rows


def build_results_table():
    all_rows = load_all_run_logs()
    print(f"Found {len(all_rows):,} run-log files in {RUN_LOGS_DIR}")

    results_df = pd.json_normalize(all_rows, sep=".")

    # Most-recent-first -- the newest runs are usually the ones actually worth looking at.
    results_df = results_df.sort_values("timestamp_utc", ascending=False).reset_index(drop=True)

    # Priority columns first, then everything else in whatever order json_normalize produced it.
    remaining_columns = [c for c in results_df.columns if c not in PRIORITY_COLUMNS]
    ordered_columns = [c for c in PRIORITY_COLUMNS if c in results_df.columns] + remaining_columns
    results_df = results_df[ordered_columns]

    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved -> {OUTPUT_CSV_PATH}")
    print(f"{len(results_df.columns)} columns, {len(results_df)} rows")

    return results_df


if __name__ == "__main__":
    build_results_table()
