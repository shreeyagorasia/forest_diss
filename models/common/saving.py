import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def save_run(output_dir, predictions_df, metrics, model_name, cohort, seed):
    """Write predictions.csv, metrics.json, and run_metadata.json for one model run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    run_metadata = {
        "model_name": model_name,
        "cohort": cohort,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)
