import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def model_output_dir(*parts, split_type):
    # Keeps every output path for a spatial_block or temporal run under its
    # own outputs/<split_type>/ subtree, so it can never overwrite the
    # plot_level results living directly under outputs/ (plot_level is the
    # only split type that gets no prefix, since it was the original
    # default before spatial_block/temporal existed). Shared by the
    # baselines and by DNN/PINN, which never run plot_level at all, so
    # their output path is always prefixed.
    if split_type in ("spatial_block", "temporal"):
        return PROJECT_ROOT / "outputs" / split_type / Path(*parts)
    return PROJECT_ROOT / "outputs" / Path(*parts)


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
