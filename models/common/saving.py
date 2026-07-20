import subprocess
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
    if split_type in ("spatial_block", "temporal", "temporal_narrow_gap"):
        return PROJECT_ROOT / "outputs" / split_type / Path(*parts)
    return PROJECT_ROOT / "outputs" / Path(*parts)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None
