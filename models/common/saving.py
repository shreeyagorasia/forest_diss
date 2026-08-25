import json
import subprocess
from pathlib import Path

from models.common.splits import SPLIT_SEED

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_cr_params(cohort, split_type, split_seed=SPLIT_SEED, held_out_fold=None):
    # Split-MATCHED CR anchor (2026-08-01 fix, switched from the pooled/"cr_pooled" version
    # every caller used before). Fit using ONLY this split_type's own train-assigned plots
    # (models/baselines/run_baselines.py's cr_train_df), not a random 60% plot_level split
    # unrelated to spatial_block/temporal. The old pooled version was confirmed to leak: its
    # random 60% training plots inevitably overlap with whichever plots a given split_type later
    # assigns to test, since the two splits were never coordinated (verified directly: the saved
    # n_rows_fit was exactly 60% of filtered rows. Plot_level_split's train share, not this
    # split's own). Read as plain floats and treated as FROZEN constants either way. Never
    # refit here, this file already exists on disk for every split_type PINN uses, built as a
    # side effect of the CR baseline's own per-split fit.
    #
    # Moved here 2026-08-02 (was duplicated verbatim in run_pinn_noenv.py and
    # run_pinn_env_terrain.py). Models/env_deviation/ became a third caller, which was the
    # trigger to de-duplicate rather than add a third copy that could silently drift from the
    # other two.
    #
    # split_seed (2026-08-02 addition): matches run_baselines.py's own "_splitseed<N>" output
    # naming (name_suffix computed the exact same way there. "" for the default SPLIT_SEED,
    # so every existing caller reads the exact same path as before; non-default reads the
    # split-seed-MATCHED anchor a robustness-check refit produces, not the mismatched default
    # one). See documentation/experiment_log.md's 2026-08-02 split-seed robustness entries.
    #
    # held_out_fold (2026-08-04 addition, for split_type="spatial_block_kfold"): a CR anchor fit
    # on a plain spatial_block split's train set would leak some of a given fold's held-out
    # compartments into the "frozen" physics anchor. Matches run_baselines.py's own
    # "_fold<N>"-suffixed output (always added under spatial_block_kfold, never omitted, since
    # held_out_fold=0 is still a genuinely different split from every other fold).
    name_suffix = "" if split_seed == SPLIT_SEED else f"_splitseed{split_seed}"
    if held_out_fold is not None:
        name_suffix = f"{name_suffix}_fold{held_out_fold}"
    # plot_level is the one split type with no outputs/<split_type>/ prefix (see
    # model_output_dir()'s own comment). Added 2026-08-08 alongside DNN/PINN's first-ever
    # plot_level run, matching that convention rather than DNN/PINN's own default of always
    # being prefixed.
    if split_type in ("spatial_block", "spatial_block_kfold", "temporal", "temporal_narrow_gap"):
        params_path = PROJECT_ROOT / "outputs" / split_type / f"chapman_richards{name_suffix}" / cohort / "params.json"
    else:
        params_path = PROJECT_ROOT / "outputs" / f"chapman_richards{name_suffix}" / cohort / "params.json"
    with open(params_path) as f:
        params = json.load(f)
    return {"y_max": params["y_max"], "k": params["k"], "p": params["p"]}


def model_output_dir(*parts, split_type):
    # Keeps every output path for a spatial_block or temporal run under its
    # own outputs/<split_type>/ subtree, so it can never overwrite the
    # plot_level results living directly under outputs/ (plot_level is the
    # only split type that gets no prefix, since it was the original
    # default before spatial_block/temporal existed). Shared by the
    # baselines and by DNN/PINN, which never run plot_level at all, so
    # their output path is always prefixed.
    if split_type in ("spatial_block", "spatial_block_kfold", "temporal", "temporal_narrow_gap"):
        return PROJECT_ROOT / "outputs" / split_type / Path(*parts)
    return PROJECT_ROOT / "outputs" / Path(*parts)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None
