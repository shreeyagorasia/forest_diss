# Run as: python -m models.baselines.rq1_pinn_split_stability_check
#
# Tests the open question flagged in the RQ1 evaluation-split paragraph: PINN shows almost no
# inflation between the random plot-level split and the spatial-block split (4survey: 0.590 vs.
# 0.584), unlike the DNN (0.831 vs. 0.634). Is this stability caused by the physics constraint,
# by PINN's generally weaker fit, or by another architectural difference? Untested until now.
#
# Method: refit PINN with the physics/trajectory loss switched OFF (physics_weight=
# trajectory_weight=0.0. The same "w=0" ablation already used and reported elsewhere this
# project, TEMP_rq1_physicsablation_results_2026-08-11.tex), under BOTH plot_level and
# spatial_block splits, same architecture, same everything else. If the physics constraint is
# what causes split-stability, removing it should make the w=0 plot_level-vs-spatial_block gap
# widen back toward the DNN's own gap. If the gap stays small even at w=0, the constraint is not
# the explanation. Something else about PINN (its weaker overall fit, or an architectural
# difference untouched by turning physics off) is more likely responsible.

import torch

from models.common.metrics import compute_metrics
from models.common.saving import load_cr_params
from models.common.splits import SPLIT_SEED
from models.common.torch_data import (
    ENV_TERRAIN_FEATURE_SETS,
    TARGET_COLUMN,
    build_pair_terrain_tensor,
    build_pair_tensors,
    build_tensors,
    build_terrain_tensor,
    encode_thinning_status,
    fit_scalers,
    fit_terrain_scaler,
    load_split_table_with_terrain,
    load_trajectory_pairs,
    select_device,
)
from models.pinn_env_terrain.pinn_env_terrain import BATCH_SIZE, PAIRS_BATCH_SIZE, LEARNING_RATE
from models.pinn_env_terrain.pinn_env_terrain import fit as fit_pinn, predict as predict_pinn

COHORT = "4survey"
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
TRAINING_SEED = 42
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 40
PHYSICS_WEIGHT = 0.0
TRAJECTORY_WEIGHT = 0.0

# Already-established w=1 (standard) reference points, same splits, same cohort. No refit
# needed, these are the numbers already in the chapter.
W1_PLOT_LEVEL_R2 = 0.590
W1_SPATIAL_BLOCK_R2 = 0.584


def fit_and_evaluate(split_type, feature_columns, device):
    cr_held_out_fold = None
    cr_params = load_cr_params(COHORT, split_type, split_seed=SPLIT_SEED, held_out_fold=cr_held_out_fold)

    full_df = load_split_table_with_terrain(COHORT, split_type, feature_columns, split_seed=SPLIT_SEED)
    train_df = full_df[full_df["split"] == "train"]
    val_df = full_df[full_df["split"] == "val"]
    test_df = full_df[full_df["split"] == "test"]

    pairs_df = load_trajectory_pairs(COHORT, full_df)

    scaler_age, scaler_other_features, scaler_height = fit_scalers(train_df)
    scaler_terrain = fit_terrain_scaler(train_df, feature_columns)
    encoded_column_names = encode_thinning_status(train_df).columns.tolist()

    age_train, other_train, target_train = build_tensors(
        train_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    age_val, other_val, target_val = build_tensors(
        val_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    terrain_train = build_terrain_tensor(train_df, scaler_terrain, feature_columns, device)

    pair_tensors = build_pair_tensors(pairs_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
    terrain_pairs = build_pair_terrain_tensor(pairs_df, scaler_terrain, feature_columns, device, COHORT)

    n_other_features = other_train.shape[1]
    n_terrain_features = terrain_train.shape[1]
    best_model, _final_model_state, history_df = fit_pinn(
        age_train, other_train, terrain_train, target_train,
        age_val, other_val, target_val,
        pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
        n_other_features, n_terrain_features, device, TRAINING_SEED,
        MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
        optimizer_name="adam",
        physics_weight=PHYSICS_WEIGHT, trajectory_weight=TRAJECTORY_WEIGHT,
        batch_size=BATCH_SIZE, pairs_batch_size=PAIRS_BATCH_SIZE,
        dropout_rate=0.0, learning_rate=LEARNING_RATE, hidden_layer_sizes=None,
    )
    print(f"  [{split_type}] trained for {len(history_df)} epochs, {len(pairs_df)} trajectory pairs")

    age_test, other_test, target_test = build_tensors(
        test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    predicted_scaled = predict_pinn(best_model, age_test, other_test)
    predicted = scaler_height.inverse_transform(predicted_scaled.cpu().numpy()).flatten()
    observed = test_df[TARGET_COLUMN].to_numpy()

    return compute_metrics(observed, predicted)["r2"]


def main():
    device = select_device()
    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]

    print("Fitting PINN (physics_weight=trajectory_weight=0.0), plot_level split...")
    plot_level_r2 = fit_and_evaluate("plot_level", feature_columns, device)
    print(f"  w=0 plot_level test R2: {plot_level_r2:.4f}")

    print("\nFitting PINN (physics_weight=trajectory_weight=0.0), spatial_block split...")
    spatial_block_r2 = fit_and_evaluate("spatial_block", feature_columns, device)
    print(f"  w=0 spatial_block test R2: {spatial_block_r2:.4f}")

    w0_gap = plot_level_r2 - spatial_block_r2
    w1_gap = W1_PLOT_LEVEL_R2 - W1_SPATIAL_BLOCK_R2

    print("\n=== Comparison ===")
    print(f"w=1 (standard PINN): plot_level={W1_PLOT_LEVEL_R2:.4f}  spatial_block={W1_SPATIAL_BLOCK_R2:.4f}  gap={w1_gap:+.4f}")
    print(f"w=0 (no physics loss): plot_level={plot_level_r2:.4f}  spatial_block={spatial_block_r2:.4f}  gap={w0_gap:+.4f}")
    if abs(w0_gap) > abs(w1_gap) + 0.03:
        print("Gap widened meaningfully at w=0 -> consistent with the physics constraint causing split-stability.")
    elif abs(w0_gap) < abs(w1_gap) - 0.03:
        print("Gap shrank further at w=0 -> not consistent with the physics constraint being the cause.")
    else:
        print("Gap stayed about the same at w=0 -> physics constraint does not explain the split-stability;")
        print("something else (weaker fit, or another architectural factor) is more likely responsible.")


if __name__ == "__main__":
    main()
