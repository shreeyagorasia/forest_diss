# MIGRATED 2026-08-25 from temp_results_pinn/pinn_env_terrain_fix/pinn_env_terrain_fix.py into
# models/ so the CORRECTED forward pass has a canonical, discoverable home alongside the other
# model families. Byte-identical to the original except for this header. No math, no logic
# changed. The original stays in place under temp_results_pinn/ (still used by ~12 diagnostic
# job scripts there, e.g. run_pinn_fix_cluster.sh, run_pinn_mechanism_checks.py) and is the full
# record of the bug discovery/fix investigation (see temp_results_pinn/PLAN.md).
#
# FIX EXPERIMENT (2026-08-20). Copy of models/pinn_env_terrain/pinn_env_terrain.py with ONE
# real change: forward() now actually routes the per-plot y_max through to the prediction,
# instead of only to the physics/trajectory losses. See temp_results_pinn/PLAN.md for the full
# plan, the 15 documented pitfalls, and why this originally lived outside models/ instead of
# editing the original file in place.
#
# Prediction becomes: H_pred_i = y_max_i * (1 - exp(-k*age_i))^p + trunk_residual_i
# (k, p stay global frozen floats, same as the original. Only y_max is per-plot here; the k
# version is a separate file, pinn_env_terrain_k_fix.py).
#
# The buggy original, models/pinn_env_terrain/pinn_env_terrain.py, is intentionally left
# untouched. It remains the correct historical basis for interpreting any existing outputs/
# results produced before this fix (see outputs/pinn_env_terrain* naming).

import itertools
import json
import time
from datetime import datetime, timezone

import pandas as pd
import torch
import torch.nn as nn

from models.common.torch_model import NoEnvNetwork, YMaxSubNetwork, chapman_richards_derivative, compute_l1_penalty

# ----- Fixed hyperparameters. Identical to the original pinn_env_terrain.py, on purpose. Any
# difference in results should come from the now-functional y_max path, not an unrelated knob. -----
L1_COEFFICIENT = 1e-5
PHYSICS_WEIGHT = 1.0
TRAJECTORY_WEIGHT = 1.0
LEARNING_RATE = 0.0001
LR_SCHEDULER_FACTOR = 0.8
LR_SCHEDULER_PATIENCE = 15
BATCH_SIZE = 256
PAIRS_BATCH_SIZE = 256
WEIGHT_DECAY = 1e-5
GRAD_CLIP_MAX_NORM = 1.0
VAL_LOSS_SMOOTHING_WINDOW = 5
PRINT_EVERY_N_EPOCHS = 10

Y_MAX_SUBNETWORK_HIDDEN_SIZE = 16


def chapman_richards_value(age, y_max, k, p):
    # FIX: new function, not in the original file (Mistake #12 in PLAN.md. Need H(a) now, not
    # just H'(a)). Mirrors chapman_richards_derivative()'s own style line-by-line on purpose:
    # same broadcasting convention (age/y_max both [batch, 1] tensors, k/p plain floats), so if
    # the derivative function's shapes are already trusted, this one's are too.
    # height = y_max * (1 - exp(-k*age))^p
    decay_term = torch.exp(-k * age)
    return y_max * (1 - decay_term) ** p


class EnvTerrainPINN(nn.Module):
    def __init__(
        self, n_other_features, n_terrain_features, cr_params, scaler_age, scaler_height,
        dropout_rate=0.0, hidden_layer_sizes=None,
    ):
        super().__init__()
        self.main_network = NoEnvNetwork(
            n_other_features=n_other_features, dropout_rate=dropout_rate, hidden_layer_sizes=hidden_layer_sizes,
        )
        self.y_max_subnetwork = YMaxSubNetwork(
            n_terrain_features=n_terrain_features, hidden_size=Y_MAX_SUBNETWORK_HIDDEN_SIZE,
            dropout_rate=dropout_rate,
        )
        # FIX: cr_params/scaler_age/scaler_height stored on the model itself, not threaded
        # through every call site's arguments. These are fixed constants for the whole run
        # (same "process model constants never get a gradient" guarantee the original file
        # already relies on for cr_params elsewhere). Only terrain_features is a genuinely new
        # required argument to forward() below.
        self.cr_params = cr_params
        self.scaler_age = scaler_age
        self.scaler_height = scaler_height

    def forward(self, other_features, age, terrain_features):
        # FIX: this is the actual fix. Original file only ever returned
        # self.main_network(other_features, age). Terrain never reached the prediction, only
        # the physics/trajectory losses (via compute_plot_specific_y_max, called separately).
        # Now: trunk output is treated as a RESIDUAL on top of the CR curve evaluated at this
        # plot's own adjusted y_max, not the whole prediction by itself.
        trunk_residual_scaled = self.main_network(other_features, age)

        y_max_per_row = compute_plot_specific_y_max(self, terrain_features, self.cr_params["y_max"])

        # Mistake #10 (PLAN.md): do NOT detach age here. The physics loss differentiates this
        # forward() output w.r.t. age via torch.autograd.grad. If the CR term's age input were
        # a different tensor (e.g. a detached copy), that call would silently miss the CR term's
        # own contribution to the derivative and only return the residual's. Same age tensor,
        # same graph, all the way through.
        age_unscaled = age * self.scaler_age.scale_[0] + self.scaler_age.mean_[0]

        cr_value = chapman_richards_value(age_unscaled, y_max_per_row, self.cr_params["k"], self.cr_params["p"])

        # Mistake #6 (PLAN.md): unit mismatch. cr_value is in REAL metres (y_max_per_row and the
        # age used here both are); trunk_residual_scaled is in SCALED units (same convention
        # NoEnvNetwork always outputs in). Convert the CR term into scaled units so the two are
        # addable, rather than silently mixing units.
        cr_value_scaled = (cr_value - self.scaler_height.mean_[0]) / self.scaler_height.scale_[0]

        return cr_value_scaled + trunk_residual_scaled


def build_model(
    n_other_features, n_terrain_features, cr_params, scaler_age, scaler_height, device, seed,
    dropout_rate=0.0, hidden_layer_sizes=None,
):
    torch.manual_seed(seed)
    model = EnvTerrainPINN(
        n_other_features=n_other_features, n_terrain_features=n_terrain_features,
        cr_params=cr_params, scaler_age=scaler_age, scaler_height=scaler_height,
        dropout_rate=dropout_rate, hidden_layer_sizes=hidden_layer_sizes,
    )
    return model.to(device)


def compute_plot_specific_y_max(model, terrain_batch, global_y_max):
    # Unchanged from the original file.
    y_max_adjustment = model.y_max_subnetwork(terrain_batch)
    return global_y_max + y_max_adjustment


def compute_physics_loss(model, age_batch, other_batch, terrain_batch, cr_params, scaler_age, scaler_height):
    # FIX: model(...) call now needs terrain_batch too (forward()'s new required argument).
    # Mistake #14 (PLAN.md): NaN from the CR term. Retained here since it's structurally the
    # same call as before, just with the extra argument.
    age_batch = age_batch.clone().requires_grad_(True)

    predicted_height_scaled = model(other_batch, age_batch, terrain_batch)

    height_wrt_age_scaled = torch.autograd.grad(
        outputs=predicted_height_scaled,
        inputs=age_batch,
        grad_outputs=torch.ones_like(predicted_height_scaled),
        create_graph=True,
    )[0]

    scale_factor = scaler_height.scale_[0] / scaler_age.scale_[0]
    height_wrt_age = height_wrt_age_scaled * scale_factor

    age_unscaled = age_batch.detach() * scaler_age.scale_[0] + scaler_age.mean_[0]
    y_max_per_row = compute_plot_specific_y_max(model, terrain_batch, cr_params["y_max"])
    cr_growth_rate = chapman_richards_derivative(age_unscaled, y_max_per_row, cr_params["k"], cr_params["p"])

    physics_loss = torch.mean((height_wrt_age - cr_growth_rate) ** 2)
    return physics_loss


def compute_trajectory_loss(
    model, age_earlier, other_earlier, age_later, other_later, delta_age, age_mid, terrain_pairs,
    cr_params, scaler_age, scaler_height,
):
    # FIX: both model(...) calls now need terrain_pairs too.
    predicted_earlier_scaled = model(other_earlier, age_earlier, terrain_pairs)
    predicted_later_scaled = model(other_later, age_later, terrain_pairs)

    predicted_earlier = predicted_earlier_scaled * scaler_height.scale_[0] + scaler_height.mean_[0]
    predicted_later = predicted_later_scaled * scaler_height.scale_[0] + scaler_height.mean_[0]

    predicted_growth_rate = (predicted_later - predicted_earlier) / delta_age
    y_max_per_row = compute_plot_specific_y_max(model, terrain_pairs, cr_params["y_max"])
    cr_growth_rate_at_mid = chapman_richards_derivative(age_mid, y_max_per_row, cr_params["k"], cr_params["p"])

    trajectory_loss = torch.mean((predicted_growth_rate - cr_growth_rate_at_mid) ** 2)
    return trajectory_loss


def train_one_epoch(
    model, optimizer,
    age_train, other_train, terrain_train, target_train,
    pair_tensors, terrain_pairs_all, cr_params, scaler_age, scaler_height,
    device, physics_weight, trajectory_weight,
    batch_size=BATCH_SIZE, pairs_batch_size=PAIRS_BATCH_SIZE,
):
    model.train()

    n_rows = age_train.shape[0]
    shuffled_row_order = torch.randperm(n_rows, device=device)
    main_batch_starts = list(range(0, n_rows, batch_size))

    age_earlier_all, other_earlier_all, age_later_all, other_later_all, delta_age_all, age_mid_all, _ = pair_tensors
    n_pairs = age_earlier_all.shape[0]
    shuffled_pair_order = torch.randperm(n_pairs, device=device)
    pair_batch_starts = list(range(0, n_pairs, pairs_batch_size))
    pair_batch_cycle = itertools.cycle(pair_batch_starts)

    totals = {"data_loss": 0.0, "physics_loss": 0.0, "trajectory_loss": 0.0, "grad_norm": 0.0}
    n_batches = 0

    for batch_start in main_batch_starts:
        batch_row_indices = shuffled_row_order[batch_start:batch_start + batch_size]
        age_batch = age_train[batch_row_indices]
        other_batch = other_train[batch_row_indices]
        terrain_batch = terrain_train[batch_row_indices]
        target_batch = target_train[batch_row_indices]

        pair_batch_start = next(pair_batch_cycle)
        pair_batch_indices = shuffled_pair_order[pair_batch_start:pair_batch_start + pairs_batch_size]

        optimizer.zero_grad()

        # FIX: needs terrain_batch now.
        predicted_height = model(other_batch, age_batch, terrain_batch)
        data_loss = torch.mean((predicted_height - target_batch) ** 2)

        physics_loss = compute_physics_loss(
            model, age_batch, other_batch, terrain_batch, cr_params, scaler_age, scaler_height,
        )

        trajectory_loss = compute_trajectory_loss(
            model,
            age_earlier_all[pair_batch_indices], other_earlier_all[pair_batch_indices],
            age_later_all[pair_batch_indices], other_later_all[pair_batch_indices],
            delta_age_all[pair_batch_indices], age_mid_all[pair_batch_indices],
            terrain_pairs_all[pair_batch_indices],
            cr_params, scaler_age, scaler_height,
        )

        l1_loss = L1_COEFFICIENT * compute_l1_penalty(model)

        total_loss = data_loss + physics_weight * physics_loss + trajectory_weight * trajectory_loss + l1_loss

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        totals["data_loss"] = totals["data_loss"] + data_loss.item()
        totals["physics_loss"] = totals["physics_loss"] + physics_loss.item()
        totals["trajectory_loss"] = totals["trajectory_loss"] + trajectory_loss.item()
        totals["grad_norm"] = totals["grad_norm"] + grad_norm.item()
        n_batches = n_batches + 1

    return {key: value / n_batches for key, value in totals.items()}


def evaluate_on_validation_set(model, age_val, other_val, terrain_val, target_val):
    # FIX: needs terrain_val now.
    model.eval()
    with torch.no_grad():
        predicted_height = model(other_val, age_val, terrain_val)
        val_loss = torch.mean((predicted_height - target_val) ** 2)
    return val_loss.item()


def build_optimizer(model, optimizer_name, learning_rate=LEARNING_RATE):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    elif optimizer_name == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer_name: {optimizer_name!r}")


def fit(
    age_train, other_train, terrain_train, target_train,
    age_val, other_val, terrain_val, target_val,
    pair_tensors, terrain_pairs_all, cr_params, scaler_age, scaler_height,
    n_other_features, n_terrain_features, device, seed,
    max_epochs, early_stopping_patience,
    optimizer_name="adam",
    physics_weight=PHYSICS_WEIGHT, trajectory_weight=TRAJECTORY_WEIGHT,
    batch_size=BATCH_SIZE, pairs_batch_size=PAIRS_BATCH_SIZE,
    dropout_rate=0.0,
    learning_rate=LEARNING_RATE,
    hidden_layer_sizes=None,
):
    # FIX: signature gains terrain_val (evaluate_on_validation_set needs it now).
    training_start_time = time.time()
    model = build_model(
        n_other_features, n_terrain_features, cr_params, scaler_age, scaler_height, device, seed,
        dropout_rate=dropout_rate, hidden_layer_sizes=hidden_layer_sizes,
    )
    optimizer = build_optimizer(model, optimizer_name, learning_rate=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )

    best_smoothed_val_loss = None
    best_model_state = None
    epochs_without_improvement = 0
    history_rows = []
    recent_val_losses = []

    for epoch in range(1, max_epochs + 1):
        epoch_losses = train_one_epoch(
            model, optimizer,
            age_train, other_train, terrain_train, target_train,
            pair_tensors, terrain_pairs_all, cr_params, scaler_age, scaler_height,
            device, physics_weight, trajectory_weight,
            batch_size=batch_size, pairs_batch_size=pairs_batch_size,
        )
        val_loss = evaluate_on_validation_set(model, age_val, other_val, terrain_val, target_val)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed_seconds = time.time() - training_start_time

        recent_val_losses.append(val_loss)
        if len(recent_val_losses) > VAL_LOSS_SMOOTHING_WINDOW:
            recent_val_losses.pop(0)
        smoothed_val_loss = sum(recent_val_losses) / len(recent_val_losses)

        history_rows.append({
            "epoch": epoch,
            "train_loss": epoch_losses["data_loss"] + physics_weight * epoch_losses["physics_loss"]
                          + trajectory_weight * epoch_losses["trajectory_loss"],
            "data_loss": epoch_losses["data_loss"],
            "physics_loss": epoch_losses["physics_loss"],
            "trajectory_loss": epoch_losses["trajectory_loss"],
            "grad_norm": epoch_losses["grad_norm"],
            "val_loss": val_loss,
            "val_loss_smoothed": smoothed_val_loss,
            "learning_rate": current_lr,
            "elapsed_seconds": elapsed_seconds,
        })

        is_new_best = best_smoothed_val_loss is None or smoothed_val_loss < best_smoothed_val_loss
        if is_new_best:
            best_smoothed_val_loss = smoothed_val_loss
            best_model_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement = epochs_without_improvement + 1

        should_print_this_epoch = (epoch == 1) or (epoch % PRINT_EVERY_N_EPOCHS == 0)
        if should_print_this_epoch:
            best_marker = " (new best)" if is_new_best else ""
            print(
                f"  epoch {epoch}/{max_epochs}  data_loss={epoch_losses['data_loss']:.4f}  "
                f"physics_loss={epoch_losses['physics_loss']:.4f}  "
                f"trajectory_loss={epoch_losses['trajectory_loss']:.4f}  "
                f"val_loss={val_loss:.4f}  val_loss_smoothed={smoothed_val_loss:.4f}  "
                f"learning_rate={current_lr:.6f}{best_marker}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} (no val_loss improvement for {early_stopping_patience} epochs).")
            break

    final_model_state = {key: value.clone() for key, value in model.state_dict().items()}

    best_model = build_model(
        n_other_features, n_terrain_features, cr_params, scaler_age, scaler_height, device, seed,
        dropout_rate=dropout_rate, hidden_layer_sizes=hidden_layer_sizes,
    )
    best_model.load_state_dict(best_model_state)

    history_df = pd.DataFrame(history_rows)
    return best_model, final_model_state, history_df


def predict(model, age, other_features, terrain_features):
    # FIX: needs terrain_features now. This is the whole point of the fix, so the prediction
    # itself can no longer be made without it.
    model.eval()
    with torch.no_grad():
        predicted_height_scaled = model(other_features, age, terrain_features)
    return predicted_height_scaled


def predict_y_max(model, terrain_features, global_y_max):
    model.eval()
    with torch.no_grad():
        y_max_per_row = compute_plot_specific_y_max(model, terrain_features, global_y_max)
    return y_max_per_row


def save_checkpoints(
    best_model, final_model_state, n_other_features, n_terrain_features, output_dir,
    hidden_layer_sizes=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), output_dir / "best_model.pt")
    torch.save(final_model_state, output_dir / "final_model.pt")
    with open(output_dir / "architecture.json", "w") as f:
        json.dump({
            "n_other_features": n_other_features, "n_terrain_features": n_terrain_features,
            "hidden_layer_sizes": hidden_layer_sizes,
        }, f, indent=2)


def save_run_metadata(cohort, n_rows_fit, hyperparameters, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cohort": cohort,
        "n_rows_fit": n_rows_fit,
        "fit_date": datetime.now(timezone.utc).isoformat(),
        **hyperparameters,
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata
