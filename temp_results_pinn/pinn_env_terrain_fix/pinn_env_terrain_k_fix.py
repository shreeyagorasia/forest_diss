# FIX EXPERIMENT (2026-08-20) -- copy of models/pinn_env_terrain_k/pinn_env_terrain_k.py with
# ONE real change: forward() now routes BOTH the per-plot y_max AND k through to the prediction,
# not just to the physics/trajectory losses. See temp_results_pinn/PLAN.md and
# pinn_env_terrain_fix.py's own header for the full reasoning -- identical mechanism here, just
# with k_per_row also feeding the CR term (p stays global/frozen, same as the original).
#
# Prediction becomes: H_pred_i = y_max_i * (1 - exp(-k_i*age_i))^p + trunk_residual_i
#
# ISOLATION: same rules as pinn_env_terrain_fix.py -- nothing under models/ or outputs/ touched.
# freeze_y_max ablation option dropped here (not needed for this quicktest; the original file's
# own default is False everywhere this experiment uses it).

import itertools
import json
import time
from datetime import datetime, timezone

import pandas as pd
import torch
import torch.nn as nn

from models.common.torch_model import NoEnvNetwork, YMaxSubNetwork, chapman_richards_derivative, compute_l1_penalty
from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_fix import chapman_richards_value

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
K_SUBNETWORK_HIDDEN_SIZE = 16


class EnvTerrainRatePINN(nn.Module):
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
        self.k_subnetwork = YMaxSubNetwork(
            n_terrain_features=n_terrain_features, hidden_size=K_SUBNETWORK_HIDDEN_SIZE,
            dropout_rate=dropout_rate,
        )
        self.cr_params = cr_params
        self.scaler_age = scaler_age
        self.scaler_height = scaler_height

    def forward(self, other_features, age, terrain_features):
        # FIX: same mechanism as pinn_env_terrain_fix.py, now with k_per_row also feeding the
        # CR term (not just y_max_per_row).
        trunk_residual_scaled = self.main_network(other_features, age)

        y_max_per_row = compute_plot_specific_y_max(self, terrain_features, self.cr_params["y_max"])
        k_per_row = compute_plot_specific_k(self, terrain_features, self.cr_params["k"])

        # Mistake #10 (PLAN.md): same age tensor, no detach -- physics loss differentiates the
        # whole forward() output w.r.t. age, needs the CR term's contribution captured too.
        age_unscaled = age * self.scaler_age.scale_[0] + self.scaler_age.mean_[0]

        cr_value = chapman_richards_value(age_unscaled, y_max_per_row, k_per_row, self.cr_params["p"])
        cr_value_scaled = (cr_value - self.scaler_height.mean_[0]) / self.scaler_height.scale_[0]

        return cr_value_scaled + trunk_residual_scaled


def build_model(
    n_other_features, n_terrain_features, cr_params, scaler_age, scaler_height, device, seed,
    dropout_rate=0.0, hidden_layer_sizes=None,
):
    torch.manual_seed(seed)
    model = EnvTerrainRatePINN(
        n_other_features=n_other_features, n_terrain_features=n_terrain_features,
        cr_params=cr_params, scaler_age=scaler_age, scaler_height=scaler_height,
        dropout_rate=dropout_rate, hidden_layer_sizes=hidden_layer_sizes,
    )
    return model.to(device)


def compute_plot_specific_y_max(model, terrain_batch, global_y_max):
    y_max_adjustment = model.y_max_subnetwork(terrain_batch)
    return global_y_max + y_max_adjustment


def compute_plot_specific_k(model, terrain_batch, global_k):
    k_log_adjustment = model.k_subnetwork(terrain_batch)
    return global_k * torch.exp(k_log_adjustment)


def compute_physics_loss(model, age_batch, other_batch, terrain_batch, cr_params, scaler_age, scaler_height):
    # FIX: model(...) call needs terrain_batch now (forward()'s new required argument).
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
    k_per_row = compute_plot_specific_k(model, terrain_batch, cr_params["k"])
    cr_growth_rate = chapman_richards_derivative(age_unscaled, y_max_per_row, k_per_row, cr_params["p"])

    physics_loss = torch.mean((height_wrt_age - cr_growth_rate) ** 2)
    return physics_loss


def compute_trajectory_loss(
    model, age_earlier, other_earlier, age_later, other_later, delta_age, age_mid, terrain_pairs,
    cr_params, scaler_age, scaler_height,
):
    # FIX: both model(...) calls need terrain_pairs now.
    predicted_earlier_scaled = model(other_earlier, age_earlier, terrain_pairs)
    predicted_later_scaled = model(other_later, age_later, terrain_pairs)

    predicted_earlier = predicted_earlier_scaled * scaler_height.scale_[0] + scaler_height.mean_[0]
    predicted_later = predicted_later_scaled * scaler_height.scale_[0] + scaler_height.mean_[0]

    predicted_growth_rate = (predicted_later - predicted_earlier) / delta_age
    y_max_per_row = compute_plot_specific_y_max(model, terrain_pairs, cr_params["y_max"])
    k_per_row = compute_plot_specific_k(model, terrain_pairs, cr_params["k"])
    cr_growth_rate_at_mid = chapman_richards_derivative(age_mid, y_max_per_row, k_per_row, cr_params["p"])

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
    # FIX: needs terrain_features now.
    model.eval()
    with torch.no_grad():
        predicted_height_scaled = model(other_features, age, terrain_features)
    return predicted_height_scaled


def predict_y_max(model, terrain_features, global_y_max):
    model.eval()
    with torch.no_grad():
        y_max_per_row = compute_plot_specific_y_max(model, terrain_features, global_y_max)
    return y_max_per_row


def predict_k(model, terrain_features, global_k):
    model.eval()
    with torch.no_grad():
        k_per_row = compute_plot_specific_k(model, terrain_features, global_k)
    return k_per_row


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
