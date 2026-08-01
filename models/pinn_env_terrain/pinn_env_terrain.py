# CR-PINN, WITH terrain/wind conditioning `y_max`. Same age/no-env-feature main network as
# pinn_noenv.py (models/common/torch_model.py::NoEnvNetwork, completely unchanged), plus a small
# extra sub-network (models/common/torch_model.py::YMaxSubNetwork) that turns the chosen
# terrain/wind feature set (models/common/torch_data.py::ENV_TERRAIN_FEATURE_SETS, picked by name
# via run_pinn_env_terrain.py's --feature-set) into a plot-specific ADJUSTMENT to the
# Chapman-Richards curve's y_max -- replacing the single global y_max constant pinn_noenv uses in
# its physics/trajectory loss with y_max_i = global_y_max + adjustment_i.
#
# Design choice (deliberate, not default -- see progress_notes.md's Env-PINN discussion and
# models/common/torch_model.py::YMaxSubNetwork's own note): terrain/wind conditions ONLY y_max,
# never the main network's inputs directly, and k/p stay global, frozen floats, never
# plot-specific. This keeps the dissertation's actual claim clean -- environment determines the
# growth CEILING, not the trajectory shape -- rather than diluting it into "terrain is just
# another generic input mixed in with everything else", which is already what XGBoost/SHAP
# established (models/xgb_environmental/) and isn't the novel contribution here.
#
# Consequence worth stating plainly: because terrain/wind ONLY reaches the model through the
# physics/trajectory loss terms, the physics_weight/trajectory_weight choice matters MORE here
# than it did for pinn_noenv -- at w=0, the y_max sub-network gets no gradient at all and
# terrain/wind would be completely unused. The no-env Stage 3 weight decision (low/zero weight
# wins) is explicitly NOT assumed to carry over here -- this file defaults to physics_weight=
# trajectory_weight=1.0 (the untested base case, same "what base case means" convention
# pinn_noenv.py uses), and a dedicated env_terrain physics-weight sweep is still-open work (see
# the handover), not resolved by this file.
#
# Same fit-only/evaluate-only split as every other model in this repo -- this file only builds,
# trains, and saves/loads; run_pinn_env_terrain.py (cluster) and evaluate_pinn_env_terrain.py
# (local CPU) are the two separate scripts that actually use it.

import itertools
import json
import time
from datetime import datetime, timezone

import pandas as pd
import torch
import torch.nn as nn

from models.common.torch_model import NoEnvNetwork, YMaxSubNetwork, chapman_richards_derivative, compute_l1_penalty

# ----- Fixed hyperparameters -- identical to pinn_noenv.py's, on purpose (same reasoning: any
# difference in results should come from the terrain-conditioned y_max, not an unrelated knob). -----
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

# The y_max sub-network's own hidden layer size -- much smaller than the main network's 128,
# since it only has 5 terrain/wind inputs and one simple job (a single scalar adjustment), not
# the main network's job of learning the whole growth trajectory shape.
Y_MAX_SUBNETWORK_HIDDEN_SIZE = 16


class EnvTerrainPINN(nn.Module):
    # Bundles the two sub-networks into one module so a single state_dict/save/load covers both
    # -- PyTorch already tracks a sub-module's parameters automatically once it's an attribute,
    # so this needs no special handling beyond normal nn.Module composition.
    def __init__(self, n_other_features, n_terrain_features, dropout_rate=0.0):
        super().__init__()
        self.main_network = NoEnvNetwork(n_other_features=n_other_features, dropout_rate=dropout_rate)
        self.y_max_subnetwork = YMaxSubNetwork(
            n_terrain_features=n_terrain_features, hidden_size=Y_MAX_SUBNETWORK_HIDDEN_SIZE,
            dropout_rate=dropout_rate,
        )

    def forward(self, other_features, age):
        # Only the main network answers an ordinary "predict height" call -- the y_max
        # sub-network is only ever called directly (see compute_physics_loss/
        # compute_trajectory_loss below), never as part of this forward pass, since it isn't
        # part of the height PREDICTION itself, only the physics loss's target.
        return self.main_network(other_features, age)


def build_model(n_other_features, n_terrain_features, device, seed, dropout_rate=0.0):
    torch.manual_seed(seed)
    model = EnvTerrainPINN(
        n_other_features=n_other_features, n_terrain_features=n_terrain_features, dropout_rate=dropout_rate,
    )
    return model.to(device)


def compute_plot_specific_y_max(model, terrain_batch, global_y_max):
    # global_y_max is a plain Python float (the frozen cr_pooled value, same source pinn_noenv
    # uses) -- adding a tensor adjustment to it is fine and keeps the RESULT a tensor (part of
    # the autograd graph, so gradients flow back into the y_max sub-network's weights), while
    # global_y_max itself is never a tensor and so can never itself accumulate a gradient --
    # same "the process model's known constants can never be updated by training" guarantee
    # pinn_noenv.py's frozen k/p already rely on.
    y_max_adjustment = model.y_max_subnetwork(terrain_batch)
    return global_y_max + y_max_adjustment


def compute_physics_loss(model, age_batch, other_batch, terrain_batch, cr_params, scaler_age, scaler_height):
    # Same instantaneous-derivative mechanism as pinn_noenv.py's compute_physics_loss -- the one
    # change is the CR target uses a PER-ROW y_max (from the sub-network) instead of a single
    # global float, which chapman_richards_derivative() supports with no change needed (elementwise
    # tensor broadcasting against age_unscaled, which is the same shape).
    age_batch = age_batch.clone().requires_grad_(True)

    predicted_height_scaled = model(other_batch, age_batch)

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
    # Same finite-difference mechanism as pinn_noenv.py -- terrain_pairs is ONE tensor per pair
    # (not separate earlier/later versions), since terrain/wind is a static per-plot property
    # that doesn't change between a pair's two survey years (see
    # models/common/torch_data.py::build_pair_terrain_tensor()).
    predicted_earlier_scaled = model(other_earlier, age_earlier)
    predicted_later_scaled = model(other_later, age_later)

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

        predicted_height = model(other_batch, age_batch)
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


def evaluate_on_validation_set(model, age_val, other_val, target_val):
    # Validation loss is data_loss only, same reasoning as pinn_noenv.py: trajectory pairs never
    # touch val/test by construction, and early stopping should track prediction accuracy, not
    # physics-term agreement.
    model.eval()
    with torch.no_grad():
        predicted_height = model(other_val, age_val)
        val_loss = torch.mean((predicted_height - target_val) ** 2)
    return val_loss.item()


def build_optimizer(model, optimizer_name):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif optimizer_name == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer_name: {optimizer_name!r}")


def fit(
    age_train, other_train, terrain_train, target_train,
    age_val, other_val, target_val,
    pair_tensors, terrain_pairs_all, cr_params, scaler_age, scaler_height,
    n_other_features, n_terrain_features, device, seed,
    max_epochs, early_stopping_patience,
    optimizer_name="adam",
    physics_weight=PHYSICS_WEIGHT, trajectory_weight=TRAJECTORY_WEIGHT,
    batch_size=BATCH_SIZE, pairs_batch_size=PAIRS_BATCH_SIZE,
    dropout_rate=0.0,
):
    training_start_time = time.time()
    model = build_model(n_other_features, n_terrain_features, device, seed, dropout_rate=dropout_rate)
    optimizer = build_optimizer(model, optimizer_name)
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
        val_loss = evaluate_on_validation_set(model, age_val, other_val, target_val)
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

    best_model = build_model(n_other_features, n_terrain_features, device, seed, dropout_rate=dropout_rate)
    best_model.load_state_dict(best_model_state)

    history_df = pd.DataFrame(history_rows)
    return best_model, final_model_state, history_df


def predict(model, age, other_features):
    model.eval()
    with torch.no_grad():
        predicted_height_scaled = model(other_features, age)
    return predicted_height_scaled


def predict_y_max(model, terrain_features, global_y_max):
    # Not used for the main height prediction (predict() above already covers that) -- exposed
    # separately so evaluate_pinn_env_terrain.py (or a future notebook) can inspect the LEARNED
    # plot-specific y_max map directly, the actual interpretable output this whole model exists
    # to produce.
    model.eval()
    with torch.no_grad():
        y_max_per_row = compute_plot_specific_y_max(model, terrain_features, global_y_max)
    return y_max_per_row


def save_checkpoints(best_model, final_model_state, n_other_features, n_terrain_features, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), output_dir / "best_model.pt")
    torch.save(final_model_state, output_dir / "final_model.pt")
    with open(output_dir / "architecture.json", "w") as f:
        json.dump({"n_other_features": n_other_features, "n_terrain_features": n_terrain_features}, f, indent=2)


def load_best_model(n_other_features, n_terrain_features, device, checkpoint_dir):
    model = EnvTerrainPINN(n_other_features=n_other_features, n_terrain_features=n_terrain_features)
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device))
    model.to(device)
    return model


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
