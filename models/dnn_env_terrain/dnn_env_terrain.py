# Plain DNN, WITH terrain/wind features -- the "same information, no physics" control for
# pinn_env_terrain, same role dnn_noenv already plays for pinn_noenv. Reuses
# models/common/torch_model.py::NoEnvNetwork completely unchanged -- that class already accepts
# any n_other_features, so this model's "other features" tensor is just the no-env features PLUS
# the chosen terrain/wind feature set (models/common/torch_data.py::ENV_TERRAIN_FEATURE_SETS,
# picked by name via run_dnn_env_terrain.py's --feature-set) concatenated together, built by
# run_dnn_env_terrain.py, not a different network architecture.
#
# Why terrain feeds this network directly, when pinn_env_terrain's main network does NOT (its
# terrain features go only to the y_max sub-network): fairness. pinn_env_terrain's total
# information budget is age + no-env features (main network) + terrain/wind (y_max
# sub-network) -- for the "does physics help, given the SAME information" comparison to be
# honest, this control needs access to the exact same total information, even though PINN
# structures its own use of it differently (a physics-informed sub-network) as part of what's
# actually being tested. See pinn_env_terrain.py's own top-of-file note for the full reasoning.
#
# This file only knows how to build, train, and save/load the DNN -- same fit-then-evaluate
# split as every other model in this repo (run_dnn_env_terrain.py / evaluate_dnn_env_terrain.py).
#
# DELIBERATE DUPLICATION, not an oversight (2026-08-01): every function below is near-identical
# to dnn_noenv.py's version (only real difference: dropout_rate threaded through) -- kept as a
# separate, self-contained file rather than importing from dnn_noenv.py, matching this project's
# convention of one fully self-contained file per model folder. Accepted cost: a future fix to
# the shared training-loop logic (early stopping, gradient clipping, val-loss smoothing) has to
# be applied by hand in both files -- nothing enforces them staying in sync.

import json
import time
from datetime import datetime, timezone

import pandas as pd
import torch

from models.common.torch_model import NoEnvNetwork, compute_l1_penalty

# ----- Fixed hyperparameters -- identical to dnn_noenv.py's, on purpose. Any difference between
# dnn_noenv and dnn_env_terrain's results should come from the extra terrain/wind features, not
# from an unrelated training-knob difference. -----
L1_COEFFICIENT = 1e-5
LEARNING_RATE = 0.0001
LR_SCHEDULER_FACTOR = 0.8
LR_SCHEDULER_PATIENCE = 15
BATCH_SIZE = 256  # matches PINN/PINN-k's default -- changed from 512 on 2026-08-22 for a fair batch-size comparison
WEIGHT_DECAY = 1e-5
GRAD_CLIP_MAX_NORM = 1.0
VAL_LOSS_SMOOTHING_WINDOW = 5
PRINT_EVERY_N_EPOCHS = 10


def build_model(n_other_features, device, seed, dropout_rate=0.0, hidden_layer_sizes=None):
    # hidden_layer_sizes=None keeps the original 3x128 network -- see
    # models/common/torch_model.py::NoEnvNetwork's own note (2026-08-02).
    torch.manual_seed(seed)
    model = NoEnvNetwork(n_other_features=n_other_features, dropout_rate=dropout_rate, hidden_layer_sizes=hidden_layer_sizes)
    return model.to(device)


def train_one_epoch(model, optimizer, age_train, other_train, target_train, batch_size, device):
    model.train()

    n_rows = age_train.shape[0]
    shuffled_row_order = torch.randperm(n_rows, device=device)

    total_data_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0

    for batch_start in range(0, n_rows, batch_size):
        batch_row_indices = shuffled_row_order[batch_start:batch_start + batch_size]

        age_batch = age_train[batch_row_indices]
        other_batch = other_train[batch_row_indices]
        target_batch = target_train[batch_row_indices]

        optimizer.zero_grad()

        predicted_height = model(other_batch, age_batch)
        data_loss = torch.mean((predicted_height - target_batch) ** 2)
        l1_loss = L1_COEFFICIENT * compute_l1_penalty(model)
        total_loss = data_loss + l1_loss

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        total_data_loss = total_data_loss + data_loss.item()
        total_grad_norm = total_grad_norm + grad_norm.item()
        n_batches = n_batches + 1

    average_data_loss = total_data_loss / n_batches
    average_grad_norm = total_grad_norm / n_batches
    return average_data_loss, average_grad_norm


def evaluate_on_validation_set(model, age_val, other_val, target_val):
    model.eval()
    with torch.no_grad():
        predicted_height = model(other_val, age_val)
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
    age_train, other_train, target_train,
    age_val, other_val, target_val,
    n_other_features, device, seed,
    max_epochs, early_stopping_patience,
    optimizer_name="adam",
    batch_size=BATCH_SIZE,
    dropout_rate=0.0,
    learning_rate=LEARNING_RATE,
    hidden_layer_sizes=None,
):
    training_start_time = time.time()
    model = build_model(n_other_features, device, seed, dropout_rate=dropout_rate, hidden_layer_sizes=hidden_layer_sizes)
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
        train_loss, grad_norm = train_one_epoch(
            model, optimizer, age_train, other_train, target_train, batch_size, device
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
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_loss_smoothed": smoothed_val_loss, "grad_norm": grad_norm,
            "learning_rate": current_lr, "elapsed_seconds": elapsed_seconds,
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
                f"  epoch {epoch}/{max_epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_loss_smoothed={smoothed_val_loss:.4f}  "
                f"learning_rate={current_lr:.6f}{best_marker}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} (no val_loss improvement for {early_stopping_patience} epochs).")
            break

    final_model_state = {key: value.clone() for key, value in model.state_dict().items()}

    best_model = build_model(n_other_features, device, seed, dropout_rate=dropout_rate, hidden_layer_sizes=hidden_layer_sizes)
    best_model.load_state_dict(best_model_state)

    history_df = pd.DataFrame(history_rows)
    return best_model, final_model_state, history_df


def predict(model, age, other_features):
    model.eval()
    with torch.no_grad():
        predicted_height_scaled = model(other_features, age)
    return predicted_height_scaled


def save_checkpoints(best_model, final_model_state, n_other_features, output_dir, hidden_layer_sizes=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), output_dir / "best_model.pt")
    torch.save(final_model_state, output_dir / "final_model.pt")
    with open(output_dir / "architecture.json", "w") as f:
        json.dump({"n_other_features": n_other_features, "hidden_layer_sizes": hidden_layer_sizes}, f, indent=2)


def load_best_model(n_other_features, device, checkpoint_dir, hidden_layer_sizes=None):
    model = NoEnvNetwork(n_other_features=n_other_features, hidden_layer_sizes=hidden_layer_sizes)
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
