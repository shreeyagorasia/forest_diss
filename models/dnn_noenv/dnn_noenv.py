# Plain DNN, no-environment feature set. Same architecture as the PINN
# (models/common/torch_model.py::NoEnvNetwork) with plain MSE loss and no
# physics term at all -- see
# documentation/model_instructions/age_only_dnn_pinn_instructions.md.

import json
from datetime import datetime, timezone

import pandas as pd
import torch

from models.common.torch_model import NoEnvNetwork, compute_l1_penalty

L1_COEFFICIENT = 1e-5
LEARNING_RATE = 0.0001
LR_SCHEDULER_FACTOR = 0.8
LR_SCHEDULER_PATIENCE = 10
BATCH_SIZE = 128


def build_model(n_other_features, device, seed):
    # torch.manual_seed controls the random starting weights, so the same
    # seed always gives the same initial network -- matches the seed=42
    # convention already used by plot_level_split/spatial_block_split/RF
    # elsewhere in this codebase.
    torch.manual_seed(seed)
    model = NoEnvNetwork(n_other_features=n_other_features)
    return model.to(device)


def train_one_epoch(model, optimizer, age_train, other_train, target_train, batch_size, device):
    # Manually shuffles and batches the training rows each epoch -- this
    # dataset is small enough (well under a million rows) to keep entirely
    # on the device already, so there is no need for a DataLoader/Dataset
    # wrapper just to iterate over batches.
    model.train()

    n_rows = age_train.shape[0]
    shuffled_row_order = torch.randperm(n_rows, device=device)

    total_data_loss = 0.0
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
        optimizer.step()

        total_data_loss = total_data_loss + data_loss.item()
        n_batches = n_batches + 1

    average_data_loss = total_data_loss / n_batches
    return average_data_loss


def evaluate_on_validation_set(model, age_val, other_val, target_val):
    model.eval()
    with torch.no_grad():
        predicted_height = model(other_val, age_val)
        val_loss = torch.mean((predicted_height - target_val) ** 2)
    return val_loss.item()


def fit(
    age_train, other_train, target_train,
    age_val, other_val, target_val,
    n_other_features, device, seed,
    max_epochs, early_stopping_patience,
):
    # Trains the DNN with early stopping on validation loss. Returns the
    # best model (by validation loss, not necessarily the final epoch) and
    # a per-epoch training history dataframe.
    model = build_model(n_other_features, device, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )

    best_val_loss = None
    best_model_state = None
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, optimizer, age_train, other_train, target_train, BATCH_SIZE, device)
        val_loss = evaluate_on_validation_set(model, age_val, other_val, target_val)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history_rows.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": current_lr,
        })

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement = epochs_without_improvement + 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} (no val_loss improvement for {early_stopping_patience} epochs).")
            break

    final_model_state = {key: value.clone() for key, value in model.state_dict().items()}

    best_model = build_model(n_other_features, device, seed)
    best_model.load_state_dict(best_model_state)

    history_df = pd.DataFrame(history_rows)
    return best_model, final_model_state, history_df


def predict(model, age, other_features):
    model.eval()
    with torch.no_grad():
        predicted_height_scaled = model(other_features, age)
    return predicted_height_scaled


def save_checkpoints(best_model, final_model_state, n_other_features, output_dir):
    # n_other_features is saved alongside the weights (not just implied by
    # them) so a fresh script -- possibly on a different machine, evaluating
    # a checkpoint trained on the SLURM cluster -- can rebuild the exact
    # same NoEnvNetwork(n_other_features=...) architecture before calling
    # load_state_dict(), without needing to re-run the data pipeline first
    # just to find that number out.
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), output_dir / "best_model.pt")
    torch.save(final_model_state, output_dir / "final_model.pt")
    with open(output_dir / "architecture.json", "w") as f:
        json.dump({"n_other_features": n_other_features}, f, indent=2)


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
