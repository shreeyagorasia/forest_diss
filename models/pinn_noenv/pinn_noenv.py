# CR-PINN Version 1, no-environment feature set. Same architecture as the
# DNN (models/common/torch_model.py::NoEnvNetwork), same data, but with two
# additional physics loss terms on top of plain MSE -- see
# documentation/model_instructions/age_only_dnn_pinn_instructions.md.
#
# Term 1 (physics_loss, weight lambda_ph): the network's own instantaneous
# growth rate (d(predicted height)/d(age), via autograd, on ordinary
# per-row data) must match the CR curve's analytical derivative at that
# same age.
#
# Term 2 (trajectory_loss, weight lambda_traj): for pairs of real,
# consecutive training-year observations of the SAME plot, the network's
# own predicted height CHANGE between the two must match the CR curve's
# derivative at the pair's mid-age. This is a finite-difference check
# across two real measurements, not an instantaneous derivative at one
# point -- see section 2 of the instructions doc for why these are
# genuinely different checks, not two versions of the same thing.
#
# CR's y_max/k/p are always plain Python floats (never tensors) so they can
# never accumulate a gradient -- training only ever updates the network,
# never the process model.
#
# This file only knows how to build, train, and save/load the PINN. It does
# NOT load data or evaluate on the test set -- that split is deliberate:
# models/pinn_noenv/run_pinn_noenv.py (fit only, meant to run on the SLURM
# cluster where the GPU is) and models/pinn_noenv/evaluate_pinn_noenv.py
# (evaluate only, cheap enough to run locally on a laptop CPU) are two
# separate scripts, so a GPU job is never spent on the cheap part.

import itertools
import json
from datetime import datetime, timezone

import pandas as pd
import torch

from models.common.torch_model import NoEnvNetwork, chapman_richards_derivative, compute_l1_penalty

# ----- Fixed hyperparameters for this model -----
# These are not swept or tuned in this project (see the instructions doc
# for why) -- they are simple constants so every part of the code, and
# every log file, uses the exact same numbers.
L1_COEFFICIENT = 1e-5
PHYSICS_WEIGHT = 1.0
TRAJECTORY_WEIGHT = 1.0
LEARNING_RATE = 0.0001
LR_SCHEDULER_FACTOR = 0.8
LR_SCHEDULER_PATIENCE = 10
BATCH_SIZE = 32
PAIRS_BATCH_SIZE = 32

# How often to PRINT progress during training (every 10 epochs, not every
# epoch). This only affects what gets printed to the screen / the SLURM
# .out file -- every single epoch's numbers (including the three separate
# loss terms) are still saved to training_history.csv regardless, so
# nothing is ever lost, this just keeps the printed log readable over a
# run of hundreds of epochs.
PRINT_EVERY_N_EPOCHS = 10


def build_model(n_other_features, device, seed):
    torch.manual_seed(seed)
    model = NoEnvNetwork(n_other_features=n_other_features)
    return model.to(device)


def compute_physics_loss(model, age_batch, other_batch, cr_params, scaler_age, scaler_height):
    # Instantaneous derivative-consistency term. age_batch must be a FRESH
    # tensor with requires_grad=True set here (not on the original data
    # tensor it was sliced from), otherwise autograd.grad has nothing to
    # differentiate through for this specific forward pass.
    age_batch = age_batch.clone().requires_grad_(True)

    predicted_height_scaled = model(other_batch, age_batch)

    height_wrt_age_scaled = torch.autograd.grad(
        outputs=predicted_height_scaled,
        inputs=age_batch,
        grad_outputs=torch.ones_like(predicted_height_scaled),
        create_graph=True,
    )[0]

    # Undo scaling so the derivative is in real units (metres/year) before
    # comparing it to the CR curve's own derivative.
    scale_factor = scaler_height.scale_[0] / scaler_age.scale_[0]
    height_wrt_age = height_wrt_age_scaled * scale_factor

    # .detach() here: the CR target must never itself be part of the graph
    # that autograd.grad already built above -- y_max/k/p are plain floats
    # with nothing to update, so there is nothing useful to backpropagate
    # through this branch, and detaching keeps the retained graph (needed
    # for create_graph=True) as small as possible.
    age_unscaled = age_batch.detach() * scaler_age.scale_[0] + scaler_age.mean_[0]
    cr_growth_rate = chapman_richards_derivative(age_unscaled, cr_params["y_max"], cr_params["k"], cr_params["p"])

    physics_loss = torch.mean((height_wrt_age - cr_growth_rate) ** 2)
    return physics_loss


def compute_trajectory_loss(
    model, age_earlier, other_earlier, age_later, other_later, delta_age, age_mid, cr_params,
    scaler_age, scaler_height,
):
    # Finite-difference-consistency term, using two REAL forward passes on
    # two real observations of the same plot -- no autograd.grad needed
    # here, since this compares two ordinary predictions, not a derivative.
    predicted_earlier_scaled = model(other_earlier, age_earlier)
    predicted_later_scaled = model(other_later, age_later)

    predicted_earlier = predicted_earlier_scaled * scaler_height.scale_[0] + scaler_height.mean_[0]
    predicted_later = predicted_later_scaled * scaler_height.scale_[0] + scaler_height.mean_[0]

    predicted_growth_rate = (predicted_later - predicted_earlier) / delta_age
    cr_growth_rate_at_mid = chapman_richards_derivative(age_mid, cr_params["y_max"], cr_params["k"], cr_params["p"])

    trajectory_loss = torch.mean((predicted_growth_rate - cr_growth_rate_at_mid) ** 2)
    return trajectory_loss


def train_one_epoch(
    model, optimizer,
    age_train, other_train, target_train,
    pair_tensors, cr_params, scaler_age, scaler_height,
    device,
):
    model.train()

    n_rows = age_train.shape[0]
    shuffled_row_order = torch.randperm(n_rows, device=device)
    main_batch_starts = list(range(0, n_rows, BATCH_SIZE))

    age_earlier_all, other_earlier_all, age_later_all, other_later_all, delta_age_all, age_mid_all, _ = pair_tensors
    n_pairs = age_earlier_all.shape[0]
    shuffled_pair_order = torch.randperm(n_pairs, device=device)
    pair_batch_starts = list(range(0, n_pairs, PAIRS_BATCH_SIZE))

    # There are usually fewer pairs than main rows (one pair per plot vs.
    # several survey-year rows per plot), so the pairs batches are cycled
    # to line up with every main batch -- every training step sees all
    # three loss terms, and "one epoch" still means one full pass over the
    # main per-row training data, same meaning as for the DNN.
    pair_batch_cycle = itertools.cycle(pair_batch_starts)

    totals = {"data_loss": 0.0, "physics_loss": 0.0, "trajectory_loss": 0.0}
    n_batches = 0

    for batch_start in main_batch_starts:
        batch_row_indices = shuffled_row_order[batch_start:batch_start + BATCH_SIZE]
        age_batch = age_train[batch_row_indices]
        other_batch = other_train[batch_row_indices]
        target_batch = target_train[batch_row_indices]

        pair_batch_start = next(pair_batch_cycle)
        pair_batch_indices = shuffled_pair_order[pair_batch_start:pair_batch_start + PAIRS_BATCH_SIZE]

        optimizer.zero_grad()

        predicted_height = model(other_batch, age_batch)
        data_loss = torch.mean((predicted_height - target_batch) ** 2)

        physics_loss = compute_physics_loss(model, age_batch, other_batch, cr_params, scaler_age, scaler_height)

        trajectory_loss = compute_trajectory_loss(
            model,
            age_earlier_all[pair_batch_indices], other_earlier_all[pair_batch_indices],
            age_later_all[pair_batch_indices], other_later_all[pair_batch_indices],
            delta_age_all[pair_batch_indices], age_mid_all[pair_batch_indices],
            cr_params, scaler_age, scaler_height,
        )

        l1_loss = L1_COEFFICIENT * compute_l1_penalty(model)

        total_loss = data_loss + PHYSICS_WEIGHT * physics_loss + TRAJECTORY_WEIGHT * trajectory_loss + l1_loss

        total_loss.backward()
        optimizer.step()

        totals["data_loss"] = totals["data_loss"] + data_loss.item()
        totals["physics_loss"] = totals["physics_loss"] + physics_loss.item()
        totals["trajectory_loss"] = totals["trajectory_loss"] + trajectory_loss.item()
        n_batches = n_batches + 1

    return {key: value / n_batches for key, value in totals.items()}


def evaluate_on_validation_set(model, age_val, other_val, target_val):
    # Validation loss is data_loss only (plain MSE), never the physics or
    # trajectory terms -- trajectory pairs never touch the validation year
    # by construction (see the leakage rule in the instructions doc), and
    # early stopping should track prediction accuracy, not physics-term
    # agreement.
    model.eval()
    with torch.no_grad():
        predicted_height = model(other_val, age_val)
        val_loss = torch.mean((predicted_height - target_val) ** 2)
    return val_loss.item()


def fit(
    age_train, other_train, target_train,
    age_val, other_val, target_val,
    pair_tensors, cr_params, scaler_age, scaler_height,
    n_other_features, device, seed,
    max_epochs, early_stopping_patience,
):
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
        epoch_losses = train_one_epoch(
            model, optimizer,
            age_train, other_train, target_train,
            pair_tensors, cr_params, scaler_age, scaler_height,
            device,
        )
        val_loss = evaluate_on_validation_set(model, age_val, other_val, target_val)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history_rows.append({
            "epoch": epoch,
            "train_loss": epoch_losses["data_loss"] + PHYSICS_WEIGHT * epoch_losses["physics_loss"]
                          + TRAJECTORY_WEIGHT * epoch_losses["trajectory_loss"],
            "data_loss": epoch_losses["data_loss"],
            "physics_loss": epoch_losses["physics_loss"],
            "trajectory_loss": epoch_losses["trajectory_loss"],
            "val_loss": val_loss,
            "learning_rate": current_lr,
        })

        # Has this epoch beaten every previous epoch's validation loss?
        is_new_best = best_val_loss is None or val_loss < best_val_loss
        if is_new_best:
            best_val_loss = val_loss
            # .clone() makes an independent copy of the weights -- without
            # it, best_model_state would just keep pointing at the SAME
            # weights that keep changing every epoch.
            best_model_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement = epochs_without_improvement + 1

        # Only PRINT every PRINT_EVERY_N_EPOCHS epochs (plus always print
        # epoch 1) -- every epoch is still in history_rows above either
        # way. Shows all three loss terms separately so it's visible at a
        # glance whether the physics/trajectory terms are doing anything.
        should_print_this_epoch = (epoch == 1) or (epoch % PRINT_EVERY_N_EPOCHS == 0)
        if should_print_this_epoch:
            best_marker = " (new best)" if is_new_best else ""
            print(
                f"  epoch {epoch}/{max_epochs}  data_loss={epoch_losses['data_loss']:.4f}  "
                f"physics_loss={epoch_losses['physics_loss']:.4f}  "
                f"trajectory_loss={epoch_losses['trajectory_loss']:.4f}  "
                f"val_loss={val_loss:.4f}  learning_rate={current_lr:.6f}{best_marker}"
            )

        # Stop early if validation loss has not improved for
        # early_stopping_patience epochs in a row.
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
    # n_other_features is saved alongside the weights so a fresh script --
    # possibly on a different machine, evaluating a checkpoint trained on
    # the SLURM cluster -- can rebuild the exact same
    # NoEnvNetwork(n_other_features=...) architecture before calling
    # load_state_dict().
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), output_dir / "best_model.pt")
    torch.save(final_model_state, output_dir / "final_model.pt")
    with open(output_dir / "architecture.json", "w") as f:
        json.dump({"n_other_features": n_other_features}, f, indent=2)


def load_best_model(n_other_features, device, checkpoint_dir):
    # The other half of save_checkpoints(): rebuilds the network
    # architecture, then loads the saved best-epoch weights into it. Used
    # by evaluate_pinn_noenv.py, which never calls fit() at all -- it only
    # needs a trained model to make predictions with. Note this loads the
    # PLAIN network only -- the physics/trajectory loss terms are a
    # TRAINING-time concept, nothing to load or reuse at evaluation time.
    model = NoEnvNetwork(n_other_features=n_other_features)
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
