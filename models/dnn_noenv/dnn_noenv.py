# Plain DNN, no-environment feature set. Same architecture as the PINN
# (models/common/torch_model.py::NoEnvNetwork) with plain MSE loss and no
# physics term at all -- see
# documentation/model_instructions/age_only_dnn_pinn_instructions.md.
#
# This file only knows how to build, train, and save/load the DNN. It does
# NOT load data or evaluate on the test set -- that split is deliberate:
# models/dnn_noenv/run_dnn_noenv.py (fit only, meant to run on the SLURM
# cluster where the GPU is) and models/dnn_noenv/evaluate_dnn_noenv.py
# (evaluate only, cheap enough to run locally on a laptop CPU) are two
# separate scripts, so a GPU job is never spent on the cheap part.

import json
import time
from datetime import datetime, timezone

import pandas as pd
import torch

from models.common.torch_model import NoEnvNetwork, compute_l1_penalty

# ----- Fixed hyperparameters for this model -----
# These are not swept or tuned in this project (see the instructions doc
# for why) -- they are simple constants so every part of the code, and
# every log file, uses the exact same numbers.
L1_COEFFICIENT = 1e-5
LEARNING_RATE = 0.0001
LR_SCHEDULER_FACTOR = 0.8
LR_SCHEDULER_PATIENCE = 15
BATCH_SIZE = 512

# L2 penalty built into the optimizer itself (distinct from L1_COEFFICIENT
# above, which is added to the loss manually) -- a second, standard way to
# discourage the network from fitting training-set noise. Kept small and the
# same order of magnitude as L1_COEFFICIENT so the two don't fight each
# other or squash genuine learning.
WEIGHT_DECAY = 1e-5

# Caps how large a single weight update can be, regardless of how big the
# loss gradient is that step. Cheap insurance against one noisy/unlucky
# batch causing a big, destabilising jump -- exactly the kind of thing that
# could explain a val_loss that suddenly climbs (see the DNN/4survey
# overfitting case in documentation/experiment_log.md).
GRAD_CLIP_MAX_NORM = 1.0

# How many of the most recent epochs' val_loss to average together before
# deciding "is this a new best" / "has training stopped improving". A
# single epoch's val_loss is noisy (see the same experiment_log.md
# discussion); averaging over a short window stops one lucky or unlucky
# epoch from being mistaken for real progress or real stalling.
VAL_LOSS_SMOOTHING_WINDOW = 5

# How often to PRINT progress during training (every 10 epochs, not every
# epoch). This only affects what gets printed to the screen / the SLURM
# .out file -- every single epoch's numbers are still saved to
# training_history.csv regardless, so nothing is ever lost, this just
# keeps the printed log readable over a run of hundreds of epochs.
PRINT_EVERY_N_EPOCHS = 10


def build_model(n_other_features, device, seed):
    # torch.manual_seed controls the random starting weights, so the same
    # seed always gives the same initial network -- matches the seed=42
    # convention already used by plot_level_split/spatial_block_split/RF
    # elsewhere in this codebase.
    torch.manual_seed(seed)
    model = NoEnvNetwork(n_other_features=n_other_features)
    return model.to(device)


def train_one_epoch(model, optimizer, age_train, other_train, target_train, batch_size, device):
    # One epoch = one full pass over every training row, split into small
    # batches. This function does that pass ONCE and returns the average
    # training loss over all the batches in that pass.
    #
    # Manually shuffles and batches the training rows -- this dataset is
    # small enough (well under a million rows) to keep entirely on the
    # device already, so there is no need for a DataLoader/Dataset wrapper
    # just to iterate over batches. torch.randperm makes a random shuffle
    # of row numbers, so every epoch sees the rows in a different order.
    model.train()  # tells the network "we are training now", not evaluating

    n_rows = age_train.shape[0]
    shuffled_row_order = torch.randperm(n_rows, device=device)

    total_data_loss = 0.0
    n_batches = 0

    # Walk through the shuffled rows, batch_size rows at a time.
    for batch_start in range(0, n_rows, batch_size):
        batch_row_indices = shuffled_row_order[batch_start:batch_start + batch_size]

        age_batch = age_train[batch_row_indices]
        other_batch = other_train[batch_row_indices]
        target_batch = target_train[batch_row_indices]

        # Standard PyTorch training step: clear old gradients, make a
        # prediction, work out how wrong it was (the loss), then use that
        # loss to update the network's weights a small amount.
        optimizer.zero_grad()

        predicted_height = model(other_batch, age_batch)
        data_loss = torch.mean((predicted_height - target_batch) ** 2)  # plain MSE
        l1_loss = L1_COEFFICIENT * compute_l1_penalty(model)
        total_loss = data_loss + l1_loss

        total_loss.backward()  # work out how much each weight contributed to the loss
        # Shrinks the gradient if its overall size is above GRAD_CLIP_MAX_NORM,
        # so one noisy batch can never cause an oversized weight update.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()  # nudge every weight a little bit to reduce the loss

        total_data_loss = total_data_loss + data_loss.item()
        n_batches = n_batches + 1

    average_data_loss = total_data_loss / n_batches
    return average_data_loss


def evaluate_on_validation_set(model, age_val, other_val, target_val):
    # Checks how well the CURRENT network does on the validation rows
    # (2021 survey), without updating any weights. Used every epoch to
    # decide whether training is still improving (early stopping) -- this
    # is NOT the final test-set evaluation, which happens later in a
    # separate script.
    model.eval()  # tells the network "we are evaluating now", not training
    with torch.no_grad():  # no need to track gradients when we're not training
        predicted_height = model(other_val, age_val)
        val_loss = torch.mean((predicted_height - target_val) ** 2)
    return val_loss.item()


def build_optimizer(model, optimizer_name):
    # "adam" is the default used everywhere so far. "sgd_momentum" is a
    # genuinely different optimizer (not an addition to Adam -- Adam
    # already has momentum built in, via its beta1=0.9 first-moment
    # estimate) offered as an easy A/B test: SGD with Nesterov momentum
    # sometimes finds flatter, better-generalizing minima than Adam does.
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif optimizer_name == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer_name: {optimizer_name!r}")


def fit(
    age_train, other_train, target_train,
    age_val, other_val, target_val,
    n_other_features, device, seed,
    max_epochs, early_stopping_patience,
    optimizer_name="adam",
):
    # Trains the DNN, one epoch at a time, stopping early if the
    # validation loss stops improving. Returns:
    #   - best_model: the network from whichever epoch had the LOWEST
    #     SMOOTHED validation loss (not necessarily the last epoch, and not
    #     necessarily the single lowest raw epoch either -- see
    #     VAL_LOSS_SMOOTHING_WINDOW above)
    #   - final_model_state: the network's weights from the VERY LAST
    #     epoch trained, saved too in case it's ever useful to compare
    #     against the best one
    #   - history_df: one row per epoch actually trained, with the loss
    #     numbers, learning rate, and elapsed training time at that point
    training_start_time = time.time()
    model = build_model(n_other_features, device, seed)
    optimizer = build_optimizer(model, optimizer_name)
    # ReduceLROnPlateau automatically shrinks the learning rate once
    # validation loss stops improving for a while -- this often helps a
    # network fine-tune once it's close to its best result.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )

    best_smoothed_val_loss = None
    best_model_state = None
    epochs_without_improvement = 0
    history_rows = []
    # Keeps only the most recent VAL_LOSS_SMOOTHING_WINDOW raw val_loss
    # values -- recent_val_losses.pop(0) below drops the oldest one once
    # the window is full, so this is always a short rolling window, not the
    # whole history.
    recent_val_losses = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, optimizer, age_train, other_train, target_train, BATCH_SIZE, device)
        val_loss = evaluate_on_validation_set(model, age_val, other_val, target_val)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # elapsed_seconds is the time since training started, not the time
        # for this one epoch -- so plotting loss against elapsed_seconds
        # shows how training speed changes on the actual machine it ran on
        # (e.g. comparing a cluster GPU run to a laptop CPU sanity check).
        elapsed_seconds = time.time() - training_start_time

        # Smoothed val_loss: the average of the last VAL_LOSS_SMOOTHING_WINDOW
        # epochs (fewer, early on, before the window fills up). Used instead
        # of the raw per-epoch val_loss for "is this a new best" / "has
        # training stalled" below, so a single noisy epoch can't trigger a
        # false best or falsely count toward early stopping.
        recent_val_losses.append(val_loss)
        if len(recent_val_losses) > VAL_LOSS_SMOOTHING_WINDOW:
            recent_val_losses.pop(0)
        smoothed_val_loss = sum(recent_val_losses) / len(recent_val_losses)

        # Save EVERY epoch's numbers to the history list -- this becomes
        # training_history.csv later, with nothing skipped, regardless of
        # how often we print below.
        history_rows.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_loss_smoothed": smoothed_val_loss,
            "learning_rate": current_lr, "elapsed_seconds": elapsed_seconds,
        })

        # Has this epoch beaten every previous epoch's SMOOTHED validation
        # loss? (Not the raw val_loss -- see above.)
        is_new_best = best_smoothed_val_loss is None or smoothed_val_loss < best_smoothed_val_loss
        if is_new_best:
            best_smoothed_val_loss = smoothed_val_loss
            # .clone() makes an independent copy of the weights -- without
            # it, best_model_state would just keep pointing at the SAME
            # weights that keep changing every epoch, and by the end it
            # would not actually be "the best" weights any more.
            best_model_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement = epochs_without_improvement + 1

        # Only PRINT every PRINT_EVERY_N_EPOCHS epochs (plus always print
        # epoch 1, so you can see straight away that training has started
        # and the numbers look sane). Every epoch is still in
        # history_rows above either way -- this only controls what shows
        # up on the screen / in the SLURM .out file.
        should_print_this_epoch = (epoch == 1) or (epoch % PRINT_EVERY_N_EPOCHS == 0)
        if should_print_this_epoch:
            best_marker = " (new best)" if is_new_best else ""
            print(
                f"  epoch {epoch}/{max_epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_loss_smoothed={smoothed_val_loss:.4f}  "
                f"learning_rate={current_lr:.6f}{best_marker}"
            )

        # Stop early if validation loss has not improved for
        # early_stopping_patience epochs in a row -- carries on training
        # for no reason once the model has stopped getting better.
        if epochs_without_improvement >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} (no val_loss improvement for {early_stopping_patience} epochs).")
            break

    final_model_state = {key: value.clone() for key, value in model.state_dict().items()}

    # Build a fresh model and load the BEST weights into it (not the final
    # epoch's weights) -- this is the model that actually gets returned
    # and used for predictions later.
    best_model = build_model(n_other_features, device, seed)
    best_model.load_state_dict(best_model_state)

    history_df = pd.DataFrame(history_rows)
    return best_model, final_model_state, history_df


def predict(model, age, other_features):
    # Runs the network forward to get predictions, without training it.
    # Returns SCALED predictions -- the caller is responsible for
    # unscaling them back into real metres using scaler_height.
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


def load_best_model(n_other_features, device, checkpoint_dir):
    # The other half of save_checkpoints(): rebuilds the network
    # architecture, then loads the saved best-epoch weights into it. Used
    # by evaluate_dnn_noenv.py, which never calls fit() at all -- it only
    # needs a trained model to make predictions with.
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
