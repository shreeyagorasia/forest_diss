# CR-PINN, WITH terrain/wind conditioning BOTH `y_max` AND `k` (the Chapman-Richards growth-rate
# parameter). A new, separate model from models/pinn_env_terrain/pinn_env_terrain.py -- not an
# edit to it, per an explicit 2026-08-03 instruction to keep this fully separate from that
# model AND from the unrelated per-plot growth-curve work in models/growth_curve_attribution/
# (see documentation/model_instructions/growth_curve_stage2_handover.md -- that is a different,
# parallel investigation, not touched by anything in this file).
#
# WHY k, not p (2026-08-03, supervisor-prompted): pinn_env_terrain.py's own YMaxSubNetwork
# docstring already documents a deliberate prior choice -- terrain/wind conditions ONLY y_max,
# citing Socha et al. 2021's ADA/GADA framework ("the asymptote parameter is the site-varying
# one"). This file is a real, deliberate departure from that citation, not a simple missing
# feature -- prompted by the supervisor asking, in plain language, whether an exponent could be
# environment-conditioned so that "all trees reach their height eventually but not at the same
# time". Checked the actual ecological literature before picking a parameter (not guessed from
# the maths alone): k is described as "an empirical growth parameter scaling the absolute growth
# rate" -- directly the kind of thing site quality/environment would plausibly affect. p is tied
# to catabolic/metabolic-scaling theory in the classical von Bertalanffy-style derivation and is
# "often restricted to a value... for theoretical, biological reasons" -- treated in the
# literature as a fixed biological constant, not something expected to vary by site. The broader
# forestry site-index/GADA literature (polymorphic curve families) independently confirms site
# quality classically varies the asymptote and/or growth rate, not the shape exponent. So: k gets
# a new per-plot adjustment here, alongside the EXISTING y_max adjustment (unchanged mechanism,
# reused from pinn_env_terrain.py) -- p stays global and frozen, same as pinn_env_terrain.py.
#
# k must stay strictly positive (it sits inside exp(-k*age); a non-positive k breaks the growth
# curve and risks numerical blow-up during training). Parameterised multiplicatively in log-space
# (k_per_row = global_k * exp(k_log_adjustment)), not additively like y_max's adjustment -- this
# guarantees positivity for any real-valued sub-network output, and a fresh sub-network
# (near-zero output at initialisation) starts at k_log_adjustment≈0, i.e. k_per_row≈global_k,
# same "start near the known-good global constant" stability reasoning y_max's adjustment already
# uses.
#
# Real, open risk worth checking once this is fit, not assumed away: y_max and k are both learned
# per-plot from the SAME terrain inputs via two separate small sub-networks -- they could become
# confounded (a lower k mimicking what should really be a lower y_max, especially for plots that
# never reach anywhere near their true asymptote within the observed age range, common in this
# dataset). predict_k()/predict_y_max() below both exist so this can be checked directly
# (correlation between the two learned adjustments across plots) rather than assumed fine.
#
# Same fit-only/evaluate-only split as every other model in this repo.

import itertools
import json
import time
from datetime import datetime, timezone

import pandas as pd
import torch
import torch.nn as nn

from models.common.torch_model import NoEnvNetwork, YMaxSubNetwork, chapman_richards_derivative, compute_l1_penalty

# ----- Fixed hyperparameters -- identical to pinn_env_terrain.py's, on purpose (same reasoning:
# any difference in results should come from the new k-conditioning, not an unrelated knob). -----
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

# Same size as pinn_env_terrain.py's y_max sub-network -- a small, separate job (one scalar
# adjustment from a handful of terrain inputs), not a reason to need more capacity.
Y_MAX_SUBNETWORK_HIDDEN_SIZE = 16
K_SUBNETWORK_HIDDEN_SIZE = 16


class EnvTerrainRatePINN(nn.Module):
    # Bundles three sub-modules (main network, y_max sub-network, k sub-network) so a single
    # state_dict/save/load covers all three, same PyTorch composition reasoning
    # pinn_env_terrain.py's EnvTerrainPINN already uses.
    def __init__(self, n_other_features, n_terrain_features, dropout_rate=0.0, hidden_layer_sizes=None):
        super().__init__()
        self.main_network = NoEnvNetwork(
            n_other_features=n_other_features, dropout_rate=dropout_rate, hidden_layer_sizes=hidden_layer_sizes,
        )
        self.y_max_subnetwork = YMaxSubNetwork(
            n_terrain_features=n_terrain_features, hidden_size=Y_MAX_SUBNETWORK_HIDDEN_SIZE,
            dropout_rate=dropout_rate,
        )
        # Reusing YMaxSubNetwork's class for the k-adjustment too -- it's already a generic
        # "terrain features -> one scalar" network, nothing about its name stops a second,
        # separately-initialised instance being used for a different scalar. Not renamed to a
        # more generic class name because pinn_env_terrain.py imports YMaxSubNetwork by this
        # exact name and that file is explicitly not being touched.
        self.k_subnetwork = YMaxSubNetwork(
            n_terrain_features=n_terrain_features, hidden_size=K_SUBNETWORK_HIDDEN_SIZE,
            dropout_rate=dropout_rate,
        )

    def forward(self, other_features, age):
        # Same as pinn_env_terrain.py: only the main network answers an ordinary "predict
        # height" call -- both sub-networks are only ever called directly (see
        # compute_physics_loss/compute_trajectory_loss below), never as part of this forward
        # pass, since neither is part of the height PREDICTION itself, only the physics loss's
        # target construction.
        return self.main_network(other_features, age)


def build_model(n_other_features, n_terrain_features, device, seed, dropout_rate=0.0, hidden_layer_sizes=None):
    torch.manual_seed(seed)
    model = EnvTerrainRatePINN(
        n_other_features=n_other_features, n_terrain_features=n_terrain_features,
        dropout_rate=dropout_rate, hidden_layer_sizes=hidden_layer_sizes,
    )
    return model.to(device)


def compute_plot_specific_y_max(model, terrain_batch, global_y_max):
    # Identical mechanism to pinn_env_terrain.py's own function of the same name -- additive
    # adjustment, global_y_max stays a plain Python float (never accumulates gradient itself).
    y_max_adjustment = model.y_max_subnetwork(terrain_batch)
    return global_y_max + y_max_adjustment


def compute_plot_specific_k(model, terrain_batch, global_k):
    # Multiplicative, log-space adjustment -- see this file's own top-of-file note for why k
    # needs a different parameterisation to y_max's additive one (k must stay strictly positive;
    # exp() of any real-valued sub-network output is always positive; near-zero output at
    # initialisation means k_per_row starts at global_k, the same "start near the known-good
    # global constant" stability property y_max's adjustment already has).
    k_log_adjustment = model.k_subnetwork(terrain_batch)
    return global_k * torch.exp(k_log_adjustment)


def compute_physics_loss(model, age_batch, other_batch, terrain_batch, cr_params, scaler_age, scaler_height, freeze_y_max=False):
    # Same instantaneous-derivative mechanism as pinn_env_terrain.py's compute_physics_loss --
    # the one change is BOTH y_max and k use a per-row value now, not just y_max. p stays the
    # single frozen global float, exactly as in pinn_env_terrain.py.
    #
    # freeze_y_max (2026-08-04 addition, default False): the council's "freeze-one-vary-other"
    # ablation -- when True, y_max is pinned to the plain global constant (never adjusted by
    # y_max_subnetwork, which then gets no gradient at all since it's never called here) so only
    # k is a per-plot free parameter. Tests whether a wide, implausible learned-k range persists
    # even with a single degree of freedom per plot -- if so, that points at overparameterisation
    # of ANY second per-plot knob, not something specific to k.
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
    if freeze_y_max:
        y_max_per_row = cr_params["y_max"]
    else:
        y_max_per_row = compute_plot_specific_y_max(model, terrain_batch, cr_params["y_max"])
    k_per_row = compute_plot_specific_k(model, terrain_batch, cr_params["k"])
    cr_growth_rate = chapman_richards_derivative(age_unscaled, y_max_per_row, k_per_row, cr_params["p"])

    physics_loss = torch.mean((height_wrt_age - cr_growth_rate) ** 2)
    return physics_loss


def compute_trajectory_loss(
    model, age_earlier, other_earlier, age_later, other_later, delta_age, age_mid, terrain_pairs,
    cr_params, scaler_age, scaler_height, freeze_y_max=False,
):
    # Same finite-difference mechanism as pinn_env_terrain.py -- terrain_pairs is ONE tensor per
    # pair, same reasoning (terrain/wind is static per plot across a pair's two survey years).
    # freeze_y_max: see compute_physics_loss's own note above -- same meaning here.
    predicted_earlier_scaled = model(other_earlier, age_earlier)
    predicted_later_scaled = model(other_later, age_later)

    predicted_earlier = predicted_earlier_scaled * scaler_height.scale_[0] + scaler_height.mean_[0]
    predicted_later = predicted_later_scaled * scaler_height.scale_[0] + scaler_height.mean_[0]

    predicted_growth_rate = (predicted_later - predicted_earlier) / delta_age
    if freeze_y_max:
        y_max_per_row = cr_params["y_max"]
    else:
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
    batch_size=BATCH_SIZE, pairs_batch_size=PAIRS_BATCH_SIZE, freeze_y_max=False,
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
            freeze_y_max=freeze_y_max,
        )

        trajectory_loss = compute_trajectory_loss(
            model,
            age_earlier_all[pair_batch_indices], other_earlier_all[pair_batch_indices],
            age_later_all[pair_batch_indices], other_later_all[pair_batch_indices],
            delta_age_all[pair_batch_indices], age_mid_all[pair_batch_indices],
            terrain_pairs_all[pair_batch_indices],
            cr_params, scaler_age, scaler_height, freeze_y_max=freeze_y_max,
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
    age_train, other_train, terrain_train, target_train,
    age_val, other_val, target_val,
    pair_tensors, terrain_pairs_all, cr_params, scaler_age, scaler_height,
    n_other_features, n_terrain_features, device, seed,
    max_epochs, early_stopping_patience,
    optimizer_name="adam",
    physics_weight=PHYSICS_WEIGHT, trajectory_weight=TRAJECTORY_WEIGHT,
    batch_size=BATCH_SIZE, pairs_batch_size=PAIRS_BATCH_SIZE,
    dropout_rate=0.0,
    learning_rate=LEARNING_RATE,
    hidden_layer_sizes=None,
    freeze_y_max=False,
):
    training_start_time = time.time()
    model = build_model(
        n_other_features, n_terrain_features, device, seed,
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
            batch_size=batch_size, pairs_batch_size=pairs_batch_size, freeze_y_max=freeze_y_max,
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

    best_model = build_model(
        n_other_features, n_terrain_features, device, seed,
        dropout_rate=dropout_rate, hidden_layer_sizes=hidden_layer_sizes,
    )
    best_model.load_state_dict(best_model_state)

    history_df = pd.DataFrame(history_rows)
    return best_model, final_model_state, history_df


def predict(model, age, other_features):
    model.eval()
    with torch.no_grad():
        predicted_height_scaled = model(other_features, age)
    return predicted_height_scaled


def predict_y_max(model, terrain_features, global_y_max):
    model.eval()
    with torch.no_grad():
        y_max_per_row = compute_plot_specific_y_max(model, terrain_features, global_y_max)
    return y_max_per_row


def predict_k(model, terrain_features, global_k):
    # Not used for height prediction -- exposed so the learned per-plot k map can be inspected
    # directly, same reasoning as predict_y_max(). Also needed to check the confound risk flagged
    # at the top of this file: correlate this against predict_y_max()'s output across the same
    # plots before trusting the two adjustments as independently meaningful.
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


def load_best_model(n_other_features, n_terrain_features, device, checkpoint_dir, hidden_layer_sizes=None):
    model = EnvTerrainRatePINN(
        n_other_features=n_other_features, n_terrain_features=n_terrain_features,
        hidden_layer_sizes=hidden_layer_sizes,
    )
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
