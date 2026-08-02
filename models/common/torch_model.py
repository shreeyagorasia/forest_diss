# Shared network architecture for both dnn_noenv and pinn_noenv, so any
# difference between their results comes from the loss function, not from a
# different number of layers or neurons. See
# documentation/model_instructions/age_only_dnn_pinn_instructions.md.

import torch
import torch.nn as nn


class NoEnvNetwork(nn.Module):
    # Default: 3 hidden layers, 128 neurons each, leaky ReLU activation.
    #
    # dropout_rate defaults to 0.0 (2026-08-01 addition) -- OPT-IN, not a default behaviour
    # change. dnn_noenv/pinn_noenv's already-completed, reported runs never pass this argument,
    # so they are completely unaffected; dropout only ever activates for a model that explicitly
    # asks for it (dnn_env_terrain/pinn_env_terrain, where it's a real hyperparameter to sweep,
    # not a guessed default -- see run_dnn_env_terrain.py/run_pinn_env_terrain.py's --dropout-rate).
    #
    # hidden_layer_sizes=None (2026-08-02 addition) is the same kind of OPT-IN: when omitted, this
    # class builds the EXACT original 3x128 structure below, with the EXACT same parameter names
    # (hidden_layer_1/2/3) -- so every already-completed checkpoint (Stage 1-4, E1/E2) still loads
    # with zero change, since a checkpoint's state_dict keys must match the class's current
    # attribute names. Passing an explicit list (e.g. [64, 32] for an architecture-sweep variant,
    # see documentation/experiment_log.md's 2026-08-02 entry) switches to the general
    # nn.ModuleList path below instead -- only ever used for NEW architecture variants that need
    # training from scratch anyway, never for reloading an existing checkpoint.
    def __init__(self, n_other_features, dropout_rate=0.0, hidden_layer_sizes=None):
        super().__init__()
        input_size = 1 + n_other_features  # age (1) + every other no-environment feature

        self.uses_legacy_layer_names = hidden_layer_sizes is None
        if self.uses_legacy_layer_names:
            self.hidden_layer_1 = nn.Linear(input_size, 128)
            self.hidden_layer_2 = nn.Linear(128, 128)
            self.hidden_layer_3 = nn.Linear(128, 128)
            self.output_layer = nn.Linear(128, 1)
        else:
            layers = []
            previous_size = input_size
            for layer_size in hidden_layer_sizes:
                layers.append(nn.Linear(previous_size, layer_size))
                previous_size = layer_size
            self.hidden_layers = nn.ModuleList(layers)
            self.output_layer = nn.Linear(previous_size, 1)

        self.activation = nn.LeakyReLU()
        # nn.Dropout(p=0.0) is a documented no-op in PyTorch (never zeroes anything, and
        # .eval()/.train() mode makes no difference at p=0) -- so this can always be in the
        # module graph without needing an "if dropout_rate > 0" branch anywhere.
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, other_features, age):
        # age is kept as its OWN argument, not pre-concatenated outside this
        # function, so the PINN can call this exact same forward() with an
        # age tensor that has requires_grad=True (for the physics loss),
        # while every other feature stays an ordinary, non-differentiated
        # input. The DNN calls it the same way and simply never asks for a
        # gradient with respect to age.
        combined_input = torch.cat([age, other_features], dim=1)
        if self.uses_legacy_layer_names:
            hidden = self.dropout(self.activation(self.hidden_layer_1(combined_input)))
            hidden = self.dropout(self.activation(self.hidden_layer_2(hidden)))
            hidden = self.dropout(self.activation(self.hidden_layer_3(hidden)))
        else:
            hidden = combined_input
            for layer in self.hidden_layers:
                hidden = self.dropout(self.activation(layer(hidden)))
        predicted_height = self.output_layer(hidden)
        return predicted_height


class YMaxSubNetwork(nn.Module):
    # Small network for dnn_env_terrain/pinn_env_terrain: terrain/wind features -> one
    # plot-specific y_max adjustment, replacing the single global y_max constant pinn_noenv uses.
    # Deliberately SEPARATE from NoEnvNetwork above, not a bigger version of it -- the design
    # intent (see progress_notes.md's Env-PINN discussion) is that terrain/wind determines only
    # the growth CEILING, not the trajectory shape, so this sub-network never sees age or the
    # no-env features at all, and its only job is producing one number per plot.
    #
    # Outputs an ADJUSTMENT to add to the already-fitted global y_max (models/chapman_richards/
    # params.json), not a from-scratch value -- starting near the known-good global constant
    # (adjustment near 0 at initialisation, since fresh linear layers start with small weights)
    # is far more stable than a randomly-initialised network trying to output ~50m from
    # scratch. k/p stay GLOBAL, frozen floats (not also plot-specific) -- a deliberate choice,
    # not a default: the dissertation's actual novel claim is that the CEILING varies with
    # environment (matching Socha et al. 2021's ADA/GADA framework, where the asymptote
    # parameter is the site-varying one), not that growth RATE/shape does. Making k/p
    # plot-specific too would blur that story and isn't supported by any evidence gathered so
    # far -- it's flagged in progress_notes.md as a possible future extension, not built here.
    # dropout_rate defaults to 0.0, same opt-in reasoning as NoEnvNetwork above.
    def __init__(self, n_terrain_features, hidden_size=16, dropout_rate=0.0):
        super().__init__()
        self.hidden_layer = nn.Linear(n_terrain_features, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, terrain_features):
        hidden = self.dropout(self.activation(self.hidden_layer(terrain_features)))
        y_max_adjustment = self.output_layer(hidden)
        return y_max_adjustment


def parse_hidden_layer_sizes(value):
    # Turns a CLI string like "64,32" into [64, 32] -- shared by all four run_*.py scripts'
    # --hidden-layer-sizes argument, so there is exactly one place this parsing logic lives, not
    # four near-identical copies. Returns None unchanged (the "use the original 3x128 network"
    # default -- see NoEnvNetwork's own note above), so argparse's own default=None still works
    # without this function needing a special case for it.
    if value is None:
        return None
    return [int(size) for size in value.split(",")]


def compute_l1_penalty(model):
    # Sum of the absolute value of every weight and bias in the network.
    # Multiplied by the L1 coefficient (1e-5, Lynch 2025) and added to the
    # loss by whichever training script calls this.
    l1_penalty = 0.0
    for parameter in model.parameters():
        l1_penalty = l1_penalty + parameter.abs().sum()
    return l1_penalty


def chapman_richards_derivative(age, y_max, k, p):
    # d(height)/d(age) for the Chapman-Richards curve
    # height = y_max * (1 - exp(-k*age))^p
    # -- the analytical growth-RATE curve both PINN physics loss terms
    # compare the network's own predictions against. age is a torch tensor
    # (so this stays inside the autograd graph if age itself requires
    # gradients); y_max/k/p are plain Python floats read from this cohort's
    # frozen params.json, never tensors -- that is what keeps them frozen:
    # a plain float can never accumulate a gradient, so training can never
    # accidentally update the process model itself, only the network.
    decay_term = torch.exp(-k * age)
    growth_rate = y_max * p * (1 - decay_term) ** (p - 1) * k * decay_term
    return growth_rate
