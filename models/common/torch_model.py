# Shared network architecture for both dnn_noenv and pinn_noenv, so any
# difference between their results comes from the loss function, not from a
# different number of layers or neurons. See
# documentation/model_instructions/age_only_dnn_pinn_instructions.md.

import torch
import torch.nn as nn


class NoEnvNetwork(nn.Module):
    # 3 hidden layers, 128 neurons each, leaky ReLU activation.
    #
    # dropout_rate defaults to 0.0 (2026-08-01 addition) -- OPT-IN, not a default behaviour
    # change. dnn_noenv/pinn_noenv's already-completed, reported runs never pass this argument,
    # so they are completely unaffected; dropout only ever activates for a model that explicitly
    # asks for it (dnn_env_terrain/pinn_env_terrain, where it's a real hyperparameter to sweep,
    # not a guessed default -- see run_dnn_env_terrain.py/run_pinn_env_terrain.py's --dropout-rate).

    def __init__(self, n_other_features, dropout_rate=0.0):
        super().__init__()
        input_size = 1 + n_other_features  # age (1) + every other no-environment feature
        self.hidden_layer_1 = nn.Linear(input_size, 128)
        self.hidden_layer_2 = nn.Linear(128, 128)
        self.hidden_layer_3 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 1)
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
        hidden_1 = self.dropout(self.activation(self.hidden_layer_1(combined_input)))
        hidden_2 = self.dropout(self.activation(self.hidden_layer_2(hidden_1)))
        hidden_3 = self.dropout(self.activation(self.hidden_layer_3(hidden_2)))
        predicted_height = self.output_layer(hidden_3)
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
