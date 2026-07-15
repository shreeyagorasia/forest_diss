# Shared network architecture for both dnn_noenv and pinn_noenv, so any
# difference between their results comes from the loss function, not from a
# different number of layers or neurons. See
# documentation/model_instructions/age_only_dnn_pinn_instructions.md.

import torch
import torch.nn as nn


class NoEnvNetwork(nn.Module):
    # 3 hidden layers, 128 neurons each, leaky ReLU activation.

    def __init__(self, n_other_features):
        super().__init__()
        input_size = 1 + n_other_features  # age (1) + every other no-environment feature
        self.hidden_layer_1 = nn.Linear(input_size, 128)
        self.hidden_layer_2 = nn.Linear(128, 128)
        self.hidden_layer_3 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, other_features, age):
        # age is kept as its OWN argument, not pre-concatenated outside this
        # function, so the PINN can call this exact same forward() with an
        # age tensor that has requires_grad=True (for the physics loss),
        # while every other feature stays an ordinary, non-differentiated
        # input. The DNN calls it the same way and simply never asks for a
        # gradient with respect to age.
        combined_input = torch.cat([age, other_features], dim=1)
        hidden_1 = self.activation(self.hidden_layer_1(combined_input))
        hidden_2 = self.activation(self.hidden_layer_2(hidden_1))
        hidden_3 = self.activation(self.hidden_layer_3(hidden_2))
        predicted_height = self.output_layer(hidden_3)
        return predicted_height


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
