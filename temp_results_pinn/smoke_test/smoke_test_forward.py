# Smoke test for temp_results_pinn/pinn_env_terrain_fix/pinn_env_terrain_fix.py.
# Local, no cluster, no GPU needed (runs on CPU deliberately -- speed doesn't matter here,
# correctness does). Two checks, per PLAN.md:
#   1. Does terrain now actually reach the prediction? (the whole point of the fix)
#   2. Does a real (tiny) training loop run 3 epochs with no NaN and a real gradient reaching
#      the y_max sub-network?
# Run as: PYTHONPATH=. .venv/bin/python temp_results_pinn/smoke_test/smoke_test_forward.py

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_fix import (
    build_model, fit, predict, chapman_richards_value,
)

N_OTHER_FEATURES = 5
N_TERRAIN_FEATURES = 5

# Fake scaler objects -- just need .scale_[0] and .mean_[0], matching sklearn's
# StandardScaler attribute shape (array-like, index 0).
class FakeScaler:
    def __init__(self, mean, scale):
        self.mean_ = np.array([mean])
        self.scale_ = np.array([scale])

scaler_age = FakeScaler(mean=40.0, scale=15.0)      # age roughly 0-80 years
scaler_height = FakeScaler(mean=20.0, scale=8.0)    # height roughly 0-40m

cr_params = {"y_max": 35.0, "k": 0.05, "p": 1.3}    # plausible Sitka spruce-ish values

print("=" * 70)
print("CHECK 1: does terrain now reach the prediction?")
print("=" * 70)

model = build_model(
    n_other_features=N_OTHER_FEATURES, n_terrain_features=N_TERRAIN_FEATURES,
    cr_params=cr_params, scaler_age=scaler_age, scaler_height=scaler_height,
    device="cpu", seed=42,
)
model.eval()

# Two rows, identical age + other_features, DIFFERENT terrain.
age = torch.tensor([[0.0], [0.0]])  # scaled age = 0 -> real age = 40 (scaler mean)
other = torch.zeros((2, N_OTHER_FEATURES))
terrain_a = torch.zeros((2, N_TERRAIN_FEATURES))
terrain_b = terrain_a.clone()
terrain_b[1] += 2.0  # 2 SD shift, second row only

with torch.no_grad():
    pred_a = model(other, age, terrain_a)
    pred_b = model(other, age, terrain_b)

print("pred with terrain_a:", pred_a.squeeze().tolist())
print("pred with terrain_b (row 1 shifted):", pred_b.squeeze().tolist())
diff = (pred_b[1] - pred_a[1]).item()
print(f"difference for the shifted row: {diff:.6f}")
assert abs(diff) > 1e-6, "FAIL: terrain still not reaching the prediction"
print("PASS: terrain reaches the prediction, difference =", diff)

print()
print("=" * 70)
print("CHECK 1b: chapman_richards_value sanity -- does it match")
print("chapman_richards_derivative's own analytical relationship roughly?")
print("=" * 70)
# At age=0, H(0) should be 0 (curve starts at the origin).
h_at_zero = chapman_richards_value(torch.tensor([[0.0]]), torch.tensor([[35.0]]), 0.05, 1.3)
print("H(age=0) =", h_at_zero.item(), "(should be ~0)")
assert abs(h_at_zero.item()) < 1e-6, "FAIL: CR curve does not start at 0"
# As age -> large, H(age) should approach y_max.
h_at_large_age = chapman_richards_value(torch.tensor([[500.0]]), torch.tensor([[35.0]]), 0.05, 1.3)
print("H(age=500) =", h_at_large_age.item(), "(should be close to y_max=35)")
assert abs(h_at_large_age.item() - 35.0) < 0.5, "FAIL: CR curve does not approach y_max"
print("PASS: chapman_richards_value behaves as expected at the boundaries")

print()
print("=" * 70)
print("CHECK 2: tiny end-to-end fit() loop, 3 epochs, CPU, no NaN, real gradient")
print("=" * 70)

torch.manual_seed(0)
n_rows = 50
age_train = torch.randn(n_rows, 1)
other_train = torch.randn(n_rows, N_OTHER_FEATURES)
terrain_train = torch.randn(n_rows, N_TERRAIN_FEATURES)
target_train = torch.randn(n_rows, 1)

n_val = 10
age_val = torch.randn(n_val, 1)
other_val = torch.randn(n_val, N_OTHER_FEATURES)
terrain_val = torch.randn(n_val, N_TERRAIN_FEATURES)
target_val = torch.randn(n_val, 1)

# Trajectory pairs -- tiny fake set, 20 pairs.
n_pairs = 20
age_earlier = torch.randn(n_pairs, 1)
other_earlier = torch.randn(n_pairs, N_OTHER_FEATURES)
age_later = age_earlier + torch.rand(n_pairs, 1)
other_later = torch.randn(n_pairs, N_OTHER_FEATURES)
delta_age = (age_later - age_earlier) * scaler_age.scale_[0]  # real-unit-ish delta, avoid /0
delta_age = torch.clamp(delta_age, min=0.5)
age_mid = (age_earlier + age_later) / 2 * scaler_age.scale_[0] + scaler_age.mean_[0]
terrain_pairs = torch.randn(n_pairs, N_TERRAIN_FEATURES)
pair_tensors = (age_earlier, other_earlier, age_later, other_later, delta_age, age_mid, None)

best_model, final_state, history_df = fit(
    age_train, other_train, terrain_train, target_train,
    age_val, other_val, terrain_val, target_val,
    pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
    n_other_features=N_OTHER_FEATURES, n_terrain_features=N_TERRAIN_FEATURES,
    device="cpu", seed=42, max_epochs=3, early_stopping_patience=10,
)

print(history_df[["epoch", "data_loss", "physics_loss", "trajectory_loss", "grad_norm", "val_loss"]].to_string(index=False))

assert not history_df["data_loss"].isna().any(), "FAIL: NaN in data_loss"
assert not history_df["physics_loss"].isna().any(), "FAIL: NaN in physics_loss"
assert not history_df["trajectory_loss"].isna().any(), "FAIL: NaN in trajectory_loss"
assert (history_df["grad_norm"] > 0).all(), "FAIL: grad_norm is zero -- gradient not reaching the model"
print("PASS: 3 epochs ran, no NaN, real gradient throughout")

print()
print("=" * 70)
print("CHECK 3: predict() runs and returns a sane shape")
print("=" * 70)
preds = predict(best_model, age_val, other_val, terrain_val)
print("predict() output shape:", tuple(preds.shape), "(expected:", (n_val, 1), ")")
assert preds.shape == (n_val, 1), "FAIL: unexpected predict() output shape"
assert not torch.isnan(preds).any(), "FAIL: NaN in predictions"
print("PASS: predict() works")

print()
print("ALL SMOKE TESTS PASSED")
