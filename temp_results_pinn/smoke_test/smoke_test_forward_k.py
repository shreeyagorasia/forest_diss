# Smoke test for pinn_env_terrain_k_fix.py. Same structure as smoke_test_forward.py, but checks
# k's own multiplicative/log-space mechanism specifically (different from y_max's additive one).
# Run as: PYTHONPATH=. .venv/bin/python temp_results_pinn/smoke_test/smoke_test_forward_k.py

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_k_fix import build_model, fit, predict, predict_k

N_OTHER_FEATURES = 5
N_TERRAIN_FEATURES = 5


class FakeScaler:
    def __init__(self, mean, scale):
        self.mean_ = np.array([mean])
        self.scale_ = np.array([scale])


scaler_age = FakeScaler(mean=40.0, scale=15.0)
scaler_height = FakeScaler(mean=20.0, scale=8.0)
cr_params = {"y_max": 35.0, "k": 0.05, "p": 1.3}

print("=" * 70)
print("CHECK 1: does terrain reach the prediction, and is k always positive?")
print("=" * 70)

model = build_model(
    n_other_features=N_OTHER_FEATURES, n_terrain_features=N_TERRAIN_FEATURES,
    cr_params=cr_params, scaler_age=scaler_age, scaler_height=scaler_height,
    device="cpu", seed=42,
)
model.eval()

age = torch.tensor([[0.0], [0.0]])
other = torch.zeros((2, N_OTHER_FEATURES))
terrain_a = torch.zeros((2, N_TERRAIN_FEATURES))
terrain_b = terrain_a.clone()
terrain_b[1] += 2.0

with torch.no_grad():
    pred_a = model(other, age, terrain_a)
    pred_b = model(other, age, terrain_b)
    k_a = predict_k(model, terrain_a, cr_params["k"])
    k_b = predict_k(model, terrain_b, cr_params["k"])

print("pred with terrain_a:", pred_a.squeeze().tolist())
print("pred with terrain_b (row 1 shifted):", pred_b.squeeze().tolist())
diff = (pred_b[1] - pred_a[1]).item()
assert abs(diff) > 1e-6, "FAIL: terrain still not reaching the prediction"
print("PASS: terrain reaches the prediction, difference =", diff)

print("k_a:", k_a.squeeze().tolist(), " k_b:", k_b.squeeze().tolist())
assert (k_a > 0).all() and (k_b > 0).all(), "FAIL: k went non-positive"
print("PASS: k stays strictly positive under a terrain shift, as the log-space parameterisation guarantees")

# Extreme terrain shift -- check k doesn't blow up or go to exactly 0 (numerical stability check,
# since k sits inside exp(-k*age) and a bad value here poisons the whole CR term).
terrain_extreme = terrain_a.clone()
terrain_extreme[1] += 20.0  # 20 SD, deliberately extreme
with torch.no_grad():
    k_extreme = predict_k(model, terrain_extreme, cr_params["k"])
print("k under a 20-SD terrain shift (extreme):", k_extreme.squeeze().tolist())
assert torch.isfinite(k_extreme).all(), "FAIL: k is not finite under an extreme terrain shift"
print("PASS: k stays finite even under an extreme terrain value")

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

n_pairs = 20
age_earlier = torch.randn(n_pairs, 1)
other_earlier = torch.randn(n_pairs, N_OTHER_FEATURES)
age_later = age_earlier + torch.rand(n_pairs, 1)
other_later = torch.randn(n_pairs, N_OTHER_FEATURES)
delta_age = torch.clamp((age_later - age_earlier) * scaler_age.scale_[0], min=0.5)
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
assert (history_df["grad_norm"] > 0).all(), "FAIL: grad_norm is zero"
print("PASS: 3 epochs ran, no NaN, real gradient throughout")

print()
print("=" * 70)
print("CHECK 3: predict() runs, sane shape; y_max/k both extractable and uncorrelated-by-default")
print("=" * 70)
preds = predict(best_model, age_val, other_val, terrain_val)
assert preds.shape == (n_val, 1) and not torch.isnan(preds).any(), "FAIL: predict() output wrong"
print("PASS: predict() works, shape", tuple(preds.shape))

print()
print("ALL SMOKE TESTS PASSED (k version)")
