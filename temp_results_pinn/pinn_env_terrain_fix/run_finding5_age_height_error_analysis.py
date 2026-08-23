# Finding 5: do prediction errors cluster at certain tree ages or heights, and does this
# differ across DNN, PINN, and PINN-k? All three models' fold-0 test-set predictions now exist
# (same 46,032 rows, confirmed identical age/height ranges) -- see
# run_finding5_plain_pinn_export.py and run_finding5_pinn_k_export.py for how PINN/PINN-k's
# were built. No new training here, this is analysis only.
#
# Run: PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_finding5_age_height_error_analysis.py

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

AGE_BINS = [15, 30, 45, 60, 75, 93]
AGE_LABELS = ["15-30", "30-45", "45-60", "60-75", "75-93"]
HEIGHT_BINS = [0, 10, 20, 30, 47]
HEIGHT_LABELS = ["0-10m", "10-20m", "20-30m", "30-47m"]


def load_models():
    dnn = pd.read_csv("outputs/spatial_block_kfold/rq1_dnn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42/4survey/fold_0/predictions.csv")
    dnn = dnn[dnn["split"] == "test"][["identification", "Age", "observed_top_height", "predicted_top_height", "residual"]].reset_index(drop=True)
    pinn = pd.read_csv("temp_results_pinn/outputs/example_curve/plain_pinn_fixed_full_predictions.csv")
    pinn_k = pd.read_csv("temp_results_pinn/outputs/example_curve/pinn_k_fixed_full_predictions.csv")
    return {"DNN": dnn, "PINN": pinn, "PINN-k": pinn_k}


def main():
    models = load_models()
    for name, df in models.items():
        df["abs_err"] = df["residual"].abs()
        df["age_band"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)
        df["height_band"] = pd.cut(df["observed_top_height"], bins=HEIGHT_BINS, labels=HEIGHT_LABELS, include_lowest=True)

    print("=== Mean absolute error by age band ===")
    age_table = pd.DataFrame({name: df.groupby("age_band", observed=True)["abs_err"].mean() for name, df in models.items()})
    age_table["n"] = models["DNN"].groupby("age_band", observed=True)["abs_err"].count()
    print(age_table.to_string())

    print("\n=== Mean absolute error by height band ===")
    height_table = pd.DataFrame({name: df.groupby("height_band", observed=True)["abs_err"].mean() for name, df in models.items()})
    height_table["n"] = models["DNN"].groupby("height_band", observed=True)["abs_err"].count()
    print(height_table.to_string())

    age_table.to_csv("temp_results_pinn/outputs/example_curve/finding5_age_band_mae.csv")
    height_table.to_csv("temp_results_pinn/outputs/example_curve/finding5_height_band_mae.csv")
    print("\nSaved -> temp_results_pinn/outputs/example_curve/finding5_age_band_mae.csv")
    print("Saved -> temp_results_pinn/outputs/example_curve/finding5_height_band_mae.csv")


if __name__ == "__main__":
    main()
