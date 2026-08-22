# Run as: PYTHONPATH=. .venv/bin/python -m models.growth_curve_attribution.build_saturation_transformed_features
#
# One-off data-prep step for the "does GNNWR do better with pre-transformed inputs" check
# (models/growth_curve_attribution/run_rq3_gnnwr_saturation_transform_check.py). GNNWR is
# structurally local-LINEAR -- it cannot represent a curve on its own, even one that varies
# smoothly across space. The no-CanopyCover XGBoost SHAP-dependence check (2026-08-22,
# figures/fig_results/q2_no_canopy_shap_dependence_check.png) found two variables with a clear
# SATURATING (not noisy) relationship to the target: slope_degrees (steep 0-15deg, flat above) and
# windward_topex (S-shaped, flat below roughly -12 and above roughly +6). This script adds
# pre-saturated versions of both, so GNNWR's local straight line only has to fit the already-linear
# middle section of each curve, not approximate the whole curve with one slope.
#
# Knot values (15 for slope, -12/+6 for windward_topex) were chosen by visual inspection of that
# SHAP-dependence plot, not a separate validation split -- a real, disclosed methodological
# shortcut appropriate for a quick "does this help at all" robustness check, not a headline claim.
#
# Output: a new parquet, same content as the real environmental-features file plus 2 new columns.
# Never overwrites the original file.

import pandas as pd

from models.xgb_environmental.data import FEATURES_PATH

OUTPUT_PATH = FEATURES_PATH.parent / "plot_environmental_features_saturation_transformed.parquet"

SLOPE_CAP_DEGREES = 15.0
WINDWARD_TOPEX_CLIP_LOW = -12.0
WINDWARD_TOPEX_CLIP_HIGH = 6.0

features = pd.read_parquet(FEATURES_PATH)
print(f"Loaded {FEATURES_PATH} -- {len(features):,} rows, {len(features.columns)} columns")

features["slope_degrees_capped15"] = features["slope_degrees"].clip(upper=SLOPE_CAP_DEGREES)
features["windward_topex_clipped"] = features["windward_topex"].clip(
    lower=WINDWARD_TOPEX_CLIP_LOW, upper=WINDWARD_TOPEX_CLIP_HIGH
)

print(f"Added slope_degrees_capped15 (cap={SLOPE_CAP_DEGREES}) and "
      f"windward_topex_clipped (clip=[{WINDWARD_TOPEX_CLIP_LOW}, {WINDWARD_TOPEX_CLIP_HIGH}])")
print(features[["slope_degrees", "slope_degrees_capped15", "windward_topex", "windward_topex_clipped"]].describe())

features.to_parquet(OUTPUT_PATH)
print(f"\nSaved {OUTPUT_PATH} ({len(features):,} rows, {len(features.columns)} columns) -- original file untouched")
