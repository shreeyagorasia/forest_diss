"""Broader environmental attribution for the per-plot growth-curve target.

This is deliberately separate from the established 17-column terrain/wind analysis.  It keeps
the same cleaned target, compartment folds, 60 m buffer, Elastic Net, and XGBoost evaluation,
but expands the candidate scope to climate, soil/site, and forest-edge variables.  A second
scope adds stand structure/management so its contribution is visible rather than silently
calling management an environmental effect.

The leaked neighbour-height features are not candidates.  Alternative wind formulae and the
250 m TPI are also left out because the established representation checks already identified
them as redundant with the selected 50 m mean wind and native/500 m TPI representations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.elasticnet_environmental.elasticnet_environmental import CATEGORICAL_COLUMNS
from models.growth_curve_attribution.explain_signal import FINAL_FEATURE_COLUMNS
from models.growth_curve_attribution.scale_comparison_check import build_plot_level_table
from models.growth_curve_attribution.spatial_cv_check import run_spatial_cv, summarize_spatial_cv
from models.xgb_environmental.data import load_environmental_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "broad_environmental_spatial_cv.csv"

# Cohort-varying HadUK values are stored as suffixed columns in the environmental export.
CLIMATE_COLUMNS = [
    "tas_mean",
    "groundfrost_mean",
    "chelsa_bio1_celsius",
    "chelsa_gdd5_degc",
    "chelsa_bio12_precip_mm",
]

SOIL_SITE_COLUMNS = [
    "soilgrids_ph",
    "ceh_pedotope",
    "ceh_subsurface_drainage",
    "ceh_textural_composition",
    "dist_to_watercourse",
]

EDGE_POSITION_COLUMNS = [
    "dist_to_cpmt_boundary",
    "dist_to_forest_perimeter",
    "dist_to_scpt_boundary",
    "dist_to_block_boundary",
    "cpmt_compactness_ratio",
    "dist_to_road",
]

MANAGEMENT_COLUMNS = [
    "CanopyCover",
    "Thin",
    "time_since_thinning",
    "time_since_thinning_missing",
    "recent_thinning_5yr",
]

FEATURE_GROUPS = {
    "terrain_wind": list(FINAL_FEATURE_COLUMNS),
    "climate": CLIMATE_COLUMNS,
    "soil_site": SOIL_SITE_COLUMNS,
    "edge_position": EDGE_POSITION_COLUMNS,
    "management": MANAGEMENT_COLUMNS,
}

SCOPE_GROUPS = {
    "terrain_wind": ["terrain_wind"],
    "terrain_wind_plus_climate": ["terrain_wind", "climate"],
    "terrain_wind_plus_soil_site": ["terrain_wind", "soil_site"],
    "terrain_wind_plus_edge_position": ["terrain_wind", "edge_position"],
    "terrain_wind_plus_management": ["terrain_wind", "management"],
    "broad_environment": ["terrain_wind", "climate", "soil_site", "edge_position"],
    "broad_environment_plus_management": [
        "terrain_wind", "climate", "soil_site", "edge_position", "management"
    ],
}

DEFAULT_COMPARISON_SCOPES = ["terrain_wind", "broad_environment", "broad_environment_plus_management"]


def columns_for_groups(group_names: list[str]) -> list[str]:
    """Return de-duplicated raw columns for a sequence of named groups."""
    return list(dict.fromkeys(column for group in group_names for column in FEATURE_GROUPS[group]))


def _resolve_cohort_columns(environment: pd.DataFrame, cohort: str, columns: list[str]) -> pd.DataFrame:
    """Expose cohort-specific HadUK columns under their unsuffixed model names."""
    environment = environment.copy()
    for column in columns:
        cohort_column = f"{column}_{cohort}"
        if column not in environment.columns and cohort_column in environment.columns:
            environment[column] = environment[cohort_column]
    missing = [column for column in columns if column not in environment.columns]
    if missing:
        raise KeyError(f"Environmental export is missing requested columns: {missing}")
    return environment


def prepare_broad_table(cohort: str, raw_feature_columns: list[str]):
    """Build the cleaned target table and encode unordered soil classes consistently.

    Complete cases are selected before one-hot encoding.  Both models therefore see identical
    rows and identical representations; raw CEH class identifiers are never treated as ordered
    continuous measurements.
    """
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    environment = _resolve_cohort_columns(load_environmental_features(), cohort, raw_feature_columns)
    table = plot_table.merge(
        environment[["identification"] + raw_feature_columns],
        on="identification",
        how="left",
        validate="one_to_one",
    )

    before = len(table)
    table = table.dropna(subset=raw_feature_columns).copy()
    dropped = before - len(table)
    if dropped:
        print(f"  Broad complete-case population: dropped {dropped:,} of {before:,} plots")

    categorical = [column for column in raw_feature_columns if column in CATEGORICAL_COLUMNS]
    continuous = [column for column in raw_feature_columns if column not in categorical]
    if categorical:
        dummy_table = pd.get_dummies(
            table[categorical].astype("string"), prefix=categorical, prefix_sep="=", dtype=float
        )
        table = pd.concat([table.drop(columns=categorical), dummy_table], axis=1)
        model_columns = continuous + list(dummy_table.columns)
    else:
        model_columns = continuous
    return table, model_columns


def model_columns_for_raw_columns(raw_columns: list[str], available_model_columns: list[str]) -> list[str]:
    """Map a raw scope to its continuous and one-hot-expanded model columns."""
    selected = []
    for column in raw_columns:
        if column in CATEGORICAL_COLUMNS:
            selected.extend(candidate for candidate in available_model_columns if candidate.startswith(f"{column}="))
        else:
            selected.append(column)
    missing = [column for column in selected if column not in available_model_columns]
    if missing:
        raise KeyError(f"Prepared table is missing model columns: {missing}")
    return selected


def run_scope(cohort: str, scope: str, k: int = 5, seed: int = 42):
    """Run the established spatial CV for one cohort and one broader feature scope."""
    raw_columns = columns_for_groups(SCOPE_GROUPS[scope])
    table, model_columns = prepare_broad_table(cohort, raw_columns)
    predictions, fold_counts = run_spatial_cv(table, model_columns, k=k, seed=seed)

    rows = []
    for prediction_column, method in [
        ("elastic_net_predicted", "Elastic Net"),
        ("xgboost_predicted", "XGBoost"),
    ]:
        summary = summarize_spatial_cv(predictions, prediction_column)
        rows.append(
            {
                "cohort": cohort,
                "scope": scope,
                "method": method,
                "n_raw_features": len(raw_columns),
                "n_model_columns": len(model_columns),
                "k": k,
                "seed": seed,
                **summary,
            }
        )
    return pd.DataFrame(rows), predictions, fold_counts


def run_comparison(
    cohorts=("4survey", "6survey"),
    scopes=("terrain_wind", "broad_environment", "broad_environment_plus_management"),
    k: int = 5,
    seed: int = 42,
):
    """Compare narrow, broad-environment, and management-augmented scopes."""
    # Prepare one common complete-case population per cohort using the union of every requested
    # scope.  This makes scope-to-scope R2 differences genuine feature additions, not changes in
    # which plots survived missing-value filtering.
    results = []
    union_groups = list(dict.fromkeys(group for scope in scopes for group in SCOPE_GROUPS[scope]))
    union_raw_columns = columns_for_groups(union_groups)
    for cohort in cohorts:
        table, all_model_columns = prepare_broad_table(cohort, union_raw_columns)
        for scope in scopes:
            print(f"\n{cohort}: {scope}")
            raw_columns = columns_for_groups(SCOPE_GROUPS[scope])
            model_columns = model_columns_for_raw_columns(raw_columns, all_model_columns)
            predictions, _ = run_spatial_cv(table, model_columns, k=k, seed=seed)
            for prediction_column, method in [
                ("elastic_net_predicted", "Elastic Net"),
                ("xgboost_predicted", "XGBoost"),
            ]:
                summary = summarize_spatial_cv(predictions, prediction_column)
                results.append(pd.DataFrame([{
                    "cohort": cohort,
                    "scope": scope,
                    "method": method,
                    "n_raw_features": len(raw_columns),
                    "n_model_columns": len(model_columns),
                    "common_complete_case_n": len(table),
                    "k": k,
                    "seed": seed,
                    **summary,
                }]))
    return pd.concat(results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", choices=["4survey", "6survey"], default=["4survey", "6survey"])
    parser.add_argument("--scopes", nargs="+", choices=list(SCOPE_GROUPS), default=DEFAULT_COMPARISON_SCOPES)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = run_comparison(args.cohorts, args.scopes, k=args.folds, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print("\n", results[["cohort", "scope", "method", "pooled_r2", "per_fold_r2_mean", "per_fold_r2_std"]])
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
