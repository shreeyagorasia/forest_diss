import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

# Reuses the SAME provenance dict and feature-set definitions XGBoost uses -- one place
# (xgb_environmental.py) decides what "all_environmental" etc actually contains, so the two
# methods are always being compared on the exact same candidate variables, never a silently
# drifted copy of the list.
from models.xgb_environmental.xgb_environmental import (
    ALL_FEATURE_COLUMNS,
    NEIGHBOUR_COLUMNS,
    TERRAIN_AND_WIND_COLUMNS,
)

FEATURE_SETS = {
    "all_environmental": ALL_FEATURE_COLUMNS,
    "all_environmental_no_neighbour": [c for c in ALL_FEATURE_COLUMNS if c not in NEIGHBOUR_COLUMNS],
    "terrain_and_wind_only": TERRAIN_AND_WIND_COLUMNS,
}

# Elastic Net is a LINEAR model: it assumes each input column is a number on a meaningful,
# ordered scale, where "twice the value" means "twice the effect". That assumption is true for
# almost every column here (elevation, slope, temperature, whcl -- a 0-6 ordinal hazard
# severity) but it is FALSE for these three: they are raster CLASS IDs (see FEATURE_PROVENANCE
# in xgb_environmental.py -- "a categorical class ID, not an ordinal scale"). Class 3 is not
# "three times" class 1, and class 4 is not necessarily "between" class 3 and class 5 in any
# real sense. Feeding a class ID straight into a linear model would silently assume a fake
# ordering and a fake straight-line effect that isn't there -- so these three are one-hot
# encoded instead (turned into a separate 0/1 column per category), same treatment a tree model
# does not need (a tree can split "is this class 4?" without caring about ordering at all,
# which is why xgb_environmental.py uses these columns raw).
CATEGORICAL_COLUMNS = ["ceh_pedotope", "ceh_subsurface_drainage", "ceh_textural_composition"]

# ElasticNetCV tries every one of these l1_ratio values (how much of the penalty is "Lasso-style"
# vs "Ridge-style") together with a range of overall penalty strengths, picking whichever
# combination scores best under its own internal cross-validation on the TRAINING data only --
# this never looks at val or test, so there is no leakage in letting the CV choose this for us.
# l1_ratio=0 is pure Ridge (spreads credit across correlated features, never zeroes any out);
# l1_ratio=1 is pure Lasso (happy to zero out a whole correlated group down to one survivor).
# Trying a range instead of picking one lets the data decide which behaviour fits best.
L1_RATIOS_TO_TRY = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]


def drop_rows_with_missing_features(df, feature_columns):
    # A handful of plots are missing one or more environmental variables (edge-of-raster-window
    # cases: soilgrids_ph, the CEH layers, and the neighbour features each have a small number of
    # NaNs -- see notebooks/environmental_data/aux_data_resolution_check.ipynb for why). XGBoost
    # can split around a NaN natively, so xgb_environmental.py never needed to touch this. Elastic
    # Net has no such handling -- sklearn's ElasticNetCV raises outright on any NaN input -- so
    # those rows are dropped here instead of silently failing or being imputed with a made-up
    # value. This affects a small fraction of rows (under 1% for every feature set tried so far).
    before = len(df)
    df = df.dropna(subset=feature_columns)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} of {before} rows with a missing value in one of {len(feature_columns)} features")
    return df


def split_feature_types(feature_columns):
    # Every feature is either "continuous" (a real number on a meaningful scale, gets
    # standardized) or "categorical" (a class ID, gets one-hot encoded instead).
    categorical_columns = [c for c in feature_columns if c in CATEGORICAL_COLUMNS]
    continuous_columns = [c for c in feature_columns if c not in CATEGORICAL_COLUMNS]
    return continuous_columns, categorical_columns


def build_design_matrix(df, continuous_columns, categorical_columns, dummy_columns=None):
    # Builds the actual numeric table Elastic Net will see: continuous columns copied as-is
    # (scaling happens separately, see fit_with_columns/predict_with_columns), categorical
    # columns expanded into one 0/1 column per category seen.
    continuous_part = df[continuous_columns].reset_index(drop=True)

    if not categorical_columns:
        return continuous_part, []

    # astype(str) first so pandas treats every categorical column as text, not a number --
    # this guarantees get_dummies expands ALL of them into 0/1 columns, never accidentally
    # leaving one as a raw (wrongly-ordinal) number if pandas otherwise treated it as numeric.
    categorical_part = pd.get_dummies(df[categorical_columns].astype(str)).reset_index(drop=True)

    if dummy_columns is None:
        # First time this is called (on the training data) -- record exactly which dummy
        # columns exist, so val/test data can be forced to match, even if a rare category is
        # missing from (or only appears in) one split.
        dummy_columns = list(categorical_part.columns)
    else:
        # reindex guarantees val/test end up with the SAME columns, in the SAME order, as
        # training -- any category training saw but this data doesn't gets filled with 0
        # (that category is simply absent here); any category this data has that training
        # never saw is dropped (the model has no coefficient for it anyway).
        categorical_part = categorical_part.reindex(columns=dummy_columns, fill_value=0)

    design_matrix = pd.concat([continuous_part, categorical_part], axis=1)
    return design_matrix, dummy_columns


def fit_with_columns(train_df, feature_columns, target_col="mean_cr_residual", seed=42):
    # Returns a plain dict bundling everything needed to reproduce the exact same
    # preprocessing at prediction time: the fitted scaler, which columns are continuous vs
    # categorical, and which dummy columns training ended up with. A fitted XGBoost model
    # carries all of this internally already -- Elastic Net needs it handed back explicitly
    # because the preprocessing (scaling, one-hot encoding) lives outside the model object.
    continuous_columns, categorical_columns = split_feature_types(feature_columns)
    design_train, dummy_columns = build_design_matrix(train_df, continuous_columns, categorical_columns)

    # StandardScaler rescales each continuous column to mean 0, std 1. This matters specifically
    # for Elastic Net: its penalty punishes large coefficients, so a column measured in metres
    # (elevation, up to ~600) would otherwise get an artificially small coefficient just because
    # of its units, while a column measured in small decimals (slope in degrees, or a Spearman-
    # scale index) would get an artificially large one -- not because either is more important,
    # just because of the units each happens to be recorded in. Scaling removes that distortion,
    # so the fitted coefficients are actually comparable to each other.
    scaler = StandardScaler()
    design_train = design_train.copy()
    design_train[continuous_columns] = scaler.fit_transform(design_train[continuous_columns])

    model = ElasticNetCV(l1_ratio=L1_RATIOS_TO_TRY, cv=5, random_state=seed, max_iter=20000)
    model.fit(design_train, train_df[target_col])

    return {
        "model": model,
        "scaler": scaler,
        "continuous_columns": continuous_columns,
        "categorical_columns": categorical_columns,
        "dummy_columns": dummy_columns,
        "design_columns": continuous_columns + dummy_columns,
    }


def fit(train_df, feature_set_name, target_col="mean_cr_residual", seed=42):
    feature_columns = FEATURE_SETS[feature_set_name]
    return fit_with_columns(train_df, feature_columns, target_col=target_col, seed=seed)


def predict_with_columns(df, fitted, feature_columns):
    # feature_columns is accepted for the same call shape as xgb_environmental's
    # predict_with_columns(), but the actual columns used are read back from `fitted` --
    # this guarantees prediction always uses the identical preprocessing (same scaler, same
    # dummy columns, same order) the model was trained with, not whatever columns happen to be
    # passed in.
    design, _ = build_design_matrix(
        df, fitted["continuous_columns"], fitted["categorical_columns"], dummy_columns=fitted["dummy_columns"],
    )
    design = design.copy()
    design[fitted["continuous_columns"]] = fitted["scaler"].transform(design[fitted["continuous_columns"]])
    return fitted["model"].predict(design)


def predict(df, fitted, feature_set_name):
    feature_columns = FEATURE_SETS[feature_set_name]
    return predict_with_columns(df, fitted, feature_columns)


def get_coefficients_table(fitted):
    # One row per design-matrix column (continuous features standardized, categorical features
    # one-hot) with its fitted coefficient. Reading this: for a CONTINUOUS feature, the
    # coefficient is "for a 1 standard-deviation increase in this variable, the CR residual
    # changes by this many metres, holding the regularized combination of everything else
    # roughly fixed". For a CATEGORICAL dummy column (e.g. "ceh_pedotope_3.0"), it is "being in
    # this specific class changes the residual by this many metres, relative to the categories
    # Elastic Net folded into the baseline". A coefficient of exactly 0.0 means Elastic Net's
    # penalty zeroed that variable out entirely -- treated as carrying no unique information once
    # everything else is accounted for, not merely "small".
    coefficients = pd.Series(fitted["model"].coef_, index=fitted["design_columns"], name="coefficient")
    return coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)
