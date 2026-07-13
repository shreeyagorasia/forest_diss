import json
from datetime import datetime, timezone
from pathlib import Path


def fit(train_df, age_col="Age", height_col="Top_Height99"):
    # Round age to the nearest whole year so ages line up into a simple table.
    rounded_ages = train_df[age_col].round().astype(int)

    # Group by age and take the mean observed height for each age.
    mean_height_by_age = train_df.groupby(rounded_ages)[height_col].mean()
    lookup_table = {int(age): float(height) for age, height in mean_height_by_age.items()}

    # If a row's age is not in the lookup table later, fall back to the
    # overall mean height across the whole training set.
    fallback_mean_height = float(train_df[height_col].mean())

    return lookup_table, fallback_mean_height


def predict(age_values, lookup_table, fallback_mean_height):
    # lookup_table keys come back from JSON as strings, so convert them to
    # integers once here rather than on every lookup.
    lookup_table = {int(age): height for age, height in lookup_table.items()}

    predicted_heights = []
    for age in age_values:
        rounded_age = int(round(age))
        if rounded_age in lookup_table:
            predicted_heights.append(lookup_table[rounded_age])
        else:
            predicted_heights.append(fallback_mean_height)

    return predicted_heights


def save_lookup(lookup_table, fallback_mean_height, cohort, n_rows_fit, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "lookup_table": lookup_table,
        "fallback_mean_height": fallback_mean_height,
        "cohort": cohort,
        "n_rows_fit": n_rows_fit,
        "fit_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
