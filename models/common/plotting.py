import matplotlib.pyplot as plt
import pandas as pd

# One fixed colour per survey year, shared by every plot in this module, so
# 2008 is always the same colour whichever model or cohort is being shown.
YEAR_COLORS = {
    2002: "#8c564b",
    2006: "#9467bd",
    2008: "#1f77b4",
    2012: "#2ca02c",
    2021: "#ff7f0e",
    2023: "#d62728",
}


def plot_predicted_vs_observed(y_true, y_pred, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=8, alpha=0.4)

    # draw a diagonal "perfect prediction" line, from the smallest value
    # in the data up to the largest value in the data
    smallest_value = min(min(y_true), min(y_pred))
    largest_value = max(max(y_true), max(y_pred))
    ax.plot(
        [smallest_value, largest_value],
        [smallest_value, largest_value],
        color="black", linewidth=1, linestyle="--",
    )

    ax.set_xlabel("Observed Top_Height99")
    ax.set_ylabel("Predicted Top_Height99")
    return ax


def plot_residuals(y_true, y_pred, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    resid = y_true - y_pred
    ax.scatter(y_pred, resid, s=8, alpha=0.4)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Predicted Top_Height99")
    ax.set_ylabel("Residual")
    return ax


def plot_error_by_age(age, y_true, y_pred, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    resid = y_true - y_pred
    ax.scatter(age, resid, s=8, alpha=0.4)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Age")
    ax.set_ylabel("Residual")
    return ax


def plot_growth_curve(age, y_true, y_pred, lidar_year, ax=None):
    # Shows the model's single fitted age -> height relationship (the
    # predicted points, which collapse onto one curve for an age-only model
    # like Chapman-Richards or average-by-age) against the observed points,
    # coloured by survey year. If the model is a good fit for every era, the
    # coloured observed points should scatter evenly above and below the
    # predicted curve at every age. If one colour sits mostly above or below
    # the curve, that survey year is where the single pooled fit is biased.
    if ax is None:
        _, ax = plt.subplots()

    for year, colour in YEAR_COLORS.items():
        year_mask = lidar_year == year
        if year_mask.sum() == 0:
            continue
        ax.scatter(age[year_mask], y_true[year_mask], s=8, alpha=0.4, color=colour, label=str(year))

    # Predicted points are drawn last, in black, on top of the observed
    # points, so the fitted curve is always visible even where colours
    # overlap heavily.
    ax.scatter(age, y_pred, s=6, alpha=0.6, color="black", label="Predicted")

    ax.set_xlabel("Age")
    ax.set_ylabel("Top_Height99")
    ax.legend(title="Survey year", markerscale=2, fontsize=8)
    return ax


def plot_bias_by_year(lidar_year, y_true, y_pred, ax=None):
    # Bias (observed - predicted) averaged within each survey year. This is
    # the numeric version of what plot_growth_curve shows visually: whether
    # the model over- or under-predicts a whole survey year's rows on
    # average, which a single pooled Age-only fit cannot correct for.
    if ax is None:
        _, ax = plt.subplots()

    residual_df = pd.DataFrame({"lidar_year": lidar_year, "residual": y_true - y_pred})
    bias_by_year = residual_df.groupby("lidar_year")["residual"].mean().sort_index()

    colours = [YEAR_COLORS.get(year, "gray") for year in bias_by_year.index]
    ax.bar(bias_by_year.index.astype(str), bias_by_year.values, color=colours)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Survey year")
    ax.set_ylabel("Bias (observed - predicted), m")
    return ax


def plot_loss_curve(history_df, ax=None):
    # history_df is training_history.csv read back in (one row per epoch).
    # Every model's history has train_loss/val_loss; the PINN's history also
    # has data_loss/physics_loss/trajectory_loss, which get drawn as thin
    # dashed lines automatically if those columns exist -- so this one
    # function works for both the DNN and the PINN's training_history.csv,
    # nothing to change when switching between them.
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(history_df["epoch"], history_df["train_loss"], label="train_loss", color="#1f77b4")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="val_loss", color="#d62728")

    for component in ["data_loss", "physics_loss", "trajectory_loss"]:
        if component in history_df.columns:
            ax.plot(
                history_df["epoch"], history_df[component],
                label=component, linestyle="--", linewidth=1, alpha=0.6,
            )

    # Mark the epoch with the lowest val_loss -- this is the epoch whose
    # weights load_best_model() actually loads, not necessarily the last
    # epoch trained (early stopping keeps training a while longer after the
    # best epoch, to make sure it really has stopped improving).
    best_epoch = history_df.loc[history_df["val_loss"].idxmin(), "epoch"]
    ax.axvline(best_epoch, color="black", linewidth=1, linestyle=":", alpha=0.6, label="best epoch")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    return ax
