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
    ax.plot(history_df["epoch"], history_df["val_loss"], label="val_loss", color="#d62728", alpha=0.4)

    # val_loss_smoothed (a short rolling average -- see fit()'s
    # VAL_LOSS_SMOOTHING_WINDOW) only exists in newer training_history.csv
    # files. Drawn on top, solid, since it's what actually decides the best
    # epoch below; val_loss itself is kept faint so the raw noise is still
    # visible underneath.
    best_loss_column = "val_loss"
    if "val_loss_smoothed" in history_df.columns:
        ax.plot(history_df["epoch"], history_df["val_loss_smoothed"], label="val_loss_smoothed", color="#d62728")
        best_loss_column = "val_loss_smoothed"

    for component in ["data_loss", "physics_loss", "trajectory_loss"]:
        if component in history_df.columns:
            ax.plot(
                history_df["epoch"], history_df[component],
                label=component, linestyle="--", linewidth=1, alpha=0.6,
            )

    # Mark the epoch with the lowest (smoothed, if available) val_loss --
    # this is the epoch whose weights load_best_model() actually loads, not
    # necessarily the last epoch trained (early stopping keeps training a
    # while longer after the best epoch, to make sure it really has stopped
    # improving).
    best_epoch = history_df.loc[history_df[best_loss_column].idxmin(), "epoch"]
    ax.axvline(best_epoch, color="black", linewidth=1, linestyle=":", alpha=0.6, label="best epoch")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    return ax


def plot_loss_curve_comparison(histories, ax=None, value_column="val_loss_smoothed"):
    # histories: dict of {label: training_history.csv DataFrame}, e.g. one
    # entry per physics_weight value in a sweep, or one entry per seed in a
    # reseed check. Plots all of them on ONE set of axes so their shapes
    # (not just their final numbers) can be compared directly -- a
    # side-by-side grid of separate panels makes it easy to compare final
    # height but hard to compare convergence speed/stability at a glance.
    #
    # Colour: labels that parse as numbers (e.g. physics_weight values)
    # get a sequential colormap ordered by value, since the weight itself
    # is ordinal and a shared colour scale makes "does a higher weight
    # trend toward a worse plateau" visible directly. Non-numeric labels
    # (e.g. seed identifiers) fall back to a fixed qualitative palette.
    if ax is None:
        _, ax = plt.subplots()

    labels = list(histories.keys())
    try:
        numeric_labels = [float(label) for label in labels]
        order = sorted(range(len(labels)), key=lambda i: numeric_labels[i])
        colormap = plt.cm.viridis
        colors = {
            labels[i]: colormap(rank / max(len(labels) - 1, 1))
            for rank, i in enumerate(order)
        }
    except (TypeError, ValueError):
        qualitative_colors = plt.cm.tab10.colors
        colors = {label: qualitative_colors[i % len(qualitative_colors)] for i, label in enumerate(labels)}

    for label, history_df in histories.items():
        if value_column not in history_df.columns:
            continue
        ax.plot(history_df["epoch"], history_df[value_column], label=str(label), color=colors[label])

    ax.set_xlabel("Epoch")
    ax.set_ylabel(value_column)
    ax.legend(fontsize=7, ncol=2)
    return ax


def plot_learning_rate_curve(history_df, ax=None):
    # The ReduceLROnPlateau scheduler shrinks the learning rate whenever
    # val_loss stalls -- this plot shows WHEN that happened, which is
    # usually the explanation for a sudden change in the loss curve's
    # slope that looking at loss alone doesn't make obvious.
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(history_df["epoch"], history_df["learning_rate"], color="#2ca02c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_yscale("log")
    return ax


def plot_generalization_gap(history_df, ax=None):
    # val_loss - train_loss, plotted directly, epoch by epoch. A gap that
    # grows over training is the overfitting signal; a gap that's flat or
    # shrinking is healthy -- more direct to read off one line than to
    # eyeball the distance between two separate lines that are also each
    # moving.
    if ax is None:
        _, ax = plt.subplots()

    gap = history_df["val_loss"] - history_df["train_loss"]
    ax.plot(history_df["epoch"], gap, color="#8c564b")
    ax.axhline(0, color="black", linewidth=1, linestyle=":", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss - train_loss")
    return ax


def plot_gradient_norm_curve(history_df, ax=None, grad_clip_max_norm=None):
    # Pre-clip gradient norm per epoch (see train_one_epoch() in
    # dnn_noenv.py/pinn_noenv.py -- clip_grad_norm_ returns this value
    # before shrinking it). If grad_clip_max_norm is given, draws it as a
    # reference line: gradient norms that sit clearly above it mean
    # clipping is engaging almost every epoch (a sign of real training
    # instability the clip is masking); norms that stay below it mean
    # clipping is rarely/never actually doing anything.
    if ax is None:
        _, ax = plt.subplots()
    if "grad_norm" not in history_df.columns:
        ax.set_title("grad_norm not logged for this run")
        return ax

    ax.plot(history_df["epoch"], history_df["grad_norm"], color="#e377c2")
    if grad_clip_max_norm is not None:
        ax.axhline(grad_clip_max_norm, color="black", linewidth=1, linestyle=":", alpha=0.6, label="clip threshold")
        ax.legend(fontsize=7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient norm (pre-clip)")
    return ax
