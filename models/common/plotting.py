import matplotlib.pyplot as plt


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
