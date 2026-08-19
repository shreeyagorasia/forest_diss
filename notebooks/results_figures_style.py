# Shared plotting style for the four results-chapter figure notebooks
# (results_figures_rq1.ipynb, results_figures_rq2a.ipynb, results_figures_rq2b.ipynb,
# results_figures_rq3.ipynb). Reuses the exact colour meanings already established in
# documentation/august_draft/4_Chapter_methodology/methodology_figures.ipynb, so figures across
# both chapters read as one consistent document rather than two different styles glued together.
#
# Import from a notebook as:
#   from notebooks.results_figures_style import *
# (works once the notebook's own path-setup cell has added the project root to sys.path).

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Split / role colours -- identical values to the methodology chapter's own palette ---
COLOR_TRAIN = "#0072B2"        # blue
COLOR_VAL = "#E69F00"          # orange
COLOR_TEST = "#009E73"         # green
COLOR_BUFFER = "#999999"       # neutral grey -- excluded/removed rows
COLOR_CURVE = "#D55E00"        # vermillion -- fitted model / analytical curve
COLOR_NEUTRAL_BOX = "#F0F0F0"
COLOR_NEUTRAL_EDGE = "#555555"
COLOR_EXAMPLE_4SURVEY = "#CC79A7"
COLOR_EXAMPLE_6SURVEY = "#56B4E9"

# --- Model colours (new for the results chapter, kept fixed everywhere a model appears) ---
COLOR_XGBOOST = COLOR_CURVE      # "#D55E00" -- vermillion
COLOR_DNN = COLOR_TRAIN          # "#0072B2" -- blue
COLOR_PINN = COLOR_TEST          # "#009E73" -- green
COLOR_PINN_K = "#00594A"         # darker tint of COLOR_TEST, for PINN's own two-parameter variant
COLOR_EN = COLOR_EXAMPLE_4SURVEY  # "#CC79A7" -- reddish purple
COLOR_GNNWR = "#F0E442"          # Okabe-Ito yellow -- unused elsewhere in the palette

MODEL_COLORS = {
    "XGBoost": COLOR_XGBOOST,
    "DNN": COLOR_DNN,
    "PINN": COLOR_PINN,
    "PINN_k": COLOR_PINN_K,
    "Elastic Net": COLOR_EN,
    "GNNWR": COLOR_GNNWR,
}

COHORT_COLORS = {
    "4survey": COLOR_EXAMPLE_4SURVEY,
    "6survey": COLOR_EXAMPLE_6SURVEY,
}

# Diverging maps (helped/hurt, above/below yldc, positive/negative residual): centre at 0 with
# mpl.colors.TwoSlopeNorm, always use this colormap so every diverging map in the chapter reads
# the same way.
DIVERGING_CMAP = "RdBu_r"


def lighten_color(hex_color, amount):
    """Blend `hex_color` toward white by `amount` (0 = no change, 1 = white). Used for schematic
    tints, e.g. distinguishing a model's own detail boxes from its main colour."""
    rgb = mcolors.to_rgb(hex_color)
    white = (1.0, 1.0, 1.0)
    return tuple(c + (w - c) * amount for c, w in zip(rgb, white))


def apply_rcparams():
    """Call once per notebook, right after import -- matches methodology_figures.ipynb's own
    settings exactly (print-oriented: no hover/interactivity, thin marks, no top/right spines)."""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def relative_tick_formatter(origin):
    """Returns a FuncFormatter that shows map-axis ticks as metres from `origin`, not raw
    six-digit OS grid references -- same convention as methodology_figures.ipynb's ORIGIN_X/Y."""
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda value, pos: f"{value - origin:,.0f}")
