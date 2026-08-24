import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common.geo import load_plot_coordinates
from models.common.splits import SPATIAL_BUFFER_METRES, SPLIT_SEED, plot_level_split, spatial_block_split

OUTPUT_PATH = PROJECT_ROOT / "figures" / "fig_methodology" / "fig_method_map_splits.png"
COHORT = "4survey"

COLOR_TRAIN = "#0072B2"
COLOR_VAL = "#E69F00"
COLOR_TEST = "#009E73"
COLOR_BUFFER = "#999999"
COLOR_NEUTRAL_EDGE = "#555555"


def load_plot_level_dataframe() -> pd.DataFrame:
    master_path = PROJECT_ROOT / "data" / "processed" / "master" / f"clean_master_{COHORT}.parquet"
    master_df = pd.read_parquet(master_path)
    coordinates_df = load_plot_coordinates()
    master_df = master_df.merge(
        coordinates_df[["identification", "x", "y"]],
        on="identification",
        how="left",
    )
    plots_df = master_df.drop_duplicates("identification").copy()
    plots_df["plot_split"] = plot_level_split(plots_df, seed=SPLIT_SEED)
    plots_df["buffered_split"] = spatial_block_split(
        plots_df,
        block_col="cpmt",
        buffer_distance=SPATIAL_BUFFER_METRES,
        coordinates_df=coordinates_df,
        seed=SPLIT_SEED,
    )
    return plots_df


def plot_split_map(df: pd.DataFrame, split_col: str, title: str, ax, origin_x: float, origin_y: float,
                   point_size: float = 1.2, alpha: float = 0.8, title_fontsize: float = 10.5) -> None:
    split_colors = {
        "train": COLOR_TRAIN,
        "val": COLOR_VAL,
        "test": COLOR_TEST,
        "buffer": COLOR_BUFFER,
    }

    shuffled_df = df.sample(frac=1.0, random_state=0)
    ax.scatter(
        shuffled_df["x"],
        shuffled_df["y"],
        c=shuffled_df[split_col].map(split_colors),
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value - origin_x:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value - origin_y:,.0f}"))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Easting from origin (m)", fontsize=9.5)
    ax.set_ylabel("Northing from origin (m)", fontsize=9.5)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=6, label=name)
        for name, color in split_colors.items()
        if (df[split_col] == name).any()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=0.85,
        borderpad=0.4,
    )


def main() -> None:
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    plots_df = load_plot_level_dataframe()
    origin_x = plots_df["x"].min()
    origin_y = plots_df["y"].min()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6))

    plot_split_map(
        plots_df,
        "plot_split",
        "A. Plot-level split (4survey only)\nrandom, neighbours end up in different splits",
        ax_a,
        origin_x,
        origin_y,
        title_fontsize=10.5,
    )

    plot_split_map(
        plots_df,
        "buffered_split",
        f"B. Spatial-block split (4survey only)\nwhole compartments + {SPATIAL_BUFFER_METRES}m train buffer",
        ax_b,
        origin_x,
        origin_y,
        point_size=2,
        alpha=0.7,
        title_fontsize=10.5,
    )

    fig.text(
        0.01,
        0.01,
        f"Origin (0, 0) = Easting {origin_x:,.0f}, Northing {origin_y:,.0f} (EPSG:27700)",
        fontsize=10,
        color=COLOR_NEUTRAL_EDGE,
    )

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
