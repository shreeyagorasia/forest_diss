"""Add validated GWA Weibull and multiscale terrain candidates to the plot export.

Run as:
    python -m data_processing.add_environmental_candidates

The Global Wind Atlas country endpoint supplies a national raster. This script crops it to a
5 km buffer around the real Aberfoyle plots, writes only that crop to the permanent cache, and
extracts values at plot coordinates. Terrain candidates reuse the cached 50 m OS Terrain tiles.
Existing columns and rows in plot_environmental_features.parquet are preserved.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.transform import rowcol
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "environmental"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "environmental" / "plot_environmental_features.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "environmental" / "candidate_feature_metadata.json"

GWA_INPUTS = {
    "gwa_weibull_a_10m": RAW_DIR / "gbr_weibull_a_10m.tif",
    "gwa_weibull_k_10m": RAW_DIR / "gbr_weibull_k_10m.tif",
    "gwa_wind_speed_50m": RAW_DIR / "gbr_wind_50m.tif",
    "gwa_weibull_a_50m": RAW_DIR / "gbr_weibull_a_50m.tif",
    "gwa_weibull_k_50m": RAW_DIR / "gbr_weibull_k_50m.tif",
}
GWA_CROPS = {
    column: RAW_DIR / f"aberfoyle_{column}.tif" for column in GWA_INPUTS
}

BUFFER_METRES = 5_000
CRITICAL_WIND_SPEED_MS = 20.0


def crop_raster_to_plot_area(input_path, output_path, plots):
    """Crop one raster to a buffered plot bounding box and return its retained metadata."""
    with rasterio.open(input_path) as source:
        # The GWA Weibull downloads omit the CRS tag, although their bounds/resolution are the
        # same longitude/latitude grid as the tagged GWA mean-wind file. Record and apply that
        # explicit WGS84 interpretation rather than letting an unlabelled raster pass silently.
        crs_was_missing = source.crs is None
        source_crs = source.crs or "EPSG:4326"
        to_source = Transformer.from_crs("EPSG:27700", source_crs, always_xy=True)
        left, bottom = to_source.transform(
            plots["x"].min() - BUFFER_METRES,
            plots["y"].min() - BUFFER_METRES,
        )
        right, top = to_source.transform(
            plots["x"].max() + BUFFER_METRES,
            plots["y"].max() + BUFFER_METRES,
        )

        window = from_bounds(left, bottom, right, top, source.transform)
        window = window.round_offsets().round_lengths()
        full_window = Window(0, 0, source.width, source.height)
        window = window.intersection(full_window)

        data = source.read(window=window)
        profile = source.profile.copy()
        profile.update(
            crs=source_crs,
            width=int(window.width),
            height=int(window.height),
            transform=source.window_transform(window),
            compress="deflate",
        )

    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(data)

    return {
        "source_crs": str(profile["crs"]),
        "original_crs_tag_missing": crs_was_missing,
        "native_pixel_size": [abs(profile["transform"].a), abs(profile["transform"].e)],
        "crop_width": profile["width"],
        "crop_height": profile["height"],
        "buffer_metres": BUFFER_METRES,
    }


def sample_raster_at_plots(path, plots):
    """Sample a raster at BNG plot coordinates, respecting its native CRS and nodata."""
    with rasterio.open(path) as source:
        transformer = Transformer.from_crs("EPSG:27700", source.crs, always_xy=True)
        source_x, source_y = transformer.transform(plots["x"].to_numpy(), plots["y"].to_numpy())
        sampled = np.array([value[0] for value in source.sample(zip(source_x, source_y))], dtype=float)
        if source.nodata is not None:
            sampled[np.isclose(sampled, source.nodata)] = np.nan
    return sampled


def odd_window_size(radius_metres, pixel_size_metres):
    """Convert a radius to a symmetric odd-width moving window."""
    radius_cells = max(1, int(round(radius_metres / pixel_size_metres)))
    return 2 * radius_cells + 1


def nodata_safe_mean(values, valid, size):
    """Moving mean that does not let nodata values contaminate neighbouring cells."""
    weighted_values = uniform_filter(np.where(valid, values, 0.0), size=size, mode="nearest")
    valid_fraction = uniform_filter(valid.astype(float), size=size, mode="nearest")
    return np.divide(
        weighted_values,
        valid_fraction,
        out=np.full(values.shape, np.nan, dtype=float),
        where=valid_fraction > 0,
    )


def sample_array(array, transform, plots):
    rows, columns = rowcol(transform, plots["x"].to_numpy(), plots["y"].to_numpy())
    rows = np.asarray(rows)
    columns = np.asarray(columns)
    in_bounds = (
        (rows >= 0) & (rows < array.shape[0])
        & (columns >= 0) & (columns < array.shape[1])
    )
    sampled = np.full(len(plots), np.nan)
    sampled[in_bounds] = array[rows[in_bounds], columns[in_bounds]]
    return sampled


def derive_multiscale_terrain(plots):
    """Derive TPI at 250/500 m and local relief at 500 m from OS Terrain 50."""
    terrain_paths = sorted(RAW_DIR.glob("N[NS][0-9][0-9].asc"))
    if not terrain_paths:
        raise FileNotFoundError("No OS Terrain ASCII tiles found in data/raw/environmental")

    raw_sources = [rasterio.open(path) for path in terrain_paths]
    # Five ASCII tiles omit a CRS tag; their British National Grid coordinates, 50 m grid and
    # alignment match the tagged NN40 tile. Apply the known OS Terrain CRS explicitly.
    sources = [
        WarpedVRT(source, src_crs=source.crs or "EPSG:27700", crs="EPSG:27700")
        for source in raw_sources
    ]
    try:
        mosaic, transform = merge(sources)
        elevation = mosaic[0].astype(float)
        nodata_values = [source.nodata for source in sources if source.nodata is not None]
        valid = np.isfinite(elevation)
        for nodata in nodata_values:
            valid &= ~np.isclose(elevation, nodata)

        pixel_size = abs(transform.a)
        output = {}
        for radius in [250, 500]:
            size = odd_window_size(radius, pixel_size)
            neighbourhood_mean = nodata_safe_mean(elevation, valid, size)
            output[f"tpi_{radius}m"] = sample_array(elevation - neighbourhood_mean, transform, plots)

        relief_size = odd_window_size(500, pixel_size)
        safe_min_input = np.where(valid, elevation, np.inf)
        safe_max_input = np.where(valid, elevation, -np.inf)
        local_minimum = minimum_filter(safe_min_input, size=relief_size, mode="nearest")
        local_maximum = maximum_filter(safe_max_input, size=relief_size, mode="nearest")
        local_relief = local_maximum - local_minimum
        local_relief[~np.isfinite(local_relief)] = np.nan
        output["local_relief_500m"] = sample_array(local_relief, transform, plots)
    finally:
        for source in sources:
            source.close()
        for source in raw_sources:
            source.close()

    metadata = {
        "source": "OS Terrain 50 cached ASCII tiles",
        "native_pixel_size_metres": pixel_size,
        "tpi_250m": "plot elevation minus circular-scale proxy moving mean; 250 m radius (11x11 cells)",
        "tpi_500m": "plot elevation minus circular-scale proxy moving mean; 500 m radius (21x21 cells)",
        "local_relief_500m": "maximum minus minimum elevation in the same 500 m-radius proxy window",
    }
    return output, metadata


def main():
    plots = pd.read_parquet(FEATURES_PATH)
    original_rows = len(plots)
    if plots["identification"].duplicated().any():
        raise ValueError("Environmental feature export must contain one row per plot")

    metadata = {"critical_wind_speed_ms": CRITICAL_WIND_SPEED_MS}
    for column, national_path in GWA_INPUTS.items():
        crop_path = GWA_CROPS[column]
        if national_path.exists():
            metadata[column] = crop_raster_to_plot_area(national_path, crop_path, plots)
        elif crop_path.exists():
            with rasterio.open(crop_path) as crop:
                metadata[column] = {
                    "source_crs": str(crop.crs),
                    "native_pixel_size": [abs(crop.transform.a), abs(crop.transform.e)],
                    "crop_width": crop.width,
                    "crop_height": crop.height,
                    "buffer_metres": BUFFER_METRES,
                    "reused_existing_aberfoyle_crop": True,
                }
        else:
            raise FileNotFoundError(
                f"Neither national download nor retained Aberfoyle crop exists for {column}"
            )
        plots[column] = sample_raster_at_plots(crop_path, plots)

    a = plots["gwa_weibull_a_10m"].to_numpy(dtype=float)
    k_w = plots["gwa_weibull_k_10m"].to_numpy(dtype=float)
    valid_weibull = np.isfinite(a) & np.isfinite(k_w) & (a > 0) & (k_w > 0)

    plots["gwa_wind_p95_10m"] = np.nan
    plots.loc[valid_weibull, "gwa_wind_p95_10m"] = (
        a[valid_weibull] * (-np.log(1 - 0.95)) ** (1 / k_w[valid_weibull])
    )
    plots["gwa_prob_above_critical_10m"] = np.nan
    plots.loc[valid_weibull, "gwa_prob_above_critical_10m"] = np.exp(
        -(CRITICAL_WIND_SPEED_MS / a[valid_weibull]) ** k_w[valid_weibull]
    )
    metadata["gwa_wind_p95_10m"] = "A * [-ln(1 - 0.95)] ** (1 / k_w)"
    metadata["gwa_prob_above_critical_10m"] = (
        "exp[-(20 m/s / A) ** k_w]; screening threshold, not a universal tree-failure threshold"
    )

    a_50m = plots["gwa_weibull_a_50m"].to_numpy(dtype=float)
    k_50m = plots["gwa_weibull_k_50m"].to_numpy(dtype=float)
    valid_50m = np.isfinite(a_50m) & np.isfinite(k_50m) & (a_50m > 0) & (k_50m > 0)
    plots["gwa_wind_p95_50m"] = np.nan
    plots.loc[valid_50m, "gwa_wind_p95_50m"] = (
        a_50m[valid_50m] * (-np.log(1 - 0.95)) ** (1 / k_50m[valid_50m])
    )
    plots["gwa_prob_above_critical_50m"] = np.nan
    plots.loc[valid_50m, "gwa_prob_above_critical_50m"] = np.exp(
        -(CRITICAL_WIND_SPEED_MS / a_50m[valid_50m]) ** k_50m[valid_50m]
    )
    metadata["gwa_wind_p95_50m"] = "A_50m * [-ln(1 - 0.95)] ** (1 / k_50m)"
    metadata["gwa_prob_above_critical_50m"] = (
        "exp[-(20 m/s / A_50m) ** k_50m]; screening threshold, not a universal tree-failure threshold"
    )

    terrain_columns, terrain_metadata = derive_multiscale_terrain(plots)
    for column, values in terrain_columns.items():
        plots[column] = values
    metadata["multiscale_terrain"] = terrain_metadata

    if len(plots) != original_rows:
        raise RuntimeError("Candidate extraction changed the number of plot rows")

    temporary_path = FEATURES_PATH.with_suffix(".parquet.tmp")
    plots.to_parquet(temporary_path, index=False)
    temporary_path.replace(FEATURES_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    candidate_columns = [
        "gwa_weibull_a_10m", "gwa_weibull_k_10m", "gwa_wind_p95_10m",
        "gwa_prob_above_critical_10m", "gwa_wind_speed_50m", "gwa_weibull_a_50m",
        "gwa_weibull_k_50m", "gwa_wind_p95_50m", "gwa_prob_above_critical_50m",
        "tpi_250m", "tpi_500m", "local_relief_500m",
    ]
    print(f"Updated {FEATURES_PATH} without changing its {len(plots):,} plot rows")
    print(plots[candidate_columns].describe().T)
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
