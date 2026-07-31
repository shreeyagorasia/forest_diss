# Run as: python -m models.common.download_haduk_multi_year
#
# One-off preprocessing step: downloads HadUK-Grid monthly climate rasters for every survey
# year, extracts each plot's value at each year, then DELETES the downloaded raster before
# moving to the next one. This is different from every other environmental source in this
# project (SoilGrids/CHELSA/CEH/GWA all keep their downloaded raw file permanently cached in
# data/raw/environmental/) -- HadUK-Grid needs 7 variables x 6 years = 42 files, and at ~41MB
# each that's over 1.5GB kept permanently, which doesn't fit on this machine's disk (14GB free
# at the time this was written). Downloading one file, extracting the small number of values
# actually needed (just this forest's ~72,000 plots, not the whole UK), then deleting it keeps
# peak usage to about one file's size (~41MB) instead of the full 1.5GB.
#
# Needs a CEDA access token (Bearer token, OAuth2) -- get one from
# https://services.ceda.ac.uk/cedasite/register/info/ once you have a free CEDA account, then
# set it as an environment variable before running this script:
#   export CEDA_ACCESS_TOKEN="your-token-here"
# Never commit this token or paste it into a notebook/script directly -- reading it from the
# environment keeps it out of git history.

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import requests

from models.common.geo import load_plot_coordinates

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_analysis_plot_set():
    # load_plot_coordinates() returns every plot project-wide (230,359) -- the environmental
    # attribution pipeline only ever analyses the 4survey plot universe (6survey is a strict
    # subset of it, see aux_data_resolution_check.ipynb's own setup cell), so restrict to that
    # here too, same as every other environmental source extraction.
    coordinates_df = load_plot_coordinates()
    master_4survey = pd.read_parquet(PROJECT_ROOT / "data/processed/master/clean_master_4survey.parquet")
    ids_4survey = set(master_4survey["identification"].unique())
    return coordinates_df[coordinates_df["identification"].isin(ids_4survey)].reset_index(drop=True)

# Where the small, final extracted table goes -- this is the only thing kept permanently for
# these new variables, everything else (the big downloaded rasters) is temporary.
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "environmental" / "haduk_multi_year.parquet"

# A temp folder for the download-extract-delete cycle -- separate from data/raw/haduk/ (where
# the already-downloaded, already-kept tas files live) so it's obvious at a glance which files
# are meant to be temporary and which are meant to be kept.
TEMP_DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "haduk_temp"

# Every survey year across both cohorts (4survey: 2008/2012/2021/2023, 6survey adds 2002/2006).
SURVEY_YEARS = [2002, 2006, 2008, 2012, 2021, 2023]

# HadUK-Grid's own variable folder names (confirmed by browsing CEDA's public archive listing
# directly, 2026-07-30) -- tas is deliberately NOT in this list, since it's already downloaded
# and extracted separately (see notebooks/environmental_data/aux_data_resolution_check.ipynb).
VARIABLES = ["rainfall", "tasmax", "tasmin", "groundfrost", "sun", "sfcWind"]

CEDA_VERSION = "v20260512"  # same version directory for every variable, confirmed by browsing

# dap.ceda.ac.uk, NOT data.ceda.ac.uk -- data.ceda.ac.uk is the plain browsing/public-listing
# host (works with no auth, but redirects an actual file GET to a login page even with a Bearer
# token, returning an empty 200 response rather than an error -- caught the hard way, see
# CEDA_ACCESS_TOKEN download failure 2026-07-30). dap.ceda.ac.uk is the host CEDA's own token
# documentation actually uses for authenticated downloads.
CEDA_BASE_URL = (
    "https://dap.ceda.ac.uk/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/"
    "v1.3.2.ceda/1km"
)

# Same British National Grid check as export_coordinates.py/aux_data_resolution_check.ipynb's
# HadUK-Grid cell -- GDAL reads this file family's CRS as an unnamed WKT, not a recognised EPSG
# code, so this checks the actual projection parameters instead of trusting a label.
EXPECTED_BNG_PARAMS = {"proj": "tmerc", "lat_0": 49, "lon_0": -2, "k": 0.9996012717, "x_0": 400000, "y_0": -100000}


def download_one_file(variable, year, token):
    filename = f"{variable}_hadukgrid_uk_1km_mon_{year}01-{year}12.nc"
    url = f"{CEDA_BASE_URL}/{variable}/mon/{CEDA_VERSION}/{filename}"

    TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    local_path = TEMP_DOWNLOAD_DIR / filename

    response = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    response.raise_for_status()

    # A wrong host/auth can come back as a 200 OK with an empty or tiny (login-page-redirect)
    # body instead of a real error status -- raise_for_status() alone won't catch that, so check
    # the size actually looks like a real NetCDF file (~40MB) before trusting it. Caught the hard
    # way (2026-07-30): the wrong host returned "0.0 MB" and rasterio's error only showed up much
    # later, deep inside extraction, far from the actual cause.
    if len(response.content) < 1_000_000:
        raise RuntimeError(
            f"Download of {filename} looks wrong -- only {len(response.content):,} bytes "
            f"(expected ~40MB). This usually means the token/URL isn't authenticating correctly "
            f"rather than a real HTTP error (which raise_for_status() would have already caught). "
            f"First 200 bytes of response: {response.content[:200]!r}"
        )

    local_path.write_bytes(response.content)
    return local_path


def extract_plot_values(local_path, plots):
    with rasterio.open(local_path) as ds:
        crs_params = ds.crs.to_dict()
        mismatches = {k: (crs_params.get(k), v) for k, v in EXPECTED_BNG_PARAMS.items() if crs_params.get(k) != v}
        if mismatches:
            raise ValueError(f"{local_path.name}'s CRS doesn't match British National Grid: {mismatches}")

        # Average the 12 monthly bands into one annual mean -- same approach already used for
        # the tas file already downloaded (see aux_data_resolution_check.ipynb).
        monthly = ds.read(masked=True).astype(float)
        annual_mean = monthly.mean(axis=0)

        rows_idx, cols_idx = rasterio.transform.rowcol(ds.transform, plots["x"].values, plots["y"].values)
        rows_idx, cols_idx = np.array(rows_idx), np.array(cols_idx)
        in_bounds = (
            (rows_idx >= 0) & (rows_idx < annual_mean.shape[0])
            & (cols_idx >= 0) & (cols_idx < annual_mean.shape[1])
        )

        values = np.full(len(plots), np.nan)
        sampled = annual_mean[rows_idx[in_bounds], cols_idx[in_bounds]]
        values[in_bounds] = np.ma.filled(sampled, np.nan)

    return values


def main():
    token = os.environ.get("CEDA_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "Set the CEDA_ACCESS_TOKEN environment variable first (export CEDA_ACCESS_TOKEN=...). "
            "Get a token from https://services.ceda.ac.uk/cedasite/register/info/ once you have "
            "a free CEDA account."
        )

    plots = load_analysis_plot_set()
    print(f"Extracting for {len(plots):,} plots.")

    all_rows = []
    for variable in VARIABLES:
        for year in SURVEY_YEARS:
            print(f"--- {variable} {year} ---")
            local_path = download_one_file(variable, year, token)
            size_mb = local_path.stat().st_size / 1e6
            print(f"  Downloaded {size_mb:.1f} MB -> {local_path}")

            try:
                values = extract_plot_values(local_path, plots)
            finally:
                # Delete regardless of whether extraction succeeded -- the whole point of this
                # script is to never let these files pile up, including on a failed run.
                local_path.unlink()
                print(f"  Deleted {local_path.name} (extraction done, not kept)")

            n_valid = int(np.isfinite(values).sum())
            print(f"  {n_valid:,} / {len(plots):,} plots sampled, "
                  f"range {np.nanmin(values):.2f} to {np.nanmax(values):.2f}")

            year_rows = pd.DataFrame({
                "identification": plots["identification"].values,
                "variable": variable,
                "year": year,
                "value": values,
            })
            all_rows.append(year_rows)

            time.sleep(1)  # be polite to CEDA's server between downloads

    combined = pd.concat(all_rows, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined):,} rows ({len(VARIABLES)} variables x {len(SURVEY_YEARS)} years) "
          f"-> {OUTPUT_PATH}")

    if TEMP_DOWNLOAD_DIR.exists() and not any(TEMP_DOWNLOAD_DIR.iterdir()):
        TEMP_DOWNLOAD_DIR.rmdir()


if __name__ == "__main__":
    main()
