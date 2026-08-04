"""Build observed wind-storm metrics for Aberfoyle survey intervals.

Input files are CEDA MIDAS Open UK mean-wind BADC-CSV files. Download only
nearby stations and keep their station metadata beside the observations. This
script does not store credentials and does not treat a station measurement as
plot-scale wind.

Example
-------
.venv/bin/python data_processing/build_midas_wind_intervals.py \
    --raw-dir data/raw/environmental/midas_wind \
    --station-metadata data/raw/environmental/midas_wind/station_metadata.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


KNOT_TO_MS = 0.514444
DEFAULT_GUST_THRESHOLDS_MS = (20.0, 25.0, 30.0)
DEFAULT_EVENT_GAP_HOURS = 48
ABERFOYLE_CENTRE_LAT = 56.17636
ABERFOYLE_CENTRE_LON = -4.46308


def _finite_quantile(values: np.ndarray, probability: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, probability)) if finite.size else np.nan


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def read_badc_csv(path: Path) -> pd.DataFrame:
    """Read the data block of a BADC-CSV file without assuming its line count."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if any("<html" in line.lower() for line in lines[:10]):
        raise ValueError(f"{path} is an HTML login page, not MIDAS data")

    data_line = next(
        (i for i, line in enumerate(lines) if line.strip().lower().startswith("data")),
        None,
    )
    if data_line is None:
        # Some exports are ordinary CSV files.
        frame = pd.read_csv(path)
    else:
        # Let pandas' C parser read the rectangular data block. The final
        # one-field ``end data`` row is harmless and is removed below. This
        # avoids materialising every CSV field as a Python string twice.
        frame = pd.read_csv(path, skiprows=data_line + 1, low_memory=False)
        if len(frame):
            first_column = frame.columns[0]
            frame = frame.loc[
                ~frame[first_column].astype(str).str.strip().str.lower().eq("end data")
            ].copy()
    frame.columns = [_normalise(c) for c in frame.columns]
    frame["source_file"] = str(path)
    return frame


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...], required: bool = True):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise KeyError(f"None of {candidates} found. Available columns: {frame.columns.tolist()}")
    return None


def standardise_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return time, station, mean wind and gust in SI units.

    MIDAS speed fields are stored in knots. Rows with a non-empty QC flag are
    excluded conservatively. The retained/excluded counts remain in the output
    summaries, so this rule can be audited.
    """
    time_col = _first_column(frame, ("ob_end_time", "ob_time", "observation_time", "datetime"))
    station_col = _first_column(frame, ("src_id", "station_id", "source_id"))
    mean_col = _first_column(frame, ("mean_wind_speed", "mean_wind_spd", "wind_speed"), False)
    gust_col = _first_column(frame, ("max_gust_speed", "max_gust_spd", "maximum_gust_speed"), False)
    duration_col = _first_column(frame, ("ob_hour_count", "observation_hour_count", "duration_hours"), False)

    station_ids = pd.to_numeric(frame[station_col], errors="coerce").astype("Int64").astype(str).str.zfill(5)
    out = pd.DataFrame({
        # MIDAS timestamps are UTC. Store timezone-naive UTC values because
        # all comparisons use the same convention and this avoids platform-
        # specific timezone-array failures in large concatenated frames.
        "time": pd.to_datetime(frame[time_col], errors="coerce", utc=True).dt.tz_localize(None),
        "station_id": station_ids,
        "mean_wind_ms": pd.to_numeric(frame[mean_col], errors="coerce") * KNOT_TO_MS if mean_col else np.nan,
        "max_gust_ms": pd.to_numeric(frame[gust_col], errors="coerce") * KNOT_TO_MS if gust_col else np.nan,
        "observation_hours": pd.to_numeric(frame[duration_col], errors="coerce") if duration_col else 1.0,
    })

    flag_columns = [c for c in frame.columns if c.endswith("_qc") or c.endswith("_qc_flag")]
    if flag_columns:
        flags = frame[flag_columns].fillna("").astype(str).apply(lambda s: s.str.strip())
        out["qc_pass"] = flags.eq("").all(axis=1)
    else:
        out["qc_pass"] = True
    out = out.loc[out["time"].notna() & out["qc_pass"]].copy()
    out = out.loc[out["station_id"] != "99999"]
    return out.sort_values(["station_id", "time"])


def _independent_event_count(times: pd.Series, gap_hours: int) -> int:
    times = pd.to_datetime(times).sort_values()
    if times.empty:
        return 0
    return int(1 + times.diff().gt(pd.Timedelta(hours=gap_hours)).sum())


def build_event_catalogue(
    observations: pd.DataFrame,
    thresholds: tuple[float, ...],
    event_gap_hours: int = DEFAULT_EVENT_GAP_HOURS,
) -> pd.DataFrame:
    """Create one row per declustered station-level gust event."""
    rows = []
    station_values = observations["station_id"].to_numpy(dtype=str)
    all_times = observations["time"].to_numpy()
    all_gusts = observations["max_gust_ms"].to_numpy(dtype=float)
    all_means = observations["mean_wind_ms"].to_numpy(dtype=float)
    all_hours = observations["observation_hours"].to_numpy(dtype=float)
    for station_id in sorted(np.unique(station_values)):
        station_mask = station_values == station_id
        for threshold in thresholds:
            exceed_mask = station_mask & np.isfinite(all_gusts) & (all_gusts > threshold)
            indices = np.flatnonzero(exceed_mask)
            if not indices.size:
                continue
            order = np.argsort(all_times[indices])
            indices = indices[order]
            times = all_times[indices]
            split_points = np.flatnonzero(
                np.diff(times).astype("timedelta64[h]").astype(int) > event_gap_hours
            ) + 1
            for event_number, event_indices in enumerate(np.split(indices, split_points), start=1):
                event_times = all_times[event_indices]
                event_gusts = all_gusts[event_indices]
                event_means = all_means[event_indices]
                event_hours = np.clip(all_hours[event_indices], 0, None)
                start = event_times.min()
                end = event_times.max()
                duration = max(
                    float((end - start) / np.timedelta64(1, "h")) + event_hours[-1],
                    float(event_hours.max()),
                )
                rows.append({
                    "event_id": f"{station_id}_{threshold:g}_{event_number:04d}",
                    "midas_station_id": station_id,
                    "threshold_ms": threshold,
                    # Store ISO text here. Inferring a mixed datetime block
                    # from hundreds of NumPy scalars is unstable in the
                    # project macOS/Python build and is unnecessary for CSV.
                    "event_start": np.datetime_as_string(start, unit="s"),
                    "event_end": np.datetime_as_string(end, unit="s"),
                    "duration_hours": duration,
                    "max_gust_ms": np.nanmax(event_gusts),
                    "gust_p95_ms": _finite_quantile(event_gusts, 0.95),
                    "mean_wind_p95_ms": _finite_quantile(event_means, 0.95),
                    "observations_above_threshold": len(event_indices),
                    "cumulative_gust_excess_ms_hours": (
                        (event_gusts - threshold) * event_hours
                    ).sum(),
                    "event_gap_hours": event_gap_hours,
                })
    return pd.DataFrame(rows)


def summarise_interval(
    observations: pd.DataFrame,
    start_year: int,
    end_year: int,
    threshold_ms: float,
    event_gap_hours: int,
    station_id: str | None = None,
) -> dict[str, float]:
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-01-01")
    times = observations["time"].to_numpy()
    mask = (times >= start.to_datetime64()) & (times < end.to_datetime64())
    if station_id is not None:
        mask &= observations["station_id"].to_numpy(dtype=str) == str(station_id)
    gust_all = observations["max_gust_ms"].to_numpy(dtype=float)
    mean_all = observations["mean_wind_ms"].to_numpy(dtype=float)
    hours_all = observations["observation_hours"].to_numpy(dtype=float)
    gust = gust_all[mask & np.isfinite(gust_all)]
    mean = mean_all[mask & np.isfinite(mean_all)]
    exceed_mask = mask & np.isfinite(gust_all) & (gust_all > threshold_ms)
    exceed_gust = gust_all[exceed_mask]
    exceed_hours = np.clip(hours_all[exceed_mask], 0, None)
    exceed_times = times[exceed_mask]
    expected_hours = (end - start).total_seconds() / 3600
    interval_years = expected_hours / (24 * 365.2425)
    observed_hours = np.clip(hours_all[mask & np.isfinite(hours_all)], 0, None).sum()
    last_event = exceed_times.max() if exceed_times.size else np.datetime64("NaT")
    storm_count = int(exceed_times.size > 0) + int(
        (np.diff(exceed_times).astype("timedelta64[h]").astype(int) > event_gap_hours).sum()
    )
    hours_above = exceed_hours.sum()
    cumulative_excess = ((exceed_gust - threshold_ms) * exceed_hours).sum()
    return {
        "previous_lidar_year": start_year,
        "LiDAR_year": end_year,
        "midas_max_gust_ms": np.nanmax(gust) if gust.size else np.nan,
        "midas_gust_p95_ms": _finite_quantile(gust, 0.95),
        "midas_mean_wind_p95_ms": _finite_quantile(mean, 0.95),
        "midas_hours_above_critical": hours_above,
        "midas_hours_above_critical_per_year": hours_above / interval_years,
        "midas_cumulative_gust_excess_ms_hours": cumulative_excess,
        "midas_cumulative_gust_excess_per_year": cumulative_excess / interval_years,
        "midas_independent_storm_count": storm_count,
        "midas_independent_storm_count_per_year": storm_count / interval_years,
        "midas_time_since_last_major_storm_days": (
            float((end.to_datetime64() - last_event) / np.timedelta64(1, "D"))
            if not np.isnat(last_event) else np.nan
        ),
        "midas_observation_coverage": min(observed_hours / expected_hours, 1.0),
        "midas_n_gust_observations": int(gust.size),
        "midas_gust_threshold_ms": threshold_ms,
        "midas_event_gap_hours": event_gap_hours,
    }


def build_metrics(
    raw_dir: Path,
    transitions: pd.DataFrame,
    thresholds: tuple[float, ...],
    observations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if observations is None:
        files = sorted(p for p in raw_dir.rglob("*.csv") if "metadata" not in p.name.lower())
        if not files:
            raise FileNotFoundError(f"No MIDAS CSV files found below {raw_dir}")
        observations = pd.concat(
            [standardise_observations(read_badc_csv(path)) for path in files],
            ignore_index=True,
        )
    intervals = (
        transitions[["previous_lidar_year", "LiDAR_year"]]
        .dropna().drop_duplicates().astype(int).sort_values(["previous_lidar_year", "LiDAR_year"])
    )
    rows = []
    for station_id in sorted(observations["station_id"].astype(str).unique()):
        for threshold in thresholds:
            for interval in intervals.itertuples(index=False):
                row = summarise_interval(
                    observations, interval.previous_lidar_year, interval.LiDAR_year,
                    threshold, DEFAULT_EVENT_GAP_HOURS, station_id,
                )
                row["midas_station_id"] = station_id
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--station-metadata", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/environmental/midas_wind_interval_metrics.parquet"))
    parser.add_argument("--event-output", type=Path, default=Path("data/processed/environmental/aberfoyle_storm_events.csv"))
    parser.add_argument("--thresholds-ms", type=float, nargs="+", default=list(DEFAULT_GUST_THRESHOLDS_MS))
    args = parser.parse_args()

    transition_files = sorted(Path("data/processed/transitions").glob("transition_growth_*survey.parquet"))
    transitions = pd.concat([pd.read_parquet(path) for path in transition_files], ignore_index=True)
    files = sorted(p for p in args.raw_dir.rglob("*.csv") if "metadata" not in p.name.lower())
    observations = pd.concat(
        [standardise_observations(read_badc_csv(path)) for path in files],
        ignore_index=True,
    )
    metrics = build_metrics(args.raw_dir, transitions, tuple(args.thresholds_ms), observations)
    events = build_event_catalogue(observations, tuple(args.thresholds_ms))

    # Metadata columns are merged when their names are unambiguous. Station
    # coordinates, distance and elevation are evidence about spatial support.
    if args.station_metadata and args.station_metadata.exists():
        metadata = read_badc_csv(args.station_metadata)
        station_col = _first_column(metadata, ("src_id", "station_id", "source_id"))
        metadata["midas_station_id"] = (
            pd.to_numeric(metadata[station_col], errors="coerce")
            .astype("Int64").astype(str).str.zfill(5)
        )
        lat = pd.to_numeric(metadata.get("station_latitude"), errors="coerce")
        lon = pd.to_numeric(metadata.get("station_longitude"), errors="coerce")
        lat1 = np.radians(ABERFOYLE_CENTRE_LAT)
        lat2 = np.radians(lat)
        delta_lat = lat2 - lat1
        delta_lon = np.radians(lon - ABERFOYLE_CENTRE_LON)
        metadata["station_distance_km"] = 2 * 6371.0 * np.arcsin(np.sqrt(
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
        ))
        metrics = metrics.merge(metadata, on="midas_station_id", how="left", suffixes=("", "_station"))
        events = events.merge(metadata, on="midas_station_id", how="left", suffixes=("", "_station"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.output, index=False)
    args.event_output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.event_output, index=False)
    print(f"Wrote {len(metrics):,} station-interval-threshold rows to {args.output}")
    print(f"Wrote {len(events):,} station-level storm events to {args.event_output}")


if __name__ == "__main__":
    main()
