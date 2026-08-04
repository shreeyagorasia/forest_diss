from __future__ import annotations

from dataclasses import dataclass, asdict

from models.growth_curve_attribution.explain_signal import FINAL_FEATURE_COLUMNS
from models.growth_curve_attribution.run_wind_height_swap_check import SWAPPED_COLUMNS
from models.xgb_environmental.xgb_environmental import ALL_FEATURE_COLUMNS, FEATURE_PROVENANCE


AV2_BROAD_CLIMATE_COLUMNS = [
    "tas_mean",
    "groundfrost_mean",
    "chelsa_bio1_celsius",
    "chelsa_gdd5_degc",
    "chelsa_bio12_precip_mm",
]

AV2_BROAD_SOIL_SITE_COLUMNS = [
    "soilgrids_ph",
    "ceh_pedotope",
    "ceh_subsurface_drainage",
    "ceh_textural_composition",
    "dist_to_watercourse",
]

AV2_BROAD_EDGE_COLUMNS = [
    "dist_to_cpmt_boundary",
    "dist_to_forest_perimeter",
    "dist_to_scpt_boundary",
    "dist_to_block_boundary",
    "cpmt_compactness_ratio",
    "dist_to_road",
]

AV2_MANAGEMENT_COLUMNS = [
    "CanopyCover",
    "Thin",
    "time_since_thinning",
    "time_since_thinning_missing",
    "recent_thinning_5yr",
]

AV2_TEMPORAL_WIND_BASE_COLUMNS = [
    "midas_max_gust_ms",
    "midas_gust_p95_ms",
    "midas_mean_wind_p95_ms",
    "midas_hours_above_critical_per_year",
    "midas_cumulative_gust_excess_per_year",
    "midas_independent_storm_count_per_year",
    "midas_time_since_last_major_storm_days",
]

AV2_TEMPORAL_WIND_EXPOSURE_COLUMNS = [
    "topex",
    "windward_topex",
    "whcl",
    "gwa_wind_speed_50m",
]

AV2_PLANNED_TEMPORAL_DROUGHT_COLUMNS = [
    "haduk_consecutive_dry_days",
    "haduk_dry_spell_count",
    "haduk_cumulative_rain_deficit",
    "haduk_spei3",
    "haduk_spei6",
    "haduk_spei12",
]

AV1_EXCLUDED_COLUMNS = {
    "era5_land_temp_k": "excluded: too coarse / too few distinct values over Aberfoyle",
    "neighbour_mean_height": "excluded: confirmed leakage from nearby real test-set heights",
    "neighbour_height_differential": "excluded: confirmed leakage from nearby real test-set heights",
    "aspect_degrees": "excluded: circular 0-360 bearing replaced by northness/eastness",
}

AV2_EXCLUDED_COLUMNS = {
    "inverse_slope_proxy": "excluded from final AV2 interpretation set: exact duplicate of slope_degrees",
    "tpi_250m": "excluded from AV2 final set after representation check favoured native TPI + 500m TPI",
    "gwa_wind_speed_10m": "excluded from AV2 final set after representation swap to 50m wind",
    "gwa_weibull_a_10m": "excluded from AV2 final set: redundant alternative wind representation",
    "gwa_weibull_k_10m": "excluded from AV2 final set: redundant alternative wind representation",
    "gwa_wind_p95_10m": "excluded from AV2 final set: derived / redundant alternative wind representation",
    "gwa_prob_above_critical_10m": "excluded from AV2 final set: derived / redundant alternative wind representation",
    "gwa_weibull_a_50m": "excluded from AV2 final set: redundant alternative wind representation",
    "gwa_weibull_k_50m": "excluded from AV2 final set: redundant alternative wind representation",
    "gwa_wind_p95_50m": "excluded from AV2 final set: redundant alternative wind representation",
    "gwa_prob_above_critical_50m": "excluded from AV2 final set: redundant alternative wind representation",
    "neighbour_mean_height": "excluded: leaky spatial-lag feature",
    "neighbour_height_differential": "excluded: leaky spatial-lag feature",
    "era5_land_temp_k": "excluded: too coarse for AV2 and not part of approved environmental export set",
}


@dataclass
class VariableRegistryRow:
    variable: str
    provenance: str
    avenue_1_status: str
    avenue_2_static_status: str
    avenue_2_temporal_status: str
    notes: str


def _contains_any(text: str, snippets: list[str]) -> bool:
    lowered = text.lower()
    return any(snippet in lowered for snippet in snippets)


def _av1_status(variable: str) -> str:
    if variable in AV1_EXCLUDED_COLUMNS:
        return AV1_EXCLUDED_COLUMNS[variable]
    if variable in ALL_FEATURE_COLUMNS:
        return "used in AV1 candidate universe"
    return "not part of AV1 export/model feature universe"


def _av2_static_status(variable: str) -> str:
    if variable in FINAL_FEATURE_COLUMNS:
        return "used in AV2 final terrain/wind set"
    if variable in SWAPPED_COLUMNS and variable not in FINAL_FEATURE_COLUMNS:
        return "used in AV2 representation checks only"
    if variable in AV2_BROAD_CLIMATE_COLUMNS:
        return "used in AV2 broad-environment extension (climate)"
    if variable in AV2_BROAD_SOIL_SITE_COLUMNS:
        return "used in AV2 broad-environment extension (soil/site)"
    if variable in AV2_BROAD_EDGE_COLUMNS:
        return "used in AV2 broad-environment extension (edge-position)"
    if variable in AV2_MANAGEMENT_COLUMNS:
        return "used in AV2 management extension"
    if variable in AV2_EXCLUDED_COLUMNS:
        return AV2_EXCLUDED_COLUMNS[variable]
    return "not used in AV2 static analyses"


def _av2_temporal_status(variable: str) -> str:
    if variable in AV2_TEMPORAL_WIND_BASE_COLUMNS:
        return "used in AV2 temporal extension as cohort-level weather summary"
    if variable in AV2_TEMPORAL_WIND_EXPOSURE_COLUMNS:
        return "used in AV2 temporal extension as plot-level exposure partner"
    if variable in AV2_PLANNED_TEMPORAL_DROUGHT_COLUMNS:
        return "planned for later AV2 temporal drought/rain extension"
    return "not part of AV2 temporal extension"


def _notes(variable: str, provenance: str) -> str:
    notes = []
    if _contains_any(provenance, ["own calculation", "derived from"]):
        notes.append("derived variable")
    if variable in AV2_TEMPORAL_WIND_BASE_COLUMNS:
        notes.append("interval weather; constant within cohort unless interacted with exposure")
    if variable in {"tas_mean", "groundfrost_mean"}:
        notes.append("cohort-specific multi-year summary in environmental export")
    if variable in {"ceh_pedotope", "ceh_subsurface_drainage", "ceh_textural_composition"}:
        notes.append("categorical; one-hot encoded in AV2 broad-environment checks")
    if variable in {"gwa_wind_p95_10m", "gwa_prob_above_critical_10m", "gwa_wind_p95_50m", "gwa_prob_above_critical_50m"}:
        notes.append("deterministic transform of GWA Weibull parameters")
    return "; ".join(notes)


def build_variable_registry_rows() -> list[VariableRegistryRow]:
    variables = list(dict.fromkeys(
        list(FEATURE_PROVENANCE)
        + AV2_TEMPORAL_WIND_BASE_COLUMNS
        + AV2_PLANNED_TEMPORAL_DROUGHT_COLUMNS
    ))
    rows = []
    for variable in variables:
        provenance = FEATURE_PROVENANCE.get(variable, "planned temporal variable; not yet in feature export")
        rows.append(
            VariableRegistryRow(
                variable=variable,
                provenance=provenance,
                avenue_1_status=_av1_status(variable),
                avenue_2_static_status=_av2_static_status(variable),
                avenue_2_temporal_status=_av2_temporal_status(variable),
                notes=_notes(variable, provenance),
            )
        )
    return rows


def build_variable_registry_table():
    import pandas as pd

    rows = [asdict(row) for row in build_variable_registry_rows()]
    return pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)
