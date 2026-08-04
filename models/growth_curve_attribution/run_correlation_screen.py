# Run as: python -m models.growth_curve_attribution.run_correlation_screen

from models.growth_curve_attribution.correlation_screen import (
    SLOPE_COLUMNS, TPI_COLUMNS, compute_correlation_matrix, summarize_pairwise,
)


def main():
    print("===== TPI at 100m (native) / 250m / 500m, and local_relief_500m =====")
    tpi_correlation, tpi_missing = compute_correlation_matrix(TPI_COLUMNS)
    if tpi_missing:
        print(f"  Missing columns (not in environmental export): {tpi_missing}")
    print(tpi_correlation.round(3))
    print()
    print(summarize_pairwise(tpi_correlation).to_string(index=False))

    print("\n\n===== slope_degrees vs inverse_slope_proxy =====")
    slope_correlation, slope_missing = compute_correlation_matrix(SLOPE_COLUMNS)
    if slope_missing:
        print(f"  Missing columns: {slope_missing}")
    print(slope_correlation.round(4))


if __name__ == "__main__":
    main()
