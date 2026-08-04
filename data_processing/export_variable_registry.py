"""Export one shared AV1/AV2 variable registry for write-up and audit use.

Run as:
    .venv/bin/python -m data_processing.export_variable_registry
"""

from pathlib import Path

from models.common.variable_registry import build_variable_registry_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "documentation" / "variable_registry_av1_av2.csv"


def main():
    registry = build_variable_registry_table()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")
    print(registry.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
