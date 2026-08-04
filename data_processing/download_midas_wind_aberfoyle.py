"""Download only nearby MIDAS Open wind stations for the Aberfoyle study.

The short-lived bearer token is read from /private/tmp/ceda_access_token. It
is never printed or copied into the project. The selected stations are the
three nearest stations with complete metadata coverage for 2002--2023.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


VERSION = "dataset-version-202507"
YEARS = set(range(2002, 2024))
TOKEN_PATH = Path("/private/tmp/ceda_access_token")
OUTPUT_ROOT = Path("data/raw/environmental/midas_wind")
LISTING_ROOT = f"https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-mean-wind-obs/{VERSION}"
DAP_ROOT = f"https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-mean-wind-obs/{VERSION}"

STATIONS = {
    "24125": "renfrewshire/24125_glasgow-bishopton/qc-version-1",
    "00212": "perthshire-in-tayside-region/00212_strathallan-airfield/qc-version-1",
    "00982": "lanarkshire/00982_salsburgh/qc-version-1",
}


def year_from_name(name: str) -> int | None:
    try:
        return int(Path(name).stem.rsplit("_", 1)[-1])
    except ValueError:
        return None


def main() -> None:
    if not TOKEN_PATH.exists():
        raise SystemExit(f"Missing temporary CEDA token: {TOKEN_PATH}")
    token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    headers = {"Authorization": f"Bearer {token}"}
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=6, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )))
    manifest = []

    for station_id, relative_directory in STATIONS.items():
        listing_url = f"{LISTING_ROOT}/{relative_directory}/?json="
        listing_response = session.get(listing_url, timeout=60)
        listing_response.raise_for_status()
        items = listing_response.json()["items"]
        selected = [item for item in items if year_from_name(item["name"]) in YEARS]
        if len(selected) != len(YEARS):
            available = sorted(filter(None, (year_from_name(item["name"]) for item in items)))
            raise RuntimeError(f"Station {station_id} lacks requested years; available={available}")

        station_output = OUTPUT_ROOT / station_id
        station_output.mkdir(parents=True, exist_ok=True)
        for item in selected:
            destination = station_output / item["name"]
            if destination.exists() and destination.stat().st_size == item["size"]:
                status = "existing"
            else:
                response = session.get(
                    f"{DAP_ROOT}/{relative_directory}/{item['name']}",
                    headers=headers,
                    timeout=180,
                )
                response.raise_for_status()
                if b"<html" in response.content[:1000].lower():
                    raise RuntimeError(f"CEDA returned HTML instead of data for {item['name']}")
                destination.write_bytes(response.content)
                status = "downloaded"
            manifest.append({
                "station_id": station_id,
                "year": year_from_name(item["name"]),
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "status": status,
                "source_url": f"{DAP_ROOT}/{relative_directory}/{item['name']}",
            })
            print(station_id, year_from_name(item["name"]), status)

    manifest_path = OUTPUT_ROOT / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} files and manifest {manifest_path}")


if __name__ == "__main__":
    main()
