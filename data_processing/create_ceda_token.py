"""Create a short-lived CEDA access-token file without storing login details."""

from getpass import getpass
from pathlib import Path

import requests


TOKEN_URL = "https://services.ceda.ac.uk/api/token/create/"
TOKEN_PATH = Path("/private/tmp/ceda_access_token")


def main() -> None:
    username = input("CEDA username: ").strip()
    password = getpass("CEDA password (input remains hidden): ")

    response = requests.post(TOKEN_URL, auth=(username, password), timeout=60)
    if not response.ok:
        print(f"CEDA login failed with HTTP status {response.status_code}.")
        raise SystemExit(1)

    access_token = response.json().get("access_token")
    if not access_token:
        print("CEDA responded successfully but did not return an access token.")
        raise SystemExit(1)

    TOKEN_PATH.write_text(access_token, encoding="utf-8")
    TOKEN_PATH.chmod(0o600)
    print(f"Temporary token created at {TOKEN_PATH}.")


if __name__ == "__main__":
    main()
