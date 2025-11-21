import requests
import pandas as pd
import os

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE = "https://api.balldontlie.io/v1"

HEADERS = {"Authorization": API_KEY}


def fetch_all(endpoint, params={}):
    """BDL pagination handler."""
    out = []
    page = 1

    while True:
        res = requests.get(
            f"{BASE}/{endpoint}",
            params={**params, "page": page, "per_page": 100},
            headers=HEADERS,
            timeout=15
        ).json()

        if "data" not in res:
            break

        out.extend(res["data"])

        if page >= res.get("meta", {}).get("total_pages", 1):
            break

        page += 1

    return out


def scrape_bdl():
    print("📥 Fetching players...")
    players = fetch_all("players")

    print("📥 Fetching 2025–26 game logs...")
    logs = fetch_all("stats", {"seasons[]": "2025"})

    os.makedirs("data", exist_ok=True)

    pd.DataFrame(players).to_csv("data/players_raw.csv", index=False)
    pd.DataFrame(logs).to_csv("data/player_logs_raw.csv", index=False)

    print("✅ Saved players_raw.csv and player_logs_raw.csv")


if __name__ == "__main__":
    scrape_bdl()
