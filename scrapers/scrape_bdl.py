import requests
import pandas as pd
import os
import time

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE = "https://api.balldontlie.io/v1"

HEADERS = {"Authorization": API_KEY}

def fetch_all_pages(endpoint, params=None):
    """Generic paginator for BDL endpoints."""
    if params is None:
        params = {}

    params["per_page"] = 100
    page = 1
    all_data = []

    while True:
        params["page"] = page
        r = requests.get(f"{BASE}/{endpoint}", headers=HEADERS, params=params)
        data = r.json()

        if "data" not in data:
            break

        all_data.extend(data["data"])

        if data["meta"]["next_page"] is None:
            break

        page += 1
        time.sleep(0.2)

    return all_data


def scrape_logs(season=2025):
    print(f"Fetching NBA logs for season {season}...")

    logs = fetch_all_pages("stats", {"seasons[]": season})

    df = pd.json_normalize(logs)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_logs.csv", index=False)

    print(f"Saved {len(df)} rows → data/raw_logs.csv")
    return df


if __name__ == "__main__":
    scrape_logs()
