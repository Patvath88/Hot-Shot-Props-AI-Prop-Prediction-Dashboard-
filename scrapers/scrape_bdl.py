import requests
import pandas as pd
import time
import os

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE = "https://api.balldontlie.io/v1"

HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def fetch_all(endpoint, params=None, max_pages=200):
    """Fetch paginated BDL endpoints reliably."""
    all_rows = []
    page = 1

    while page <= max_pages:
        p = params.copy() if params else {}
        p["page"] = page

        r = requests.get(f"{BASE}/{endpoint}", headers=HEADERS, params=p)
        if r.status_code != 200:
            break

        data = r.json()
        rows = data.get("data", [])
        if not rows:
            break

        all_rows.extend(rows)
        page += 1
        time.sleep(0.25)

    return all_rows


def scrape_season(season=2025):
    print("🔍 Fetching games…")
    games = fetch_all("games", {"seasons[]": season})
    game_map = {g["id"]: g for g in games}

    print("🔍 Fetching stats…")
    stats = fetch_all("stats", {"seasons[]": season}, max_pages=350)

    rows = []
    for s in stats:
        gid = s.get("game", {}).get("id")
        if gid not in game_map:
            continue

        g = game_map[gid]
        row = {
            "GAME_ID": gid,
            "GAME_DATE": g["date"][:10],
            "player_name": s["player"]["first_name"] + " " + s["player"]["last_name"],
            "team": s["team"]["full_name"],
            "points": s["pts"],
            "rebounds": s["reb"],
            "assists": s["ast"],
            "minutes": s.get("min", 0) or 0
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_bdl_logs.csv", index=False)
    print("Saved → data/raw_bdl_logs.csv")
    return df


if __name__ == "__main__":
    scrape_season(2025)