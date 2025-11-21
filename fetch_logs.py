import os
import time
import requests
import pandas as pd
from pathlib import Path

# -----------------------------------------
# CONFIG
# -----------------------------------------
# Hardcoded API key (replace with your actual key)
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RAW_LOGS = DATA_DIR / "raw_logs.csv"

BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": API_KEY}

# -----------------------------------------
# HELPERS
# -----------------------------------------
def get_paginated(endpoint, params=None, max_pages=100):
    """Handles pagination for Ball Don't Lie API requests."""
    all_data = []
    page = 1
    while page <= max_pages:
        if params:
            params["page"] = page
        else:
            params = {"page": page}

        res = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params)
        if res.status_code != 200:
            print(f"⚠️ Failed to fetch {endpoint} page {page}: {res.status_code}")
            break

        data = res.json()
        if "data" not in data or not data["data"]:
            break

        all_data.extend(data["data"])
        print(f"✅ Fetched page {page} ({len(data['data'])} records)")
        if page >= data.get("meta", {}).get("total_pages", 1):
            break

        page += 1
        time.sleep(0.5)  # be polite to API
    return all_data


# -----------------------------------------
# FETCH ACTIVE PLAYERS
# -----------------------------------------
print("🏀 Fetching active players...")
players = get_paginated("players", params={"per_page": 100})
players_df = pd.DataFrame(players)
active_players = players_df[
    players_df["team"].apply(lambda x: isinstance(x, dict) and x.get("full_name"))
]
print(f"✅ Found {len(active_players)} active players.")


# -----------------------------------------
# FETCH GAME LOGS
# -----------------------------------------
all_games = []
for i, player in active_players.iterrows():
    pid = player["id"]
    pname = f"{player['first_name']} {player['last_name']}"
    print(f"Fetching games for {pname} (ID: {pid})")

    games = get_paginated("stats", params={"player_ids[]": pid, "per_page": 100})
    if not games:
        continue

    for g in games:
        game = g.get("game", {})
        all_games.append({
            "GAME_DATE": game.get("date", "").split("T")[0],
            "player_name": pname,
            "points": g.get("pts", 0),
            "rebounds": g.get("reb", 0),
            "assists": g.get("ast", 0),
            "threept_fg": g.get("fg3m", 0),
            "steals": g.get("stl", 0),
            "blocks": g.get("blk", 0),
            "minutes": int(g.get("min", "0").split(":")[0]) if g.get("min") else 0,
        })

    if i % 10 == 0:
        print(f"🌀 Progress: {i}/{len(active_players)} players processed.")
    time.sleep(1.5)  # avoid API rate limit

# -----------------------------------------
# SAVE TO CSV
# -----------------------------------------
if all_games:
    df = pd.DataFrame(all_games)
    df = df.sort_values(["player_name", "GAME_DATE"])
    df.to_csv(RAW_LOGS, index=False)
    print(f"✅ Saved {len(df)} rows to {RAW_LOGS}")
else:
    print("⚠️ No game data fetched.")
