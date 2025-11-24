import requests
import pandas as pd
from pathlib import Path
import time
import json
from datetime import datetime

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RAW_LOGS_FILE = DATA_DIR / "raw_logs.csv"
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # replace if needed
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# -------------------------------------------------
# FETCH UTILITIES
# -------------------------------------------------
def fetch_active_players():
    """Fetch list of active NBA players with cursor pagination."""
    print("🏀 Fetching active players...")
    players = []
    cursor = None
    while True:
        params = {"per_page": 100}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{BASE_URL}/players/active", headers=HEADERS, params=params)
        if r.status_code != 200:
            print(f"⚠️ Player fetch failed: {r.status_code}")
            break
        data = r.json()
        players.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.4)
    print(f"✅ Retrieved {len(players)} active players")
    return pd.DataFrame(players)


def fetch_player_games(player_id, season=2025):
    """Fetch game logs for a player using cursor pagination."""
    logs = []
    cursor = None
    while True:
        params = {"player_ids[]": player_id, "seasons[]": season, "per_page": 100}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{BASE_URL}/stats", headers=HEADERS, params=params)
        if r.status_code == 429:
            print(f"⚠️ Rate limited, pausing 3s for player {player_id}")
            time.sleep(3)
            continue
        if r.status_code != 200:
            print(f"⚠️ Failed for player {player_id}: {r.status_code}")
            break
        data = r.json()
        logs.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.4)
    return logs

# -------------------------------------------------
# MAIN FETCH
# -------------------------------------------------
def main():
    print("🚀 Starting data refresh...")
    players_df = fetch_active_players()

    # Load previous logs to avoid duplicate fetching
    if RAW_LOGS_FILE.exists():
        old_df = pd.read_csv(RAW_LOGS_FILE)
        fetched_players = set(old_df["player_name"].unique())
    else:
        old_df = pd.DataFrame()
        fetched_players = set()

    all_logs = []

    for _, player in players_df.iterrows():
        name = f"{player['first_name']} {player['last_name']}"
        pid = player["id"]

        # Skip players already in dataset if they have recent logs
        if name in fetched_players:
            continue

        print(f"⛓ Fetching logs for {name} (ID {pid})")
        stats = fetch_player_games(pid)
        for s in stats:
            game = s.get("game", {})
            log = {
                "GAME_DATE": game.get("date", "").split("T")[0],
                "player_name": name,
                "points": s.get("pts", 0),
                "rebounds": s.get("reb", 0),
                "assists": s.get("ast", 0),
                "threept_fg": s.get("fg3m", 0),
                "steals": s.get("stl", 0),
                "blocks": s.get("blk", 0),
                "minutes": int(s.get("min", "0:00").split(":")[0]) if s.get("min") else 0,
            }
            all_logs.append(log)

        time.sleep(0.4)

    # Combine new + old
    new_df = pd.DataFrame(all_logs)
    if not new_df.empty:
        full_df = pd.concat([old_df, new_df]).drop_duplicates(subset=["player_name", "GAME_DATE"])
        full_df.to_csv(RAW_LOGS_FILE, index=False)
        print(f"✅ Updated raw logs ({len(full_df)} total rows)")
    else:
        print("⚠️ No new data found.")

# -------------------------------------------------
# EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    main()