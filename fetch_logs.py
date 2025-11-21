import requests
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RAW_LOGS_FILE = DATA_DIR / "raw_logs.csv"

# 🔒 Your Ball Don't Lie API key (hardcoded for GitHub Actions)
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # Replace with your key
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE_URL = "https://api.balldontlie.io/v1"


# -------------------------------------------------
# FETCH FUNCTIONS
# -------------------------------------------------
def fetch_active_players():
    """Fetch all active NBA players using cursor pagination."""
    print("🏀 Fetching all active players...")
    players = []
    cursor = None

    while True:
        params = {"per_page": 100}
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(f"{BASE_URL}/players/active", params=params, headers=HEADERS)
        if resp.status_code != 200:
            print(f"⚠️ Failed fetching active players: {resp.status_code}")
            print(resp.text)
            break

        data = resp.json()
        players.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.3)

    print(f"✅ Retrieved {len(players)} active players")
    return pd.DataFrame(players)


def fetch_new_games_only(player_id, last_date):
    """Fetch only new games for a player after the provided date."""
    logs = []
    cursor = None
    while True:
        params = {
            "player_ids[]": player_id,
            "per_page": 100,
            "seasons[]": 2025
        }
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE_URL}/stats", params=params, headers=HEADERS)
        if resp.status_code != 200:
            print(f"⚠️ Error fetching stats for player {player_id}: {resp.status_code}")
            break
        data = resp.json()
        new_logs = [
            g for g in data.get("data", [])
            if "game" in g and g["game"].get("date", "").split("T")[0] > last_date
        ]
        logs.extend(new_logs)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.3)
    return logs


# -------------------------------------------------
# MAIN FETCH PIPELINE
# -------------------------------------------------
def main():
    # Load existing logs if present
    if RAW_LOGS_FILE.exists():
        existing = pd.read_csv(RAW_LOGS_FILE)
        print(f"🔁 Found {len(existing)} existing rows in raw_logs.csv")
    else:
        existing = pd.DataFrame(columns=["GAME_DATE", "player_name"])

    players_df = fetch_active_players()
    if players_df.empty:
        print("⚠️ No active players found.")
        return

    all_new_logs = []

    for _, row in players_df.iterrows():
        pid = row["id"]
        name = f"{row['first_name']} {row['last_name']}"
        if not existing.empty and name in existing["player_name"].values:
            last_date = existing.loc[existing["player_name"] == name, "GAME_DATE"].max()
        else:
            last_date = "1900-01-01"

        new_logs = fetch_new_games_only(pid, last_date)
        if new_logs:
            print(f"🆕 Added {len(new_logs)} new games for {name}")
            for g in new_logs:
                game = g.get("game", {})
                stats = {
                    "GAME_DATE": game.get("date", "").split("T")[0],
                    "player_name": name,
                    "points": g.get("pts", 0),
                    "rebounds": g.get("reb", 0),
                    "assists": g.get("ast", 0),
                    "threept_fg": g.get("fg3m", 0),
                    "steals": g.get("stl", 0),
                    "blocks": g.get("blk", 0),
                    "minutes": int(g["min"].split(":")[0]) if g.get("min") else 0,
                }
                all_new_logs.append(stats)
        else:
            print(f"⏭ No new games for {name}")

    if all_new_logs:
        new_df = pd.DataFrame(all_new_logs)
        combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates()
        combined.to_csv(RAW_LOGS_FILE, index=False)
        print(f"✅ Updated raw_logs.csv with {len(new_df)} new rows")
    else:
        print("✅ No new data found — raw_logs.csv is already up to date")


if __name__ == "__main__":
    main()
