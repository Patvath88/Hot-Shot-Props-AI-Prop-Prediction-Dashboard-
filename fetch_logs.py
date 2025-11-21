import requests
import pandas as pd
from pathlib import Path
import time

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RAW_LOGS_FILE = DATA_DIR / "raw_logs.csv"

# Hardcode your Ball Don't Lie API key here:
API_KEY = "YOUR_API_KEY_HERE"  # 🔒 Replace with your real key

if not API_KEY:
    raise ValueError("❌ Missing BALL_DONT_LIE_API_KEY — please set it in the code.")

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE_URL = "https://api.balldontlie.io/v1"


# -------------------------------------------------
# FETCH FUNCTIONS
# -------------------------------------------------
def fetch_active_players():
    """Fetch list of all active NBA players."""
    players = []
    page = 1
    while True:
        resp = requests.get(f"{BASE_URL}/players", params={"page": page, "per_page": 100}, headers=HEADERS)
        if resp.status_code != 200:
            print(f"⚠️ Failed fetching players page {page}")
            break
        data = resp.json()
        players.extend(data["data"])
        if data["meta"]["next_page"] is None:
            break
        page += 1
        time.sleep(0.5)
    return pd.DataFrame(players)


def fetch_player_game_logs(player_id, season=2025):
    """Fetch game logs for a given player."""
    logs = []
    page = 1
    while True:
        resp = requests.get(
            f"{BASE_URL}/stats",
            params={"player_ids[]": player_id, "seasons[]": season, "per_page": 100, "page": page},
            headers=HEADERS,
        )
        if resp.status_code != 200:
            print(f"⚠️ Error fetching stats for player {player_id} page {page}")
            break
        data = resp.json()
        logs.extend(data["data"])
        if data["meta"]["next_page"] is None:
            break
        page += 1
        time.sleep(0.5)
    return logs


# -------------------------------------------------
# MAIN FETCH PIPELINE
# -------------------------------------------------
def main():
    print("🏀 Fetching all active players...")
    players_df = fetch_active_players()
    print(f"✅ Found {len(players_df)} active players")

    all_logs = []

    for _, row in players_df.iterrows():
        pid = row["id"]
        name = f"{row['first_name']} {row['last_name']}"
        print(f"⛓ Fetching logs for {name} (ID {pid})")
        logs = fetch_player_game_logs(pid)
        for g in logs:
            game = g["game"]
            stats = {
                "GAME_DATE": game["date"].split("T")[0],
                "player_name": name,
                "points": g["pts"],
                "rebounds": g["reb"],
                "assists": g["ast"],
                "threept_fg": g["fg3m"],
                "steals": g["stl"],
                "blocks": g["blk"],
                "minutes": int(g["min"].split(":")[0]) if g["min"] else 0,
            }
            all_logs.append(stats)
        time.sleep(0.5)

    if not all_logs:
        print("⚠️ No logs fetched!")
        return

    df = pd.DataFrame(all_logs)
    df.to_csv(RAW_LOGS_FILE, index=False)
    print(f"✅ Saved raw logs to {RAW_LOGS_FILE}")


if __name__ == "__main__":
    main()
