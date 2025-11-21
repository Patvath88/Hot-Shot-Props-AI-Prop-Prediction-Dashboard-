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

# 🔒 Your Ball Don't Lie API key (hardcoded for GitHub Actions)
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # Replace if needed

if not API_KEY:
    raise ValueError("❌ Missing BALL_DONT_LIE_API_KEY — please set it in the code.")

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE_URL = "https://api.balldontlie.io/v1"


# -------------------------------------------------
# FETCH FUNCTIONS
# -------------------------------------------------
def fetch_active_players():
    """Fetch list of all active NBA players (cursor-based pagination)."""
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
            break

        data = resp.json()
        players.extend(data.get("data", []))

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(0.4)

    print(f"✅ Retrieved {len(players)} active players")
    return pd.DataFrame(players)


def fetch_player_game_logs(player_id, season=2025):
    """Fetch game logs for a given player (cursor-based pagination)."""
    logs = []
    cursor = None

    while True:
        params = {"player_ids[]": player_id, "seasons[]": season, "per_page": 100}
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(f"{BASE_URL}/stats", params=params, headers=HEADERS)
        if resp.status_code != 200:
            print(f"⚠️ Error fetching stats for player {player_id}")
            break

        data = resp.json()
        logs.extend(data.get("data", []))

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(0.4)

    return logs


# -------------------------------------------------
# MAIN FETCH PIPELINE
# -------------------------------------------------
def main():
    players_df = fetch_active_players()
    if players_df.empty:
        print("⚠️ No active players found.")
        return

    all_logs = []

    for _, row in players_df.iterrows():
        pid = row["id"]
        name = f"{row['first_name']} {row['last_name']}"
        print(f"⛓ Fetching logs for {name} (ID {pid})")

        logs = fetch_player_game_logs(pid)
        for g in logs:
            game = g.get("game", {})
            stats = {
                "GAME_DATE": game.get("date", "").split("T")[0] if "date" in game else "",
                "player_name": name,
                "points": g.get("pts", 0),
                "rebounds": g.get("reb", 0),
                "assists": g.get("ast", 0),
                "threept_fg": g.get("fg3m", 0),
                "steals": g.get("stl", 0),
                "blocks": g.get("blk", 0),
                "minutes": int(g["min"].split(":")[0]) if g.get("min") else 0,
            }
            all_logs.append(stats)

        time.sleep(0.4)

    if not all_logs:
        print("⚠️ No logs fetched!")
        return

    df = pd.DataFrame(all_logs)
    df.to_csv(RAW_LOGS_FILE, index=False)
    print(f"✅ Saved raw logs to {RAW_LOGS_FILE}")


if __name__ == "__main__":
    main()
