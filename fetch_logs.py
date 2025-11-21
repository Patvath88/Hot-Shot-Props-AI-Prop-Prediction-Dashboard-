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

# 🔒 Ball Don't Lie API key (hardcoded for GitHub Actions)
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # Replace if needed

if not API_KEY:
    raise ValueError("❌ Missing BALL_DONT_LIE_API_KEY — please set it in the code.")

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE_URL = "https://api.balldontlie.io/v1"

# -------------------------------------------------
# RATE-LIMIT SAFE GET REQUEST
# -------------------------------------------------
def safe_api_get(url, params=None, headers=None, retries=5):
    """Handles API rate limits and retries automatically."""
    for attempt in range(retries):
        resp = requests.get(url, params=params, headers=headers)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 10))
            print(f"⏳ Rate limited. Waiting {retry_after} seconds before retry...")
            time.sleep(retry_after)
            continue
        elif resp.status_code >= 500:
            print(f"⚠️ Server error {resp.status_code}, retrying in 3s...")
            time.sleep(3)
            continue
        return resp
    print(f"❌ Failed after {retries} retries: {url}")
    return resp

# -------------------------------------------------
# FETCH FUNCTIONS
# -------------------------------------------------
def fetch_active_players():
    """Fetch all active NBA players (cursor-based pagination)."""
    print("🏀 Fetching all active players...")
    players = []
    cursor = None

    while True:
        params = {"per_page": 100}
        if cursor:
            params["cursor"] = cursor

        resp = safe_api_get(f"{BASE_URL}/players/active", params=params, headers=HEADERS)
        if resp.status_code != 200:
            print(f"⚠️ Failed fetching active players: {resp.status_code}")
            break

        data = resp.json()
        players.extend(data.get("data", []))

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(0.5)

    print(f"✅ Retrieved {len(players)} active players")
    return pd.DataFrame(players)

def fetch_player_game_logs(player_id, season=2025):
    """Fetch game logs for a given player with retry + rate-limit handling."""
    logs = []
    cursor = None

    while True:
        params = {"player_ids[]": player_id, "seasons[]": season, "per_page": 100}
        if cursor:
            params["cursor"] = cursor

        resp = safe_api_get(f"{BASE_URL}/stats", params=params, headers=HEADERS)
        if resp.status_code != 200:
            print(f"⚠️ Error fetching stats for player {player_id}: {resp.status_code}")
            break

        data = resp.json()
        player_logs = data.get("data", [])
        if not player_logs:
            break

        logs.extend(player_logs)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(0.6)

    return logs

# -------------------------------------------------
# MAIN FETCH PIPELINE
# -------------------------------------------------
def main():
    players_df = fetch_active_players()
    if players_df.empty:
        print("⚠️ No active players found.")
        return

    # Load existing logs if available to prevent duplicates
    existing_data = pd.read_csv(RAW_LOGS_FILE) if RAW_LOGS_FILE.exists() else pd.DataFrame()
    all_logs = []

    for _, row in players_df.iterrows():
        pid = row["id"]
        name = f"{row['first_name']} {row['last_name']}"
        print(f"⛓ Fetching logs for {name} (ID {pid})")

        logs = fetch_player_game_logs(pid)
        if not logs:
            print(f"⏭ No new games for {name}")
            continue

        new_entries = []
        for g in logs:
            game = g.get("game", {})
            game_date = game.get("date", "").split("T")[0] if "date" in game else ""
            if not game_date:
                continue

            # Skip if already logged
            if not existing_data.empty and (
                (existing_data["player_name"] == name) &
                (existing_data["GAME_DATE"] == game_date)
            ).any():
                continue

            stats = {
                "GAME_DATE": game_date,
                "player_name": name,
                "points": g.get("pts", 0),
                "rebounds": g.get("reb", 0),
                "assists": g.get("ast", 0),
                "threept_fg": g.get("fg3m", 0),
                "steals": g.get("stl", 0),
                "blocks": g.get("blk", 0),
                "minutes": int(g["min"].split(":")[0]) if g.get("min") else 0,
            }
            new_entries.append(stats)

        if new_entries:
            all_logs.extend(new_entries)
            print(f"✅ Added {len(new_entries)} new logs for {name}")
        else:
            print(f"⏭ No new games for {name}")

        time.sleep(0.5)

    if not all_logs:
        print("⚠️ No new logs fetched.")
        return

    new_df = pd.DataFrame(all_logs)
    combined = pd.concat([existing_data, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["player_name", "GAME_DATE"], inplace=True)
    combined.to_csv(RAW_LOGS_FILE, index=False)

    print(f"✅ Saved {len(new_df)} new logs (total {len(combined)} entries) → {RAW_LOGS_FILE}")

if __name__ == "__main__":
    main()
