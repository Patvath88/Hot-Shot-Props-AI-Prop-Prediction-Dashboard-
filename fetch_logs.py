import requests
import pandas as pd
import os

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def fetch_game_logs(season=2025, pages=200):
    all_rows = []
    for page in range(1, pages + 1):
        url = f"{BASE}/stats?page={page}&per_page=100&seasons[]={season}"
        r = requests.get(url, headers=HEADERS, timeout=20)

        if r.status_code != 200:
            print(f"❌ Failed page {page}")
            break

        data = r.json()
        rows = data.get("data", [])
        if not rows:
            break

        for s in rows:
            game = s["game"]
            player = s["player"]
            row = {
                "GAME_DATE": game["date"][:10],
                "player_name": f"{player['first_name']} {player['last_name']}",
                "points": s.get("pts", 0),
                "rebounds": s.get("reb", 0),
                "assists": s.get("ast", 0),
                "threept_fg": s.get("fg3m", 0),
                "steals": s.get("stl", 0),
                "blocks": s.get("blk", 0),
                "minutes": int(s.get("min", "0").split(":")[0]) if s.get("min") else 0,
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_logs.csv", index=False)
    print(f"✅ Saved data/raw_logs.csv with {len(df)} rows")
    return df

if __name__ == "__main__":
    fetch_game_logs()
