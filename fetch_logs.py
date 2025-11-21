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
            print(f"❌ Failed on page {page}")
            break

        data = r.json()
        rows = data.get("data", [])
        if not rows:
            break

        for s in rows:
            mins = s.get("min", "0")
            mins = int(mins.split(":")[0]) if isinstance(mins, str) and ":" in mins else int(mins or 0)
            all_rows.append({
                "GAME_DATE": s["game"]["date"][:10],
                "player_name": f"{s['player']['first_name']} {s['player']['last_name']}",
                "points": s.get("pts", 0),
                "rebounds": s.get("reb", 0),
                "assists": s.get("ast", 0),
                "minutes": mins,
            })

    df = pd.DataFrame(all_rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_logs.csv", index=False)
    print("✅ Saved data/raw_logs.csv")
    return df


if __name__ == "__main__":
    fetch_game_logs()
