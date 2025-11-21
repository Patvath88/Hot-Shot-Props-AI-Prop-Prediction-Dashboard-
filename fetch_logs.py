import requests
import pandas as pd
import os
import time

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def fetch_game_logs(season=2025, pages=200, retries=3, delay=5):
    """Fetch NBA game logs with retry & fail-safe continuation."""
    all_rows = []

    for page in range(1, pages + 1):
        url = f"{BASE}/stats?page={page}&per_page=100&seasons[]={season}"
        success = False

        for attempt in range(1, retries + 1):
            try:
                r = requests.get(url, headers=HEADERS, timeout=20)
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    if not data:
                        print(f"✅ Finished early at page {page}")
                        success = True
                        break

                    for s in data:
                        mins_str = s.get("min", "0")
                        try:
                            mins = int(mins_str.split(":")[0]) if ":" in mins_str else int(mins_str)
                        except Exception:
                            mins = 0

                        all_rows.append({
                            "GAME_DATE": s["game"]["date"][:10],
                            "player_name": f"{s['player']['first_name']} {s['player']['last_name']}",
                            "points": s.get("pts", 0),
                            "rebounds": s.get("reb", 0),
                            "assists": s.get("ast", 0),
                            "threept_fg": s.get("fg3m", 0),
                            "steals": s.get("stl", 0),
                            "blocks": s.get("blk", 0),
                            "minutes": mins
                        })

                    if page % 25 == 0:
                        print(f"📦 Progress: {page * 100} rows fetched...")
                    success = True
                    break
                else:
                    print(f"⚠️ API returned {r.status_code} on page {page} (attempt {attempt})")
            except Exception as e:
                print(f"⚠️ Error on page {page}, attempt {attempt}: {e}")
            time.sleep(delay)

        if not success:
            print(f"❌ Failed on page {page} after {retries} attempts — continuing to next page.")
            continue

    df = pd.DataFrame(all_rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_logs.csv", index=False)
    print(f"✅ Saved data/raw_logs.csv with {len(df)} rows.")
    return df


if __name__ == "__main__":
    fetch_game_logs()
