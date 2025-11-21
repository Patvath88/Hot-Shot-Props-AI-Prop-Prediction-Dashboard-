
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "raw_logs.csv"
OUTPUT_FILE = DATA_DIR / "model_dataset.csv"

def build_dataset():
    if not RAW_FILE.exists():
        print("❌ No raw_logs.csv found — run fetch_logs.py first.")
        return

    print("📊 Loading raw logs...")
    df = pd.read_csv(RAW_FILE)

    # Clean up and ensure correct columns
    essential_cols = ["player_name", "TEAM", "OPP", "GAME_DATE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "FG3M"]
    df = df[[col for col in essential_cols if col in df.columns]].copy()

    df.rename(columns={
        "PTS": "points",
        "REB": "rebounds",
        "AST": "assists",
        "STL": "steals",
        "BLK": "blocks",
        "FG3M": "threept_fg",
        "MIN": "minutes"
    }, inplace=True)

    # Sort and calculate rolling stats
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df.sort_values(["player_name", "GAME_DATE"], inplace=True)

    for stat in ["points", "rebounds", "assists", "minutes"]:
        df[f"{stat}_rolling5"] = df.groupby("player_name")[stat].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    # Derived combination features
    df["points_assists"] = df["points"] + df["assists"]
    df["points_rebounds"] = df["points"] + df["rebounds"]
    df["rebounds_assists"] = df["rebounds"] + df["assists"]
    df["points_rebounds_assists"] = df["points"] + df["rebounds"] + df["assists"]

    print("✅ Built enriched dataset with rolling averages.")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved enriched dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_dataset()
