import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RAW_FILE = DATA_DIR / "raw_logs.csv"
OUTPUT_FILE = DATA_DIR / "model_dataset.csv"

def build_dataset():
    print("📊 Building enriched dataset...")

    if not RAW_FILE.exists():
        raise FileNotFoundError("raw_logs.csv not found. Run fetch_logs.py first.")

    df = pd.read_csv(RAW_FILE)

    # Validate minimal required columns
    required = ["player_name", "TEAM", "GAME_DATE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "FG3M"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by player and date
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["player_name", "GAME_DATE"])

    # Rolling averages
    for col in ["PTS", "REB", "AST", "MIN"]:
        df[f"{col.lower()}_rolling5"] = df.groupby("player_name")[col].transform(lambda x: x.rolling(5, min_periods=1).mean())

    # Derived combined targets
    df["points_assists"] = df["PTS"] + df["AST"]
    df["points_rebounds"] = df["PTS"] + df["REB"]
    df["rebounds_assists"] = df["REB"] + df["AST"]
    df["points_rebounds_assists"] = df["PTS"] + df["REB"] + df["AST"]
    df["threept_fg"] = df["FG3M"]
    df["minutes"] = df["MIN"]

    keep_cols = [
        "player_name", "TEAM", "GAME_DATE", "points", "rebounds", "assists",
        "steals", "blocks", "threept_fg", "points_assists", "points_rebounds",
        "rebounds_assists", "points_rebounds_assists", "minutes",
        "points_rolling5", "reb_rolling5", "ast_rolling5", "min_rolling5"
    ]

    # Rename columns to standardized lowercase names
    rename_map = {
        "PTS": "points",
        "REB": "rebounds",
        "AST": "assists",
        "STL": "steals",
        "BLK": "blocks",
        "MIN": "minutes"
    }
    df = df.rename(columns=rename_map)

    # Ensure all expected columns exist
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[keep_cols].dropna(subset=["player_name"])

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {OUTPUT_FILE} with {len(df)} rows and {len(df.columns)} columns.")

if __name__ == "__main__":
    build_dataset()
