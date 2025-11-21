import pandas as pd
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_DIR = Path("data")
RAW_LOGS_FILE = DATA_DIR / "raw_logs.csv"
MODEL_DATASET_FILE = DATA_DIR / "model_dataset.csv"

# -------------------------------------------------
# MAIN DATASET BUILDER
# -------------------------------------------------
def build_dataset():
    print("📊 Loading raw logs...")
    if not RAW_LOGS_FILE.exists():
        print("❌ No raw_logs.csv found — run fetch_logs.py first.")
        return

    df = pd.read_csv(RAW_LOGS_FILE)
    print(f"✅ Loaded {len(df)} rows from raw logs")

    # Ensure consistent column naming
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename to match expected format if needed
    rename_map = {
        "pts": "points",
        "reb": "rebounds",
        "ast": "assists",
        "fg3m": "threept_fg",
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure all expected stat columns exist
    required_cols = [
        "game_date", "player_name", "points", "rebounds", "assists",
        "threept_fg", "steals", "blocks", "minutes"
    ]

    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ Missing column '{col}' — filling with 0s")
            df[col] = 0

    # Convert numeric columns safely
    numeric_cols = [
        "points", "rebounds", "assists", "threept_fg",
        "steals", "blocks", "minutes"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Sort by player and game date
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["player_name", "game_date"])

    # Rolling averages for last 5 games
    print("🔁 Calculating rolling averages...")
    for stat in ["points", "rebounds", "assists", "minutes"]:
        df[f"{stat}_rolling5"] = (
            df.groupby("player_name")[stat]
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )

    # Combined stats
    print("🧮 Computing combined features...")
    df["points_assists"] = df["points"] + df["assists"]
    df["points_rebounds"] = df["points"] + df["rebounds"]
    df["rebounds_assists"] = df["rebounds"] + df["assists"]
    df["points_rebounds_assists"] = df["points"] + df["rebounds"] + df["assists"]

    # Drop any duplicates or invalid rows
    df.drop_duplicates(subset=["player_name", "game_date"], inplace=True)
    df = df.dropna(subset=["player_name", "game_date"])

    # Save dataset
    df.to_csv(MODEL_DATASET_FILE, index=False)
    print(f"✅ Dataset built successfully and saved to {MODEL_DATASET_FILE}")
    print(f"📊 Final shape: {df.shape}")

    # Display sample output
    print(df.head())

# -------------------------------------------------
# RUN SCRIPT
# -------------------------------------------------
if __name__ == "__main__":
    build_dataset()
