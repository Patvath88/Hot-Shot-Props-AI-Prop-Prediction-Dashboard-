import pandas as pd
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_DIR = Path("data")
RAW_LOGS_FILE = DATA_DIR / "raw_logs.csv"
MODEL_DATASET_FILE = DATA_DIR / "model_dataset.csv"

# -------------------------------------------------
# BUILD FUNCTION
# -------------------------------------------------
def build_dataset():
    print("📊 Building enriched dataset...")

    if not RAW_LOGS_FILE.exists():
        raise FileNotFoundError(f"❌ Missing {RAW_LOGS_FILE}. Run fetch_logs.py first.")

    df = pd.read_csv(RAW_LOGS_FILE)
    print(f"📥 Loaded {len(df)} rows from raw logs")

    # Ensure required columns exist
    required = ["player_name", "GAME_DATE", "points", "rebounds", "assists", "steals", "blocks", "minutes"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Sort and compute rolling averages
    df = df.sort_values(["player_name", "GAME_DATE"])
    for stat in ["points", "rebounds", "assists", "minutes"]:
        df[f"{stat}_rolling5"] = (
            df.groupby("player_name")[stat]
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )

    # Combo stats
    df["points_assists"] = df["points"] + df["assists"]
    df["points_rebounds"] = df["points"] + df["rebounds"]
    df["rebounds_assists"] = df["rebounds"] + df["assists"]
    df["points_rebounds_assists"] = df["points"] + df["rebounds"] + df["assists"]

    # Clean & save
    df = df.dropna(subset=["player_name"])
    df.to_csv(MODEL_DATASET_FILE, index=False)
    print(f"✅ Saved cleaned dataset to {MODEL_DATASET_FILE}")

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    build_dataset()
