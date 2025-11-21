import pandas as pd
import os

RAW_PATH = "data/raw_logs.csv"
MODEL_PATH = "data/model_dataset.csv"

def build_dataset():
    if not os.path.exists(RAW_PATH):
        raise Exception("Raw logs missing. Run fetch_logs.py first.")

    df = pd.read_csv(RAW_PATH)
    df = df.sort_values(["player_name", "GAME_DATE"])

    # Compute rolling averages (5-game)
    for col in ["points", "rebounds", "assists", "threept_fg", "steals", "blocks", "minutes"]:
        roll_col = f"{col}_rolling5"
        df[roll_col] = (
            df.groupby("player_name")[col]
              .rolling(window=5, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )

    # Derived combined stats
    df["points_assists"] = df["points"] + df["assists"]
    df["points_rebounds"] = df["points"] + df["rebounds"]
    df["rebounds_assists"] = df["rebounds"] + df["assists"]
    df["points_rebounds_assists"] = df["points"] + df["rebounds"] + df["assists"]

    df = df.dropna().reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv(MODEL_PATH, index=False)
    print(f"✅ Saved dataset to {MODEL_PATH} with {len(df)} rows and {len(df.columns)} columns.")

if __name__ == "__main__":
    build_dataset()
