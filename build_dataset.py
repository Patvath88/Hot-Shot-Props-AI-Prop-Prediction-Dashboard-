import pandas as pd
import os
import sys

def build_dataset():
    raw_path = "data/raw_logs.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError("❌ raw_logs.csv not found. Run fetch first.")

    df = pd.read_csv(raw_path)
    if df.empty:
        print("❌ raw_logs.csv is empty — aborting dataset build.")
        sys.exit(1)

    df = df.sort_values(["player_name", "GAME_DATE"])

    # Rolling averages
    for col in ["points", "rebounds", "assists", "threept_fg", "steals", "blocks", "minutes"]:
        df[f"{col}_rolling5"] = df.groupby("player_name")[col].rolling(5).mean().reset_index(drop=True)

    # Derived features
    df["points_assists"] = df["points"] + df["assists"]
    df["points_rebounds"] = df["points"] + df["rebounds"]
    df["rebounds_assists"] = df["rebounds"] + df["assists"]
    df["points_rebounds_assists"] = df["points"] + df["rebounds"] + df["assists"]

    df = df.dropna()
    if df.empty:
        print("❌ No rows left after rolling computation — likely missing enough games per player.")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/model_dataset.csv", index=False)
    print(f"✅ Saved data/model_dataset.csv with {len(df)} rows.")
    return df


if __name__ == "__main__":
    build_dataset()
