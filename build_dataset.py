import pandas as pd
import os

def build_dataset():
    if not os.path.exists("data/raw_logs.csv"):
        raise FileNotFoundError("Run fetch_logs.py first!")
    df = pd.read_csv("data/raw_logs.csv")
    if df.empty:
        raise ValueError("data/raw_logs.csv is empty!")

    df = df.sort_values(["player_name", "GAME_DATE"])

    df["points_rolling5"] = df.groupby("player_name")["points"].rolling(5).mean().reset_index(drop=True)
    df["reb_rolling5"] = df.groupby("player_name")["rebounds"].rolling(5).mean().reset_index(drop=True)
    df["ast_rolling5"] = df.groupby("player_name")["assists"].rolling(5).mean().reset_index(drop=True)
    df["min_rolling5"] = df.groupby("player_name")["minutes"].rolling(5).mean().reset_index(drop=True)

    df["points_assists"] = df["points"] + df["assists"]
    df["points_rebounds"] = df["points"] + df["rebounds"]
    df["rebounds_assists"] = df["rebounds"] + df["assists"]
    df["points_rebounds_assists"] = df["points"] + df["rebounds"] + df["assists"]

    df = df.dropna()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/model_dataset.csv", index=False)
    print(f"✅ Saved data/model_dataset.csv with {len(df)} rows")
    return df

if __name__ == "__main__":
    build_dataset()
