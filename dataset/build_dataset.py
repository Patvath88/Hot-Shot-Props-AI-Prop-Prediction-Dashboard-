import pandas as pd
import os


def add_features(df):
    df = df.sort_values(["player_name", "GAME_DATE"])

    df["points_rolling5"] = df.groupby("player_name")["points"].rolling(5).mean().reset_index(0, drop=True)
    df["reb_rolling5"] = df.groupby("player_name")["rebounds"].rolling(5).mean().reset_index(0, drop=True)
    df["ast_rolling5"] = df.groupby("player_name")["assists"].rolling(5).mean().reset_index(0, drop=True)
    df["min_rolling5"] = df.groupby("player_name")["minutes"].rolling(5).mean().reset_index(0, drop=True)

    df = df.dropna()
    return df


def build_dataset():
    raw_path = "data/raw_bdl_logs.csv"
    if not os.path.exists(raw_path):
        raise Exception("Run scraper first.")

    df = pd.read_csv(raw_path)
    df = add_features(df)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/model_dataset.csv", index=False)
    print("Saved → data/model_dataset.csv")
    return df


if __name__ == "__main__":
    build_dataset()