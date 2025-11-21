import pandas as pd
import os

def build_dataset():
    raw_path = "data/raw_logs.csv"
    if not os.path.exists(raw_path):
        raise Exception("Run scraper first.")

    df = pd.read_csv(raw_path)

    # standard columns
    df["GAME_DATE"] = pd.to_datetime(df["game.date"])
    df = df.sort_values(["player.id", "GAME_DATE"])

    # rename important fields
    df["player_name"] = df["player.first_name"] + " " + df["player.last_name"]
    df["points"] = df["pts"]
    df["rebounds"] = df["reb"]
    df["assists"] = df["ast"]
    df["minutes"] = df["min"]

    # feature engineering
    df["points_rolling5"] = df.groupby("player.id")["points"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df["reb_rolling5"] = df.groupby("player.id")["rebounds"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df["ast_rolling5"] = df.groupby("player.id")["assists"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df["min_rolling5"] = df.groupby("player.id")["minutes"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)

    # save output
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/model_dataset.csv", index=False)
    print("Saved dataset → data/model_dataset.csv")

    return df


if __name__ == "__main__":
    build_dataset()
