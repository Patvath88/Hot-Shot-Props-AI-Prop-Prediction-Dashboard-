import pandas as pd
import os

def build_dataset():
    logs = pd.read_csv("data/player_logs_raw.csv")

    # Normalize nested dictionaries
    logs["player_name"] = logs["player"].apply(lambda x: eval(x)["first_name"] + " " + eval(x)["last_name"])
    logs["team"] = logs["team"].apply(lambda x: eval(x)["full_name"])
    logs["GAME_DATE"] = logs["game"].apply(lambda x: eval(x)["date"].split("T")[0])

    df = logs[[
        "player_name",
        "team",
        "GAME_DATE",
        "pts",
        "reb",
        "ast",
        "min"
    ]].rename(columns={
        "pts": "points",
        "reb": "rebounds",
        "ast": "assists",
        "min": "minutes"
    })

    df = df.sort_values(["player_name", "GAME_DATE"])

    # Rollings
    df["points_rolling5"] = df.groupby("player_name")["points"].rolling(5).mean().reset_index(0, drop=True)
    df["reb_rolling5"] = df.groupby("player_name")["rebounds"].rolling(5).mean().reset_index(0, drop=True)
    df["ast_rolling5"] = df.groupby("player_name")["assists"].rolling(5).mean().reset_index(0, drop=True)
    df["min_rolling5"] = df.groupby("player_name")["minutes"].rolling(5).mean().reset_index(0, drop=True)

    df.to_csv("data/model_dataset.csv", index=False)
    print("✅ model_dataset.csv created")


if __name__ == "__main__":
    build_dataset()
