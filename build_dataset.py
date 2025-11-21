import pandas as pd
import os

def build_dataset():
    raw_path = "data/raw_logs.csv"
    if not os.path.exists(raw_path):
        raise Exception("Raw logs missing. Run fetch_logs.py first.")

    df = pd.read_csv(raw_path)
    df = df.sort_values(["player_name", "GAME_DATE"])

    # Convert to datetime
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # -------------------------------
    # Rolling Averages and Variances
    # -------------------------------
    stat_cols = ["points", "rebounds", "assists", "threept_fg", "steals", "blocks", "minutes"]
    for col in stat_cols:
        df[f"{col}_rolling5"] = df.groupby("player_name")[col].rolling(5).mean().reset_index(drop=True)
        df[f"{col}_var5"] = df.groupby("player_name")[col].rolling(5).var().reset_index(drop=True)

    # -------------------------------
    # Rest days between games
    # -------------------------------
    df["rest_days"] = df.groupby("player_name")["GAME_DATE"].diff().dt.days.fillna(0)

    # -------------------------------
    # Combined metrics for props
    # -------------------------------
    df["points_assists"] = df["points"] + df["assists"]
    df["points_rebounds"] = df["points"] + df["rebounds"]
    df["rebounds_assists"] = df["rebounds"] + df["assists"]
    df["points_rebounds_assists"] = df["points"] + df["rebounds"] + df["assists"]

    # -------------------------------
    # Season Averages (long-term baseline)
    # -------------------------------
    season_avg = df.groupby("player_name")[["points", "rebounds", "assists", "minutes"]].expanding().mean().reset_index()
    season_avg = season_avg.rename(columns={
        "points": "season_points_avg",
        "rebounds": "season_reb_avg",
        "assists": "season_ast_avg",
        "minutes": "season_min_avg"
    })
    df = pd.concat([df.reset_index(drop=True), season_avg[["season_points_avg","season_reb_avg","season_ast_avg","season_min_avg"]]], axis=1)

    # Drop incomplete rows
    df = df.dropna()

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/model_dataset.csv", index=False)
    print("✅ Saved enriched data/model_dataset.csv")
    return df


if __name__ == "__main__":
    build_dataset()
