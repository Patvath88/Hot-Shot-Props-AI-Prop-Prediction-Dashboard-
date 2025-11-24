import pandas as pd
import os


def build_dataset():
    raw_path = "data/raw_logs.csv"
    if not os.path.exists(raw_path):
        raise Exception("Raw logs missing. Run fetch_logs.py first.")

    df = pd.read_csv(raw_path)
    df = df.sort_values(["player_name", "GAME_DATE"])

    for col in ["points", "rebounds", "assists", "minutes"]:
        df[f"{col}_rolling5"] = (
            df.groupby("player_name")[col].rolling(5).mean().reset_index(drop=True)
        )

    df = df.dropna()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/model_dataset.csv", index=False)
    print("✅ Saved data/model_dataset.csv")
    return df


if __name__ == "__main__":
    build_dataset()
