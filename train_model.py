import pandas as pd
import xgboost as xgb
import os

FEATURES = ["points_rolling5", "rebounds_rolling5", "assists_rolling5", "minutes_rolling5", "minutes"]

def train_stat_model(target):
    df = pd.read_csv("data/model_dataset.csv")
    X = df[FEATURES]
    y = df[target]

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    model.save_model(f"models/{target}.json")
    print(f"✅ Saved models/{target}.json")

def main():
    for stat in ["points", "rebounds", "assists"]:
        train_stat_model(stat)

if __name__ == "__main__":
    main()
