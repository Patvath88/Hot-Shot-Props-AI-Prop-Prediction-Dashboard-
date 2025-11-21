import pandas as pd
import xgboost as xgb
import os

def train_stat(target):
    df = pd.read_csv("data/model_dataset.csv")

    features = [
        "points_rolling5",
        "reb_rolling5",
        "ast_rolling5",
        "min_rolling5",
        "minutes"
    ]

    X = df[features]
    y = df[target]

    model = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9
    )

    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    model.save_model(f"models/{target}.json")

    print(f"Model saved: models/{target}.json")
    return model


if __name__ == "__main__":
    train_stat("points")
