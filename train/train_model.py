import xgboost as xgb
import pandas as pd
import os

def train_all():
    df = pd.read_csv("data/model_dataset.csv").dropna()

    features = [
        "points_rolling5",
        "reb_rolling5",
        "ast_rolling5",
        "min_rolling5",
        "minutes"
    ]

    os.makedirs("models", exist_ok=True)

    targets = {
        "points": "points",
        "rebounds": "rebounds",
        "assists": "assists"
    }

    for name, target in targets.items():
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8
        )
        model.fit(df[features], df[target])
        model.save_model(f"models/{name}.json")
        print(f"✅ Trained {name}")

if __name__ == "__main__":
    train_all()
