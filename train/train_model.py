import pandas as pd
import xgboost as xgb
import os

FEATURES = [
    "points_rolling5",
    "reb_rolling5",
    "ast_rolling5",
    "min_rolling5",
    "minutes"
]


def train_stat(df, target, outname):
    X = df[FEATURES]
    y = df[target]

    model = xgb.XGBRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    model.save_model(f"models/{outname}.json")
    print(f"Saved model → models/{outname}.json")
    return model