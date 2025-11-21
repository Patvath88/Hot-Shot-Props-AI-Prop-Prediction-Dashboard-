import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# --------------------------------------------
# List of stats to train models for
# --------------------------------------------
TARGETS = [
    "points", "rebounds", "assists",
    "threept_fg", "steals", "blocks",
    "points_assists", "points_rebounds", "rebounds_assists",
    "points_rebounds_assists", "minutes"
]

# --------------------------------------------
# Feature set (covers rolling averages, variance, and context)
# --------------------------------------------
FEATURES = [
    "points_rolling5", "rebounds_rolling5", "assists_rolling5",
    "threept_fg_rolling5", "steals_rolling5", "blocks_rolling5", "minutes_rolling5",
    "points_var5", "rebounds_var5", "assists_var5",
    "threept_fg_var5", "steals_var5", "blocks_var5", "minutes_var5",
    "rest_days",
    "season_points_avg", "season_reb_avg", "season_ast_avg", "season_min_avg"
]


# --------------------------------------------
# Unified training routine
# --------------------------------------------
def train_models_for_stat(target):
    print(f"⚙️ Training models for {target.upper()}")

    df = pd.read_csv("data/model_dataset.csv")
    X = df[FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    os.makedirs("models", exist_ok=True)

    # -----------------------
    # XGBoost
    # -----------------------
    xgb_model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_lambda=2,
        n_jobs=-1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    preds_xgb = xgb_model.predict(X_test)
    rmse_xgb = mean_squared_error(y_test, preds_xgb, squared=False)
    print(f"✅ XGBoost RMSE ({target}): {rmse_xgb:.3f}")
    xgb_model.save_model(f"models/{target}_xgb.json")

    # -----------------------
    # LightGBM
    # -----------------------
    lgb_model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    preds_lgb = lgb_model.predict(X_test)
    rmse_lgb = mean_squared_error(y_test, preds_lgb, squared=False)
    print(f"✅ LightGBM RMSE ({target}): {rmse_lgb:.3f}")
    joblib.dump(lgb_model, f"models/{target}_lgb.pkl")

    # -----------------------
    # Random Forest
    # -----------------------
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    preds_rf = rf_model.predict(X_test)
    rmse_rf = mean_squared_error(y_test, preds_rf, squared=False)
    print(f"✅ RandomForest RMSE ({target}): {rmse_rf:.3f}")
    joblib.dump(rf_model, f"models/{target}_rf.pkl")

    print(f"💾 Models saved for {target.upper()}!\n")


def main():
    for stat in TARGETS:
        train_models_for_stat(stat)


if __name__ == "__main__":
    main()
