import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import joblib

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "model_dataset.csv"

def train_stat_model(stat):
    print(f"🤖 Training {stat} model...")

    df = pd.read_csv(DATA_FILE)
    features = ["points_rolling5", "reb_rolling5", "ast_rolling5", "min_rolling5", "minutes"]
    df = df.dropna(subset=features + [stat])

    X = df[features]
    y = df[stat]

    if len(X) < 20:
        print(f"⚠️ Not enough data to train {stat} model.")
        return

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
    xgb_model.fit(X, y)
    xgb_model.save_model(MODELS_DIR / f"{stat}_xgb.json")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
    lgb_model.fit(X, y)
    joblib.dump(lgb_model, MODELS_DIR / f"{stat}_lgb.pkl")

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    rf_model.fit(X, y)
    joblib.dump(rf_model, MODELS_DIR / f"{stat}_rf.pkl")

    print(f"✅ Saved models for {stat}")

def main():
    stats = [
        "points", "rebounds", "assists", "threept_fg", "steals", "blocks",
        "points_assists", "points_rebounds", "rebounds_assists", "points_rebounds_assists", "minutes"
    ]
    for s in stats:
        try:
            train_stat_model(s)
        except Exception as e:
            print(f"⚠️ Error training {s}: {e}")

if __name__ == "__main__":
    main()
