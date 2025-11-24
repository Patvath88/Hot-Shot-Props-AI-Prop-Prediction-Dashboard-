import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_PATH = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
print("📦 Loading dataset...")
df = pd.read_csv(DATA_PATH)

target_stats = [
    "points", "rebounds", "assists", "threept_fg",
    "steals", "blocks", "minutes",
    "points_assists", "points_rebounds",
    "rebounds_assists", "points_rebounds_assists"
]

feature_cols = [
    c for c in df.columns if c not in ["game_date", "player_name"]
]

print(f"✅ Found {len(df)} records and {len(feature_cols)} features")

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
for target in target_stats:
    if target not in df.columns:
        print(f"⚠️ Skipping missing stat: {target}")
        continue

    print(f"\n🎯 Training models for {target}...")

    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=6, subsample=0.8, random_state=42)
    lgbm = LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=31, random_state=42)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    joblib.dump(rf, MODELS_DIR / f"rf_{target}.pkl")
    joblib.dump(xgb, MODELS_DIR / f"xgb_{target}.pkl")
    joblib.dump(lgbm, MODELS_DIR / f"lgbm_{target}.pkl")

    print(f"✅ Saved rf_{target}.pkl, xgb_{target}.pkl, lgbm_{target}.pkl")

print("\n🏁 All models trained and saved successfully.")