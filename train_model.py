import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_PATH = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

STATS = [
    "points", "rebounds", "assists", "threept_fg",
    "steals", "blocks", "minutes",
    "points_assists", "points_rebounds",
    "rebounds_assists", "points_rebounds_assists"
]

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
print("📊 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Filter invalid rows
df = df.dropna(subset=["player_name"]).fillna(0)

# All features except player_name and game_date
ignore_cols = ["player_name", "game_date"]
features = [col for col in df.columns if col not in ignore_cols and col not in STATS]

print(f"🧮 Using {len(features)} features for training")

# -------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------
for target in STATS:
    if target not in df.columns:
        print(f"⚠️ Skipping {target}: column not found")
        continue

    X = df[features]
    y = df[target]

    if y.sum() == 0:
        print(f"⚠️ Skipping {target}: all zero values")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "rf": RandomForestRegressor(n_estimators=200, random_state=42),
        "xgb": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, verbosity=0),
        "lgbm": LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=-1, random_state=42)
    }

    print(f"\n🚀 Training models for {target}...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"   ✅ {name.upper()} RMSE: {rmse:.3f}")

        save_path = MODELS_DIR / f"{name}_{target}.pkl"
        joblib.dump(model, save_path)
        print(f"💾 Saved {save_path}")

print("\n✅ All available models trained successfully.")
