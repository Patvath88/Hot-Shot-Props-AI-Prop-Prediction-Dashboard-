import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_FILE = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
print("📊 Loading dataset...")
if not DATA_FILE.exists():
    raise FileNotFoundError("❌ Dataset not found! Run build_dataset.py first.")

df = pd.read_csv(DATA_FILE)

# Check required columns
required_cols = {"player_name", "GAME_DATE", "points", "rebounds", "assists"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"❌ Missing required columns: {required_cols - set(df.columns)}")

# Drop any rows with missing numeric data
df = df.dropna(subset=["points", "rebounds", "assists"])

# Encode player names numerically (for model input)
df["player_encoded"] = df["player_name"].astype("category").cat.codes

# -------------------------------------------------
# FEATURES AND TARGETS
# -------------------------------------------------
feature_cols = ["player_encoded", "points_rolling5", "rebounds_rolling5", "assists_rolling5"]
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0  # fallback in case of missing rolling averages

X = df[feature_cols]
y_points = df["points"]
y_rebounds = df["rebounds"]
y_assists = df["assists"]

# -------------------------------------------------
# TRAIN FUNCTION
# -------------------------------------------------
def train_and_save_model(X, y, model_name):
    """Train and save a RandomForest model."""
    print(f"🤖 Training {model_name} model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"✅ {model_name} model trained | MAE: {mae:.2f}")

    out_path = MODELS_DIR / f"{model_name}_model.pkl"
    joblib.dump(model, out_path)
    print(f"💾 Saved {model_name} model → {out_path}")

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
train_and_save_model(X, y_points, "points")
train_and_save_model(X, y_rebounds, "rebounds")
train_and_save_model(X, y_assists, "assists")

print("🎯 All models trained and saved successfully!")
