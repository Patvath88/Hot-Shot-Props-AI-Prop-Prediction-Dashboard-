import streamlit as st
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from pathlib import Path
import json
import os
import time
# -------------------------------------------------
# PATHS
# -------------------------------------------------
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
# ----------------------------------------------------
# CONFIG & DIAGNOSTIC LOADER
# ----------------------------------------------------
st.set_page_config(page_title="🏀 NBA AI Prop Predictor", layout="wide")
DATA_DIR = Path("data")
FAV_FILE = DATA_DIR / "favorites.json"
SAVED_FILE = DATA_DIR / "saved_predictions.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# Diagnostics
start_time = time.time()
st.sidebar.info("🔍 Checking data and model files...")

if not (DATA_DIR / "model_dataset.csv").exists():
    st.error("❌ model_dataset.csv missing. Run pipeline first.")
    st.stop()

try:
    df_diag = pd.read_csv(DATA_DIR / "model_dataset.csv")
    st.sidebar.success(f"✅ Loaded dataset ({len(df_diag)} rows, {len(df_diag.columns)} cols)")
except Exception as e:
    st.error(f"⚠️ Error loading dataset: {e}")
    st.stop()

st.sidebar.write(f"⏱ Load time: {time.time() - start_time:.2f}s")

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def load_json(file_path):
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return []

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def load_saved_predictions():
    if SAVED_FILE.exists():
        return pd.read_csv(SAVED_FILE)
    return pd.DataFrame(columns=["timestamp", "player_name", "stat", "prediction"])

def save_prediction(player_name, stat, prediction):
    df = load_saved_predictions()
    new_row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "player_name": player_name,
        "stat": stat,
        "prediction": round(prediction, 2)
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(SAVED_FILE, index=False)

def delete_prediction(index):
    df = load_saved_predictions()
    if 0 <= index < len(df):
        df = df.drop(index)
        df.to_csv(SAVED_FILE, index=False)

# ----------------------------------------------------
# DATA LOADING
# ----------------------------------------------------
@st.cache_data
def load_df():
    path = DATA_DIR / "model_dataset.csv"
    df = pd.read_csv(path)
    if df.empty:
        st.error("❌ model_dataset.csv is empty.")
        st.stop()
    return df

df = load_df()

FEATURES = ["points_rolling5", "reb_rolling5", "ast_rolling5", "min_rolling5", "minutes"]

TARGETS = [
    "points", "rebounds", "assists", "threept_fg", "steals", "blocks",
    "points_assists", "points_rebounds", "rebounds_assists",
    "points_rebounds_assists", "minutes"
]

# ----------------------------------------------------
# MODEL TRAINING + PREDICTION
# ----------------------------------------------------
def train_player_models(player_df):
    results = {}
    player_df = player_df.tail(20)
    if len(player_df) < 10:
        return None
    X = player_df[FEATURES]
    latest = player_df.iloc[-1][FEATURES].values.reshape(1, -1)
    for target in TARGETS:
        y = player_df[target]
        if y.nunique() < 2:
            continue
        model_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
        model_xgb.fit(X, y)
        pred_xgb = model_xgb.predict(latest)[0]
        model_lgb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
        model_lgb.fit(X, y)
        pred_lgb = model_lgb.predict(latest)[0]
        model_rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        model_rf.fit(X, y)
        pred_rf = model_rf.predict(latest)[0]
        avg_pred = (pred_xgb + pred_lgb + pred_rf) / 3
        results[target] = {
            "XGBoost": round(pred_xgb, 2),
            "LightGBM": round(pred_lgb, 2),
            "RandomForest": round(pred_rf, 2),
            "Average": round(avg_pred, 2)
        }
    return results

# ----------------------------------------------------
# DISPLAY PREDICTIONS
# ----------------------------------------------------
def display_predictions(player, results):
    if not results:
        st.warning("No predictions available for this player.")
        return
    st.subheader(f"📊 Projections for {player}")
    df_preds = pd.DataFrame(results).T
    st.dataframe(df_preds.style.format("{:.2f}"))
    save_col, fav_col = st.columns(2)
    if save_col.button(f"💾 Save Predictions for {player}"):
        for stat, vals in results.items():
            save_prediction(player, stat, vals["Average"])
        st.success(f"✅ Predictions for {player} saved!")
    favorites = load_json(FAV_FILE)
    if player in favorites:
        if fav_col.button("💔 Remove from Favorites"):
            favorites.remove(player)
            save_json(FAV_FILE, favorites)
            st.info(f"Removed {player} from favorites.")
    else:
        if fav_col.button("⭐ Add to Favorites"):
            favorites.append(player)
            save_json(FAV_FILE, favorites)
            st.success(f"Added {player} to favorites!")

# ----------------------------------------------------
# MULTI-TAB UI
# ----------------------------------------------------
st.title("🏀 Hot Shot Props AI Dashboard")
tabs = st.tabs(["🏠 Home", "🔍 Player Search", "💾 Saved Predictions"])

with tabs[0]:
    st.header("⭐ Favorite Players")
    favorites = load_json(FAV_FILE)
    if not favorites:
        st.info("You have no favorite players yet. Add some in 'Player Search'.")
    else:
        for fav_player in favorites:
            st.subheader(f"📈 {fav_player}")
            pdf = df[df["player_name"] == fav_player].sort_values("GAME_DATE")
            results = train_player_models(pdf)
            display_predictions(fav_player, results)
            st.markdown("---")

with tabs[1]:
    st.header("🔍 Search Player")
    players = sorted(df["player_name"].unique())
    player = st.selectbox("Select a Player", ["Select Player From Dropdown"] + players, key="player_select")
    if player != "Select Player From Dropdown":
        pdf = df[df["player_name"] == player].sort_values("GAME_DATE")
        if len(pdf) < 10:
            st.warning("Not enough games to train a model for this player.")
        else:
            st.info(f"Training models for {player} (last 20 games)...")
            results = train_player_models(pdf)
            display_predictions(player, results)

with tabs[2]:
    st.header("💾 Saved Predictions")
    df_saved = load_saved_predictions()
    if df_saved.empty:
        st.info("No saved predictions yet.")
    else:
        st.dataframe(df_saved)
        delete_index = st.number_input("Enter row index to delete", min_value=0,
                                       max_value=len(df_saved) - 1, step=1, key="del_idx")
        if st.button("🗑 Delete Selected Prediction"):
            delete_prediction(int(delete_index))
            st.success("Deleted prediction successfully.")
            st.experimental_rerun()

with st.sidebar:
    st.markdown("### 📅 Data Info")
    model_dataset_path = DATA_DIR / "model_dataset.csv"
    if model_dataset_path.exists():
        mod_time = datetime.fromtimestamp(model_dataset_path.stat().st_mtime)
        st.caption(f"Last dataset update: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.caption("Dataset not found.")
    favs = load_json(FAV_FILE)
    st.markdown("---")
    st.caption(f"⭐ {len(favs)} favorites saved.")
    st.caption(f"💾 {len(load_saved_predictions())} saved predictions.")
