import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_PATH = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
PREDICTIONS_FILE = Path("data/predictions.json")
FAVORITES_FILE = Path("data/favorites.json")

for path in [PREDICTIONS_FILE, FAVORITES_FILE]:
    if not path.exists():
        path.write_text("[]")

FEATURES = [
    "points_rolling5", "reb_rolling5", "ast_rolling5", "min_rolling5",
    "points_assists", "points_rebounds", "rebounds_assists", "points_rebounds_assists"
]

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        st.error("❌ model_dataset.csv not found. Run your dataset workflow first.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    return df

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------
def train_models(df, player_name):
    player_df = df[df["player_name"] == player_name].copy()
    if len(player_df) < 5:
        st.warning("Not enough data to train models for this player.")
        return None

    X = player_df[FEATURES]
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=300, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X, player_df["points"])
        results[name] = model

    # Save the models
    MODELS_DIR.mkdir(exist_ok=True)
    for name, model in results.items():
        joblib.dump(model, MODELS_DIR / f"{player_name}_{name.replace(' ', '_')}.pkl")

    return results

# -------------------------------------------------
# FAVORITES & SAVED PREDICTIONS
# -------------------------------------------------
def load_json(file_path):
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return []

def save_json(file_path, data):
    file_path.write_text(json.dumps(data, indent=4))

def add_favorite(player):
    favorites = load_json(FAVORITES_FILE)
    if player not in favorites:
        favorites.append(player)
        save_json(FAVORITES_FILE, favorites)

def remove_favorite(player):
    favorites = load_json(FAVORITES_FILE)
    if player in favorites:
        favorites.remove(player)
        save_json(FAVORITES_FILE, favorites)

def save_prediction(entry):
    preds = load_json(PREDICTIONS_FILE)
    preds.append(entry)
    save_json(PREDICTIONS_FILE, preds)

def delete_prediction(timestamp):
    preds = load_json(PREDICTIONS_FILE)
    preds = [p for p in preds if p["timestamp"] != timestamp]
    save_json(PREDICTIONS_FILE, preds)

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props AI", layout="wide")

tabs = st.tabs(["🏠 Home", "📊 Generate Projections", "⭐ Favorites", "🗂 Prediction Tracker"])

df = load_dataset()
players = sorted(df["player_name"].unique())

# HOME TAB
with tabs[0]:
    st.title("🏀 Hot Shot Props AI Dashboard")
    st.write("Welcome! Your daily projections refresh automatically at **7 AM**.")
    st.write("Use the tabs above to generate, favorite, and track player projections.")
    favorites = load_json(FAVORITES_FILE)

    if favorites:
        st.subheader("⭐ Favorite Players (Auto-Trained)")
        for fav in favorites:
            st.write(f"📈 {fav}")
    else:
        st.info("No favorite players yet. Add some from the 'Generate Projections' tab!")

# GENERATE PROJECTIONS TAB
with tabs[1]:
    st.header("📊 Generate Player Projections")

    player_name = st.selectbox("Select Player From Dropdown", ["Select Player From Dropdown"] + players)

    if player_name != "Select Player From Dropdown":
        st.write(f"Training models for **{player_name}**...")

        models = train_models(df, player_name)
        if models:
            latest_data = df[df["player_name"] == player_name].tail(1)[FEATURES]
            preds = {name: mdl.predict(latest_data)[0] for name, mdl in models.items()}
            avg_pred = sum(preds.values()) / len(preds)

            st.success(f"**Predictions for {player_name}:**")
            for name, value in preds.items():
                st.write(f"- {name}: {value:.2f} points")
            st.write(f"**Average Prediction:** {avg_pred:.2f} points")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("⭐ Add to Favorites"):
                    add_favorite(player_name)
                    st.success(f"Added {player_name} to favorites!")
            with col2:
                if st.button("💾 Save Prediction"):
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "player_name": player_name,
                        "predictions": preds,
                        "average": avg_pred
                    }
                    save_prediction(entry)
                    st.success("Prediction saved!")
            with col3:
                if st.button("🗑 Remove from Favorites"):
                    remove_favorite(player_name)
                    st.warning(f"Removed {player_name} from favorites.")

# FAVORITES TAB
with tabs[2]:
    st.header("⭐ Your Favorite Players")
    favorites = load_json(FAVORITES_FILE)
    if not favorites:
        st.info("No favorites yet.")
    else:
        for fav in favorites:
            st.write(f"📈 {fav}")

# PREDICTION TRACKER TAB
with tabs[3]:
    st.header("🗂 Saved Predictions")
    saved = load_json(PREDICTIONS_FILE)
    if not saved:
        st.info("No saved predictions.")
    else:
        for p in saved:
            st.write(f"**{p['player_name']}** — saved at {p['timestamp']}")
            st.json(p["predictions"])
            st.write(f"Average: {p['average']:.2f}")
            if st.button(f"🗑 Delete {p['player_name']} ({p['timestamp']})"):
                delete_prediction(p["timestamp"])
                st.rerun()
