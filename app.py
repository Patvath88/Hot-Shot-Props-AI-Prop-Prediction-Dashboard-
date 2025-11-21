import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import numpy as np

# -------------------------------------------------
# PATHS
# -------------------------------------------------
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DATASET_FILE = DATA_DIR / "model_dataset.csv"

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    if not DATASET_FILE.exists():
        st.error("❌ No dataset found. Please run the daily refresh or manual build workflow.")
        st.stop()
    df = pd.read_csv(DATASET_FILE)
    if df.empty:
        st.error("⚠️ Dataset is empty. Rebuild the dataset.")
        st.stop()
    return df

df = load_data()

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    for stat in ["points", "rebounds", "assists"]:
        model_path = MODELS_DIR / f"{stat}_model.pkl"
        if model_path.exists():
            try:
                models[stat] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"⚠️ Failed to load {stat} model: {e}")
        else:
            st.warning(f"⚠️ Model for {stat} not found. Please run training workflow.")
    return models

models = load_models()

# -------------------------------------------------
# PLAYER SEARCH
# -------------------------------------------------
st.title("🏀 Hot Shot Props AI - Player Projection Dashboard")
st.markdown("Use the dropdown to select a player and view AI-generated stat predictions.")

players = sorted(df["player_name"].unique())
selected_player = st.selectbox("Select Player From Dropdown", ["Select Player From Dropdown"] + players)

if selected_player == "Select Player From Dropdown":
    st.info("👆 Choose a player to see their projections.")
    st.stop()

player_df = df[df["player_name"] == selected_player]

if player_df.empty:
    st.warning("⚠️ No game data available for this player yet.")
    st.stop()

# -------------------------------------------------
# GENERATE PREDICTIONS
# -------------------------------------------------
latest_game = player_df.iloc[-1:]
X = latest_game[
    ["points_rolling5", "reb_rolling5", "ast_rolling5", "min_rolling5", "points_assists", "points_rebounds"]
]

st.subheader(f"📈 Predicted Stats for {selected_player}")

for stat in ["points", "rebounds", "assists"]:
    if stat in models:
        model = models[stat]
        prediction = model.predict(X)[0]
        st.metric(label=f"Predicted {stat.capitalize()}", value=f"{prediction:.1f}")
    else:
        st.warning(f"No model found for {stat}.")

# -------------------------------------------------
# FAVORITES / SAVED PREDICTIONS (Local)
# -------------------------------------------------
FAV_FILE = Path("data/favorites.csv")

def save_favorite(player):
    favs = []
    if FAV_FILE.exists():
        favs = pd.read_csv(FAV_FILE)["player_name"].tolist()
    if player not in favs:
        favs.append(player)
        pd.DataFrame({"player_name": favs}).to_csv(FAV_FILE, index=False)
        st.success(f"⭐ Added {player} to favorites!")

def view_favorites():
    if FAV_FILE.exists():
        favs = pd.read_csv(FAV_FILE)
        st.write("⭐ Favorite Players:")
        st.table(favs)
    else:
        st.info("No favorites saved yet.")

col1, col2 = st.columns(2)
if col1.button("⭐ Add to Favorites"):
    save_favorite(selected_player)
if col2.button("📋 View Favorites"):
    view_favorites()
