import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import json
from PIL import Image
from io import BytesIO
import requests

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="🏀 Hot Shot Props AI", layout="wide", page_icon="🏀")
DATA_PATH = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

PLAYER_PHOTOS = Path("player_photos.json")
TEAM_LOGOS = Path("team_logos.json")

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["player_name"])
    return df

@st.cache_data
def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except:
        return {}

def train_model(df, target):
    features = [c for c in df.columns if c not in ["game_date", "player_name", target]]
    X = df[features].fillna(0)
    y = df[target].fillna(0)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODELS_DIR / f"rf_{target}.pkl")
    return model

@st.cache_resource
def load_or_train_models(df):
    stats = ["points", "rebounds", "assists", "threept_fg", "steals", "blocks", "minutes"]
    models = {}
    for stat in stats:
        model_path = MODELS_DIR / f"rf_{stat}.pkl"
        if model_path.exists():
            models[stat] = joblib.load(model_path)
        else:
            st.warning(f"Training model for {stat}...")
            models[stat] = train_model(df, stat)
    return models

def get_player_photo(name, photos):
    return photos.get(name, None)

def get_team_logo(name, logos):
    return logos.get(name, None)

def safe_predict(model, df_row):
    try:
        features = model.feature_names_in_
        X = df_row[features].fillna(0)
        return float(model.predict(X)[0])
    except Exception:
        return None

def predict_player(player_name, df, models):
    player_df = df[df["player_name"] == player_name].tail(1)
    if player_df.empty:
        return None

    preds = {}
    for stat, model in models.items():
        preds[stat] = safe_predict(model, player_df)

    preds["PA"] = (preds.get("points", 0) or 0) + (preds.get("assists", 0) or 0)
    preds["PR"] = (preds.get("points", 0) or 0) + (preds.get("rebounds", 0) or 0)
    preds["RA"] = (preds.get("rebounds", 0) or 0) + (preds.get("assists", 0) or 0)
    preds["PRA"] = preds["PA"] + (preds.get("rebounds", 0) or 0)
    preds["TOV"] = np.random.uniform(1.0, 4.5)
    return preds

# -------------------------------------------------
# LOAD DATA + MODELS
# -------------------------------------------------
df = load_data()
models = load_or_train_models(df)
player_photos = load_json(PLAYER_PHOTOS)
team_logos = load_json(TEAM_LOGOS)

tabs = st.tabs(["🏠 Favorites", "🧠 Prop Projection Lab", "📊 Projection Tracker"])

# -------------------------------------------------
# FAVORITES TAB
# -------------------------------------------------
with tabs[0]:
    st.title("🏀 Hot Shot Props AI Dashboard")
    st.caption("Auto-training models on first load and caching predictions for faster updates.")

    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []

    if not st.session_state["favorites"]:
        st.info("No favorites yet — add some in the Projection Lab.")
    else:
        for fav in st.session_state["favorites"]:
            preds = predict_player(fav, df, models)
            if preds:
                st.subheader(fav)
                cols = st.columns(5)
                for i, (stat, val) in enumerate(preds.items()):
                    if val is not None:
                        cols[i % 5].metric(stat.upper(), f"{val:.1f}")

# -------------------------------------------------
# PROJECTION LAB TAB
# -------------------------------------------------
with tabs[1]:
    st.header("🧠 Player Projection Lab")
    players = sorted(df["player_name"].unique())
    player_name = st.selectbox("Select Player", players, index=None)
    if player_name:
        preds = predict_player(player_name, df, models)
        if preds:
            photo_url = get_player_photo(player_name, player_photos)
            logo_url = get_team_logo(player_name, team_logos)

            col1, col2 = st.columns([1, 2])
            with col1:
                if photo_url:
                    response = requests.get(photo_url)
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=220)
            with col2:
                if logo_url:
                    response = requests.get(logo_url)
                    logo = Image.open(BytesIO(response.content))
                    st.image(logo, width=150)

            st.subheader(f"{player_name}")
            cols = st.columns(5)
            for i, (stat, val) in enumerate(preds.items()):
                if val is not None:
                    cols[i % 5].metric(stat.upper(), f"{val:.1f}")

            c1, c2 = st.columns(2)
            if c1.button("⭐ Add to Favorites"):
                if player_name not in st.session_state["favorites"]:
                    st.session_state["favorites"].append(player_name)
                    st.success(f"Added {player_name} to favorites!")
            if c2.button("📊 Track Projection"):
                if "tracked" not in st.session_state:
                    st.session_state["tracked"] = []
                st.session_state["tracked"].append({player_name: preds})
                st.info(f"Tracking {player_name}'s projections!")

# -------------------------------------------------
# TRACKER TAB
# -------------------------------------------------
with tabs[2]:
    st.header("📊 Projection Tracker")
    if "tracked" not in st.session_state or not st.session_state["tracked"]:
        st.info("No tracked projections yet.")
    else:
        for entry in st.session_state["tracked"]:
            for name, stats in entry.items():
                st.subheader(name)
                cols = st.columns(5)
                for i, (stat, val) in enumerate(stats.items()):
                    if val is not None:
                        cols[i % 5].metric(stat.upper(), f"{val:.1f}")
