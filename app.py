import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import requests
from datetime import datetime

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props AI Dashboard", layout="wide")

DATA_PATH = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
IMAGES_DIR = Path("data/player_images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

BALL_API = "https://api.balldontlie.io/v1"
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error("Dataset not found. Please run the dataset build workflow first.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["player_name"])
    return df

df = load_data()
players = sorted(df["player_name"].unique())

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def get_player_image(player_name):
    safe_name = player_name.replace(" ", "_").lower()
    img_path = IMAGES_DIR / f"{safe_name}.jpg"
    if img_path.exists():
        return str(img_path)
    try:
        resp = requests.get(f"{BALL_API}/players?search={player_name.split()[0]}", headers=HEADERS, timeout=5)
        data = resp.json()
        if "data" in data and len(data["data"]) > 0:
            # use NBA headshot pattern if available
            player_id = data["data"][0]["id"]
            url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"
            img = requests.get(url)
            if img.status_code == 200:
                with open(img_path, "wb") as f:
                    f.write(img.content)
                return str(img_path)
    except Exception as e:
        st.warning(f"Image load failed: {e}")
    return None

@st.cache_resource
def train_models(player_name, df):
    player_df = df[df["player_name"] == player_name].copy()
    features = [
        "points_rolling5", "reb_rolling5", "ast_rolling5", "min_rolling5",
        "points_assists", "points_rebounds", "rebounds_assists", "points_rebounds_assists"
    ]
    targets = ["points", "assists", "rebounds", "minutes", "steals", "blocks"]
    models = {}
    for target in targets:
        X = player_df[features].fillna(0)
        y = player_df[target].fillna(0)
        if len(y) < 5:
            continue
        rf = RandomForestRegressor(n_estimators=100)
        xgb = XGBRegressor(n_estimators=100, eval_metric="rmse")
        lgbm = lgb.LGBMRegressor(n_estimators=100)
        rf.fit(X, y)
        xgb.fit(X, y)
        lgbm.fit(X, y)
        models[target] = {"rf": rf, "xgb": xgb, "lgb": lgbm}
    return models

def predict_player(player_name, df):
    player_df = df[df["player_name"] == player_name]
    if player_df.empty:
        return None
    latest = player_df.tail(1)
    models = train_models(player_name, df)
    preds = {}
    for stat, model_set in models.items():
        vals = [
            model_set["rf"].predict(latest)[0],
            model_set["xgb"].predict(latest)[0],
            model_set["lgb"].predict(latest)[0],
        ]
        preds[stat] = np.mean(vals)
    return preds

# -------------------------------------------------
# UI
# -------------------------------------------------
st.markdown("""
<style>
body { background-color: #0E1117; color: white; }
[data-testid="stMetricValue"] { color: #00ffcc; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Home / Favorites", "🧪 Prop Projection Lab", "📊 Projection Tracker", "🔬 Prop Research Lab"])

# ---------------- HOME ----------------
with tabs[0]:
    st.header("🏠 Favorites")
    favs_path = Path("data/favorites.json")
    if not favs_path.exists():
        favs_path.write_text(json.dumps([]))
    favorites = json.loads(favs_path.read_text())

    if favorites:
        for p in favorites:
            st.markdown(f"### ⭐ {p}")
    else:
        st.info("No favorite players yet — add one from the Projection Lab.")

# ---------------- PROP PROJECTION LAB ----------------
with tabs[1]:
    st.header("🧪 Prop Projection Lab")
    player = st.selectbox("Select Player", ["Select Player From Dropdown"] + players)

    if player != "Select Player From Dropdown":
        st.write(f"### {player}")
        img = get_player_image(player)
        if img:
            st.image(img, width=180)

        preds = predict_player(player, df)
        if preds:
            st.subheader("📈 Projected Stats (Model Avg)")
            cols = st.columns(5)
            for i, (k, v) in enumerate(preds.items()):
                cols[i % 5].metric(label=k.capitalize(), value=round(v, 1))

            # save favorite
            if st.button("⭐ Add to Favorites"):
                if player not in favorites:
                    favorites.append(player)
                    favs_path.write_text(json.dumps(favorites))
                    st.success(f"{player} added to favorites!")
        else:
            st.warning("No predictions available.")

# ---------------- TRACKER ----------------
with tabs[2]:
    st.header("📊 Prediction Tracker")
    tracker_path = Path("data/tracker.json")
    if not tracker_path.exists():
        tracker_path.write_text(json.dumps([]))
    tracker = json.loads(tracker_path.read_text())

    if tracker:
        st.dataframe(pd.DataFrame(tracker))
        if st.button("🗑 Clear Tracker"):
            tracker_path.write_text(json.dumps([]))
            st.experimental_rerun()
    else:
        st.info("No saved predictions yet.")

# ---------------- RESEARCH ----------------
with tabs[3]:
    st.header("🔬 Prop Research Lab")
    player = st.selectbox("Select Player for Research", ["Select Player From Dropdown"] + players, key="research")

    if player != "Select Player From Dropdown":
        st.subheader(player)
        img = get_player_image(player)
        if img:
            st.image(img, width=180)

        player_df = df[df["player_name"] == player]
        exp = st.expander("📊 Last 10 Games")
        with exp:
            fig = px.bar(player_df.tail(10), x="GAME_DATE", y="points", title="Points - Last 10 Games")
            st.plotly_chart(fig, use_container_width=True)
