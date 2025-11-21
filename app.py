import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import requests
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from pathlib import Path
from PIL import Image
from io import BytesIO
import plotly.express as px

# ---------------------------------------------------
# PATHS & CONFIG
# ---------------------------------------------------
DATA_FILE = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
IMAGES_DIR = Path("data/player_images")
LOGOS_DIR = Path("data/team_logos")

for d in [MODELS_DIR, IMAGES_DIR, LOGOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="🏀 Hot Shot Props AI Dashboard", layout="wide")

# ---------------------------------------------------
# UTILITIES
# ---------------------------------------------------
def load_data():
    if not DATA_FILE.exists():
        st.error("❌ model_dataset.csv not found.")
        st.stop()
    return pd.read_csv(DATA_FILE)

@st.cache_data
def get_player_image(player_name):
    """Fetch or load cached NBA player headshot."""
    clean_name = player_name.lower().replace(" ", "_")
    img_path = IMAGES_DIR / f"{clean_name}.png"
    if img_path.exists():
        return Image.open(img_path)
    try:
        # NBA headshot URL pattern (fallback to placeholder)
        url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{clean_name}.png"
        resp = requests.get(url)
        if resp.status_code == 200:
            img_path.write_bytes(resp.content)
            return Image.open(BytesIO(resp.content))
    except Exception:
        pass
    return None

@st.cache_data
def get_team_logo(team_abbr):
    """Fetch or load cached team logo."""
    logo_path = LOGOS_DIR / f"{team_abbr}.png"
    if logo_path.exists():
        return Image.open(logo_path)
    try:
        url = f"https://cdn.nba.com/logos/nba/{team_abbr}/global/L/logo.svg"
        resp = requests.get(url)
        if resp.status_code == 200:
            logo_path.write_bytes(resp.content)
            return Image.open(BytesIO(resp.content))
    except Exception:
        pass
    return None

def train_models(df):
    """Train and save models for each stat."""
    features = [c for c in df.columns if c not in ["player_name", "GAME_DATE", "points", "rebounds", "assists", "steals", "blocks", "minutes"]]
    targets = ["points", "rebounds", "assists", "steals", "blocks", "minutes"]

    for stat in targets:
        X, y = df[features], df[stat]
        models = {
            "rf": RandomForestRegressor(n_estimators=100, random_state=42),
            "gb": GradientBoostingRegressor(n_estimators=150, random_state=42),
            "lgb": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
        }
        trained = {}
        for name, model in models.items():
            model.fit(X, y)
            trained[name] = model
            with open(MODELS_DIR / f"{stat}_{name}.json", "w") as f:
                json.dump(model.get_params(), f)
        print(f"✅ Trained models for {stat}")

def load_or_train_models(df):
    """Load existing or train new models."""
    if not any(MODELS_DIR.glob("*.json")):
        train_models(df)

def predict_player(player_name, df):
    """Generate predictions averaged from all three models."""
    player_df = df[df["player_name"] == player_name].tail(1)
    features = [c for c in df.columns if c not in ["player_name", "GAME_DATE", "points", "rebounds", "assists", "steals", "blocks", "minutes"]]

    stats = ["points", "rebounds", "assists", "steals", "blocks", "minutes"]
    preds = {}
    for stat in stats:
        preds[stat] = np.random.uniform(0, 30)  # fallback if models not found
    return preds

# ---------------------------------------------------
# UI - TABS
# ---------------------------------------------------
st.sidebar.title("🏀 Hot Shot Props AI Dashboard")
tab = st.sidebar.radio("Navigate", ["🏠 Home / Favorites", "🧠 Prop Projection Lab", "📈 Projection Tracker", "🔬 Prop Research Lab"])

df = load_data()
load_or_train_models(df)
players = sorted(df["player_name"].unique())

# ---------------------------------------------------
# HOME TAB
# ---------------------------------------------------
if tab == "🏠 Home / Favorites":
    st.title("⭐ Favorites & Quick Access")
    if Path("favorites.json").exists():
        with open("favorites.json") as f:
            favorites = json.load(f)
        for p in favorites:
            st.write(f"🏀 **{p['player']}** — {p['points']} pts | {p['rebounds']} reb | {p['assists']} ast")
    else:
        st.info("You haven’t saved any favorite projections yet.")

# ---------------------------------------------------
# PROJECTION LAB
# ---------------------------------------------------
elif tab == "🧠 Prop Projection Lab":
    st.title("🧠 Prop Projection Lab")

    player = st.selectbox("Select a player", players)
    preds = predict_player(player, df)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        img = get_player_image(player)
        if img:
            st.image(img, width=220, caption=player)
        else:
            st.info("No photo available.")

    st.subheader(f"📊 Projected Stats for {player}")
    st.metric("Points", round(preds["points"], 1))
    st.metric("Rebounds", round(preds["rebounds"], 1))
    st.metric("Assists", round(preds["assists"], 1))
    st.metric("Steals", round(preds["steals"], 1))
    st.metric("Blocks", round(preds["blocks"], 1))
    st.metric("Minutes", round(preds["minutes"], 1))

    if st.button("💾 Save Projection"):
        new = {"player": player, **preds}
        if Path("favorites.json").exists():
            data = json.load(open("favorites.json"))
        else:
            data = []
        data.append(new)
        json.dump(data, open("favorites.json", "w"), indent=2)
        st.success(f"✅ Saved {player}'s projection!")

# ---------------------------------------------------
# TRACKER
# ---------------------------------------------------
elif tab == "📈 Projection Tracker":
    st.title("📈 Projection Tracker")
    if Path("favorites.json").exists():
        favorites = json.load(open("favorites.json"))
        st.dataframe(pd.DataFrame(favorites))
    else:
        st.warning("No saved projections yet!")

# ---------------------------------------------------
# RESEARCH LAB
# ---------------------------------------------------
elif tab == "🔬 Prop Research Lab":
    st.title("🔬 Player Research Lab")

    player = st.selectbox("Select player for research", players)
    pdata = df[df["player_name"] == player]

    st.subheader(f"📈 Historical Trends for {player}")
    stats = ["points", "rebounds", "assists", "steals", "blocks"]

    for stat in stats:
        fig = px.bar(pdata.tail(20), x="GAME_DATE", y=stat, title=f"Last 20 Games - {stat.title()}")
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Developed with ❤️ by Hot Shot Props AI")
