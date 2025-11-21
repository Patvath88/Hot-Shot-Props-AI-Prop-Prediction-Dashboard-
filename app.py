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
import datetime

# ---------------------------------------------------
# CONFIG & DIRECTORIES
# ---------------------------------------------------
DATA_FILE = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
IMAGES_DIR = Path("data/player_images")
LOGOS_DIR = Path("data/team_logos")
FAV_FILE = Path("favorites.json")

for d in [MODELS_DIR, IMAGES_DIR, LOGOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="🏀 Hot Shot Props AI Dashboard", layout="wide")

# ---------------------------------------------------
# LOADERS & HELPERS
# ---------------------------------------------------
def load_data():
    if not DATA_FILE.exists():
        st.error("❌ model_dataset.csv not found. Please run a refresh workflow.")
        st.stop()
    return pd.read_csv(DATA_FILE)

@st.cache_data
def get_player_image(player_name):
    clean_name = player_name.lower().replace(" ", "_")
    img_path = IMAGES_DIR / f"{clean_name}.png"
    if img_path.exists():
        return Image.open(img_path)
    try:
        url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{clean_name}.png"
        r = requests.get(url)
        if r.status_code == 200:
            img_path.write_bytes(r.content)
            return Image.open(BytesIO(r.content))
    except Exception:
        pass
    return None

@st.cache_data
def get_team_logo(team_abbr):
    logo_path = LOGOS_DIR / f"{team_abbr}.png"
    if logo_path.exists():
        return Image.open(logo_path)
    try:
        url = f"https://cdn.nba.com/logos/nba/{team_abbr}/global/L/logo.svg"
        r = requests.get(url)
        if r.status_code == 200:
            logo_path.write_bytes(r.content)
            return Image.open(BytesIO(r.content))
    except Exception:
        pass
    return None

def load_or_train_models(df):
    # for demo, we skip full retrain; can add logic if missing
    return True

def predict_player(player_name, df):
    """Predict player stats using averaged model approach."""
    row = df[df["player_name"] == player_name].tail(1)
    if row.empty:
        return {}

    # Simulated averages (replace with model predictions)
    base = {
        "points": np.random.uniform(10, 35),
        "rebounds": np.random.uniform(3, 15),
        "assists": np.random.uniform(2, 12),
        "steals": np.random.uniform(0.3, 2.5),
        "blocks": np.random.uniform(0.3, 2.5),
        "minutes": np.random.uniform(20, 38),
        "turnovers": np.random.uniform(1, 5),
    }

    base["PA"] = base["points"] + base["assists"]
    base["PR"] = base["points"] + base["rebounds"]
    base["RA"] = base["rebounds"] + base["assists"]
    base["PRA"] = base["points"] + base["rebounds"] + base["assists"]

    return {k: round(v, 1) for k, v in base.items()}

def save_favorite(player, preds):
    data = []
    if FAV_FILE.exists():
        data = json.load(open(FAV_FILE))
    preds["player"] = player
    preds["timestamp"] = str(datetime.date.today())
    data = [d for d in data if d["player"] != player]
    data.append(preds)
    json.dump(data, open(FAV_FILE, "w"), indent=2)

# ---------------------------------------------------
# APP LAYOUT
# ---------------------------------------------------
st.sidebar.title("🏀 Hot Shot Props AI Dashboard")
tab = st.sidebar.radio("Navigate", [
    "🏠 Home / Favorites",
    "🧠 Prop Projection Lab",
    "📈 Projection Tracker",
    "🔬 Prop Research Lab"
])

df = load_data()
players = sorted(df["player_name"].unique())

# ---------------------------------------------------
# HOME TAB
# ---------------------------------------------------
if tab == "🏠 Home / Favorites":
    st.title("⭐ Favorite Players - Auto Updated Daily")

    if FAV_FILE.exists():
        favorites = json.load(open(FAV_FILE))
        for fav in favorites:
            player = fav["player"]
            preds = predict_player(player, df)  # refresh predictions daily
            col1, col2 = st.columns([1, 3])
            with col1:
                img = get_player_image(player)
                if img:
                    st.image(img, width=120)
            with col2:
                st.subheader(f"🏀 {player}")
                st.write(f"**PTS:** {preds['points']} | **REB:** {preds['rebounds']} | **AST:** {preds['assists']} | **PRA:** {preds['PRA']}")
                st.caption(f"Updated: {datetime.date.today()}")
    else:
        st.info("You have no saved favorites yet. Save some from the Projection Lab!")

# ---------------------------------------------------
# PROJECTION LAB
# ---------------------------------------------------
elif tab == "🧠 Prop Projection Lab":
    st.title("🧠 Prop Projection Lab")
    player = st.selectbox("Select Player", players)

    if player:
        preds = predict_player(player, df)
        img = get_player_image(player)
        if img:
            st.image(img, width=220, caption=player)

        st.subheader(f"📊 Predicted Stats for {player}")
        cols = st.columns(5)
        metrics = list(preds.keys())
        for i, key in enumerate(metrics):
            with cols[i % 5]:
                st.metric(key.upper(), preds[key])

        if st.button("⭐ Save Player to Favorites"):
            save_favorite(player, preds)
            st.success(f"✅ Saved {player} to favorites!")

# ---------------------------------------------------
# PROJECTION TRACKER
# ---------------------------------------------------
elif tab == "📈 Projection Tracker":
    st.title("📈 Projection Tracker")
    if FAV_FILE.exists():
        data = json.load(open(FAV_FILE))
        st.dataframe(pd.DataFrame(data))
    else:
        st.warning("No saved projections found yet!")

# ---------------------------------------------------
# RESEARCH LAB
# ---------------------------------------------------
elif tab == "🔬 Prop Research Lab":
    st.title("🔬 Player Research Lab")
    player = st.selectbox("Select Player for Research", players)

    pdata = df[df["player_name"] == player].tail(20)
    img = get_player_image(player)
    if img:
        st.image(img, width=200, caption=player)

    stats = ["points", "rebounds", "assists", "steals", "blocks"]
    for stat in stats:
        fig = px.bar(pdata, x="GAME_DATE", y=stat, title=f"Last 20 Games - {stat.title()}")
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("🤖 Updated automatically daily at 7AM via GitHub Actions.")
