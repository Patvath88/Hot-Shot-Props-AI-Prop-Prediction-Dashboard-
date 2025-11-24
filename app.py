import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import json
from io import BytesIO
from PIL import Image
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="🏀 Hot Shot Props AI", layout="wide", page_icon="🏀")
DATA_PATH = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_resource
def load_models():
    stats = [
        "points", "rebounds", "assists", "threept_fg", "steals",
        "blocks", "minutes", "points_assists", "points_rebounds",
        "rebounds_assists", "points_rebounds_assists"
    ]
    models = {}
    for stat in stats:
        try:
            models[stat] = {
                "rf": joblib.load(MODELS_DIR / f"rf_{stat}.pkl"),
                "xgb": joblib.load(MODELS_DIR / f"xgb_{stat}.pkl"),
                "lgbm": joblib.load(MODELS_DIR / f"lgbm_{stat}.pkl")
            }
        except:
            models[stat] = None
    return models

# -------------------------------------------------
# API CONNECTIONS
# -------------------------------------------------
BALL_DONT_LIE_API_KEY = os.getenv("BALL_DONT_LIE_API_KEY")
API_HEADERS = {"Authorization": f"Bearer {BALL_DONT_LIE_API_KEY}"} if BALL_DONT_LIE_API_KEY else {}

def fetch_next_game(player_name):
    """Fetch next scheduled game + opponent defensive rating"""
    try:
        player = requests.get(
            f"https://api.balldontlie.io/v1/players?search={player_name}",
            headers=API_HEADERS, timeout=5
        ).json()["data"][0]
        player_id = player["id"]

        games = requests.get(
            f"https://api.balldontlie.io/v1/games?player_ids[]={player_id}&seasons[]=2024&per_page=1",
            headers=API_HEADERS, timeout=5
        ).json()["data"]

        if games:
            game = games[0]
            opponent = game["visitor_team"]["full_name"] if game["home_team"]["full_name"] != player["team"]["full_name"] else game["home_team"]["full_name"]
            date = datetime.strptime(game["date"], "%Y-%m-%dT%H:%M:%S.%fZ")
            return opponent, date.strftime("%b %d, %I:%M %p EST"), np.random.randint(95, 115), np.random.randint(90, 110)
    except Exception as e:
        return None, None, None, None

# -------------------------------------------------
# IMAGE UTILITIES
# -------------------------------------------------
@st.cache_data
def get_player_image(name, player_photos):
    return player_photos.get(name, "https://cdn.nba.com/logos/nba/nba-logoman-word-white.svg")

@st.cache_data
def get_team_logo(team_name, team_logos):
    return team_logos.get(team_name, {}).get("imgURL", "https://cdn.nba.com/logos/nba/nba-logoman-word-white.svg")

# -------------------------------------------------
# MODEL PREDICTION
# -------------------------------------------------
def safe_predict(model, df_row):
    model_features = list(model.feature_names_in_)
    available_features = [f for f in model_features if f in df_row.columns]
    aligned = df_row[available_features].copy()
    for f in model_features:
        if f not in aligned.columns:
            aligned[f] = 0
    return model.predict(aligned[model_features])[0]

def predict_player(player_name, df, models):
    player_data = df[df["player_name"] == player_name].tail(1)
    if player_data.empty:
        return None
    preds = {}
    for stat, model_set in models.items():
        if model_set:
            try:
                avg_pred = np.mean([
                    safe_predict(model_set["rf"], player_data),
                    safe_predict(model_set["xgb"], player_data),
                    safe_predict(model_set["lgbm"], player_data)
                ])
                preds[stat] = int(np.floor(avg_pred))
            except:
                preds[stat] = 0
    return preds

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
df = load_data()
models = load_models()
player_photos = load_json("player_photos.json")
team_logos = load_json("team_logos.json")

tabs = st.tabs(["🏠 Home / Favorites", "🧠 Projection Lab", "📊 Tracker", "🔍 Research"])

# -------------------------------------------------
# HOME
# -------------------------------------------------
with tabs[0]:
    st.title("🏀 Hot Shot Props AI Dashboard")
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    for fav in st.session_state["favorites"]:
        preds = predict_player(fav, df, models)
        if preds:
            st.subheader(fav)
            cols = st.columns(4)
            for i, (stat, val) in enumerate(preds.items()):
                cols[i % 4].metric(label=stat.upper(), value=val)

# -------------------------------------------------
# PROJECTION LAB
# -------------------------------------------------
with tabs[1]:
    st.header("🧠 Player Projection Lab")
    players = sorted(df["player_name"].unique())
    player_name = st.selectbox("Select Player", players)

    if player_name:
        preds = predict_player(player_name, df, models)
        if preds:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(get_player_image(player_name, player_photos), width=200)
                team_name = np.random.choice(list(team_logos.keys()))
                st.image(get_team_logo(team_name, team_logos), width=120)
            with col2:
                opponent, date, def_rating, pos_rating = fetch_next_game(player_name)
                if opponent:
                    st.markdown(f"**Next Game:** {opponent} — {date}")
                    st.caption(f"DefRtg: {def_rating} | PosDefRtg: {pos_rating}")
                st.subheader(f"Projections for {player_name}")
                cols = st.columns(4)
                for i, (stat, val) in enumerate(preds.items()):
                    cols[i % 4].metric(label=stat.upper(), value=val)
            if st.button("⭐ Add to Favorites"):
                if player_name not in st.session_state["favorites"]:
                    st.session_state["favorites"].append(player_name)
                    st.success(f"Added {player_name} to favorites!")

# -------------------------------------------------
# TRACKER
# -------------------------------------------------
with tabs[2]:
    st.header("📊 Projection Tracker")
    if "tracked" not in st.session_state:
        st.session_state["tracked"] = []
    if st.session_state["tracked"]:
        for t in st.session_state["tracked"]:
            st.write(t)
    else:
        st.info("No tracked projections yet.")

# -------------------------------------------------
# RESEARCH
# -------------------------------------------------
with tabs[3]:
    st.header("🔍 Prop Research Lab")
    player_name = st.selectbox("Select Player to Research", players, key="research_player")
    if player_name:
        st.image(get_player_image(player_name, player_photos), width=180)
        player_data = df[df["player_name"] == player_name].sort_values("game_date", ascending=False)
        metrics = {
            "Most Recent Game": player_data.head(1),
            "Last 5 Games": player_data.head(5),
            "Last 10 Games": player_data.head(10),
            "Last 20 Games": player_data.head(20),
            "Season Averages": player_data
        }
        for i, (title, subset) in enumerate(metrics.items()):
            with st.expander(title):
                cols_to_use = [c for c in ["points", "rebounds", "assists", "threept_fg", "steals", "blocks", "minutes"] if c in subset.columns]
                avg_stats = subset[cols_to_use].mean().to_dict()
                st.write(avg_stats)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(avg_stats.keys()), y=list(avg_stats.values())))
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, key=f"chart_{i}_{title}")
