import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="🏀 Hot Shot Props AI", layout="wide", page_icon="🏀")

DATA_PATH = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
PLAYER_JSON = Path("player_photos.json")
TEAM_JSON = Path("team_logos.json")

# -------------------------------------------------
# LOADERS
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df

@st.cache_resource
def load_models():
    models = {}
    stats = [
        "points", "rebounds", "assists", "threept_fg",
        "steals", "blocks", "minutes", "points_assists",
        "points_rebounds", "rebounds_assists", "points_rebounds_assists"
    ]
    for stat in stats:
        try:
            models[stat] = {
                "rf": joblib.load(MODELS_DIR / f"rf_{stat}.pkl"),
                "xgb": joblib.load(MODELS_DIR / f"xgb_{stat}.pkl"),
                "lgbm": joblib.load(MODELS_DIR / f"lgbm_{stat}.pkl"),
            }
        except:
            models[stat] = None
    return models

@st.cache_data
def load_json(path):
    import json
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def fetch_next_game(player_name):
    try:
        url = f"https://www.balldontlie.io/api/v1/players?search={player_name}"
        player_resp = requests.get(url).json()
        if "data" in player_resp and len(player_resp["data"]) > 0:
            player_id = player_resp["data"][0]["id"]
            game_url = f"https://www.balldontlie.io/api/v1/games?player_ids[]={player_id}&seasons[]=2025"
            games = requests.get(game_url).json()
            if games["data"]:
                last_game = games["data"][-1]
                opp = last_game["visitor_team"]["full_name"] if last_game["home_team"]["full_name"] != player_resp["data"][0]["team"]["full_name"] else last_game["home_team"]["full_name"]
                return {
                    "team": player_resp["data"][0]["team"]["full_name"],
                    "opponent": opp,
                    "date": last_game["date"],
                    "time": "7:30 PM EST",
                }
    except:
        pass
    return {"team": "Unknown", "opponent": "TBD", "date": "TBD", "time": "TBD"}

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def safe_predict(model, df_row):
    model_features = list(model.feature_names_in_)
    available = [f for f in model_features if f in df_row.columns]
    aligned = df_row[available].copy()
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
                    safe_predict(model_set["lgbm"], player_data),
                ])
                preds[stat] = int(np.floor(avg_pred))  # Round down to whole number
            except:
                preds[stat] = None
    return preds

# -------------------------------------------------
# LOAD EVERYTHING
# -------------------------------------------------
df = load_data()
models = load_models()
player_photos = load_json(PLAYER_JSON)
team_logos = load_json(TEAM_JSON)

tabs = st.tabs([
    "🏠 Home / Favorites",
    "🧠 Prop Projection Lab",
    "📊 Projection Tracker",
    "🔍 Prop Research Lab"
])

# -------------------------------------------------
# HOME
# -------------------------------------------------
with tabs[0]:
    st.title("🏀 Hot Shot Props AI")
    st.caption("Automatically updated daily projections for your favorite players.")

    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []

    if st.session_state["favorites"]:
        for player in st.session_state["favorites"]:
            preds = predict_player(player, df, models)
            if preds:
                game = fetch_next_game(player)
                st.subheader(f"{player} — {game['team']} vs {game['opponent']}")
                st.caption(f"Next Game: {game['date']} at {game['time']} EST")
                cols = st.columns(4)
                for i, (stat, val) in enumerate(preds.items()):
                    cols[i % 4].metric(stat.upper(), val)
    else:
        st.info("No favorites yet! Add some from the Projection Lab.")

# -------------------------------------------------
# PROJECTION LAB
# -------------------------------------------------
with tabs[1]:
    st.header("🧠 Prop Projection Lab")
    players = sorted(df["player_name"].unique())
    player_name = st.selectbox("Select Player", players)

    if player_name:
        preds = predict_player(player_name, df, models)
        if preds:
            game = fetch_next_game(player_name)
            photo_url = player_photos.get(player_name, {}).get("imgURL")
            team_code = game.get("team", "")[:3].upper()
            logo_url = team_logos.get(team_code, {}).get("imgURL")

            col1, col2 = st.columns([1, 3])
            with col1:
                if photo_url:
                    st.image(photo_url, width=200)
                if logo_url:
                    st.image(logo_url, width=100)
            with col2:
                st.subheader(f"{player_name}")
                st.caption(f"Next Game: {game['team']} vs {game['opponent']} — {game['date']} at {game['time']}")
                cols = st.columns(4)
                for i, (stat, val) in enumerate(preds.items()):
                    cols[i % 4].metric(stat.upper(), val)

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
# TRACKER
# -------------------------------------------------
with tabs[2]:
    st.header("📊 Projection Tracker")
    if "tracked" not in st.session_state:
        st.session_state["tracked"] = []
    if st.session_state["tracked"]:
        for entry in st.session_state["tracked"]:
            for name, stats in entry.items():
                st.subheader(name)
                cols = st.columns(4)
                for i, (stat, val) in enumerate(stats.items()):
                    cols[i % 4].metric(stat.upper(), val)
    else:
        st.info("No tracked projections yet.")

# -------------------------------------------------
# RESEARCH
# -------------------------------------------------
with tabs[3]:
    st.header("🔍 Prop Research Lab")
    player_name = st.selectbox("Select Player to Research", players, key="research_select")
    if player_name:
        player_data = df[df["player_name"] == player_name].sort_values("game_date", ascending=False)
        st.subheader(f"{player_name} Historical Breakdown")

        metrics = {
            "Most Recent Game": player_data.head(1),
            "Last 5 Games": player_data.head(5),
            "Last 10 Games": player_data.head(10),
            "Last 20 Games": player_data.head(20),
            "Season Averages": player_data,
        }

        for i, (title, subset) in enumerate(metrics.items()):
            with st.expander(title):
                cols_to_use = [c for c in [
                    "points", "rebounds", "assists", "threept_fg",
                    "steals", "blocks", "minutes"
                ] if c in subset.columns]
                avg_stats = subset[cols_to_use].mean().to_dict()
                st.write(avg_stats)
                if avg_stats:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(avg_stats.keys()), y=list(avg_stats.values())))
                    fig.update_layout(
                        template="plotly_dark",
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, width=600, key=f"chart_{i}_{title}")
