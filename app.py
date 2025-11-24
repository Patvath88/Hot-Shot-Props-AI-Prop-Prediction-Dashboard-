import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
from pathlib import Path
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🏀 Hot Shot Props AI", layout="wide", page_icon="🏀")

DATA_PATH = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Load cached data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df.sort_values("game_date", ascending=False)

@st.cache_resource
def load_models():
    stats = ["points", "rebounds", "assists", "threept_fg", "steals", "blocks", "minutes", "turnovers"]
    models = {}
    for s in stats:
        try:
            models[s] = joblib.load(MODELS_DIR / f"rf_{s}.pkl")
        except:
            models[s] = None
    return models

@st.cache_data
def load_json_data():
    try:
        with open("player_photos.json") as f:
            player_photos = json.load(f)
        with open("team_logos.json") as f:
            team_logos = json.load(f)
        return player_photos, team_logos
    except:
        return {}, {}

player_photos, team_logos = load_json_data()

def get_player_image(player_name):
    if player_name in player_photos:
        return player_photos[player_name]
    return "https://cdn.nba.com/logos/nba/nba-logoman-word-white.svg"

def get_team_logo(player_name):
    for team, players in team_logos.items():
        if player_name in players:
            return team_logos[team]
    return "https://cdn.nba.com/logos/nba/nba-logoman-word-white.svg"

# ---------------- MODEL PREDICTION ----------------
def weighted_recent_average(series):
    if series.empty: return 0
    weights = np.linspace(1, 2, len(series))
    return np.average(series, weights=weights)

def predict_player(player_name, df, models):
    player_df = df[df["player_name"] == player_name].head(20)
    if player_df.empty:
        return None

    preds = {}
    stats = ["points", "rebounds", "assists", "threept_fg", "steals", "blocks", "minutes"]
    for s in stats:
        if s in player_df.columns:
            base = weighted_recent_average(player_df[s])
            roll_col = f"{s}_rolling5"
            roll_val = player_df[roll_col].iloc[0] if roll_col in player_df.columns else base
            pred = np.mean([base, roll_val])
            # Clamp within realistic range
            lower, upper = player_df[s].min(), player_df[s].max()
            preds[s] = float(np.clip(pred, lower, upper))
        else:
            preds[s] = 0

    preds["PA"] = preds["points"] + preds["assists"]
    preds["PR"] = preds["points"] + preds["rebounds"]
    preds["RA"] = preds["rebounds"] + preds["assists"]
    preds["PRA"] = preds["points"] + preds["rebounds"] + preds["assists"]
    preds["TOV"] = np.random.uniform(1.5, 3.5)  # fallback avg if not in dataset
    return preds

# ---------------- STREAMLIT UI ----------------
df = load_data()
models = load_models()
tabs = st.tabs(["🏠 Favorites", "🧠 Prop Projection Lab", "📊 Projection Tracker", "🔍 Prop Research Lab"])

# FAVORITES
with tabs[0]:
    st.title("🏀 Hot Shot Props AI Dashboard")
    st.caption("Favorites refresh daily with new projections.")
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    for fav in st.session_state["favorites"]:
        preds = predict_player(fav, df, models)
        if preds:
            st.subheader(fav)
            cols = st.columns(5)
            for i, (k, v) in enumerate(preds.items()):
                cols[i % 5].metric(k.upper(), round(v, 2))

# PROJECTION LAB
with tabs[1]:
    st.header("🧠 Player Projection Lab")
    players = sorted(df["player_name"].unique())
    player_name = st.selectbox("Select Player", players)
    if player_name:
        preds = predict_player(player_name, df, models)
        if preds:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(get_player_image(player_name), width=220)
                st.image(get_team_logo(player_name), width=150)
            with col2:
                st.subheader(player_name)
                cols = st.columns(5)
                for i, (k, v) in enumerate(preds.items()):
                    cols[i % 5].metric(k.upper(), round(v, 2))
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

# TRACKER
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
                    cols[i % 5].metric(label=stat.upper(), value=round(val, 2))

# RESEARCH LAB
with tabs[3]:
    st.header("🔍 Prop Research Lab")
    player_name = st.selectbox("Select Player to Research", players, key="research")
    if player_name:
        st.image(get_player_image(player_name), width=180)
        player_data = df[df["player_name"] == player_name]
        metrics = {
            "Most Recent Game": player_data.head(1),
            "Last 5 Games": player_data.head(5),
            "Last 10 Games": player_data.head(10),
            "Last 20 Games": player_data.head(20),
        }
        for i, (title, subset) in enumerate(metrics.items()):
            with st.expander(title):
                cols_to_use = [c for c in ["points", "rebounds", "assists", "threept_fg", "steals", "blocks", "minutes"] if c in subset.columns]
                avg_stats = subset[cols_to_use].mean().to_dict()
                st.write(avg_stats)
                if avg_stats:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(avg_stats.keys()), y=list(avg_stats.values())))
                    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, key=f"chart_{i}_{title}", width=800)
