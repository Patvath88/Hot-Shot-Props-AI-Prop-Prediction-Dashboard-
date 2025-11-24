import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import time
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="🏀 Hot Shot Props AI", layout="wide", page_icon="🏀")

DATA_PATH = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
PLAYER_PHOTOS = Path("player-photos.json")
TEAM_LOGOS = Path("team_logos.json")
LAST_TRAIN_FILE = Path("data/last_train.txt")

# -------------------------------------------------
# CACHED HELPERS
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def load_json(file_path):
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

@st.cache_resource
def load_models():
    models = {}
    stats = ["points", "rebounds", "assists", "threept_fg", "steals", "blocks", "minutes"]
    for stat in stats:
        try:
            models[stat] = {
                "rf": joblib.load(MODELS_DIR / f"rf_{stat}.pkl"),
                "xgb": joblib.load(MODELS_DIR / f"xgb_{stat}.pkl"),
                "lgbm": joblib.load(MODELS_DIR / f"lgbm_{stat}.pkl")
            }
        except Exception as e:
            st.warning(f"⚠️ Missing model for {stat}: {e}")
            models[stat] = None
    return models

# -------------------------------------------------
# UTILITIES
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
                preds[stat] = avg_pred
            except:
                preds[stat] = None

    # Combo stats
    preds["PA"] = (preds.get("points", 0) or 0) + (preds.get("assists", 0) or 0)
    preds["PR"] = (preds.get("points", 0) or 0) + (preds.get("rebounds", 0) or 0)
    preds["RA"] = (preds.get("rebounds", 0) or 0) + (preds.get("assists", 0) or 0)
    preds["PRA"] = preds["PA"] + (preds.get("rebounds", 0) or 0)
    return preds

def retrain_if_needed():
    now = datetime.utcnow()
    if LAST_TRAIN_FILE.exists():
        last_run = datetime.fromtimestamp(LAST_TRAIN_FILE.stat().st_mtime)
        if now - last_run < timedelta(hours=24):
            return  # Skip retrain

    import subprocess
    st.write("🔄 Retraining models (once every 24 hours)...")
    subprocess.run(["python", "train_model.py"], check=False)
    LAST_TRAIN_FILE.touch()

# -------------------------------------------------
# APP LOAD
# -------------------------------------------------
st.title("🏀 Hot Shot Props AI Dashboard")

# Automatically retrain if older than 24h
retrain_if_needed()

df = load_data()
models = load_models()
player_photos = load_json(PLAYER_PHOTOS)
team_logos = load_json(TEAM_LOGOS)

tabs = st.tabs(["🏠 Home / Favorites", "🧠 Prop Projection Lab", "📊 Projection Tracker", "🔍 Prop Research Lab"])

# -------------------------------------------------
# HOME TAB
# -------------------------------------------------
with tabs[0]:
    st.header("⭐ Favorites Dashboard")
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    if not st.session_state["favorites"]:
        st.info("No favorites yet — add some in the Projection Lab.")
    for fav in st.session_state["favorites"]:
        preds = predict_player(fav, df, models)
        if preds:
            st.subheader(fav)
            img_url = player_photos.get(fav)
            if img_url:
                st.image(img_url, width=150)
            cols = st.columns(5)
            for i, (stat, val) in enumerate(preds.items()):
                cols[i % 5].metric(stat.upper(), round(val, 2))

# -------------------------------------------------
# PROP PROJECTION LAB
# -------------------------------------------------
with tabs[1]:
    st.header("🧠 Player Projection Lab")
    players = sorted(df["player_name"].unique())
    player_name = st.selectbox("Select Player", players, index=None, placeholder="Choose a player...")

    if player_name:
        preds = predict_player(player_name, df, models)
        if preds:
            col1, col2 = st.columns([1, 3])
            with col1:
                img_url = player_photos.get(player_name)
                if img_url:
                    st.image(img_url, width=200)
                team_logo_url = team_logos.get(player_name)
                if team_logo_url:
                    st.image(team_logo_url, width=100)
            with col2:
                st.subheader(f"Projected Stats for {player_name}")
                cols = st.columns(5)
                for i, (stat, val) in enumerate(preds.items()):
                    cols[i % 5].metric(stat.upper(), round(val, 2))

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
# PROJECTION TRACKER
# -------------------------------------------------
with tabs[2]:
    st.header("📊 Projection Tracker")
    if "tracked" not in st.session_state:
        st.session_state["tracked"] = []
    if st.session_state["tracked"]:
        for entry in st.session_state["tracked"]:
            for name, stats in entry.items():
                st.subheader(name)
                cols = st.columns(5)
                for i, (stat, val) in enumerate(stats.items()):
                    cols[i % 5].metric(label=stat.upper(), value=round(val, 2))
    else:
        st.info("No tracked projections yet.")

# -------------------------------------------------
# RESEARCH LAB
# -------------------------------------------------
with tabs[3]:
    st.header("🔍 Prop Research Lab")
    player_name = st.selectbox("Select Player to Research", players, index=None)
    if player_name:
        img_url = player_photos.get(player_name)
        if img_url:
            st.image(img_url, width=180)
        player_data = df[df["player_name"] == player_name]
        if "game_date" in player_data.columns:
            player_data = player_data.sort_values("game_date", ascending=False)

        metrics = {
            "Most Recent Game": player_data.head(1),
            "Last 5 Games": player_data.head(5),
            "Last 10 Games": player_data.head(10),
            "Last 20 Games": player_data.head(20),
            "Season Averages": player_data,
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
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}_{title}")