import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path
import plotly.graph_objects as go

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="🏀 Hot Shot Props AI", layout="wide", page_icon="🏀")
DATA_PATH = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

BALLDONTLIE_API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    stats = ["points", "rebounds", "assists", "minutes", "steals", "blocks", "threept_fg", "turnovers"]
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

@st.cache_data
def fetch_player_info(player_name):
    """Fetch player + team info from Ball Don't Lie"""
    headers = {"Authorization": f"Bearer {BALLDONTLIE_API_KEY}"}
    first, last = player_name.split(" ", 1)
    resp = requests.get(f"{BALLDONTLIE_BASE}/players", params={"search": f"{first} {last}"}, headers=headers)
    if resp.status_code == 200 and resp.json()["data"]:
        data = resp.json()["data"][0]
        team = data["team"]["full_name"]
        abbr = data["team"]["abbreviation"]
        team_logo = f"https://cdn.nba.com/logos/nba/{data['team']['id']}/primary/L/logo.svg"
        player_img = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{data['id']}.png"
        return {
            "team": team,
            "team_abbr": abbr,
            "player_img": player_img,
            "team_logo": team_logo
        }
    return None

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
    # Combo projections
    preds["PA"] = (preds.get("points", 0) or 0) + (preds.get("assists", 0) or 0)
    preds["PR"] = (preds.get("points", 0) or 0) + (preds.get("rebounds", 0) or 0)
    preds["RA"] = (preds.get("rebounds", 0) or 0) + (preds.get("assists", 0) or 0)
    preds["PRA"] = preds["PA"] + (preds.get("rebounds", 0) or 0)
    return preds

# -------------------------------------------------
# APP CONTENT
# -------------------------------------------------
df = load_data()
models = load_models()
tabs = st.tabs(["🏠 Home / Favorites", "🧠 Prop Projection Lab", "📊 Projection Tracker", "🔍 Prop Research Lab"])

# -------------------------------------------------
# HOME TAB
# -------------------------------------------------
with tabs[0]:
    st.title("🏀 Hot Shot Props AI Dashboard")
    st.caption("Favorites auto-refresh daily with updated projections.")
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    for fav in st.session_state["favorites"]:
        preds = predict_player(fav, df, models)
        info = fetch_player_info(fav)
        if preds:
            st.subheader(f"{fav} — {info['team'] if info else ''}")
            if info:
                st.image(info["player_img"], width=140)
            cols = st.columns(5)
            for i, (stat, val) in enumerate(preds.items()):
                cols[i % 5].metric(stat.upper(), round(val, 2))

# -------------------------------------------------
# PROP PROJECTION LAB
# -------------------------------------------------
with tabs[1]:
    st.header("🧠 Player Projection Lab")
    players = sorted(df["player_name"].unique())
    player_name = st.selectbox("Select Player", players)
    if player_name:
        preds = predict_player(player_name, df, models)
        info = fetch_player_info(player_name)
        if preds:
            col1, col2 = st.columns([1, 3])
            with col1:
                if info:
                    st.image(info["player_img"], width=200)
                    st.image(info["team_logo"], width=80)
                    st.markdown(f"**{info['team']}**")
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
# TRACKER TAB
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
# RESEARCH TAB
# -------------------------------------------------
with tabs[3]:
    st.header("🔍 Prop Research Lab")
    player_name = st.selectbox("Select Player to Research", players)
    if player_name:
        info = fetch_player_info(player_name)
        if info:
            st.image(info["player_img"], width=180)
            st.image(info["team_logo"], width=60)
        player_data = df[df["player_name"] == player_name]
        date_col = next((c for c in ["GAME_DATE", "game_date", "date"] if c in player_data.columns), None)
        if date_col:
            player_data = player_data.sort_values(date_col, ascending=False)
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
                if avg_stats:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(avg_stats.keys()), y=list(avg_stats.values())))
                    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}_{title.replace(' ', '_')}")
