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

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    models = {}
    for stat in ["points", "rebounds", "assists", "minutes", "steals", "blocks", "turnovers"]:
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
def get_player_image(name):
    try:
        first, last = name.split(" ", 1)
        url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{first.lower()}_{last.lower()}.png"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except:
        pass
    return Image.open(BytesIO(requests.get("https://cdn.nba.com/logos/nba/nba-logoman-word-white.svg").content))

@st.cache_data
def get_team_logo(team_name):
    try:
        base_url = "https://cdn.nba.com/logos/nba"
        team_map = {
            "Los Angeles Lakers": "1610612747", "Golden State Warriors": "1610612744",
            "Boston Celtics": "1610612738", "Dallas Mavericks": "1610612742",
            "Miami Heat": "1610612748", "Milwaukee Bucks": "1610612749"
        }
        team_id = team_map.get(team_name)
        if team_id:
            return f"{base_url}/{team_id}/primary/L/logo.svg"
    except:
        pass
    return None

def predict_player(player_name, df, models):
    player_data = df[df["player_name"] == player_name].tail(1)
    if player_data.empty:
        return None
    preds = {}
    for stat, model_set in models.items():
        if model_set:
            try:
                latest = player_data[model_set["rf"].feature_names_in_]
                avg_pred = np.mean([
                    model_set["rf"].predict(latest)[0],
                    model_set["xgb"].predict(latest)[0],
                    model_set["lgbm"].predict(latest)[0]
                ])
                preds[stat] = avg_pred
            except Exception:
                preds[stat] = None
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
    st.write("Your saved favorite players refresh daily with new projections.")
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    for fav in st.session_state["favorites"]:
        preds = predict_player(fav, df, models)
        if preds:
            st.subheader(f"{fav}")
            cols = st.columns(4)
            for i, (stat, value) in enumerate(preds.items()):
                cols[i % 4].metric(label=stat.upper(), value=round(value, 2))

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
                img = get_player_image(player_name)
                st.image(img, width=220)
            with col2:
                st.subheader(f"Projected Stats for {player_name}")
                team_name = df[df["player_name"] == player_name]["team_name"].iloc[-1] if "team_name" in df.columns else "Unknown"
                team_logo = get_team_logo(team_name)
                if team_logo:
                    st.image(team_logo, width=120)
                cols = st.columns(4)
                for i, (stat, val) in enumerate(preds.items()):
                    cols[i % 4].metric(label=stat.upper(), value=round(val, 2))
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
                cols = st.columns(4)
                for i, (stat, val) in enumerate(stats.items()):
                    cols[i % 4].metric(label=stat.upper(), value=round(val, 2))
    else:
        st.info("No tracked projections yet.")

# -------------------------------------------------
# RESEARCH TAB
# -------------------------------------------------
with tabs[3]:
    st.header("🔍 Prop Research Lab")
    player_name = st.selectbox("Select Player to Research", players, index=None, placeholder="Choose a player...")
    if player_name:
        img = get_player_image(player_name)
        st.image(img, width=200)
        player_data = df[df["player_name"] == player_name].sort_values("GAME_DATE", ascending=False)
        metrics = {
            "Most Recent Game": player_data.head(1),
            "Last 5 Games": player_data.head(5),
            "Last 10 Games": player_data.head(10),
            "Last 20 Games": player_data.head(20),
            "Season Averages": player_data,
        }
        for title, subset in metrics.items():
            with st.expander(title):
                avg_stats = subset[["points", "rebounds", "assists", "steals", "blocks", "minutes"]].mean().to_dict()
                st.write(avg_stats)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(avg_stats.keys()), y=list(avg_stats.values())))
                fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
