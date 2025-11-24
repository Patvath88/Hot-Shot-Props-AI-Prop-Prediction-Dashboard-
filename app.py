import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
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
PLAYERS_JSON = Path("player-photos.json")
TEAMS_JSON = Path("team_logos.json")

MODELS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "player_name" not in df.columns:
        st.error("❌ 'player_name' column missing from dataset.")
    return df

@st.cache_resource
def load_models():
    stats = [
        "points", "rebounds", "assists", "threept_fg",
        "steals", "blocks", "minutes",
        "points_assists", "points_rebounds",
        "rebounds_assists", "points_rebounds_assists"
    ]
    models = {}
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
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

PLAYER_PHOTOS = load_json(PLAYERS_JSON)
TEAM_LOGOS = load_json(TEAMS_JSON)

@st.cache_data
def get_player_image(player_name):
    url = PLAYER_PHOTOS.get(player_name)
    if url:
        return url
    return "https://cdn.nba.com/logos/nba/nba-logoman-word-white.svg"

@st.cache_data
def get_team_logo(team_name):
    return TEAM_LOGOS.get(team_name, "https://cdn.nba.com/logos/nba/nba-logoman-word-white.svg")

def safe_predict(model, df_row):
    model_features = list(model.feature_names_in_)
    aligned = df_row.reindex(columns=model_features, fill_value=0)
    return model.predict(aligned)[0]

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
            except Exception:
                preds[stat] = 0
    return preds

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = load_data()
models = load_models()

# -------------------------------------------------
# UI LAYOUT
# -------------------------------------------------
tabs = st.tabs(["🏠 Home / Favorites", "🧠 Prop Projection Lab", "📊 Projection Tracker", "🔍 Prop Research Lab"])

# -------------------------------------------------
# HOME TAB
# -------------------------------------------------
with tabs[0]:
    st.title("🏀 Hot Shot Props AI Dashboard")
    st.caption("Your saved favorite players auto-refresh daily with updated projections.")
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    for fav in st.session_state["favorites"]:
        preds = predict_player(fav, df, models)
        if preds:
            st.subheader(fav)
            cols = st.columns(5)
            for i, (stat, val) in enumerate(preds.items()):
                cols[i % 5].metric(stat.upper(), round(val, 2))

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
            col1, col2 = st.columns([1, 3])
            with col1:
                img = get_player_image(player_name)
                st.image(img, width=200)
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
        img = get_player_image(player_name)
        st.image(img, width=180)
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
                cols_to_use = [
                    c for c in ["points", "rebounds", "assists", "threept_fg", "steals", "blocks", "minutes"]
                    if c in subset.columns
                ]
                avg_stats = subset[cols_to_use].mean().to_dict()
                st.write(avg_stats)
               if avg_stats:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(avg_stats.keys()), y=list(avg_stats.values())))
                    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}_{title.replace(' ', '_')}")
