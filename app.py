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
    df = pd.read_csv(DATA_PATH)
    if "player_name" not in df.columns:
        st.error("❌ 'player_name' column missing from dataset.")
    return df

@st.cache_resource
def load_models():
    stats = ["points", "rebounds", "assists", "minutes", "steals", "blocks", "turnovers"]
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
def get_player_image(name):
    try:
        formatted_name = name.lower().replace(" ", "_")
        url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{formatted_name}.png"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content))
    except:
        pass
    placeholder = "https://cdn.nba.com/logos/nba/nba-logoman-word-white.svg"
    return Image.open(BytesIO(requests.get(placeholder).content))

@st.cache_data
def get_team_logo(team_name):
    team_map = {
        "LAL": "1610612747", "GSW": "1610612744", "BOS": "1610612738", "MIA": "1610612748",
        "MIL": "1610612749", "DAL": "1610612742", "PHX": "1610612756", "DEN": "1610612743",
        "NYK": "1610612752", "PHI": "1610612755"
    }
    try:
        team_id = team_map.get(team_name, "1610612747")
        return f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"
    except:
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
    preds["PA"] = (preds.get("points", 0) or 0) + (preds.get("assists", 0) or 0)
    preds["PR"] = (preds.get("points", 0) or 0) + (preds.get("rebounds", 0) or 0)
    preds["RA"] = (preds.get("rebounds", 0) or 0) + (preds.get("assists", 0) or 0)
    preds["PRA"] = preds["PA"] + (preds.get("rebounds", 0) or 0)
    return preds

# -------------------------------------------------
# APP
# -------------------------------------------------
df = load_data()
models = load_models()
tabs = st.tabs(["🏠 Home / Favorites", "🧠 Prop Projection Lab", "📊 Projection Tracker", "🔍 Prop Research Lab"])

# HOME TAB
with tabs[0]:
    st.title("🏀 Hot Shot Props AI Dashboard")
    st.caption("Favorites auto-refresh daily with updated projections.")
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    for fav in st.session_state["favorites"]:
        preds = predict_player(fav, df, models)
        if preds:
            st.subheader(fav)
            cols = st.columns(5)
            for i, (stat, val) in enumerate(preds.items()):
                cols[i % 5].metric(stat.upper(), round(val, 2))

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
                img = get_player_image(player_name)
                st.image(img, width=200)
                team_name = df[df["player_name"] == player_name]["team_abbreviation"].iloc[-1] if "team_abbreviation" in df.columns else "LAL"
                team_logo = get_team_logo(team_name)
                if team_logo:
                    st.image(team_logo, width=100)
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

# TRACKER TAB
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
# RESEARCH TAB
with tabs[3]:
    st.header("🔍 Prop Research Lab")
    player_name = st.selectbox("Select Player to Research", players)
    if player_name:
        img = get_player_image(player_name)
        st.image(img, width=180)
        player_data = df[df["player_name"] == player_name]

        # ✅ Handle flexible date column names
        date_col = None
        for col in ["GAME_DATE", "game_date", "date"]:
            if col in player_data.columns:
                date_col = col
                break

        if date_col:
            player_data = player_data.sort_values(date_col, ascending=False)
        else:
            st.warning("No date column found in dataset — showing unsorted data.")

        metrics = {
            "Most Recent Game": player_data.head(1),
            "Last 5 Games": player_data.head(5),
            "Last 10 Games": player_data.head(10),
            "Last 20 Games": player_data.head(20),
            "Season Averages": player_data,
        }

        for i, (title, subset) in enumerate(metrics.items()):
        with st.expander(title):
        cols_to_use = [c for c in ["points", "rebounds", "assists", "steals", "blocks", "minutes"] if c in subset.columns]
        avg_stats = subset[cols_to_use].mean().to_dict()
        st.write(avg_stats)
        if avg_stats:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(avg_stats.keys()), y=list(avg_stats.values())))
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0))
            # ✅ Add unique key for each chart to fix duplicate element ID errors
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}_{title.replace(' ', '_')}")
