import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="🏀 Hot Shot Props AI", layout="wide", page_icon="🏀")

DATA_PATH = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
PLAYER_PHOTOS = Path("player-photos.json")
TEAM_LOGOS = Path("team_logos.json")

MODELS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower() for c in df.columns]  # standardize column names
    return df

@st.cache_resource
def load_models():
    model_dict = {}
    stats = [
        "points", "rebounds", "assists", "threept_fg",
        "steals", "blocks", "minutes",
        "points_assists", "points_rebounds",
        "rebounds_assists", "points_rebounds_assists"
    ]
    for stat in stats:
        try:
            model_dict[stat] = {
                "rf": joblib.load(MODELS_DIR / f"rf_{stat}.pkl"),
                "xgb": joblib.load(MODELS_DIR / f"xgb_{stat}.pkl"),
                "lgbm": joblib.load(MODELS_DIR / f"lgbm_{stat}.pkl")
            }
        except:
            model_dict[stat] = None
    return model_dict

@st.cache_data
def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except:
        return {}

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
def get_player_image(player_name, photos_dict):
    url = photos_dict.get(player_name)
    if url:
        return url
    return "https://cdn.nba.com/logos/nba/nba-logoman-word-white.svg"

def get_team_logo(team_name, logos_dict):
    return logos_dict.get(team_name, "https://cdn.nba.com/logos/nba/nba-logoman-word-white.svg")

def safe_predict(model, df_row):
    model_features = list(model.feature_names_in_)
    aligned = pd.DataFrame([{
        f: df_row[f].values[0] if f in df_row.columns else 0 for f in model_features
    }])
    return model.predict(aligned)[0]

def predict_player(player_name, df, models):
    row = df[df["player_name"] == player_name].tail(1)
    if row.empty:
        return None

    preds = {}
    for stat, model_set in models.items():
        if model_set:
            try:
                rf_pred = safe_predict(model_set["rf"], row)
                xgb_pred = safe_predict(model_set["xgb"], row)
                lgbm_pred = safe_predict(model_set["lgbm"], row)
                preds[stat] = np.mean([rf_pred, xgb_pred, lgbm_pred])
            except:
                preds[stat] = None
        else:
            preds[stat] = None
    return preds

# -------------------------------------------------
# LOAD EVERYTHING
# -------------------------------------------------
df = load_data()
models = load_models()
player_photos = load_json(PLAYER_PHOTOS)
team_logos = load_json(TEAM_LOGOS)

tabs = st.tabs([
    "🏠 Home / Favorites",
    "🧠 Prop Projection Lab",
    "📊 Projection Tracker",
    "🔍 Prop Research Lab"
])

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
        if preds:
            st.subheader(fav)
            cols = st.columns(5)
            for i, (stat, val) in enumerate(preds.items()):
                cols[i % 5].metric(label=stat.upper(), value=round(val, 2))

# -------------------------------------------------
# PROP PROJECTION LAB
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
                photo_url = get_player_image(player_name, player_photos)
                st.image(photo_url, width=200)
                st.markdown("#### Team Logo")
                st.image(get_team_logo("default", team_logos), width=100)
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
# RESEARCH LAB TAB
# -------------------------------------------------
with tabs[3]:
    st.header("🔍 Prop Research Lab")
    player_name = st.selectbox("Select Player to Research", players)

    if player_name:
        photo_url = get_player_image(player_name, player_photos)
        st.image(photo_url, width=180)

        player_data = df[df["player_name"] == player_name].sort_values("game_date", ascending=False)
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
                    c for c in [
                        "points", "rebounds", "assists", "steals", "blocks", "minutes", "threept_fg"
                    ] if c in subset.columns
                ]
                avg_stats = subset[cols_to_use].mean().to_dict()
                st.write(avg_stats)
                if avg_stats:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(avg_stats.keys()), y=list(avg_stats.values())))
                    fig.update_layout(
                        template="plotly_dark",
                        height=300,
                        width=800,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, key=f"chart_{i}_{title.replace(' ', '_')}")
