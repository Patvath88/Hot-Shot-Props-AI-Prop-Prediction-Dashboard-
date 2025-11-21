import streamlit as st
import pandas as pd
import requests
import os
import joblib
from pathlib import Path
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="🏀 Hot Shot Props AI", layout="wide")

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
IMAGES_DIR = DATA_DIR / "images"
for d in [DATA_DIR, MODELS_DIR, IMAGES_DIR]:
    d.mkdir(exist_ok=True)

BASE_URL = "https://api.balldontlie.io/v1"
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# -------------------------------------------------
# IMAGE CACHING
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def get_player_image(player_id, name):
    image_path = IMAGES_DIR / f"{player_id}.jpg"
    if image_path.exists():
        return str(image_path)

    # Fetch image dynamically (NBA CDN)
    try:
        url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
        r = requests.get(url)
        if r.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(r.content)
            return str(image_path)
    except Exception:
        pass
    return None

# -------------------------------------------------
# DATA FETCH
# -------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_player_data(player_name):
    params = {"search": player_name}
    r = requests.get(f"{BASE_URL}/players", headers=HEADERS, params=params)
    if r.status_code != 200:
        return None
    data = r.json().get("data", [])
    return data[0] if data else None

@st.cache_data(ttl=3600)
def fetch_player_stats(player_id, num_games=20):
    r = requests.get(f"{BASE_URL}/stats", headers=HEADERS, params={"player_ids[]": player_id, "per_page": num_games})
    if r.status_code != 200:
        return None
    return pd.DataFrame(r.json().get("data", []))

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
def load_model(stat):
    try:
        return joblib.load(MODELS_DIR / f"{stat}_model.pkl")
    except:
        return None

# -------------------------------------------------
# PREDICTIONS
# -------------------------------------------------
def generate_predictions(player_name, df):
    predictions = {}
    feature_cols = ["points_rolling5", "reb_rolling5", "ast_rolling5", "min_rolling5"]

    for stat in ["points", "rebounds", "assists"]:
        model = load_model(stat)
        if model is None:
            predictions[stat] = "⚠️ Model not found"
            continue
        try:
            latest = df[df["player_name"] == player_name].tail(1)[feature_cols]
            predictions[stat] = round(model.predict(latest)[0], 1)
        except Exception as e:
            predictions[stat] = f"Error: {e}"

    # Derived props
    if all(isinstance(predictions.get(s), (int, float)) for s in ["points", "rebounds", "assists"]):
        predictions["PA"] = predictions["points"] + predictions["assists"]
        predictions["PR"] = predictions["points"] + predictions["rebounds"]
        predictions["RA"] = predictions["rebounds"] + predictions["assists"]
        predictions["PRA"] = predictions["points"] + predictions["rebounds"] + predictions["assists"]
    else:
        predictions["PA"] = predictions["PR"] = predictions["RA"] = predictions["PRA"] = "—"

    return predictions

# -------------------------------------------------
# UI: SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.title("🏀 Hot Shot Props AI")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Generate Projections", "⭐ Favorites", "📈 Prediction Tracker", "📚 Research"])

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
if page == "🏠 Home":
    st.title("🏀 Hot Shot Props AI - Player Projection Dashboard")
    st.markdown("Welcome! Use the sidebar to generate AI-driven stat predictions, track favorites, or research player performance.")
    st.image("https://cdn.nba.com/logos/nba/nba-logoman-75.svg", width=150)

# -------------------------------------------------
# GENERATE PROJECTIONS
# -------------------------------------------------
elif page == "📊 Generate Projections":
    st.title("📊 Generate Player Projections")
    df = pd.read_csv(DATA_DIR / "model_dataset.csv")

    players = sorted(df["player_name"].unique())
    player_name = st.selectbox("Select Player", players)

    if player_name:
        player_data = fetch_player_data(player_name)
        if player_data:
            pid = player_data["id"]
            image_path = get_player_image(pid, player_name)
            team = player_data["team"]["full_name"]

            col1, col2 = st.columns([1, 3])
            with col1:
                if image_path:
                    st.image(image_path, width=180)
            with col2:
                st.subheader(f"{player_name} - {team}")

            preds = generate_predictions(player_name, df)
            st.metric("Points", preds.get("points"))
            st.metric("Rebounds", preds.get("rebounds"))
            st.metric("Assists", preds.get("assists"))
            st.metric("PRA", preds.get("PRA"))

# -------------------------------------------------
# FAVORITES PAGE
# -------------------------------------------------
elif page == "⭐ Favorites":
    st.title("⭐ Saved Favorites")
    fav_file = DATA_DIR / "favorites.csv"

    if fav_file.exists():
        favs = pd.read_csv(fav_file)
        st.dataframe(favs)
    else:
        st.info("No favorites yet. Add one from the Projections page!")

# -------------------------------------------------
# PREDICTION TRACKER
# -------------------------------------------------
elif page == "📈 Prediction Tracker":
    st.title("📈 Prediction Tracker")
    tracker_file = DATA_DIR / "tracker.csv"

    if tracker_file.exists():
        tracker = pd.read_csv(tracker_file)
        st.dataframe(tracker)
    else:
        st.info("No tracked predictions yet.")

# -------------------------------------------------
# RESEARCH TAB
# -------------------------------------------------
elif page == "📚 Research":
    st.title("📚 Player Research Center")

    name = st.text_input("Search player name")
    if name:
        p_data = fetch_player_data(name)
        if not p_data:
            st.warning("No player found.")
        else:
            pid = p_data["id"]
            img = get_player_image(pid, name)
            team = p_data["team"]["full_name"]

            col1, col2 = st.columns([1, 3])
            with col1:
                if img:
                    st.image(img, width=180)
            with col2:
                st.subheader(f"{name} - {team}")

            stats_df = fetch_player_stats(pid, 100)
            if stats_df is not None and not stats_df.empty:
                st.success("📊 Player stats loaded successfully!")

                # Expanders
                for label, n in [("Last 5 Games", 5), ("Last 10 Games", 10), ("Last 20 Games", 20)]:
                    with st.expander(label):
                        recent = stats_df.tail(n)
                        for col in ["pts", "reb", "ast"]:
                            st.bar_chart(recent[col])
            else:
                st.warning("No stats available.")
