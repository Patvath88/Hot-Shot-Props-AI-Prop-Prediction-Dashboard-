import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_FILE = Path("data/model_dataset.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="🏀 Hot Shot Props AI", layout="wide")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error("❌ No dataset found. Run the data build workflow first.")
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILE)
    return df

df = load_data()
if df.empty:
    st.stop()

# -------------------------------------------------
# TRAIN MODEL ON DEMAND
# -------------------------------------------------
def train_player_model(player_name):
    player_data = df[df["player_name"] == player_name].copy()
    if len(player_data) < 5:
        st.warning("⚠️ Not enough data to train a model for this player.")
        return None

    features = ["points_rolling5", "rebounds_rolling5", "assists_rolling5"]
    for f in features:
        if f not in player_data.columns:
            player_data[f] = 0

    # Train models for each stat
    models = {}
    for stat in ["points", "rebounds", "assists"]:
        if stat not in player_data.columns:
            continue
        X = player_data[features]
        y = player_data[stat]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        models[stat] = model
        joblib.dump(model, MODELS_DIR / f"{player_name}_{stat}_model.pkl")

    return models

# -------------------------------------------------
# LOAD OR TRAIN MODEL
# -------------------------------------------------
def get_or_train_model(player_name):
    models = {}
    for stat in ["points", "rebounds", "assists"]:
        model_path = MODELS_DIR / f"{player_name}_{stat}_model.pkl"
        if model_path.exists():
            models[stat] = joblib.load(model_path)
        else:
            st.info(f"⚙️ Training new {stat} model for {player_name}...")
            trained = train_player_model(player_name)
            if trained and stat in trained:
                models[stat] = trained[stat]
    return models

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🏀 Hot Shot Props AI - Player Projection Dashboard")
st.write("Use the dropdown to select a player and view AI-generated stat predictions.")

player_list = sorted(df["player_name"].unique())
player_name = st.selectbox("Select Player From Dropdown", player_list)

if player_name:
    st.subheader(f"📈 Predicted Stats for {player_name}")
    models = get_or_train_model(player_name)

    if models:
        features = ["points_rolling5", "rebounds_rolling5", "assists_rolling5"]
        latest_data = df[df["player_name"] == player_name].tail(1)[features]
        for stat, model in models.items():
            pred = model.predict(latest_data)[0]
            st.success(f"**Predicted {stat.capitalize()}:** {pred:.1f}")
    else:
        st.warning("⚠️ No model available for this player yet.")
