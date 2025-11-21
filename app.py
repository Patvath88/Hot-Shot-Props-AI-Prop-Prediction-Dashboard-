import streamlit as st
import pandas as pd
import xgboost as xgb
import subprocess
import os
import sys
from pathlib import Path

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="NBA Projections", layout="wide")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

# -------------------------------
# Cached Loaders
# -------------------------------
@st.cache_data
def load_df():
    """Load processed dataset"""
    return pd.read_csv(DATA_DIR / "model_dataset.csv")

@st.cache_resource
def load_model(name):
    """Load XGBoost model"""
    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / f"{name}.json")
    return model


# -------------------------------
# Full Pipeline Runner (uses same Python env)
# -------------------------------
def run_full_pipeline():
    commands = [
        ("Fetching raw game logs...", [sys.executable, "fetch_logs.py"]),
        ("Building dataset...", [sys.executable, "build_dataset.py"]),
        ("Training models...", [sys.executable, "train_model.py"]),
    ]

    for desc, cmd in commands:
        st.info(f"🌀 {desc}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"❌ Error during {desc}\n\n{result.stderr}")
            st.stop()
        else:
            st.success(f"✅ {desc} complete.")
            st.code(result.stdout[-800:], language="text")

    # Clear caches and reload
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("✅ Pipeline finished and reloaded successfully.")
    st.rerun()


# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Admin Controls")

if st.sidebar.button("🚀 Run Full Pipeline"):
    with st.spinner("Running full pipeline... please wait..."):
        run_full_pipeline()


# -------------------------------
# Verify Data & Models
# -------------------------------
required_models = ["points", "rebounds", "assists"]
missing_data = not (DATA_DIR / "model_dataset.csv").exists()
missing_models = any(not (MODEL_DIR / f"{m}.json").exists() for m in required_models)

if missing_data or missing_models:
    st.warning("⚠️ Data or models missing. Please run the full pipeline from the sidebar.")
    st.stop()


# -------------------------------
# Load Models and Data
# -------------------------------
df = load_df()
model_pts = load_model("points")
model_reb = load_model("rebounds")
model_ast = load_model("assists")


# -------------------------------
# Main Dashboard UI
# -------------------------------
st.title("🏀 NBA Player Projections Dashboard")

players = sorted(df["player_name"].unique())
player = st.selectbox("Select Player", players)

pdf = df[df["player_name"] == player].sort_values("GAME_DATE")
latest = pdf.iloc[-1]

FEATURES = [
    "points_rolling5",
    "rebounds_rolling5",
    "assists_rolling5",
    "minutes_rolling5",
    "minutes",
]
X = latest[FEATURES].values.reshape(1, -1)

st.subheader(player)

# Prediction Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Projected Points", f"{model_pts.predict(X)[0]:.1f}")
col2.metric("Projected Rebounds", f"{model_reb.predict(X)[0]:.1f}")
col3.metric("Projected Assists", f"{model_ast.predict(X)[0]:.1f}")

# Recent Performance Chart
st.line_chart(pdf.tail(10).set_index("GAME_DATE")[["points", "rebounds", "assists"]])

st.caption("📊 Data powered by balldontlie.io | Models trained using XGBoost")
