import streamlit as st
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import subprocess
import sys
from pathlib import Path

# -------------------------------
# Streamlit Configuration
# -------------------------------
st.set_page_config(page_title="NBA Multi-Model Projections", layout="wide")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

# -------------------------------
# Caching Utilities
# -------------------------------
@st.cache_data
def load_df():
    return pd.read_csv(DATA_DIR / "model_dataset.csv")

@st.cache_resource
def load_xgb_model(target):
    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / f"{target}_xgb.json")
    return model

@st.cache_resource
def load_lgb_model(target):
    return joblib.load(MODEL_DIR / f"{target}_lgb.pkl")

@st.cache_resource
def load_rf_model(target):
    return joblib.load(MODEL_DIR / f"{target}_rf.pkl")

# -------------------------------
# Pipeline Rerun Utility
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

    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("✅ Pipeline finished successfully.")
    st.rerun()

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Admin Controls")

if st.sidebar.button("🚀 Run Full Pipeline"):
    with st.spinner("Running full pipeline... please wait..."):
        run_full_pipeline()

# -------------------------------
# Check prerequisites
# -------------------------------
if not (DATA_DIR / "model_dataset.csv").exists() or not any(MODEL_DIR.glob("*.json")):
    st.warning("⚠️ Models or data missing. Run the full pipeline first.")
    st.stop()

# -------------------------------
# Load Data + Models
# -------------------------------
df = load_df()

TARGETS = [
    "points", "rebounds", "assists",
    "threept_fg", "steals", "blocks",
    "points_assists", "points_rebounds", "rebounds_assists",
    "points_rebounds_assists", "minutes"
]

xgb_models = {t: load_xgb_model(t) for t in TARGETS}
lgb_models = {t: load_lgb_model(t) for t in TARGETS}
rf_models = {t: load_rf_model(t) for t in TARGETS}

# -------------------------------
# Dashboard
# -------------------------------
st.title("🏀 NBA Multi-Model Player Projections Dashboard")

players = sorted(df["player_name"].unique())
player = st.selectbox("Select Player", players)

pdf = df[df["player_name"] == player].sort_values("GAME_DATE")
latest = pdf.iloc[-1]

FEATURES = [
    "points_rolling5", "rebounds_rolling5", "assists_rolling5",
    "threept_fg_rolling5", "steals_rolling5", "blocks_rolling5", "minutes_rolling5",
    "points_var5", "rebounds_var5", "assists_var5",
    "threept_fg_var5", "steals_var5", "blocks_var5", "minutes_var5",
    "rest_days",
    "season_points_avg", "season_reb_avg", "season_ast_avg", "season_min_avg"
]
X = latest[FEATURES].values.reshape(1, -1)

# -------------------------------
# Predictions Table
# -------------------------------
st.subheader(f"📊 {player} — Projected Stats")

table_data = []
for stat in TARGETS:
    xgb_pred = xgb_models[stat].predict(X)[0]
    lgb_pred = lgb_models[stat].predict(X)[0]
    rf_pred = rf_models[stat].predict(X)[0]
    avg_pred = (xgb_pred + lgb_pred + rf_pred) / 3
    table_data.append({
        "Stat": stat.replace("_", " ").title(),
        "XGBoost": round(xgb_pred, 2),
        "LightGBM": round(lgb_pred, 2),
        "RandomForest": round(rf_pred, 2),
        "Ensemble Avg": round(avg_pred, 2)
    })

table_df = pd.DataFrame(table_data)
st.dataframe(table_df, use_container_width=True)

# -------------------------------
# Recent Trend Chart
# -------------------------------
st.subheader("📈 Recent Performance (Last 10 Games)")
chart_cols = ["points", "rebounds", "assists", "threept_fg", "steals", "blocks"]
st.line_chart(pdf.tail(10).set_index("GAME_DATE")[chart_cols])

st.caption("🤖 All models auto-refresh daily at 7 AM EST | Built with XGBoost, LightGBM & RandomForest")
