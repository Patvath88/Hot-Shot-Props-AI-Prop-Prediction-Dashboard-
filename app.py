import streamlit as st
import pandas as pd
import xgboost as xgb
import subprocess
import os
from pathlib import Path

# --- Streamlit Page Setup ---
st.set_page_config(page_title="NBA Projections", layout="wide")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

# --- Cached Data Loading ---
@st.cache_data
def load_df():
    return pd.read_csv(DATA_DIR / "model_dataset.csv")

@st.cache_resource
def load_model(name):
    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / f"{name}.json")
    return model


# --- Run Full Pipeline ---
def run_full_pipeline():
    st.write("⏳ Running full pipeline... this might take a few minutes.")

    steps = [
        ("Fetching raw game logs...", ["python", "fetch_logs.py"]),
        ("Building dataset...", ["python", "build_dataset.py"]),
        ("Training models...", ["python", "train_model.py"]),
    ]

    for desc, cmd in steps:
        with st.spinner(desc):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                st.error(f"❌ Error during: {desc}\n\n{result.stderr}")
                return False
            else:
                st.success(f"✅ {desc} complete.")
                st.text(result.stdout)

    st.success("🎉 Full pipeline complete!")
    return True


# --- Sidebar Actions ---
st.sidebar.header("⚙️ Admin Controls")
if st.sidebar.button("🚀 Run Full Pipeline"):
    success = run_full_pipeline()
    if success:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

# --- Ensure Data Exists ---
if not (DATA_DIR / "model_dataset.csv").exists() or not all(
    (MODEL_DIR / f"{m}.json").exists() for m in ["points", "rebounds", "assists"]
):
    st.warning("Data or models missing. Run the full pipeline using the sidebar button.")
    st.stop()


# --- Load Data & Models ---
df = load_df()
model_pts = load_model("points")
model_reb = load_model("rebounds")
model_ast = load_model("assists")

# --- Main UI ---
st.title("🏀 NBA Player Projections")

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

col1, col2, col3 = st.columns(3)
col1.metric("Projected Points", f"{model_pts.predict(X)[0]:.1f}")
col2.metric("Projected Rebounds", f"{model_reb.predict(X)[0]:.1f}")
col3.metric("Projected Assists", f"{model_ast.predict(X)[0]:.1f}")

st.line_chart(pdf.tail(10).set_index("GAME_DATE")[["points", "rebounds", "assists"]])

st.caption("Data powered by balldontlie.io | Models trained using XGBoost")
