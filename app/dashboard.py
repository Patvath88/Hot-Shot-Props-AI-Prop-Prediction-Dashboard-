import streamlit as st
import pandas as pd
import xgboost as xgb
import os

st.set_page_config(page_title="NBA Projections", layout="wide")

# -------------------------------------------------
# Corrected dataset location: dataset/model_dataset.csv
# -------------------------------------------------
@st.cache_data
def load_df():
    path = "dataset/model_dataset.csv"
    if not os.path.exists(path):
        st.error("❌ model_dataset.csv not found.\nRun the scraper + dataset builder first.")
        st.stop()
    return pd.read_csv(path)


# -------------------------------------------------
# Corrected model loading — ensure models/ folder exists
# -------------------------------------------------
@st.cache_resource
def load_model(name):
    model_path = f"models/{name}.json"
    if not os.path.exists(model_path):
        st.error(f"❌ Model {name}.json missing. Train models first.")
        st.stop()

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


# Load data
df = load_df()

# Load models
model_pts = load_model("points")
model_reb = load_model("rebounds")
model_ast = load_model("assists")

# -------------------------------------------------
# UI Layout
# -------------------------------------------------
st.title("🏀 NBA Player Projections — BallDontLie All-Star Tier")

players = sorted(df["player_name"].unique())
player = st.selectbox("Select Player", players)

pdf = df[df["player_name"] == player].sort_values("GAME_DATE")

if pdf.empty:
    st.error("❌ No data available for this player.")
    st.stop()

latest = pdf.iloc[-1]

# Features expected by model
features = [
    "points_rolling5",
    "reb_rolling5",
    "ast_rolling5",
    "min_rolling5",
    "minutes"
]

# Ensure all required columns exist
missing = [f for f in features if f not in df.columns]
if missing:
    st.error(f"❌ Missing feature columns in dataset: {missing}")
    st.stop()

# Model input
X = latest[features].values.reshape(1, -1)

pred_pts = float(model_pts.predict(X)[0])
pred_reb = float(model_reb.predict(X)[0])
pred_ast = float(model_ast.predict(X)[0])

# Projection cards
col1, col2, col3 = st.columns(3)
col1.metric("Projected Points", f"{pred_pts:.1f}")
col2.metric("Projected Rebounds", f"{pred_reb:.1f}")
col3.metric("Projected Assists", f"{pred_ast:.1f}")

# Chart
st.subheader("📈 Last 10 Games")
chart = pdf.tail(10).set_index("GAME_DATE")[["points", "rebounds", "assists"]]
st.line_chart(chart)
