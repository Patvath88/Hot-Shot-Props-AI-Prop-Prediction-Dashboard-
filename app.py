import streamlit as st
import pandas as pd
import xgboost as xgb
import os

st.set_page_config(page_title="NBA Projections", layout="wide")

@st.cache_data
def load_df():
    return pd.read_csv("data/model_dataset.csv")

@st.cache_resource
def load_model(name):
    model = xgb.XGBRegressor()
    model.load_model(f"models/{name}.json")
    return model


# --- Load Data and Models ---
df = load_df()
model_pts = load_model("points")
model_reb = load_model("rebounds")
model_ast = load_model("assists")

# --- UI ---
st.title("🏀 NBA Player Projections")

players = sorted(df["player_name"].unique())
player = st.selectbox("Select Player", players)

pdf = df[df["player_name"] == player].sort_values("GAME_DATE")
latest = pdf.iloc[-1]

FEATURES = ["points_rolling5", "rebounds_rolling5", "assists_rolling5", "minutes_rolling5", "minutes"]
X = latest[FEATURES].values.reshape(1, -1)

st.subheader(player)

col1, col2, col3 = st.columns(3)
col1.metric("Projected Points", f"{model_pts.predict(X)[0]:.1f}")
col2.metric("Projected Rebounds", f"{model_reb.predict(X)[0]:.1f}")
col3.metric("Projected Assists", f"{model_ast.predict(X)[0]:.1f}")

st.line_chart(pdf.tail(10).set_index("GAME_DATE")[["points", "rebounds", "assists"]])
