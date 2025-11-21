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

df = load_df()

try:
    model_pts = load_model("points")
    model_reb = load_model("rebounds")
    model_ast = load_model("assists")
except:
    st.error("Models missing. Train them first.")
    st.stop()

st.title("🏀 NBA Player Projections (BallDontLie All-Star)")

players = sorted(df["player_name"].unique())
player = st.selectbox("Select Player", players)

pdf = df[df["player_name"] == player].sort_values("GAME_DATE")
latest = pdf.iloc[-1]

features = [
    "points_rolling5",
    "reb_rolling5",
    "ast_rolling5",
    "min_rolling5",
    "minutes"
]

X = latest[features].values.reshape(1, -1)

pred_pts = model_pts.predict(X)[0]
pred_reb = model_reb.predict(X)[0]
pred_ast = model_ast.predict(X)[0]

col1, col2, col3 = st.columns(3)
col1.metric("Projected Points", f"{pred_pts:.1f}")
col2.metric("Projected Rebounds", f"{pred_reb:.1f}")
col3.metric("Projected Assists", f"{pred_ast:.1f}")

st.subheader("📈 Last 10 Games")
st.line_chart(pdf.tail(10).set_index("GAME_DATE")[["points", "rebounds", "assists"]])
