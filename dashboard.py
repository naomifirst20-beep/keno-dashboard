import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scraper import scrape_latest_draw
from pipeline import load_draw_data, run_pipeline, predict_next, build_leaderboard

st.set_page_config(page_title="Keno QuickCheck", layout="centered")
st.title("🎯 Keno QuickCheck")

# Refresh button
if st.button("🔄 Refresh Prediction"):
    st.experimental_rerun()

# Load and train
data = load_draw_data("data/draws.csv")
model, _, _ = run_pipeline("data/draws.csv")
latest_draws = data.tail(5)
predicted = predict_next(model, latest_draws)
latest_actual = [i+1 for i, val in enumerate(data.iloc[-1].tolist()) if val == 1]
match = set(predicted) & set(latest_actual)

# Display
st.subheader("📜 Latest Draw")
st.markdown(f"<h3 style='text-align: center;'>{', '.join(map(str, latest_actual))}</h3>", unsafe_allow_html=True)

st.subheader("🔮 Prediction")
st.markdown(f"<h3 style='text-align: center;'>{', '.join(map(str, predicted))}</h3>", unsafe_allow_html=True)

st.subheader("✅ Match Count")
st.markdown(f"<h2 style='text-align: center;'>{len(match)} matched</h2>", unsafe_allow_html=True)

# Leaderboard
st.subheader("🏆 Top Predictions")
leaderboard_df = build_leaderboard(model, data)
top_matches = leaderboard_df.sort_values(by="Match Count", ascending=False).head(5)

for _, row in top_matches.iterrows():
    st.markdown(f"""
    <div style='padding:10px; border:1px solid #ccc; border-radius:10px; margin-bottom:10px;'>
        <strong>Draw #{row['Draw Index']}</strong><br>
        🎯 Predicted: {', '.join(map(str, row['Predicted']))}<br>
        ✅ Matched: {len(row['Matched'])} numbers
    </div>
    """, unsafe_allow_html=True)

# Match rate chart
st.subheader("📊 Match Rate Over Time")
plt.figure(figsize=(6, 3))
plt.plot(leaderboard_df["Match Count"], color='green')
plt.xlabel("Draw Index")
plt.ylabel("Matched Numbers")
plt.title("Prediction Accuracy Trend")
st.pyplot(plt)

