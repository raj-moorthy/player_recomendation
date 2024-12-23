import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the data
try:
    scaler = joblib.load("scaler_model.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    d = pd.read_csv("IPL 2024 Statistics _ Team and Player Stats.xlsx")  
except Exception as e:
    st.error(f"Error loading required files: {e}")
    st.stop()

# Modified recommend_players function
def recommend_players(runs, strike_rate, scaler, kmeans):
    input_scaled = scaler.transform([[runs, strike_rate]])
    group = kmeans.predict(input_scaled)[0]
    if group not in d['Group'].unique():
        return None
    recommended_players = d[d['Group'] == group]
    return recommended_players[['PLAYER1', 'AVG', 'SR']]

# Streamlit app
st.title("IPL Player Recommendation")

target_runs = st.number_input("Enter the Average score", min_value=0, step=1)
target_strike_rate = st.number_input("Enter the strike rate", min_value=0, step=1)

if st.button("Recommend Players"):
    if target_runs < 0 or target_strike_rate < 0:
        st.error("Please enter valid non-negative inputs.")
    else:
        recommended = recommend_players(target_runs, target_strike_rate, scaler, kmeans)
        if recommended is None or recommended.empty:
            st.warning("No recommendations available for the given inputs.")
        else:
            st.write("Recommended Players:")
            st.dataframe(recommended)
