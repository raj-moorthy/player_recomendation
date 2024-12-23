import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


# Load the data
scaler = joblib.load("C:/Users/rajre/OneDrive/Documents/Player_recommendation/scaler_model.pkl")
kmeans = joblib.load("C:/Users/rajre/OneDrive/Documents/Player_recommendation/kmeans_model.pkl")


# Modified recommend_players function
def recommend_players(d, runs, strike_rate, scaler, kmeans):
    input_scaled = scaler.transform([[runs, strike_rate]])
    group = kmeans.predict(input_scaled)[0]
    recommended_players = d[d['Group'] == group]
    return recommended_players[['PLAYER1', 'AVG', 'SR']]

# Streamlit app
st.title("IPL Player Recommendation")

target_runs = st.number_input("Enter the Average score", min_value=0)
target_strike_rate = st.number_input("Enter the strike rate", min_value=0)

if st.button("Recommend Players"):
    recommended = recommend_players(target_runs, target_strike_rate, scaler, kmeans)  
    st.write("Recommended Players:")
    st.dataframe(recommended)