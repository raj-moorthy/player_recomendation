

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved models
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler_model = pickle.load(open('scaler_model.pkl', 'rb'))
d=pd.read_excel("IPL 2024 Statistics _ Team and Player Stats.xlsx")


# Streamlit app
st.title("IPL Player Recommendation App")

# Input fields
runs = st.number_input("Enter player runs:", value=1000, min_value=0)
strike_rate = st.number_input("Enter player strike rate:", value=125, min_value=0)

if st.button("Recommend Players"):
    try:
        input_scaled = scaler_model.transform([[runs, strike_rate]])
        group = kmeans_model.predict(input_scaled)[0]
        recommended_players = d[d['Group'] == group]
        st.write("Recommended Players:")
        st.dataframe(recommended_players[['PLAYER1', 'RUNS', 'SR']])


    except Exception as e:
        st.error(f"An error occurred: {e}")