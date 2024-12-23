import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the saved models and dataset
try:
    kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))
    scaler_model = pickle.load(open('scaler_model.pkl', 'rb'))
    d = pd.read_excel("IPL 2024 Statistics _ Team and Player Stats.xlsx")
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Ensure 'Group' column exists
if 'Group' not in d.columns:
    st.warning("'Group' column is missing. Recomputing clusters...")
    try:
        if {'RUNS', 'SR'}.issubset(d.columns):  # Validate required columns
            features = d[['RUNS', 'SR']]
            features_scaled = scaler_model.transform(features)
            d['Group'] = kmeans_model.predict(features_scaled)
            st.success("'Group' column added successfully.")
        else:
            st.error("The dataset does not contain required columns: 'RUNS' and 'SR'.")
            st.stop()
    except Exception as e:
        st.error(f"Error during cluster computation: {e}")
        st.stop()

# Streamlit app
st.title("IPL Player Recommendation App")

# Input fields
runs = st.number_input("Enter player runs:", value=1000, min_value=0, step=1)
strike_rate = st.number_input("Enter player strike rate:", value=125, min_value=0, step=1)

# Recommendation logic
if st.button("Recommend Players"):
    try:
        # Scale the input values
        input_scaled = scaler_model.transform([[runs, strike_rate]])
        # Predict the group
        group = kmeans_model.predict(input_scaled)[0]
        # Filter dataset based on the group
        recommended_players = d[d['Group'] == group]
        if not recommended_players.empty:
            st.write("Recommended Players:")
            st.dataframe(recommended_players[['PLAYER1', 'RUNS', 'SR']])
        else:
            st.warning("No players found for the given inputs.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
