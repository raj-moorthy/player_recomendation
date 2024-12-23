import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load models and dataset
try:
    scaler = joblib.load("scaler_model.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    d = pd.read_excel("IPL 2024 Statistics _ Team and Player Stats.xlsx")  # Adjust path if needed
except Exception as e:
    st.error(f"Error loading required files: {e}")
    st.stop()

# Verify dataset structure
if 'Group' not in d.columns:
    st.error("Dataset does not contain the 'Group' column. Recomputing clusters...")
    try:
        if {'AVG', 'SR'}.issubset(d.columns):
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(d[['AVG', 'SR']])
            kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust number of clusters
            d['Group'] = kmeans.fit_predict(features_scaled)
        else:
            st.error("Required columns ('AVG', 'SR') are missing from the dataset.")
            st.stop()
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        st.stop()

# Recommend players function
def recommend_players(runs, strike_rate, scaler, kmeans, dataset):
    input_scaled = scaler.transform([[runs, strike_rate]])
    group = kmeans.predict(input_scaled)[0]
    if 'Group' not in dataset.columns:
        st.error("Dataset is missing the 'Group' column.")
        return None
    if group not in dataset['Group'].unique():
        return None
    recommended_players = dataset[dataset['Group'] == group]
    return recommended_players[['PLAYER1', 'AVG', 'SR']]

# Streamlit app
st.title("IPL Player Recommendation")

target_runs = st.number_input("Enter the Average score", min_value=0, step=1)
target_strike_rate = st.number_input("Enter the strike rate", min_value=0, step=1)

if st.button("Recommend Players"):
    if target_runs < 0 or target_strike_rate < 0:
        st.error("Please enter valid non-negative inputs.")
    else:
        recommended = recommend_players(target_runs, target_strike_rate, scaler, kmeans, d)
        if recommended is None or recommended.empty:
            st.warning("No recommendations available for the given inputs.")
        else:
            st.write("Recommended Players:")
            st.dataframe(recommended)
