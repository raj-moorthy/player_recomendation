import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the data
d = pd.read_excel('IPL 2024 Statistics _ Team and Player Stats.xlsx')

# Preprocessing (if not already done and saved)
# prompt: code for label encoder

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
d['AVG'] = le.fit_transform(d['AVG'])
print(d.head())

d.head()
scaler = StandardScaler()
features = d[['AVG', 'SR']]
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=2, random_state=42)
d['Group'] = kmeans.fit_predict(scaled_features)

# ... (rest of your code for saving the model and scaler) ...

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
    recommended = recommend_players(d, target_runs, target_strike_rate, scaler, kmeans)  
    st.write("Recommended Players:")
    st.dataframe(recommended)