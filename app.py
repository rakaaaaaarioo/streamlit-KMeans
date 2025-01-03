import streamlit as st
import pickle
import pandas as pd

with open('fruit_KMeans.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Aplikasi KMeans Clustering")
uploaded_file = st.file_uploader("wine-clustering.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv("wine-clustering.csv")
    predictions = model.predict(data)
    data['Cluster'] = predictions
    st.write(data)