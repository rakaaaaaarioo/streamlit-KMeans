import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.title("Aplikasi Klastering Wine Dengan KMeans")

wine = pd.read_csv('wine-clustering.csv')
x = wine.iloc[:]

st.header("Isi data set")
st.write(wine)
st.write("Dimensi data:", wine.shape)

# Proses KMeans
inertias = []
k_range = range(1, 11)

progress = st.progress(0)  # Progress bar

for i, k in enumerate(k_range):
    km = KMeans(n_clusters=k, random_state=42).fit(x)
    inertias.append(km.inertia_)
    progress.progress((i + 1) / len(k_range))

# Plot hasil
plt.figure(figsize=(12, 8))
plt.xlabel('Nilai k')
plt.ylabel('Sum of Squared Errors')
plt.plot(k_range, inertias, marker='o')
plt.grid()
st.pyplot(plt)

st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih Jumlah Klaster :", 2,10,3,1)
