import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm

st.title("Aplikasi Klastering Wine Dengan KMeans")

wine = pd.read_csv('wine-clustering.csv')
x = wine.iloc[:]

st.header("Isi data set")
st.write(wine)
st.write("Dimensi data:", wine.shape)

inertias = []
k_range = range(1, 11)

progress = st.progress(0)  # Progress bar

#Tampilan Nilai K terbaik
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

# Slider untuk memilih jumlah kluster
st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih Jumlah Klaster :", 2, 10, 3, 1)

def k_means(n_clust):
    kmeans = KMeans(n_clusters=n_clust, random_state=42) 
    y_kmeans = kmeans.fit_predict(x)

    plt.figure(figsize=(12, 8))
    
    # Mendapatkan palet warna berdasarkan jumlah kluster
    colors = cm.get_cmap('tab10', n_clust)  # Menggunakan colormap 'tab10'
    
    for i in range(n_clust):
        plt.scatter(x.iloc[y_kmeans == i, 0], x.iloc[y_kmeans == i, 1], 
                    s=100, c=colors(i), label=f'Cluster {i + 1}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=300, c='black', label='Centroids')
    
    plt.title('Clusters of Wine')
    plt.xlabel('Index (Baris)')
    plt.ylabel('Value')
    plt.legend(loc='upper right')

    st.header('Cluster Plot')
    st.pyplot(plt)

k_means(clust)
