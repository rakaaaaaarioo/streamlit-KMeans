import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd

st.title("Aplikasi Klastering Wine Dengan KMeans")

wine = pd.read_csv('wine-clustering.csv')

x = wine.iloc[:]

st.header("Isi data set")
st.write(wine)

inertias = []
k_range = range (1,11)
for k in k_range:
    km = KMeans(n_clusters=k).fit(x)
    inertias.append(km.inertia_)
plt.figure(figsize=(12, 8))
plt.xlabel('nilai k')
plt.ylabel('sum of errors')
plt.plot(k_range,inertias)
plt.grid()

st.set_option('deprecation.showPyplotGlobalUse', False)
inertiass = st.pyplot
