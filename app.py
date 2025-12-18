import streamlit as st
import pandas as pd
import numpy as np

# Fix for Streamlit Cloud matplotlib error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Student Performance Clustering",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Clustering")
st.write("Hierarchical Clustering using Kaggle Dataset")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("StudentsPerformance.csv")

df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Feature Selection
# -------------------------------------------------
st.subheader("⚙️ Select Features for Clustering")

features = st.multiselect(
    "Choose score columns:",
    options=["math score", "reading score", "writing score"],
    default=["math score", "reading score", "writing score"]
)

if len(features) < 2:
    st.warning("⚠️ Please select at least two features.")
    st.stop()

X = df[features]

# -------------------------------------------------
# Data Scaling
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# Dendrogram
# -------------------------------------------------
st.subheader("🌳 Hierarchical Dendrogram")

linked = linkage(X_scaled, method="ward")

fig1, ax1 = plt.subplots(figsize=(8, 4))
dendrogram(linked, ax=ax1)
ax1.set_xlabel("Students")
ax1.set_ylabel("Distance")
st.pyplot(fig1)

# -------------------------------------------------
# Clustering
# -------------------------------------------------
st.subheader("🔢 Choose Number of Clusters")

k = st.slider("Number of clusters", min_value=2, max_value=6, value=3)

hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
clusters = hc.fit_predict(X_scaled)

df["Cluster"] = clusters

# -------------------------------------------------
# Results
# -------------------------------------------------
st.subheader("📊 Clustered Student Data")
st.dataframe(df.head(10))

st.success("✅ Hierarchical Clustering completed successfully!")

# -------------------------------------------------
# Cluster Summary
# -------------------------------------------------
st.subheader("📈 Cluster Summary (Average Scores)")

summary = df.groupby("Cluster")[features].mean().round(2)
st.dataframe(summary)
