import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Student Clustering", page_icon="🎓")

st.title("🎓 Student Performance Clustering")
st.write("Hierarchical Clustering using Kaggle Dataset")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("StudentsPerformance.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Feature selection
# -------------------------------
features = st.multiselect(
    "Select score columns",
    ["math score", "reading score", "writing score"],
    default=["math score", "reading score", "writing score"]
)

if len(features) < 2:
    st.warning("Select at least two features")
    st.stop()

X = df[features]

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Clustering
# -------------------------------
k = st.slider("Number of clusters", 2, 5, 3)

model = AgglomerativeClustering(n_clusters=k, linkage="ward")
df["Cluster"] = model.fit_predict(X_scaled)

# -------------------------------
# Results
# -------------------------------
st.subheader("Clustered Data")
st.dataframe(df.head(10))

st.subheader("Cluster Summary (Mean Scores)")
summary = df.groupby("Cluster")[features].mean().round(2)
st.dataframe(summary)

st.success("✅ App running successfully!")
