import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Student Performance Clustering",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Clustering")
st.write("Hierarchical Clustering using Kaggle Dataset (Streamlit Safe)")

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
st.subheader("⚙️ Feature Selection")

features = st.multiselect(
    "Select score columns:",
    ["math score", "reading score", "writing score"],
    default=["math score", "reading score", "writing score"]
)

if len(features) < 2:
    st.warning("Please select at least two features.")
    st.stop()

X = df[features]

# -------------------------------------------------
# Scaling
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# Select Number of Clusters
# -------------------------------------------------
st.subheader("🔢 Choose Number of Clusters")
k = st.slider("Number of clusters", 2, 6, 3)

# -------------------------------------------------
# Hierarchical Clustering
# -------------------------------------------------
model = AgglomerativeClustering(
    n_clusters=k,
    linkage="ward"
)

df["Cluster"] = model.fit_predict(X_scaled)

# -------------------------------------------------
# Results
# -------------------------------------------------
st.subheader("📊 Clustered Student Data")
st.dataframe(df.head(10))

# -------------------------------------------------
# Cluster Summary
# -------------------------------------------------
st.subheader("📈 Average Scores per Cluster")

summary = df.groupby("Cluster")[features].mean()
st.dataframe(summary)

# -------------------------------------------------
# Visualization (Streamlit Native)
# -------------------------------------------------
st.subheader("📉 Cluster Visualization")

df_plot = df.copy()
df_plot["Cluster"] = df_plot["Cluster"].astype(str)

st.scatter_chart(
    df_plot,
    x=features[0],
    y=features[1],
    color="Cluster"
)

st.success("✅ App executed successfully without matplotlib!")
