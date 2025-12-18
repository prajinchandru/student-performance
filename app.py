import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Student Performance Clustering", layout="centered")

st.title("🎓 Student Performance Clustering")
st.write("Hierarchical Clustering using Kaggle Dataset")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("StudentsPerformance.csv")

df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Select features
st.subheader("⚙️ Feature Selection")
features = st.multiselect(
    "Select score columns for clustering:",
    ['math score', 'reading score', 'writing score'],
    default=['math score', 'reading score', 'writing score']
)

if len(features) < 2:
    st.warning("Please select at least two features.")
else:
    X = df[features]

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    # Dendrogram
    st.subheader("🌳 Dendrogram")
    linked = linkage(scaled_data, method='ward')
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(linked, ax=ax)
    plt.xlabel("Students")
    plt.ylabel("Distance")
    st.pyplot(fig)

    # Number of clusters
    k = st.slider("Select number of clusters", 2, 6, 3)

    # Clustering
    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
    df['Cluster'] = hc.fit_predict(scaled_data)

    st.subheader("📊 Clustered Data")
    st.dataframe(df.head())

    st.success("Clustering completed successfully!")
