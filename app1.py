import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
def load_data():
    file_path = "amazon[1].csv"
    df = pd.read_csv(file_path)
    return df

# Data Preprocessing
def preprocess_data(df):
    df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['review_id'] = df['review_id'].apply(lambda x: len(str(x).split(',')))
    
    user_group = df.groupby('user_id').agg({
        'discounted_price': 'sum',  # Monetary
        'rating': 'mean',           # Avg Rating
        'review_id': 'sum'          # Frequency
    }).reset_index()
    
    user_group.rename(columns={
        'discounted_price': 'monetary',
        'rating': 'avg_rating_given',
        'review_id': 'frequency'
    }, inplace=True)
    
    np.random.seed(42)
    user_group['review_date'] = pd.to_datetime('2024-03-01') - pd.to_timedelta(np.random.randint(1, 365, size=len(user_group)), unit='D')
    user_group['recency'] = (pd.to_datetime('2024-03-01') - user_group['review_date']).dt.days
    
    return user_group[['user_id', 'recency', 'frequency', 'monetary']]

# Clustering function
def perform_clustering(rfm_df, n_clusters):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df['segment'] = kmeans.fit_predict(rfm_scaled)
    return rfm_df

# Streamlit UI
st.title("Amazon User Segmentation")

# Load data
df = load_data()
st.write("### Raw Dataset Sample")
st.dataframe(df.head())

# Preprocess data
rfm_df = preprocess_data(df)
st.write("### Processed RFM Data")
st.dataframe(rfm_df.head())

# Select number of clusters
n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

# Perform Clustering
segmented_users = perform_clustering(rfm_df, n_clusters)
st.write("### User Segmentation Results")
st.dataframe(segmented_users.head())
