import pandas as pd
import numpy as np

# Assuming 'user_id' is present in the dataset
user_group = df.groupby('user_id').agg({
    'discounted_price': 'sum',  # Total spending (Monetary)
    'rating': 'mean',           # Avg rating given
    'review_id': 'count'        # Total reviews (Frequency)
}).reset_index()

# Rename columns for RFM analysis
user_group.rename(columns={
    'discounted_price': 'monetary',
    'rating': 'avg_rating_given',
    'review_id': 'frequency'
}, inplace=True)

# Create a random review date (For Recency calculation)
np.random.seed(42)
user_group['review_date'] = pd.to_datetime('2024-03-01') - pd.to_timedelta(np.random.randint(1, 365, size=len(user_group)), unit='D')

# Compute Recency
user_group['recency'] = (pd.to_datetime('2024-03-01') - user_group['review_date']).dt.days

# Keep only RFM columns
rfm_df = user_group[['user_id', 'recency', 'frequency', 'monetary']]

# Verify dataset
print(rfm_df.head())

# Standardize the RFM values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])

# Apply K-Means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm_df['segment'] = kmeans.fit_predict(rfm_scaled)

# Display the updated RFM dataset
print(rfm_df.head())
