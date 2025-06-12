import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Read check-in logs
df = pd.read_csv('analysis_new_dataset_12thJune/latest_dataset/places_sg.csv')  # Update with your file path

# 2. Convert timestamps to hour-of-week (0–167)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour_of_week'] = df['timestamp'].dt.dayofweek * 24 + df['timestamp'].dt.hour

# 3. Build features for each user
user_features = []
for user_id, group in df.groupby('user_id'):
    # Temporal activity vector (168-dim)
    temporal_vec, _ = np.histogram(group['hour_of_week'], bins=168, range=(0, 168))
    # POI category histogram (top 10)
    top_cats = df['poi_category'].value_counts().index[:10]
    cat_hist = group['poi_category'].value_counts().reindex(top_cats, fill_value=0).values
    # Spatial entropy
    spatial_counts = group['poi_id'].value_counts()
    spatial_probs = spatial_counts / spatial_counts.sum()
    spatial_entropy = -np.sum(spatial_probs * np.log2(spatial_probs + 1e-9))
    # Temporal entropy
    temporal_probs = temporal_vec / (temporal_vec.sum() + 1e-9)
    temporal_entropy = -np.sum(temporal_probs * np.log2(temporal_probs + 1e-9))
    # Combine features
    feats = np.concatenate([temporal_vec, cat_hist, [spatial_entropy, temporal_entropy]])
    user_features.append((user_id, feats))

user_ids, feature_matrix = zip(*user_features)
feature_matrix = np.vstack(feature_matrix)

# 4. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_matrix)

# 5. Compute pairwise distance (cosine)
distance_matrix = cosine_distances(X_scaled)

# 6. Cluster users
n_clusters = 5  # Set as needed
clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
labels = clustering.fit_predict(distance_matrix)

# 7. Visualize with t-SNE
tsne = TSNE(n_components=2, random_state=42, metric='precomputed')
X_embedded = tsne.fit_transform(distance_matrix)
plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='tab10', s=10)
plt.title('User Clusters (t-SNE projection)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig('user_clusters_tsne.png')
plt.show()

# Save cluster assignments
user_clusters = pd.DataFrame({'user_id': user_ids, 'cluster': labels})
user_clusters.to_csv('user_clusters.csv', index=False)

# Optional: Visualize heatmap of average temporal activity per cluster
cluster_temporal = []
for k in range(n_clusters):
    idx = np.where(labels == k)[0]
    cluster_temporal.append(feature_matrix[idx, :168].mean(axis=0))
plt.figure(figsize=(12, 6))
plt.imshow(cluster_temporal, aspect='auto', cmap='viridis')
plt.colorbar(label='Avg. Activity')
plt.xlabel('Hour of Week')
plt.ylabel('Cluster')
plt.title('Cluster Temporal Activity Heatmap')
plt.tight_layout()
plt.savefig('cluster_temporal_heatmap.png')
plt.show()