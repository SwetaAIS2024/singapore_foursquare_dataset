import pandas as pd
import numpy as np
from tensorly.decomposition import tucker
import tensorly as tl
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import datetime
from sklearn.mixture import GaussianMixture

# 1. Load POI mapping (place_id -> category)
poi_map = pd.read_csv('analysis_set2/Tensor_factorisation_and_clustering/sg_place_id_to_category.csv')
placeid2cat = dict(zip(poi_map['place_id'], poi_map['category']))

# 2. Load check-in data
checkins = []
with open('singapore_checkins_filtered_with_locations_coord.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 6:
            continue
        user_id, place_id, timestamp, _, lat, lon = parts[:6]
        if place_id not in placeid2cat:
            continue
        # Parse time
        try:
            dt = datetime.datetime.strptime(timestamp, '%a %b %d %H:%M:%S %z %Y')
            hour = dt.hour
        except Exception:
            continue
        poi_cat = placeid2cat[place_id]
        checkins.append((user_id, hour, poi_cat, lat, lon))

# 3. Encode users, hours, POI categories
df = pd.DataFrame(checkins, columns=['user_id','hour','poi_cat','lat','lon'])
users = df['user_id'].unique()
hours = list(range(24))
poi_cats = df['poi_cat'].unique()
user_idx = {u: i for i, u in enumerate(users)}
poi_idx = {cat: i for i, cat in enumerate(poi_cats)}

# 4. Discretize spatial (grid cell)
lat_bins = np.linspace(df['lat'].astype(float).min(), df['lat'].astype(float).max(), 20)
lon_bins = np.linspace(df['lon'].astype(float).min(), df['lon'].astype(float).max(), 20)
df['grid_cell'] = pd.cut(df['lat'].astype(float), bins=lat_bins, labels=False).astype(str) + '_' + pd.cut(df['lon'].astype(float), bins=lon_bins, labels=False).astype(str)

# 5. Build tensor: users x hours x POI category
T = np.zeros((len(users), 24, len(poi_cats)), dtype=float)
for row in df.itertuples():
    T[user_idx[row.user_id], int(row.hour), poi_idx[row.poi_cat]] += 1

# 6. Tucker decomposition (increase rank for hour and POI dims)
core, factors = tucker(tl.tensor(T), rank=[len(users), 8, 8])  # Increased from 4,4 to 8,8
user_tensor_features = core.reshape((len(users), -1))

# 7. Spatial features (main grid cell for each user)
main_grid = df.groupby('user_id')['grid_cell'].agg(lambda x: x.value_counts().idxmax())
le_grid = LabelEncoder()
main_grid_encoded = le_grid.fit_transform(main_grid)

# 8. Clustering (combine tensor and spatial features)
# Use more tensor features for clustering (e.g., first 10 tensor features)
features = np.column_stack([user_tensor_features[:, :10], main_grid_encoded])

# KMeans clustering
kmeans = KMeans(n_clusters=9, random_state=42)
kmeans_clusters = kmeans.fit_predict(features)

# GMM clustering
gmm = GaussianMixture(n_components=9, random_state=42)
gmm_clusters = gmm.fit_predict(features)

# 9. Save results
result_df = pd.DataFrame({
    'user_id': users,
    'kmeans_cluster': kmeans_clusters,
    'gmm_cluster': gmm_clusters,
    'main_grid_cell_encoded': main_grid_encoded
})
for i in range(user_tensor_features.shape[1]):
    result_df[f'tensor_feat_{i+1}'] = user_tensor_features[:, i]
result_df.to_csv('analysis_set2/Tensor_factorisation_and_clustering/user_tensor_tucker_clustered_from_raw.csv', index=False)
print('Saved user tensor factorization and clustering results to user_tensor_tucker_clustered_from_raw.csv')

# --- Visualization: Compare KMeans and GMM clusters in 2D ---
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(user_tensor_features[:, 0], user_tensor_features[:, 1], c=kmeans_clusters, cmap='tab10', s=20, alpha=0.7)
plt.title('KMeans Clusters')
plt.xlabel('Tensor Feature 1')
plt.ylabel('Tensor Feature 2')
plt.subplot(1, 2, 2)
plt.scatter(user_tensor_features[:, 0], user_tensor_features[:, 1], c=gmm_clusters, cmap='tab10', s=20, alpha=0.7)
plt.title('GMM Clusters')
plt.xlabel('Tensor Feature 1')
plt.ylabel('Tensor Feature 2')
plt.tight_layout()
plt.savefig('analysis_set2/Tensor_factorisation_and_clustering/compare_kmeans_gmm_clusters.png')
plt.show()
