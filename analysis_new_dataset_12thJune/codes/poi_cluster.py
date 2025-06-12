import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load POI features
features = pd.read_csv('poi_spatial_semantic_features.csv')

# Merge with places_sg.csv to get primary_category
places = pd.read_csv('analysis_new_dataset_12thJune/latest_dataset/places_sg.csv')
features = features.merge(places[['fsq_place_id', 'primary_category']], on='fsq_place_id', how='left')

# Merge with categories_sg.csv to get readable category name
categories = pd.read_csv('analysis_new_dataset_12thJune/latest_dataset/categories_sg.csv')
features = features.merge(categories[['category_id', 'category_name']], left_on='primary_category', right_on='category_id', how='left')
features = features.rename(columns={'category_name': 'primary_category_name'})

# Optionally, you can one-hot encode the readable primary_category_name and add to features
category_dummies = pd.get_dummies(features['primary_category_name'], prefix='cat')
X = pd.concat([features.drop(columns=['fsq_place_id', 'primary_category', 'category_id', 'primary_category_name']), category_dummies], axis=1)
poi_ids = features['fsq_place_id']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality with PCA
pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Cluster POIs (MiniBatchKMeans for scalability)
n_clusters = 50  # Adjust as needed
clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
labels = clustering.fit_predict(X_pca)

# Visualize with t-SNE
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_pca)
plt.figure(figsize=(10, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='tab20', s=2)
plt.title('POI Clusters (t-SNE projection)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig('poi_clusters_tsne.png')
plt.show()

# Save cluster assignments
poi_clusters = pd.DataFrame({'fsq_place_id': poi_ids, 'cluster': labels})
poi_clusters.to_csv('poi_clusters.csv', index=False)
print('POI clustering complete. Results saved to poi_clusters.csv and poi_clusters_tsne.png')
