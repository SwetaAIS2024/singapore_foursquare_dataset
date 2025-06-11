import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load clustered user features from tensor factorization on raw check-ins
df = pd.read_csv('analysis_set2/Tensor_factorisation_and_clustering/user_tensor_tucker_clustered_from_raw.csv')

x = df['tensor_feat_1'].values
y = df['tensor_feat_2'].values
kmeans_clusters = df['kmeans_cluster'].values
gmm_clusters = df['gmm_cluster'].values
spatial = df['main_grid_cell_encoded'].values

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x, y, c=kmeans_clusters, cmap='tab10', s=50, alpha=0.8)
plt.xlabel('Tensor Feature 1 (Temporal/POI)')
plt.ylabel('Tensor Feature 2 (Temporal/POI)')
plt.title('KMeans Clusters (2D: Temporal/POI)')
plt.colorbar(label='Cluster')

plt.subplot(1, 2, 2)
plt.scatter(x, y, c=gmm_clusters, cmap='tab10', s=50, alpha=0.8)
plt.xlabel('Tensor Feature 1 (Temporal/POI)')
plt.ylabel('Tensor Feature 2 (Temporal/POI)')
plt.title('GMM Clusters (2D: Temporal/POI)')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.savefig('analysis_set2/Tensor_factorisation_and_clustering/user_clusters_2d_kmeans_vs_gmm.png')
plt.show()

# Cluster size barplots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.countplot(x=kmeans_clusters, palette='tab10')
plt.xlabel('KMeans Cluster')
plt.ylabel('Number of Users')
plt.title('KMeans Cluster Sizes')
plt.subplot(1, 2, 2)
sns.countplot(x=gmm_clusters, palette='tab10')
plt.xlabel('GMM Cluster')
plt.ylabel('Number of Users')
plt.title('GMM Cluster Sizes')
plt.tight_layout()
plt.savefig('analysis_set2/Tensor_factorisation_and_clustering/user_cluster_sizes_kmeans_vs_gmm.png')
plt.show()

# Additional: Temporal vs Spatial features colored by cluster (KMeans and GMM)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x, spatial, c=kmeans_clusters, cmap='tab10', s=50, alpha=0.8)
plt.xlabel('Tensor Feature 1 (Temporal/POI)')
plt.ylabel('Main Grid Cell Encoded (Spatial)')
plt.title('KMeans: Temporal vs Spatial')
plt.colorbar(label='Cluster')

plt.subplot(1, 2, 2)
plt.scatter(x, spatial, c=gmm_clusters, cmap='tab10', s=50, alpha=0.8)
plt.xlabel('Tensor Feature 1 (Temporal/POI)')
plt.ylabel('Main Grid Cell Encoded (Spatial)')
plt.title('GMM: Temporal vs Spatial')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.savefig('analysis_set2/Tensor_factorisation_and_clustering/user_clusters_temporal_vs_spatial_kmeans_vs_gmm.png')
plt.show()

# Print variance of tensor features to diagnose collapse
feature_cols = [col for col in df.columns if col.startswith('tensor_feat_')]
variances = df[feature_cols].var().values
print('Variance of tensor features:')
for i, v in enumerate(variances):
    print(f'tensor_feat_{i+1}: {v}')

# Plot variance of tensor features
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(variances)+1), variances)
plt.xlabel('Tensor Feature Index')
plt.ylabel('Variance')
plt.title('Variance of Tensor Features')
plt.tight_layout()
plt.savefig('analysis_set2/Tensor_factorisation_and_clustering/tensor_feature_variance.png')
plt.show()

# Try higher Tucker rank for hour and POI dims (e.g., 8, 8 instead of 4, 4)
# This code is for TF_tucker_from_raw.py, not the visualization script
# Please update and rerun the tensor factorization script as well
