import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Paths
pca_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Dimension_reduction_graph_construction/embeddings/user_pca_embeddings.csv'
cluster_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Clustering_Profiling/user_louvain_clusters.csv'

# Load data
pca_df = pd.read_csv(pca_path, index_col=0)
clu_df = pd.read_csv(cluster_path)

# Merge on user_id (index in PCA, user_id col in clusters)
pca_df = pca_df.reset_index().rename(columns={'index': 'user_id'})
merged = pd.merge(pca_df, clu_df, on='user_id')

# Normalize PCA components 0, 1, 2
#scaler = StandardScaler() # points are transformed to have mean=0 and std=1, and can have negative values
scaler = MinMaxScaler() # min-max scaling to [0, 1] range
merged[['0', '1', '2']] = scaler.fit_transform(merged[['0', '1', '2']])

# 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(merged['0'], merged['1'], merged['2'], c=merged['cluster'], cmap='tab20', s=20)
ax.set_xlabel('PCA 1 (normalized)')
ax.set_ylabel('PCA 2 (normalized)')
ax.set_zlabel('PCA 3 (normalized)')
ax.set_title('User Clusters in Normalized PCA Space (Components 1, 2, 3)')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
cbar = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=10, label='Cluster')
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Clustering_Profiling/user_clusters_pca123_scatter.png')
plt.close()
print('Saved 3D PCA cluster scatter plot to user_clusters_pca123_scatter.png')

# Compute cluster centroids in normalized PCA space
centroids = merged.groupby('cluster')[['0', '1', '2']].mean().reset_index()

# 3D scatter plot (all users, faded) + centroids (highlighted)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
# Plot all users (faded)
ax.scatter(merged['0'], merged['1'], merged['2'], c=merged['cluster'], cmap='tab20', s=10, alpha=0.2)
# Plot centroids (highlighted)
ax.scatter(centroids['0'], centroids['1'], centroids['2'], c=centroids['cluster'], cmap='tab20', s=120, edgecolor='k', marker='o', label='Cluster Centroid')
ax.set_xlabel('PCA 1 (normalized)')
ax.set_ylabel('PCA 2 (normalized)')
ax.set_zlabel('PCA 3 (normalized)')
ax.set_title('User Clusters in Normalized PCA Space (Centroids Highlighted)')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
cbar = fig.colorbar(ax.collections[0], ax=ax, shrink=0.8, aspect=10, label='Cluster')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Clustering_Profiling/user_clusters_pca123_centroids.png')
plt.close()
print('Saved 3D PCA cluster centroid scatter plot to user_clusters_pca123_centroids.png')
