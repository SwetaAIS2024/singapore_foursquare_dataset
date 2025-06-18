import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load embeddings
embedding_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Dimension_reduction_graph_construction/embeddings/user_pca_embeddings.csv'
df = pd.read_csv(embedding_path, index_col=0)

# Optionally normalize PCA embeddings (unit variance for each component)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.values), index=df.index, columns=df.columns)

# Run KMeans (choose n_clusters as needed)
n_clusters = 20  # You can change this as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(df_scaled.values)

# Save cluster assignments
out_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/user_kmeans_clusters.csv'
df['cluster'] = labels
df.to_csv(out_path)
print(f'Saved KMeans user clusters to {out_path}')

# 2D Visualization (first two PCA components)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('KMeans Clusters (PCA 2D)')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/kmeans_clusters_pca2d.png')
plt.show()

# 3D Visualization (first three PCA components)
if df.shape[1] >= 3:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=labels, cmap='tab10', alpha=0.7)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('KMeans Clusters (PCA 3D)')
    fig.colorbar(p, label='Cluster')
    plt.tight_layout()
    plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/kmeans_clusters_pca3d.png')
    plt.show()
else:
    print('Not enough PCA components for 3D plot.')
