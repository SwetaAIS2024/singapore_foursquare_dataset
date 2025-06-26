import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from s00_config_paths import MATRIX_PATH

# Load sparse user matrix
SPARSE_PATH = MATRIX_PATH.replace('.npy', '_normalized_flattened_sparse.npz')
user_vectors_sparse = sparse.load_npz(SPARSE_PATH)
print(f"Loaded sparse user matrix: {user_vectors_sparse.shape}")

# Optional: Remove all-zero features (columns)
nonzero_cols = user_vectors_sparse.getnnz(axis=0) > 0
user_vectors_sparse = user_vectors_sparse[:, nonzero_cols]
print(f"After removing all-zero features: {user_vectors_sparse.shape}")

# Dimensionality reduction (TruncatedSVD for sparse data)
svd_components = 100  # You can adjust this
svd = TruncatedSVD(n_components=svd_components, random_state=42)
user_vectors_reduced = svd.fit_transform(user_vectors_sparse)
print(f"Reduced user vectors shape: {user_vectors_reduced.shape}")

# Clustering (KMeans)
n_clusters = 10  # You can adjust this
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(user_vectors_reduced)
print(f"KMeans clustering complete. Cluster sizes:")
for i in range(n_clusters):
    print(f"  Cluster {i}: {(labels == i).sum()} users")

# Save cluster labels
np.save(MATRIX_PATH.replace('.npy', '_user_cluster_labels.npy'), labels)
print(f"Saved user cluster labels to {MATRIX_PATH.replace('.npy', '_user_cluster_labels.npy')}")
