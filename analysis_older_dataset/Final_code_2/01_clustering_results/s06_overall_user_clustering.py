import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import json
import os
from s00_config_paths import MATRIX_PATH

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Parameters from config
svd_components = int(config.get('svd_components', 100))  # fallback to 100 if not present
n_clusters = int(config.get('n_clusters', 10))  # fallback to 10 if not present

# Output path for cluster labels
CLUSTER_LABELS_PATH = config.get('output_cluster_labels', MATRIX_PATH.replace('.npy', '_user_cluster_labels.npy'))

# Load sparse user matrix
SPARSE_PATH = MATRIX_PATH.replace('.npy', '_normalized_flattened_sparse.npz')
user_vectors_sparse = sparse.load_npz(SPARSE_PATH)

# Remove all-zero features (columns)
nonzero_cols = user_vectors_sparse.getnnz(axis=0) > 0
user_vectors_sparse = user_vectors_sparse[:, nonzero_cols]

# Dimensionality reduction (TruncatedSVD for sparse data)
svd = TruncatedSVD(n_components=svd_components, random_state=42)
user_vectors_reduced = svd.fit_transform(user_vectors_sparse)

# Clustering (KMeans)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(user_vectors_reduced)

# Save cluster labels
np.save(CLUSTER_LABELS_PATH, labels)
