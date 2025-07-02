import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import json
import os
import sys
import traceback
from s00_config_paths import MATRIX_PATH
import glob

def load_config():
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in config file: {e}", file=sys.stderr)
        return {}

def main():
    """Main function to handle configuration and execute the clustering"""
    try:
        # Load configuration
        config = load_config()
        svd_components = int(config.get('svd_components', 100))
        n_clusters = int(config.get('n_clusters', 10))
        batch_size = int(config.get('batch_size', 100))
        cluster_labels_path = config.get('output_cluster_labels', MATRIX_PATH.replace('.npy', '_user_cluster_labels.npy'))

        # Load metadata to find all batch files
        meta_path = os.path.join(os.path.dirname(MATRIX_PATH), "matrix_metadata.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        batch_files = [f.replace('.npz', '_flattened.npy') for f in metadata.get("batch_files", [])]
        print(f"[INFO] Found {len(batch_files)} batch files for clustering.")

        # --- First pass: Fit SVD, scaler, selector, and MiniBatchKMeans incrementally ---
        svd = None
        scaler = None
        selector = None
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init=10, random_state=42)
        n_features = None
        for i, batch_file in enumerate(batch_files):
            print(f"[INFO] [First pass] Processing batch {i+1}/{len(batch_files)}: {batch_file}")
            X = np.load(batch_file)
            X_sparse = sparse.csr_matrix(X)
            # Remove all-zero columns (features)
            if n_features is None:
                nonzero_cols = X_sparse.getnnz(axis=0) > 0
                n_features = nonzero_cols.sum()
            X_sparse = X_sparse[:, :n_features]
            # Fit/transform SVD
            if svd is None:
                svd = TruncatedSVD(n_components=min(svd_components, X_sparse.shape[1]-1), random_state=42)
                X_reduced = svd.fit_transform(X_sparse)
                print(f"[INFO] SVD explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_reduced)
                selector = VarianceThreshold(threshold=0.01)
                X_high_var = selector.fit_transform(X_scaled)
            else:
                X_reduced = svd.transform(X_sparse)
                X_scaled = scaler.transform(X_reduced)
                X_high_var = selector.transform(X_scaled)
            kmeans.partial_fit(X_high_var)
        print("[INFO] First pass complete. MiniBatchKMeans fitted.")

        # --- Second pass: Assign cluster labels batch-wise ---
        all_labels = []
        for i, batch_file in enumerate(batch_files):
            print(f"[INFO] [Second pass] Predicting batch {i+1}/{len(batch_files)}: {batch_file}")
            X = np.load(batch_file)
            X_sparse = sparse.csr_matrix(X)
            X_sparse = X_sparse[:, :n_features]
            X_reduced = svd.transform(X_sparse)
            X_scaled = scaler.transform(X_reduced)
            X_high_var = selector.transform(X_scaled)
            labels = kmeans.predict(X_high_var)
            all_labels.append(labels)
        all_labels = np.concatenate(all_labels)
        print(f"[INFO] Saving cluster labels to {cluster_labels_path}...")
        np.save(cluster_labels_path, all_labels)
        # Save user-to-cluster mapping as JSON and TXT
        user_cluster_json = cluster_labels_path.replace('.npy', '_user_to_cluster.json')
        user_cluster_txt = cluster_labels_path.replace('.npy', '_user_to_cluster.txt')
        user_to_cluster = {str(i): int(label) for i, label in enumerate(all_labels)}
        with open(user_cluster_json, 'w') as f:
            json.dump(user_to_cluster, f, indent=2)
        with open(user_cluster_txt, 'w') as f:
            for user, label in user_to_cluster.items():
                f.write(f"{user}\t{label}\n")
        print(f"[INFO] User-to-cluster mapping saved to {user_cluster_json} and {user_cluster_txt}")
        print("[INFO] Clustering complete!")
        return 0
    except Exception as e:
        print(f"[ERROR] Clustering failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
