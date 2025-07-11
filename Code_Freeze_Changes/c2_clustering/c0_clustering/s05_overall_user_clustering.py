import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import json
import os
import sys
import traceback
from c0_config.s00_config_paths import FINAL_INPUT_DATASET, CLUSTER_OUTPUT_DIR
import joblib


def load_config():
    """Load configuration from c0_config/config.json"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'c0_config', 'config.json')
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
        algo = config.get('clustering_algo', 'kmeans').lower()  # 'kmeans', 'dbscan', 'agg', 'gmm', 'all'
        # Load metadata to find all batch files
        meta_path = os.path.join(FINAL_INPUT_DATASET, "matrix_metadata.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        batch_files = [os.path.join(FINAL_INPUT_DATASET, os.path.basename(f)) for f in metadata.get("batch_files", [])]
        print(f"[INFO] Found {len(batch_files)} batch files for clustering.")
        # --- First pass: Fit SVD, scaler, selector ---
        svd = None
        scaler = None
        selector = None
        n_features = None
        all_X = []
        for i, batch_file in enumerate(batch_files):
            print(f"[INFO] [First pass] Processing batch {i+1}/{len(batch_files)}: {batch_file}")
            X_sparse = sparse.load_npz(batch_file)
            X = X_sparse.toarray()
            if n_features is None:
                nonzero_cols = X_sparse.getnnz(axis=0) > 0
                n_features = nonzero_cols.sum()
            X_sparse = X_sparse[:, :n_features]
            X = X[:, :n_features]
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
            all_X.append(X_high_var)
        X_all = np.vstack(all_X)
        # Save SVD, scaler, and selector for downstream use
        dimred_model_path = os.path.join(CLUSTER_OUTPUT_DIR, 'svd_model_kmeans.pkl')
        joblib.dump({'svd': svd, 'scaler': scaler, 'selector': selector}, dimred_model_path)
        print(f"[INFO] Saved SVD, scaler, selector to {dimred_model_path}")
        # --- Clustering ---
        algos_to_run = [algo] if algo != 'all' else ['kmeans', 'dbscan', 'agg', 'gmm']
        for algo_name in algos_to_run:
            print(f"[INFO] Running clustering algorithm: {algo_name}")
            algo_output_dir = os.path.join(CLUSTER_OUTPUT_DIR, algo_name)
            os.makedirs(algo_output_dir, exist_ok=True)
            cluster_labels_path = os.path.join(algo_output_dir, "user_cluster_labels.npy")
            if algo_name == 'kmeans':
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init=10, random_state=42)
                kmeans.fit(X_all)
                all_labels = kmeans.labels_
            elif algo_name == 'dbscan':
                dbscan = DBSCAN(eps=float(config.get('dbscan_eps', 0.5)), min_samples=int(config.get('dbscan_min_samples', 5)), n_jobs=-1)
                all_labels = dbscan.fit_predict(X_all)
            elif algo_name == 'agg':
                agg = AgglomerativeClustering(n_clusters=n_clusters)
                all_labels = agg.fit_predict(X_all)
            elif algo_name == 'gmm':
                gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
                all_labels = gmm.fit_predict(X_all)
            else:
                raise ValueError(f"Unknown clustering algorithm: {algo_name}")
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
