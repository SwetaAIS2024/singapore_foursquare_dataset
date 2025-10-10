import os
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
import sys
import matplotlib.pyplot as plt
from c0_config.s00_config_paths import FINAL_INPUT_DATASET, CLUSTER_OUTPUT_DIR

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

def load_matrix(FINAL_INPUT_DATASET):
    meta_path = os.path.join(FINAL_INPUT_DATASET, "matrix_metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    batch_files = metadata["batch_files"]
    X_list = []
    for batch_file in batch_files:
        batch_path = os.path.join(FINAL_INPUT_DATASET, os.path.basename(batch_file))
        # Use scipy.sparse to load .npz sparse matrix
        X = sparse.load_npz(batch_path)
        #X = X.toarray()  # Convert to dense numpy array
        X_list.append(X)
    X_all = sparse.vstack(X_list)
    return X_all

def preprocess_matrix(X, svd_components):
    scaler = StandardScaler(with_mean=False)  # Fix: cannot center sparse matrices
    X_scaled = scaler.fit_transform(X)
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_reduced = svd.fit_transform(X_scaled)
    return X_reduced

def main():
    config = load_config()
    FINAL_INPUT_DATASET_PATH = FINAL_INPUT_DATASET
    svd_components_list = [100, 200, 300, 500]  # Add more values as needed
    n_clusters_list = [5, 8, 11, 15, 20]
    n_init_list = [10, 20]
    batch_size_list = [100, 200]
    results = []

    print("[INFO] Loading matrix...")
    X = load_matrix(FINAL_INPUT_DATASET_PATH)
    print(f"[INFO] Matrix shape: {X.shape}")

    for svd_components in svd_components_list:
        print(f"[INFO] Preprocessing matrix with svd_components={svd_components}...")
        X_proc = preprocess_matrix(X, svd_components)

        for n_clusters in n_clusters_list:
            for n_init in n_init_list:
                for batch_size in batch_size_list:
                    print(f"[INFO] Running MiniBatchKMeans: n_clusters={n_clusters}, n_init={n_init}, batch_size={batch_size}, svd_components={svd_components}")
                    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init, batch_size=batch_size, random_state=42)
                    labels = kmeans.fit_predict(X_proc)
                    silhouette = silhouette_score(X_proc, labels)
                    calinski = calinski_harabasz_score(X_proc, labels)
                    davies = davies_bouldin_score(X_proc, labels)
                    results.append({
                        "svd_components": svd_components,
                        "n_clusters": n_clusters,
                        "n_init": n_init,
                        "batch_size": batch_size,
                        "silhouette": silhouette,
                        "calinski_harabasz": calinski,
                        "davies_bouldin": davies
                    })
                    print(f"Silhouette: {silhouette:.4f}, Calinski-Harabasz: {calinski:.2f}, Davies-Bouldin: {davies:.4f}")

                    # Plot clusters using PCA for 2D visualization
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X_proc)
                    plt.figure(figsize=(8,6))
                    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='tab10', alpha=0.7)
                    plt.title(f"Clusters: n_clusters={n_clusters}, svd_components={svd_components}, batch_size={batch_size}")
                    plt.xlabel("PC1")
                    plt.ylabel("PC2")
                    plt.colorbar(scatter, ticks=range(n_clusters))
                    plot_name = f"clusters_svd{svd_components}_k{n_clusters}_b{batch_size}_init{n_init}.png"
                    plt.tight_layout()
                    plt.savefig(os.path.join(CLUSTER_OUTPUT_DIR, plot_name))
                    plt.close()

    results_df = pd.DataFrame(results)
    results_csv = os.path.join(CLUSTER_OUTPUT_DIR, "hyperparam_results_minibatchkmeans.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"[INFO] Saved hyperparameter tuning results to {results_csv}")

if __name__ == "__main__":
    main()
