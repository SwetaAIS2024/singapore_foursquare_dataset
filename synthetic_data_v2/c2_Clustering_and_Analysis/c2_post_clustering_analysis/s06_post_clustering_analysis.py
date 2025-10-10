import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import json
import csv
import gc
import traceback

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from c0_Configuration.s00_config_paths import CATEGORIES_XLSX, FINAL_INPUT_DATASET, CLUSTER_OUTPUT_DIR, PLACE_ID_POI_CAT, CHECKINS_PATH
from c1_Data_Collection_and_Processing.c1_Feature_Extraction.c2_utils import load_config


POST_CLUSTER_ANALYSIS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "c1_output")
os.makedirs(POST_CLUSTER_ANALYSIS_OUTPUT_DIR, exist_ok=True)

# Redirect stdout and stderr to a log file
log_file_path = os.path.join(POST_CLUSTER_ANALYSIS_OUTPUT_DIR, "overall_analysis.log")
log_file = open(log_file_path, "a")
sys.stdout = log_file
sys.stderr = log_file


def load_data_batchwise(algo_name):
    """Load cluster labels, user vectors, and user IDs batch-wise, yielding batches."""
    # Load metadata for batch file info
    meta_path = os.path.join(FINAL_INPUT_DATASET, "matrix_metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    batch_files = [os.path.join(FINAL_INPUT_DATASET, os.path.basename(f)) for f in metadata.get("batch_files", [])]
    user_ids = metadata.get("user_ids", None)
    if user_ids is not None:
        user_ids = np.array(user_ids)
    else:
        user_ids = None
    # Load cluster labels (should be in user order)
    cluster_labels_path = os.path.join(CLUSTER_OUTPUT_DIR,algo_name, "user_cluster_labels.npy")
    labels = np.load(cluster_labels_path)
    # Yield batches
    user_idx = 0
    for batch_file in batch_files:
        if not os.path.exists(batch_file):
            continue
        from scipy import sparse
        batch_vectors = sparse.load_npz(batch_file).toarray()
        batch_size = batch_vectors.shape[0]
        batch_labels = labels[user_idx:user_idx+batch_size]
        if user_ids is not None:
            batch_user_ids = user_ids[user_idx:user_idx+batch_size]
        else:
            batch_user_ids = np.arange(user_idx, user_idx+batch_size)
        yield batch_vectors, batch_labels, batch_user_ids
        user_idx += batch_size

def analyze_cluster_quality(user_vectors, labels, output_prefix):
    """Analyze clustering quality using multiple metrics and visualize results"""
    print("[INFO] Analyzing cluster quality...")
    n_users = user_vectors.shape[0]
    print(f"[INFO] Total number of users: {n_users}")
    
    # Process the vectors the same way as in clustering
    print("[INFO] Processing user vectors...")
    
    # 1. Dimensionality reduction
    print("[INFO] Performing dimensionality reduction...")
    # svd = TruncatedSVD(n_components=min(100, user_vectors.shape[1]-1), random_state=42)
    config = load_config()
    svd_components = config.get('post_clustering_svd_components')
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    vectors_reduced = svd.fit_transform(user_vectors)
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"[INFO] Explained variance ratio: {explained_var:.4f}")
    
    # 2. Scale features
    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors_reduced)
    
    # 3. Feature selection
    print("[INFO] Selecting features with high variance...")
    var_th = float(config.get('post_clustering_variance_threshold'))
    selector = VarianceThreshold(threshold=var_th)
    vectors_processed = selector.fit_transform(vectors_scaled)
    print(f"[INFO] Final processed shape: {vectors_processed.shape}")
    
    # Calculate metrics on processed data
    print("\n[INFO] Calculating clustering metrics...")
    try:
        silhouette = silhouette_score(vectors_processed, labels)
        calinski = calinski_harabasz_score(vectors_processed, labels)
        davies = davies_bouldin_score(vectors_processed, labels)
        
        print("\nClustering Quality Metrics:")
        print(f"Silhouette Score: {silhouette:.4f} (range: [-1, 1], higher is better)")
        print(f"Calinski-Harabasz Score: {calinski:.4f} (higher is better)")
        print(f"Davies-Bouldin Score: {davies:.4f} (lower is better)")
        
    except Exception as e:
        print(f"[ERROR] Failed to calculate metrics: {e}")
        return None
    
    # Analyze cluster sizes and balance
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    total_points = len(labels)
    proportions = cluster_sizes / total_points
    
    print("\n[INFO] Cluster size distribution:")
    for cluster, size in cluster_sizes.items():
        print(f"Cluster {cluster}: {size} users ({(size/total_points)*100:.1f}%)")
    
    # Create cluster size distribution plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=cluster_sizes.index, y=proportions.values)
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Proportion of Users')
    
    # Add percentage labels on bars
    for i, v in enumerate(proportions):
        ax.text(i, v, f'{v:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_cluster_size_distribution.png")
    plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score',
                  'Largest Cluster Size', 'Smallest Cluster Size', 'Size Ratio'],
        'Value': [silhouette, calinski, davies,
                 cluster_sizes.max(), cluster_sizes.min(),
                 cluster_sizes.max() / cluster_sizes.min()],
        'Interpretation': [
            'Range: [-1, 1], higher is better. Values > 0.5 indicate good separation',
            'Higher values indicate better-defined clusters',
            'Lower values indicate better cluster separation',
            f'{(cluster_sizes.max()/total_points)*100:.1f}% of total users',
            f'{(cluster_sizes.min()/total_points)*100:.1f}% of total users',
            'Ratio > 20 indicates high imbalance'
        ]
    })
    metrics_df.to_csv(f"{output_prefix}_cluster_quality_metrics.csv", index=False)
    
    # Clean up
    gc.collect()
    
    return {
        'silhouette': silhouette,
        'calinski': calinski,
        'davies': davies,
        'cluster_sizes': cluster_sizes.tolist(),
        'size_std': np.std(cluster_sizes),
        'size_ratio': cluster_sizes.max() / cluster_sizes.min()
    }

def filer_category_distribution(dataframe_user):
    relevant_cats_df = pd.read_excel(CATEGORIES_XLSX)
    cat_col = 'POI Category in Singapore'
    yes_col = 'Relevant to use case '
    # Filtering the relevant categories based on the yes_col
    relevant_categories = [cat.strip().lower() for cat, flag in zip(relevant_cats_df[cat_col], relevant_cats_df[yes_col]) if str(flag).strip().lower() == 'yes' and cat and str(cat).strip()]
    relevant_categories = list(dict.fromkeys(relevant_categories))
    ordered_categories = [cat.title() for cat in relevant_categories]
    # ...existing code...
    dataframe_user['category'] = dataframe_user['category'].astype(str).str.strip().str.lower()
    # Filter to only relevant categories
    dataframe_user = dataframe_user[dataframe_user['category'].isin(relevant_categories)]
    # ...existing code...
    return dataframe_user

def analyze_poi_category_by_cluster(user_cluster_df, checkins_path, poi_path, output_dir=POST_CLUSTER_ANALYSIS_OUTPUT_DIR):
    """
    Analyze POI category distribution within each cluster.
    """
    # Load check-ins and POI category mapping
    cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
    checkins = pd.read_csv(checkins_path, sep='\t', names=cols)  # must have 'user_id', 'place_id'
    poi_df = pd.read_csv(poi_path)         # must have 'place_id', 'category'

    # Merge to get user_id, place_id, category
    user_poi = pd.merge(checkins, poi_df, on='place_id', how='inner')

    # --- Filter to only relevant categories ---
    user_poi = filer_category_distribution(user_poi)
    # ------------------------------------------

    # Merge with cluster assignments
    merged = pd.merge(user_poi, user_cluster_df, on='user_id', how='inner')

    # Group by cluster and category
    cluster_cat_counts = merged.groupby(['cluster', 'category']).size().reset_index(name='count')

    # Pivot for heatmap or table
    pivot = cluster_cat_counts.pivot(index='cluster', columns='category', values='count').fillna(0)

    # Save as CSV
    pivot.to_csv(os.path.join(output_dir, "poi_category_by_cluster.csv"))
    print(f"[INFO] Saved POI category-by-cluster table to {os.path.join(output_dir, 'poi_category_by_cluster.csv')}")

    # Optional: plot heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.5)
    plt.title("POI Category Distribution by Cluster")
    plt.xlabel("POI Category")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "poi_category_by_cluster_heatmap.png"))
    plt.close()
    print(f"[INFO] Saved POI category-by-cluster heatmap to {os.path.join(output_dir, 'poi_category_by_cluster_heatmap.png')}")

def Post_Clustering():
    try:
        # Load config to get clustering_algo
        config = load_config()
        algo = config.get('clustering_algo', 'kmeans').lower()  # 'kmeans', 'dbscan', 'agg', 'gmm', 'all'
        algos_to_analyze = [algo] if algo != 'all' else ['kmeans', 'dbscan', 'agg', 'gmm']
        for algo_name in algos_to_analyze:
            output_dir = os.path.join(POST_CLUSTER_ANALYSIS_OUTPUT_DIR, algo_name)
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Analyzing clustering results for: {algo_name} (results will be saved in {output_dir})")
            # Load metadata for cluster size and user IDs
            meta_path = os.path.join(FINAL_INPUT_DATASET, "matrix_metadata.json")
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            n_users = metadata['shape'][0]
            user_ids = metadata.get('user_ids', list(range(n_users)))
            cluster_labels_path = os.path.join(CLUSTER_OUTPUT_DIR, algo_name, "user_cluster_labels.npy")
            if not os.path.exists(cluster_labels_path):
                print(f"[WARNING] Cluster labels not found for {algo_name}: {cluster_labels_path}")
                continue
            labels = np.load(cluster_labels_path)
            n_clusters = len(np.unique(labels))
            from collections import Counter
            cluster_sizes = Counter(labels)
            total_points = n_users
            print("\n[INFO] Cluster size distribution:")
            for cluster in range(n_clusters):
                size = cluster_sizes.get(cluster, 0)
                print(f"Cluster {cluster}: {size} users ({(size/total_points)*100:.1f}%)")
            # debug prints 
            print("user_ids length:", len(user_ids))
            print("labels length:", len(labels))
            user_cluster_df = pd.DataFrame({'user_id': user_ids, 'cluster': labels})
            mapping_path = os.path.join(output_dir, "user_cluster_mapping.csv")
            user_cluster_df.to_csv(mapping_path, index=False)
            print(f"\n[INFO] Saved user-cluster mapping to {mapping_path}")

            # --- ADD POI CATEGORY-WISE CLUSTER ANALYSIS HERE ---
            # Example:
            # 1. Load user-POI/category mapping (e.g., from a CSV or DataFrame)
            # 2. Merge with user_cluster_df on 'user_id'
            # 3. Group by 'cluster' and 'category' to get counts/distributions
            # 4. Save or plot the results
            print(f"\n[INFO] Analyzing POI category distribution by cluster for {algo_name}...")
            analyze_poi_category_by_cluster(user_cluster_df, CHECKINS_PATH, PLACE_ID_POI_CAT, output_dir=output_dir)

            # --- END OF POI CATEGORY-WISE ANALYSIS ---


            # --- Compute clustering quality metrics on 100 users ---
            try:
                # sample_size = config.get('n_users') - OOM issue - 62 GB for (2000, 4536000)
                sample_size = 200
                sampled_vectors = []
                sampled_labels = []
                rng = np.random.default_rng(42)
                meta_path = os.path.join(FINAL_INPUT_DATASET, "matrix_metadata.json")
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                batch_files = [os.path.join(FINAL_INPUT_DATASET, os.path.basename(f)) for f in metadata.get("batch_files", [])]
                user_indices = np.arange(n_users)
                sample_indices = rng.choice(user_indices, size=min(sample_size, n_users), replace=False)
                sample_indices_set = set(sample_indices)
                user_idx = 0
                for batch_file in batch_files:
                    if not os.path.exists(batch_file):
                        continue
                    from scipy import sparse
                    batch_vectors = sparse.load_npz(batch_file).toarray()
                    batch_size = batch_vectors.shape[0]
                    batch_labels = labels[user_idx:user_idx+batch_size]
                    for i in range(batch_size):
                        global_idx = user_idx + i
                        if global_idx in sample_indices_set:
                            sampled_vectors.append(batch_vectors[i])
                            sampled_labels.append(batch_labels[i])
                    user_idx += batch_size
                    if len(sampled_vectors) >= sample_size:
                        break
                sampled_vectors = np.stack(sampled_vectors)
                sampled_labels = np.array(sampled_labels)
                metrics = analyze_cluster_quality(sampled_vectors, sampled_labels, os.path.join(output_dir, "cluster_quality_sample"))
                if metrics is not None:
                    print(f"\n[INFO] Clustering quality metrics (sample of 100 users) for {algo_name}:")
                    for k, v in metrics.items():
                        print(f"{k}: {v}")
            except Exception as e:
                print(f"[WARNING] Could not compute clustering quality metrics for {algo_name}: {e}")
            # Optionally, add more batch-wise visualizations/analysis here
            try:
                sample_vectors = []
                sample_labels = []
                for batch_vectors, batch_labels, _ in load_data_batchwise(algo_name):
                    n = min(20, batch_vectors.shape[0])
                    idx = np.random.choice(batch_vectors.shape[0], n, replace=False)
                    sample_vectors.append(batch_vectors[idx])
                    sample_labels.append(batch_labels[idx])
                sample_vectors = np.vstack(sample_vectors)
                sample_labels = np.concatenate(sample_labels)
                pca = PCA(n_components=2)
                user_vec_2d = pca.fit_transform(sample_vectors)
                plt.figure(figsize=(10,7))
                scatter = plt.scatter(user_vec_2d[:,0], user_vec_2d[:,1], c=sample_labels, cmap='tab10', alpha=0.7)
                plt.title(f'User Clusters (PCA 2D, batchwise sample) - {algo_name}')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                cbar = plt.colorbar(scatter, ticks=np.unique(sample_labels))
                cbar.set_label('Cluster')
                plt.tight_layout()
                scatter_plot_path = os.path.join(output_dir, "cluster_scatter.png")
                plt.savefig(scatter_plot_path)
                plt.close()
                print(f"[INFO] Saved batchwise PCA scatter plot to {scatter_plot_path}")
            except Exception as e:
                print(f"[WARNING] Could not create batchwise PCA scatter plot for {algo_name}: {e}")
        print("[INFO] Batch-wise post-clustering analysis complete!")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Post-clustering analysis failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(Post_Clustering())
    # Cleanup
    gc.collect()
    log_file.close()
