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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from c0_config.s00_config_paths import FINAL_INPUT_DATASET, CLUSTER_OUTPUT_DIR, CATEGORIES_XLSX

POST_CLUSTER_ANALYSIS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "c1_output")
os.makedirs(POST_CLUSTER_ANALYSIS_OUTPUT_DIR, exist_ok=True)

# Redirect stdout and stderr to a log file
log_file_path = os.path.join(POST_CLUSTER_ANALYSIS_OUTPUT_DIR, "overall_analysis.log")
log_file = open(log_file_path, "a")
sys.stdout = log_file
sys.stderr = log_file

def load_config():
    """Load configuration from c0_config/config.json"""
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../c0_config/config.json'))
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

def load_data_batchwise():
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
    cluster_labels_path = os.path.join(CLUSTER_OUTPUT_DIR, "user_cluster_labels.npy")
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

def get_all_vectors_labels_users():
    """Concatenate all batches into single arrays (if fits in memory)."""
    batches = list(load_data_batchwise())
    vectors = np.concatenate([b[0] for b in batches], axis=0)
    labels = np.concatenate([b[1] for b in batches], axis=0)
    users = np.concatenate([b[2] for b in batches], axis=0)
    return vectors, labels, users

def load_poi_categories():
    """Load POI category names from Excel file"""
    try:
        cats_df = pd.read_excel(CATEGORIES_XLSX)
        cat_col = 'POI Category in Singapore'
        return [c for c in cats_df[cat_col] if pd.notnull(c)]
    except Exception as e:
        print(f"[WARNING] Failed to load POI categories: {e}")
        print("[INFO] Using generic category names")
        return [f'Cat_{i}' for i in range(180)]  # Assuming 180 categories

def create_cluster_scatter_plot(user_vectors, labels, output_path):
    """Create and save 2D scatter plot of clusters using PCA"""
    print("[INFO] Creating cluster scatter plot...")
    
    # Process in batches to reduce memory usage
    total_samples = min(2000, user_vectors.shape[0])  # Limit total samples if dataset is very large
    
    # Randomly sample if we have more than total_samples
    if user_vectors.shape[0] > total_samples:
        indices = np.random.choice(user_vectors.shape[0], total_samples, replace=False)
        vectors_subset = user_vectors[indices]
        labels_subset = labels[indices]
    else:
        vectors_subset = user_vectors
        labels_subset = labels
    
    # Fit PCA on the subset
    print("[INFO] Fitting PCA...")
    pca = PCA(n_components=2)
    user_vec_2d = pca.fit_transform(vectors_subset)
    
    # Create plot
    plt.figure(figsize=(10,7))
    scatter = plt.scatter(user_vec_2d[:,0], user_vec_2d[:,1], 
                         c=labels_subset, cmap='tab10', alpha=0.7)
    plt.title('User Clusters (PCA 2D)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    cbar = plt.colorbar(scatter, ticks=np.unique(labels_subset))
    cbar.set_label('Cluster')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Clean up
    del user_vec_2d
    gc.collect()
    
    print(f"[INFO] Saved cluster scatter plot to {output_path}")

def create_cluster_heatmaps(user_vectors, labels, poi_categories, output_prefix):
    """Create heatmaps for each cluster showing POI category vs time patterns"""
    print("[INFO] Creating cluster heatmaps...")
    n_clusters = len(np.unique(labels))
    n_spatial, n_cat, n_time = 50, 180, 168  # Standard dimensions
    
    for cl in range(n_clusters):
        print(f"[INFO] Processing cluster {cl}...")
        idxs = np.where(labels == cl)[0]
        if len(idxs) == 0:
            continue
            
        # Select representative users (up to 10)
        rep_idxs = np.random.choice(idxs, min(10, len(idxs)), replace=False)
        
        # Process representative users in batches
        agg_vector = None
        for batch_idx in range(0, len(rep_idxs), 2):  # Process 2 users at a time
            end_idx = min(batch_idx + 2, len(rep_idxs))
            curr_idxs = rep_idxs[batch_idx:end_idx]
            
            # Load and process batch
            batch_vectors = user_vectors[curr_idxs].astype(np.float32)
            if agg_vector is None:
                agg_vector = batch_vectors.mean(axis=0)
            else:
                agg_vector += batch_vectors.mean(axis=0)
            
            # Clean up batch
            del batch_vectors
            gc.collect()
        
        # Average the accumulated sum
        agg_vector /= len(rep_idxs)
        
        # Create and save heatmap
        agg_3d = agg_vector.reshape((n_spatial, n_cat, n_time))
        agg_cat_time = agg_3d.sum(axis=0)
        
        plt.figure(figsize=(16,8))
        sns.heatmap(agg_cat_time, cmap='viridis', cbar=True)
        plt.title(f'Cluster {cl} Representative Users: POI Category vs Hour Heatmap')
        plt.xlabel('Hour of Week')
        plt.ylabel('POI Category')
        plt.yticks(ticks=np.arange(n_cat)+0.5, labels=poi_categories[:n_cat], fontsize=6)
        plt.tight_layout()
        
        output_path = f"{output_prefix}_cluster{cl}_10randuser_cat_time_heatmap.png"
        plt.savefig(output_path)
        plt.close()
        
        # Clean up
        del agg_vector, agg_3d, agg_cat_time
        gc.collect()

def analyze_cluster_poi_categories(user_matrix, labels, users, poi_categories, output_path):
    """Analyze and save top POI categories for representative users in each cluster"""
    print("[INFO] Analyzing cluster POI categories...")
    n_clusters = len(np.unique(labels))
    output_rows = []
    
    for cl in range(n_clusters):
        idxs = np.where(labels == cl)[0]
        if len(idxs) == 0:
            continue
            
        # Select representative users
        rep_idxs = np.random.choice(idxs, min(10, len(idxs)), replace=False)
        rep_user_ids = [users[i] for i in rep_idxs]
        rep_matrices = user_matrix[rep_idxs]
        
        # Find top categories
        agg = rep_matrices.mean(axis=0)
        top_cats = np.argsort(agg.sum(axis=1))[::-1][:10]
        top_cats = np.array(top_cats).flatten().tolist()
        top_cat_names = [poi_categories[int(i)] for i in top_cats]
        
        output_rows.append({
            'cluster': cl,
            'user_ids': ', '.join(map(str, rep_user_ids)),
            'top_poi_categories': ', '.join(top_cat_names)
        })
        
        # Clean up memory
        del rep_matrices, agg, top_cats, top_cat_names
        gc.collect()
    
    # Save results
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['cluster', 'user_ids', 'top_poi_categories']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[INFO] Saved cluster analysis to {output_path}")

def analyze_cluster_quality(user_vectors, labels, output_prefix):
    """Analyze clustering quality using multiple metrics and visualize results"""
    print("[INFO] Analyzing cluster quality...")
    n_users = user_vectors.shape[0]
    print(f"[INFO] Total number of users: {n_users}")
    
    # Process the vectors the same way as in clustering
    print("[INFO] Processing user vectors...")
    
    # 1. Dimensionality reduction
    print("[INFO] Performing dimensionality reduction...")
    svd = TruncatedSVD(n_components=min(100, user_vectors.shape[1]-1), random_state=42)
    vectors_reduced = svd.fit_transform(user_vectors)
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"[INFO] Explained variance ratio: {explained_var:.4f}")
    
    # 2. Scale features
    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors_reduced)
    
    # 3. Feature selection
    print("[INFO] Selecting features with high variance...")
    selector = VarianceThreshold(threshold=0.01)
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

def process_vectors_in_batches(vectors, batch_size=100):
    """Process large vector arrays in batches to reduce memory usage"""
    total_vectors = vectors.shape[0]
    processed = np.zeros((total_vectors, 2))
    
    for i in range(0, total_vectors, batch_size):
        end_idx = min(i + batch_size, total_vectors)
        batch = vectors[i:end_idx].copy()  # Load only a batch into memory
        yield batch, i, end_idx

def main():
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
            user_cluster_df = pd.DataFrame({'user_id': user_ids, 'cluster': labels})
            mapping_path = os.path.join(output_dir, "user_cluster_mapping.csv")
            user_cluster_df.to_csv(mapping_path, index=False)
            print(f"\n[INFO] Saved user-cluster mapping to {mapping_path}")
            # --- Compute clustering quality metrics on a random sample of 100 users ---
            try:
                sample_size = 100
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
                from sklearn.decomposition import PCA
                import matplotlib.pyplot as plt
                sample_vectors = []
                sample_labels = []
                for batch_vectors, batch_labels, _ in load_data_batchwise():
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
    sys.exit(main())
