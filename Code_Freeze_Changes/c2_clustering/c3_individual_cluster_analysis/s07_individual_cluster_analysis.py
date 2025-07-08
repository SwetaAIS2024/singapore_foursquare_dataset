import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest, anderson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
from distfit import distfit
import os
import sys
import json
import gc
import warnings
from c0_config.s00_config_paths import CLUSTER_OUTPUT_DIR, FINAL_INPUT_DATASET

# --- Config loading ---
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../c0_config/config.json'))
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"[ERROR] Could not load config: {e}", file=sys.stderr)
        return {}

# Constants
SMALL_CLUSTER_THRESHOLD = 50  # Clusters smaller than this will use random sampling
DISTRIBUTIONS = ['norm', 'gamma', 'beta', 'lognorm']  # Base distributions to try
N_COMPONENTS_PCA = 10  # Number of PCs to compute

def load_data(algo_name):
    """Load cluster labels from the selected algorithm and user vectors from batch .npz files"""
    try:
        # Load cluster labels for the selected algorithm
        cluster_labels_path = os.path.join(CLUSTER_OUTPUT_DIR, algo_name, "user_cluster_labels.npy")
        if not os.path.exists(cluster_labels_path):
            raise FileNotFoundError(f"Cluster labels not found: {cluster_labels_path}")
        labels = np.load(cluster_labels_path)
        # Load user vectors from all batch .npz files in FINAL_INPUT_DATASET/c2_output
        meta_path = os.path.join(FINAL_INPUT_DATASET, "matrix_metadata.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        batch_files = [os.path.join(FINAL_INPUT_DATASET, os.path.basename(f)) for f in metadata.get("batch_files", [])]
        user_vectors = []
        for batch_file in batch_files:
            if not os.path.exists(batch_file):
                continue
            from scipy import sparse
            batch_vectors = sparse.load_npz(batch_file).toarray()
            user_vectors.append(batch_vectors)
        user_vectors = np.vstack(user_vectors)
        # Process vectors the same way as in clustering
        print("[INFO] Processing user vectors...")
        nonzero_cols = np.any(user_vectors != 0, axis=0)
        user_vectors = user_vectors[:, nonzero_cols]
        print(f"[INFO] Shape after removing zero features: {user_vectors.shape}")
        print("[INFO] Performing dimensionality reduction...")
        svd = TruncatedSVD(n_components=min(100, user_vectors.shape[1]-1), random_state=42)
        vectors_reduced = svd.fit_transform(user_vectors)
        explained_var = svd.explained_variance_ratio_.sum()
        print(f"[INFO] Explained variance ratio: {explained_var:.4f}")
        print("[INFO] Scaling features...")
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors_reduced)
        print("[INFO] Selecting features with high variance...")
        selector = VarianceThreshold(threshold=0.01)
        vectors_processed = selector.fit_transform(vectors_scaled)
        print(f"[INFO] Final processed shape: {vectors_processed.shape}")
        return labels, vectors_processed
    except Exception as e:
        raise RuntimeError(f"Failed to load data for {algo_name}: {e}")

def process_cluster(cluster_id, processed_vectors, labels, output_dir, scaled_rss_threshold=5, algo_name=None):
    """Process a single cluster's data"""
    print(f"\n[INFO] Processing Cluster {cluster_id}")
    
    # Extract cluster data
    cluster_mask = labels == cluster_id
    cluster_vectors = processed_vectors[cluster_mask]
    cluster_size = len(cluster_vectors)
    print(f"[INFO] Cluster size: {cluster_size}")
    
    # Agglomerative: sample if cluster is too large
    MAX_CLUSTER_SIZE_FOR_FIT_AGG = 50
    if algo_name == 'agg' and cluster_vectors.shape[0] > MAX_CLUSTER_SIZE_FOR_FIT_AGG:
        print(f"[INFO] Sampling {MAX_CLUSTER_SIZE_FOR_FIT_AGG} users from large AGG cluster of size {cluster_vectors.shape[0]}")
        idx = np.random.choice(cluster_vectors.shape[0], MAX_CLUSTER_SIZE_FOR_FIT_AGG, replace=False)
        cluster_vectors = cluster_vectors[idx]
        cluster_size = cluster_vectors.shape[0]

    print(f"[INFO] Fitting distributions on shape: {cluster_vectors.shape} (users x features)")
    # Create output directory for this cluster
    cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Skip small clusters
    if cluster_size < SMALL_CLUSTER_THRESHOLD:
        print(f"[WARNING] Cluster {cluster_id} is too small ({cluster_size} < {SMALL_CLUSTER_THRESHOLD})")
        print("[INFO] Will use random sampling for synthetic data generation")
        return {
            'cluster_id': int(cluster_id),
            'size': int(cluster_size),
            'method': 'random_sampling',
            'params': None
        }
    
    # Distribution Fitting on each dimension
    print("[INFO] Fitting distributions...")
    dimension_fits = []
    successful_fits = 0
    
    # Calculate variance for each dimension
    variances = np.var(cluster_vectors, axis=0)
    print(f"[INFO] Variance range: {np.min(variances):.4f} to {np.max(variances):.4f}")
    
    # Get dimensions with variance above 40th percentile (more lenient)
    var_threshold = np.percentile(variances, 40)  # Changed from 80 to 40 percentile
    significant_dims = np.where(variances > var_threshold)[0]
    
    print(f"[INFO] Analyzing {len(significant_dims)} dimensions with variance > {var_threshold:.4f}")
    
    for dim_idx, dim in enumerate(significant_dims):
        print(f"\n[INFO] Processing dimension {dim} (variance: {variances[dim]:.4f})...")
        dim_data = cluster_vectors[:, dim]
        # Debug: print shape and stats of dim_data
        print(f"[DEBUG] dim_data shape: {dim_data.shape}, min: {np.min(dim_data)}, max: {np.max(dim_data)}, any NaN: {np.isnan(dim_data).any()}, any inf: {np.isinf(dim_data).any()}")
        # Check for invalid/extreme values
        if np.isnan(dim_data).any() or np.isinf(dim_data).any() or np.abs(dim_data).max() > 1e6:
            print(f"[WARNING] Skipping dimension {dim} due to invalid/extreme values.")
            continue

        # Additional range and bin checks to prevent distfit OOM
        data_range = np.max(dim_data) - np.min(dim_data)
        # Stricter for AGG: skip if range > 30 or inter-percentile > 30
        if algo_name == 'agg':
            if data_range > 30:
                print(f"[WARNING] [AGG] Skipping dimension {dim} due to excessive data range ({data_range:.2f}).")
                continue
            lower, upper = np.percentile(dim_data, [2, 98])
            if upper - lower > 30:
                print(f"[WARNING] [AGG] Skipping dimension {dim} due to excessive inter-percentile range ({upper - lower:.2f}).")
                continue
            # Clip data for AGG
            dim_data = np.clip(dim_data, lower, upper)
        else:
            if data_range > 60 and len(dim_data) < 1000:
                print(f"[WARNING] Skipping dimension {dim} due to excessive data range ({data_range:.2f}) for small sample size.")
                continue
            lower, upper = np.percentile(dim_data, [2, 98])
            if upper - lower > 60:
                print(f"[WARNING] Skipping dimension {dim} due to excessive inter-percentile range ({upper - lower:.2f}).")
                continue
        # Remove problematic distributions for AGG if data has negative values
        distrs = ['norm', 'expon', 'uniform', 'cauchy', 'laplace', 'genextreme']
        if np.all(dim_data > 0):
            distrs += ['gamma', 'beta', 'lognorm', 'rayleigh']

        # Basic statistics for this dimension
        dim_mean = np.mean(dim_data)
        dim_std = np.std(dim_data)
        
        # Skip only if almost no variation
        if dim_std < 1e-8:  # Very lenient threshold
            print("[INFO] Skipping dimension due to near-zero variation")
            continue
        
        try:
            # Initialize distfit with tuned distribution set and max bins=5
            dfit = distfit(
                bins=5,  # Further reduce number of bins
                smooth=5,
                n_boots=0,  # no bootstrapping, to speed up the fitting
                method='parametric'  # Use parametric fit (supported by installed distfit)
            )
            # Fit distributions
            results = dfit.fit_transform(dim_data)
            # --- DEBUG: Print structure of results and summary ---
            print(f"[DEBUG] distfit results keys: {list(results.keys())}")
            print(f"[DEBUG] type(results['summary']): {type(results['summary'])}")
            print(f"[DEBUG] results['summary']: {results['summary']}")
            summary = results['summary']
            # Handle possible nested summary (distfit API changes)
            if isinstance(summary, dict) and 'summary' in summary:
                summary = summary['summary']
            # Now check for DataFrame and score column (distfit >=1.5)
            if summary is None or not hasattr(summary, 'columns') or 'score' not in summary.columns or summary.empty:
                print(f"[WARNING] No valid summary/score for dimension {dim}. Skipping.")
                continue
            best_row = summary.loc[summary['score'].idxmin()]
            rss = float(best_row['score'])
            scaled_rss = rss / (dim_std ** 2) if dim_std > 0 else rss
            if scaled_rss < scaled_rss_threshold:
                successful_fits += 1
                # Plot distribution fit with histogram
                plt.figure(figsize=(12, 6))
                plt.hist(dim_data, bins='auto', density=True, alpha=0.6, color='gray', label='Data')
                dfit.plot()
                plt.title(f'Distribution Fitting - Cluster {cluster_id}, Dimension {dim}\n'
                         f'RSS: {rss:.4f}, Scaled RSS: {scaled_rss:.4f}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(cluster_dir, f'distribution_fit_dim_{dim}.png'))
                plt.close()
                # Store fit results
                param_dict = {}
                if isinstance(best_row['params'], dict):
                    param_dict = {k: float(v) for k, v in best_row['params'].items()}
                elif hasattr(best_row['params'], '__iter__'):
                    param_dict = {str(i): float(v) for i, v in enumerate(best_row['params'])}
                dimension_fits.append({
                    'dimension': int(dim),
                    'distribution': {
                        'name': best_row['name'],
                        'parameters': param_dict,
                        'rss': float(rss),
                        'scaled_rss': float(scaled_rss),
                        'quality': 'good' if scaled_rss < 10 else 'fair' if scaled_rss < 50 else 'poor'
                    },
                    'summary_stats': {
                        'mean': float(dim_mean),
                        'std': float(dim_std),
                        'variance': float(variances[dim]),
                        'variance_percentile': float(stats.percentileofscore(variances, variances[dim]))
                    }
                })
            else:
                print(f"[INFO] Skipping dimension {dim} due to poor fit (scaled RSS: {scaled_rss:.4f})")
            
        except Exception as e:
            print(f"[WARNING] Failed to fit distribution for dimension {dim}: {str(e)}")
            continue
    
    print(f"\n[INFO] Successfully fitted {successful_fits} out of {len(significant_dims)} dimensions")
    
    # 4. Compute geometric properties
    centroid = np.mean(cluster_vectors, axis=0)
    covariance = np.cov(cluster_vectors.T)
    
    # Save results
    results = {
        'cluster_id': int(cluster_id),
        'size': int(cluster_size),
        'method': 'distribution_fitting',
        'successful_fits': successful_fits,
        'dimensions_analyzed': len(significant_dims),
        'dimensions': dimension_fits,
        'geometric_properties': {
            'centroid': [float(x) for x in centroid],
            'covariance_matrix': [[float(x) for x in row] for row in covariance]
        }
    }
    
    # Save to JSON
    with open(os.path.join(cluster_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def analyze_clusters():
    config = load_config()
    algo = config.get('clustering_algo', 'kmeans').lower()
    algos_to_analyze = [algo] if algo != 'all' else ['kmeans', 'dbscan', 'agg', 'gmm']
    base_scaled_rss_threshold = config.get('distfit_scaled_rss_threshold', 100)
    all_results = {}
    for algo_name in algos_to_analyze:
        print(f"\n[INFO] Running individual cluster analysis for: {algo_name}")
        # Use a higher threshold for agg
        if algo_name == 'agg':
            scaled_rss_threshold = 100
        else:
            scaled_rss_threshold = base_scaled_rss_threshold
        try:
            labels, user_vectors = load_data(algo_name)
            # Output dir for this algorithm
            output_dir = os.path.join(os.path.dirname(CLUSTER_OUTPUT_DIR), 'c3_individual_cluster_analysis', 'cluster_analysis', algo_name)
            os.makedirs(output_dir, exist_ok=True)
            # Process each cluster
            unique_labels = np.unique(labels)
            algo_results = []
            for cluster_id in unique_labels:
                try:
                    results = process_cluster(cluster_id, user_vectors, labels, output_dir, scaled_rss_threshold=scaled_rss_threshold, algo_name=algo_name)
                    algo_results.append(results)
                except Exception as e:
                    print(f"[ERROR] Failed to process cluster {cluster_id} for {algo_name}: {e}")
                    continue
            # Save overall results for this algorithm
            with open(os.path.join(output_dir, 'all_clusters_analysis.json'), 'w') as f:
                json.dump(algo_results, f, indent=4)
            create_summary_plots(algo_results, output_dir)
            all_results[algo_name] = algo_results
        except Exception as e:
            print(f"[ERROR] Could not analyze clusters for {algo_name}: {e}")
            continue
    print("\n[INFO] Individual cluster analysis complete!")
    return 0

def create_summary_plots(results, output_dir):
    """Create summary visualizations for all clusters"""
    # Extract key metrics
    data = []
    
    for r in results:
        cluster_info = {
            'Cluster': r['cluster_id'],
            'Size': r['size'],
            'Method': r['method'],
            'Dimensions_Fitted': 0,
            'Average_RSS': np.nan,
            'Best_Distribution': 'N/A',
            'Best_RSS': np.nan
        }
        
        if r['method'] == 'distribution_fitting':
            # Count successful fits
            cluster_info['Dimensions_Fitted'] = r['successful_fits']
            
            # Get RSS scores for successful fits
            if r['dimensions']:
                rss_values = [d['distribution']['rss'] for d in r['dimensions']]
                cluster_info['Average_RSS'] = np.mean(rss_values)
                
                # Find best fit
                best_fit = min(r['dimensions'], key=lambda x: x['distribution']['rss'])
                cluster_info['Best_Distribution'] = best_fit['distribution']['name']
                cluster_info['Best_RSS'] = best_fit['distribution']['rss']
        
        data.append(cluster_info)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(data)
    
    # Plot cluster sizes
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=summary_df, x='Cluster', y='Size')
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Users')
    
    # Add size labels on bars
    for i, v in enumerate(summary_df['Size']):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_sizes.png'))
    plt.close()
    
    # Plot distribution fitting success
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=summary_df, x='Cluster', y='Dimensions_Fitted')
    plt.title('Number of Successfully Fitted Dimensions per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Dimensions')
    
    # Add labels on bars
    for i, v in enumerate(summary_df['Dimensions_Fitted']):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_fits.png'))
    plt.close()
    
    # Create detailed summary table
    summary_df['Size_Percentage'] = (summary_df['Size'] / summary_df['Size'].sum() * 100).round(2)
    summary_df['Fit_Success_Rate'] = (summary_df['Dimensions_Fitted'] / summary_df['Size'] * 100).round(2)
    
    # Save detailed summary
    summary_path = os.path.join(output_dir, 'cluster_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[INFO] Saved detailed summary to {summary_path}")
    
    # Print summary
    print("\nCluster Analysis Summary:")
    for _, row in summary_df.iterrows():
        print(f"\nCluster {row['Cluster']}:")
        print(f"- Size: {row['Size']} users ({row['Size_Percentage']}% of total)")
        print(f"- Method: {row['Method']}")
        if row['Method'] == 'distribution_fitting':
            print(f"- Dimensions successfully fitted: {row['Dimensions_Fitted']}")
            if not np.isnan(row['Average_RSS']):
                print(f"- Average RSS: {row['Average_RSS']:.4f}")
            if row['Best_Distribution'] != 'N/A':
                print(f"- Best fitting distribution: {row['Best_Distribution']} (RSS: {row['Best_RSS']:.4f})")
            
    return summary_df

if __name__ == "__main__":
    sys.exit(analyze_clusters())