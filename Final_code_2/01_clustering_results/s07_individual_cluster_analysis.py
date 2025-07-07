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
from s00_config_paths import MATRIX_PATH
warnings.filterwarnings('ignore')

# Constants
SMALL_CLUSTER_THRESHOLD = 50  # Clusters smaller than this will use random sampling
DISTRIBUTIONS = ['norm', 'gamma', 'beta', 'lognorm']  # Base distributions to try
N_COMPONENTS_PCA = 10  # Number of PCs to compute

def load_data():
    """Load cluster labels, user vectors, and processed vectors"""
    try:
        # Load labels and raw vectors
        labels = np.load(MATRIX_PATH.replace('.npy', '_user_cluster_labels.npy'))
        user_vectors = np.load(MATRIX_PATH.replace('.npy', '_normalized_flattened.npy'))
        
        # Process vectors the same way as in clustering
        print("[INFO] Processing user vectors...")
        
        # 1. Remove all-zero features
        nonzero_cols = np.any(user_vectors != 0, axis=0)
        user_vectors = user_vectors[:, nonzero_cols]
        print(f"[INFO] Shape after removing zero features: {user_vectors.shape}")
        
        # 2. Dimensionality reduction
        print("[INFO] Performing dimensionality reduction...")
        svd = TruncatedSVD(n_components=min(100, user_vectors.shape[1]-1), random_state=42)
        vectors_reduced = svd.fit_transform(user_vectors)
        explained_var = svd.explained_variance_ratio_.sum()
        print(f"[INFO] Explained variance ratio: {explained_var:.4f}")
        
        # 3. Scale features
        print("[INFO] Scaling features...")
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors_reduced)
        
        # 4. Feature selection
        print("[INFO] Selecting features with high variance...")
        selector = VarianceThreshold(threshold=0.01)
        vectors_processed = selector.fit_transform(vectors_scaled)
        print(f"[INFO] Final processed shape: {vectors_processed.shape}")
        
        return labels, vectors_processed
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

def process_cluster(cluster_id, processed_vectors, labels, output_dir):
    """Process a single cluster's data"""
    print(f"\n[INFO] Processing Cluster {cluster_id}")
    
    # Extract cluster data
    cluster_mask = labels == cluster_id
    cluster_vectors = processed_vectors[cluster_mask]
    cluster_size = len(cluster_vectors)
    print(f"[INFO] Cluster size: {cluster_size}")
    
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
        
        # Basic statistics for this dimension
        dim_mean = np.mean(dim_data)
        dim_std = np.std(dim_data)
        
        # Skip only if almost no variation
        if dim_std < 1e-8:  # Very lenient threshold
            print("[INFO] Skipping dimension due to near-zero variation")
            continue
        
        try:
            # Initialize distfit with expanded distribution set and relaxed parameters
            dfit = distfit(
                distr=['norm', 'gamma', 'expon', 'uniform', 'beta', 'cauchy', 
                       'laplace', 'lognorm', 'rayleigh', 'genextreme'],
                bins='auto',
                smooth=5,  # Increased smoothing
                n_boots=100,  # Reduced bootstrapping for faster processing
                method='parametric'  # Use parametric method for more stable results
            )
            
            # Fit distributions
            results = dfit.fit_transform(dim_data)
            best_dist = results['model']
            
            # Get RSS and scale it by variance to make it relative
            rss = float(best_dist['RSS'])
            scaled_rss = rss / (dim_std ** 2) if dim_std > 0 else rss
            
            # Accept fit if scaled RSS is reasonable
            if scaled_rss < 100:  # Very lenient threshold
                successful_fits += 1
                
                # Plot distribution fit with histogram
                plt.figure(figsize=(12, 6))
                plt.hist(dim_data, bins='auto', density=True, alpha=0.6, color='gray', label='Data')
                dfit.plot(color='red', label=f'Fitted {best_dist["name"]}')
                plt.title(f'Distribution Fitting - Cluster {cluster_id}, Dimension {dim}\n'
                         f'RSS: {rss:.4f}, Scaled RSS: {scaled_rss:.4f}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(cluster_dir, f'distribution_fit_dim_{dim}.png'))
                plt.close()
                
                # Store fit results
                dimension_fits.append({
                    'dimension': int(dim),
                    'distribution': {
                        'name': best_dist['name'],
                        'parameters': {k: float(v) for k, v in best_dist['params'].items()},
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
    """Main function to analyze all clusters"""
    try:
        # Load data
        print("[INFO] Loading data...")
        labels, user_vectors = load_data()
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(MATRIX_PATH), 'cluster_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each cluster
        unique_labels = np.unique(labels)
        all_results = []
        
        for cluster_id in unique_labels:
            try:
                results = process_cluster(cluster_id, user_vectors, labels, output_dir)
                all_results.append(results)
            except Exception as e:
                print(f"[ERROR] Failed to process cluster {cluster_id}: {e}")
                continue
        
        # Save overall results
        with open(os.path.join(output_dir, 'all_clusters_analysis.json'), 'w') as f:
            json.dump(all_results, f, indent=4)
        
        # Create summary visualization
        create_summary_plots(all_results, output_dir)
        
        print("\n[INFO] Cluster analysis complete!")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}", file=sys.stderr)
        return 1

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