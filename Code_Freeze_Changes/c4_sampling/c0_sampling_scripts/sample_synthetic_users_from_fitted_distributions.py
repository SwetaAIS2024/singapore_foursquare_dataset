"""
Script: sample_synthetic_users_from_fitted_distributions.py

This script samples synthetic user vectors from fitted distributions for each cluster, agnostic to the clustering algorithm and number of clusters.
- Loads per-cluster analysis_results.json files from the selected algorithm's cluster analysis directory.
- For each cluster with 'distribution_fitting', samples synthetic users by drawing one value per fitted dimension from the best-fit distribution.
- Pads missing dimensions with zeros if not all dimensions are fitted.
- Combines all synthetic users into a single matrix and saves as .npy and .csv (with dummy UserID column).

Usage:
- Set the clustering algorithm in the config file (c0_config/config.json or s00_config_paths.py).
- Run this script from the workspace root or as a module.
"""

import os
import json
import numpy as np
from scipy import stats

# --- Configurable paths ---
from c0_config.s00_config_paths import CLUSTER_OUTPUT_DIR

# User config: select algorithm (should match the clustering run)
algo = os.environ.get('CLUSTER_ALGO', 'kmeans')  # or set via config file

# Directory containing per-cluster analysis_results.json
ANALYSIS_DIR = os.path.join(
    os.path.dirname(__file__),
    '../../c2_clustering/c3_individual_cluster_analysis/cluster_analysis',
    algo
)

# Output directory
OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../c1_sampling_outputs'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of synthetic users per cluster (can be made configurable)
N_SYNTH = 100

# Helper: sample from a fitted scipy distribution
SUPPORTED_DISTS = set(stats._continuous_distns._distn_names + stats._discrete_distns._distn_names)
def sample_from_distribution(dist_name, params, size=1):
    if dist_name not in SUPPORTED_DISTS:
        raise ValueError(f"Distribution {dist_name} not supported by scipy.stats.")
    # Convert param dict to ordered tuple (by key order)
    if isinstance(params, dict):
        # Sort keys numerically if possible
        try:
            param_tuple = tuple(params[str(i)] for i in range(len(params)))
        except Exception:
            param_tuple = tuple(params.values())
    else:
        param_tuple = tuple(params)
    dist = getattr(stats, dist_name)
    return dist.rvs(*param_tuple, size=size)

# First pass: find global max dimension index across all clusters
max_dim_global = 0
cluster_dim_info = []  # Store (cluster_id, dims) for second pass
for fname in os.listdir(ANALYSIS_DIR):
    if fname.startswith('cluster_') and os.path.isdir(os.path.join(ANALYSIS_DIR, fname)):
        cluster_id = int(fname.split('_')[1])
        analysis_path = os.path.join(ANALYSIS_DIR, fname, 'analysis_results.json')
        if not os.path.exists(analysis_path):
            continue
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)
        if analysis.get('method') != 'distribution_fitting':
            continue
        dims = analysis.get('dimensions', [])
        if not dims:
            continue
        max_dim = max(d['dimension'] for d in dims)
        if max_dim > max_dim_global:
            max_dim_global = max_dim
        cluster_dim_info.append((cluster_id, dims))

synthetic_users = []
synthetic_labels = []

# Second pass: sample synthetic users with fixed vector length
for cluster_id, dims in cluster_dim_info:
    for _ in range(N_SYNTH):
        user_vec = np.zeros(max_dim_global + 1)
        for dim in dims:
            dim_idx = dim['dimension']
            dist_name = dim['distribution']['name']
            params = dim['distribution']['parameters']
            try:
                val = sample_from_distribution(dist_name, params, size=1)[0]
            except Exception as e:
                print(f"[WARNING] Could not sample from {dist_name} for cluster {cluster_id} dim {dim_idx}: {e}")
                val = 0.0
            user_vec[dim_idx] = val
        synthetic_users.append(user_vec)
        synthetic_labels.append(cluster_id)

synthetic_users = np.array(synthetic_users)
synthetic_labels = np.array(synthetic_labels)

# Save as .npy
npy_path = os.path.join(OUTPUT_DIR, f'synthetic_users_{algo}.npy')
np.save(npy_path, synthetic_users)
print(f"Saved synthetic users to {npy_path}")

# Save as .csv with dummy UserID
csv_path = os.path.join(OUTPUT_DIR, f'synthetic_users_{algo}.csv')
user_ids = np.arange(1, synthetic_users.shape[0] + 1)
all_with_ids = np.column_stack((user_ids, synthetic_labels, synthetic_users))
header = 'UserID,ClusterID,' + ','.join([f'Feature_{i}' for i in range(synthetic_users.shape[1])])
np.savetxt(csv_path, all_with_ids, delimiter=',', fmt='%d,%d' + ',%.6f'*synthetic_users.shape[1], header=header, comments='')
print(f"Saved synthetic users CSV to {csv_path}")
